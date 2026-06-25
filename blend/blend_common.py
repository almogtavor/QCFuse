"""
Shared constants, utilities, and base class for blend test scripts.
"""

import time
from pathlib import Path
from typing import List, Tuple

from transformers import AutoConfig
import sglang as sgl
from sglang.srt.utils.triton_attention_score import warmup_triton_kernels

from qcfuse_config import (
    DEFAULT_CRITICAL_LAYERS,
    MODEL_TOP10_CRITICAL_LAYERS,
    SUPPORTED_BASELINES,
)


DEFAULT_DATA_DIR = Path(__file__).parent
# Frontend-only delimiter. The tokenizer path splits on this string before
# tokenization, so it must not rely on whitespace to avoid token merges.
BLEND_SEP = "<|blendsep|>"

# Model template configurations
TEMPLATES = {
    "llama": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "mistral": ("<s>[INST]", "", "[/INST]"),
    "qwen": (
        "<|im_start|>system\n",
        "<|im_end|>\n<|im_start|>user\n",
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
}

# Blend baseline configurations: (style, start, method)
BLEND_CONFIG = {
    "fullcomp": ("FULLCOMPUTE", 0, "none"),
    "ours": ("KVCOMPUTE", 0, "attn"),
    "fuserag": ("KVCOMPUTE", 0, "attn"),
    "prophetkv": ("KVCOMPUTE", 0, "attn"),
}


def _critical_model_key(model_name: str) -> str:
    name_lower = model_name.lower()
    if name_lower.startswith("qwen3-8b"):
        return "qwen3-8b"
    elif name_lower.startswith("qwen3-14b"):
        return "qwen3-14b"
    elif name_lower.startswith("qwen3-32b"):
        return "qwen3-32b"
    elif name_lower.startswith("llama"):
        return "llama3.1-8b"
    elif name_lower.startswith("mistral"):
        return "mistral-7b"
    raise ValueError(f"critical layers are not configured for model {model_name}")

def get_critical_layers(
    model_name: str, num_layers: int, critical_layers: int = DEFAULT_CRITICAL_LAYERS
) -> List[int]:
    """Return Top-K critical layers as 0-based indices.

    critical_layers=-1 selects every layer.
    """
    critical_layers = int(critical_layers)
    if critical_layers == -1:
        return list(range(num_layers))

    model_key = _critical_model_key(model_name)
    top_layers = MODEL_TOP10_CRITICAL_LAYERS[model_key]
    if critical_layers < 1 or critical_layers > len(top_layers):
        raise ValueError(
            f"critical_layers must be -1 or an integer in [1, {len(top_layers)}], "
            f"got {critical_layers}"
        )

    return _validate_explicit_layers(
        model_name,
        num_layers,
        top_layers[:critical_layers],
    )


def _validate_explicit_layers(
    model_name: str, num_layers: int, layers: List[int]
) -> List[int]:
    invalid_layers = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if invalid_layers:
        raise ValueError(
            f"critical layers {invalid_layers} are out of range for "
            f"model {model_name} with {num_layers} layers"
        )
    return layers


def _set_critical_layers(engine, model_name: str, layers: List[int]) -> None:
    layers = _validate_explicit_layers(
        model_name,
        engine._get_model_config()["num_layers"],
        [int(layer) for layer in layers],
    )
    engine.critical_layers = layers
    engine.attn_start, engine.attn_end = 0, max(layers) + 1


def set_ours_layers(
    engine, model_name: str, critical_layers: int = DEFAULT_CRITICAL_LAYERS
):
    """Set the critical layer Top-K used by the ours baseline."""
    num_layers = engine._get_model_config()["num_layers"]
    layers = get_critical_layers(
        model_name, num_layers, critical_layers=critical_layers
    )
    _set_critical_layers(engine, model_name, layers)


def set_fuserag_layers(engine, model_name: str):
    """Set the last-layer signal used by the FUSE-RAG baseline."""
    num_layers = engine._get_model_config()["num_layers"]
    _set_critical_layers(engine, model_name, [num_layers - 1])


def set_prophetkv_layers(engine, model_name: str):
    """Set all-layer signals used by the ProphetKV baseline."""
    num_layers = engine._get_model_config()["num_layers"]
    _set_critical_layers(engine, model_name, list(range(num_layers)))


class BlendEngineBase:
    """Base class with shared blend engine functionality."""

    def __init__(self, model_path: str, baseline: str = "ours"):
        self.model_name = Path(model_path).name.lower()
        self.model_path = model_path
        self.context_length = 16000
        self.attn_start = 0
        self.attn_end = -1
        self.critical_layers = None
        self._model_config = None

        self.llm = sgl.Engine(
            model_path=model_path,
            mem_fraction_static=0.6,
            context_length=self.context_length,
            tp_size=1,
            disable_cuda_graph=True,
            trust_remote_code=True,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            dtype="bfloat16",
            attention_backend="triton",
        )

        self.set_baseline(baseline)

    def set_baseline(self, baseline: str):
        """Switch baseline configuration."""
        if baseline not in SUPPORTED_BASELINES:
            raise ValueError(
                f"Unsupported baseline={baseline!r}; expected one of "
                f"{SUPPORTED_BASELINES}"
            )
        self.baseline = baseline
        self.critical_layers = None
        cfg = BLEND_CONFIG[baseline]
        self.first_style, self.start, self.method = cfg

    def _get_model_config(self) -> dict:
        """Get model architecture parameters (cached)."""
        if self._model_config is not None:
            return self._model_config

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        if getattr(config, "multi_query_attention", False):
            num_kv_heads = getattr(config, "multi_query_group_num", 1)
        else:
            num_kv_heads = getattr(
                config,
                "num_key_value_heads",
                getattr(
                    config,
                    "multi_query_group_num",
                    config.num_attention_heads,
                ),
            )

        self._model_config = {
            "head_dim": head_dim,
            "num_layers": getattr(config, "num_hidden_layers", 32),
            "num_heads": getattr(config, "num_attention_heads", 32),
            "num_kv_heads": num_kv_heads,
        }
        return self._model_config

    def _get_template(self) -> Tuple[str, str, str]:
        """Get model template based on model name."""
        for prefix, template in TEMPLATES.items():
            if self.model_name.startswith(prefix):
                return template
        return ("", "", "")

    def _build_prompt(
        self, system_prompt: str, docs: List[str], q_prompt: List[str], use_sep: bool
    ) -> Tuple[str, str]:
        """Build complete prompt from components."""
        sys_h, sys_e, asst_h = self._get_template()
        prefix = sys_h + system_prompt + sys_e
        suffix = "".join(q_prompt) + "\n\n## Answer\n" + asst_h

        if use_sep:
            query_sep = BLEND_SEP.join(q_prompt)
            return BLEND_SEP.join([prefix] + docs + [suffix]), query_sep
        return prefix + "".join(docs) + suffix, suffix

    def _timed_generate(self, prompt: str, params: dict, **kwargs) -> dict:
        """Run streaming generate and return generated text plus TTFT."""
        start = time.time()
        ttft, text = None, ""
        for out in self.llm.generate(prompt, params, stream=True, **kwargs):
            if ttft is None and out.get("text"):
                ttft = time.time() - start
            text = out.get("text", "")
        ttft = ttft or (time.time() - start)
        return {"text": text, "ttft": ttft}

    def warmup(self, num_warmup: int = 3):
        cfg = self._get_model_config()
        warmup_triton_kernels(
            head_dims=[cfg["head_dim"]],
            num_warmup_iters=3,
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
        )

        sys_h, sys_e, asst_h = self._get_template()
        warmup_prompt = (
            sys_h + "You are a helpful assistant." + sys_e
            + "Hello, how are you?" + asst_h
        )
        for _ in range(num_warmup):
            for _ in self.llm.generate(
                warmup_prompt,
                {"temperature": 0, "max_new_tokens": 1},
                stream=True,
                blend_style=None,
            ):
                pass

    def shutdown(self):
        if hasattr(self, "llm"):
            self.llm.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False
