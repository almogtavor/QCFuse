"""QCFuse OpenAI proxy — runs the REAL query-aware token-selective recompute
(blend/) behind /v1/chat/completions, so `sglang-qcfuse` actually exercises the
method from the paper (not a scaffold).

Why a proxy: SGLang's stock OpenAI route has no blend plumbing; the method is only
reachable via Engine.generate(blend_style=..., is_contextblend=True, critical_layers=...,
ratio=rho) as a stateful 3-call sequence (KVCOMPUTE -> QCOMPUTE -> DO_BLEND_FINISH),
single-request-at-a-time, tp=1, triton backend, no radix/cuda-graph. This proxy wraps
sgl.Engine, replays that sequence per request, and shapes an OpenAI response + Qwen
tool-call parse. rho (recompute ratio) = QCFUSE_RATIO env (default 0.2), the legolink-K analog.

Run: python3 qcfuse_proxy.py --model-path /model-cache --port 8000
"""
import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# blend/ uses bare intra-package imports (from blend_common import …), so it must be
# on sys.path — same as how blend/sglang_blend_ssd.py is run. QCFUSE_SRC=/opt/qcfuse-src.
_SRC = os.environ.get("QCFUSE_SRC", "/opt/qcfuse-src")
sys.path.insert(0, os.path.join(_SRC, "blend"))
sys.path.insert(0, _SRC)

from blend_common import BlendEngineBase, BLEND_SEP, get_critical_layers  # noqa: E402
from qcfuse_config import DEFAULT_CRITICAL_LAYERS  # noqa: E402

DIGEST_ZIP_PROMPT = "\n\nRepeat the previous context exactly."
RATIO = float(os.environ.get("QCFUSE_RATIO", "0.2"))       # rho: recompute ratio
DIGEST_RATIO = float(os.environ.get("QCFUSE_DIGEST_RATIO", "0.1"))  # KVzip anchor ratio
N_SINK = int(os.environ.get("QCFUSE_N_SINK", "4"))
TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class QCFuseProxyEngine(BlendEngineBase):
    """BlendEngineBase wired for on-the-fly single-request serving (no offline dataset)."""

    def __init__(self, model_path: str):
        # Replicate BlendEngineBase.__init__ but with env-tunable tp/mem_fraction/ctx —
        # Qwen3-32B (~64GB bf16) OOMs on 1 GPU at the base's mem_fraction=0.6, so allow
        # TP>1 and a bigger fraction. Blend REQUIRES: triton backend, radix+cudagraph off,
        # chunked_prefill=-1 (kept fixed).
        import sglang as sgl
        self.model_name = os.path.basename(model_path.rstrip("/")).lower()
        # model_name is used for critical-layer keying; /model-cache basename isn't "qwen3-32b".
        self.model_name = os.environ.get("SGLANG_MODEL_KEY", self.model_name)
        self.model_path = model_path
        self.context_length = int(os.environ.get("QCFUSE_CTX", "16000"))
        self.attn_start = 0
        self.attn_end = -1
        self.critical_layers = None
        self._model_config = None
        self.llm = sgl.Engine(
            model_path=model_path,
            mem_fraction_static=float(os.environ.get("QCFUSE_MEM_FRACTION", "0.85")),
            context_length=self.context_length,
            tp_size=int(os.environ.get("QCFUSE_TP", "2")),
            disable_cuda_graph=True,
            trust_remote_code=True,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            dtype="bfloat16",
            attention_backend="triton",
        )
        self.set_baseline("ours")
        # ContextBlend / query-source config (mirrors the runner's "ours" path).
        self.context_enhance = True
        self.context_cache_source = "query"
        self.digest_ratio = DIGEST_RATIO
        self.digest_index_method = "kvzip"
        self.context_n_sink = N_SINK
        n_layers = self._get_model_config()["num_layers"]
        self.critical_layers = get_critical_layers(
            self.model_name, n_layers, DEFAULT_CRITICAL_LAYERS
        )
        self._lock = threading.Lock()  # singletons -> single-flight

    # -- prompt assembly: OpenAI messages -> blendsep-delimited RAG-style prompt --
    def build_prompt(self, messages, tools):
        sys_h, sys_e, asst_h = self._get_template()
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        # everything except the final user turn = the reusable "context" chunk;
        # the final user turn = the query. Tool defs are appended to system.
        if tools:
            system += "\n\n# Tools\n" + json.dumps(tools)
        convo = [m for m in messages if m.get("role") != "system"]
        query = ""
        if convo and convo[-1].get("role") == "user":
            query = convo[-1]["content"]
            convo = convo[:-1]
        context = "\n".join(
            "%s: %s" % (m.get("role"), _content(m)) for m in convo
        )
        prefix = sys_h + system + sys_e
        suffix = query + "\n\n## Answer\n" + asst_h
        # one context chunk (+ KVzip reconstruction probe), like _build_augmented_prompt
        parts = [prefix, context, DIGEST_ZIP_PROMPT, suffix]
        return BLEND_SEP.join(parts)

    def _blend_args(self, blend_style, ratio, save_query_cache=False):
        args = {"blend_style": blend_style, "separator": BLEND_SEP,
                "start": self.start, "ratio": ratio, "method": self.method}
        if self.method == "attn":
            args["attn_start"] = self.attn_start
            args["attn_end"] = self.attn_end
        uses_cb = save_query_cache or (self.context_enhance and blend_style != "KVCOMPUTE")
        if uses_cb:
            args["is_contextblend"] = True
            args["context_cache_source"] = "query"
            args["digest_ratio"] = self.digest_ratio
            args["digest_index_method"] = self.digest_index_method
        if self.critical_layers:
            args["critical_layers"] = [int(x) for x in self.critical_layers]
        if save_query_cache or blend_style == "KVCOMPUTE":
            args["context_n_sink"] = self.context_n_sink
        return args

    def _drain(self, prompt, params, ssd, **kw):
        print("[blend] drain start style=%s" % kw.get("blend_style"), flush=True)
        for _ in self.llm.generate(prompt, params, stream=True,
                                   ssd_cache_path_chunk=ssd["c"],
                                   ssd_cache_path_query=ssd["q"], **kw):
            pass
        print("[blend] drain done style=%s" % kw.get("blend_style"), flush=True)

    def run(self, prompt, max_new_tokens, temperature):
        """The real 3-call blend sequence for one request. Returns (text, timings)."""
        with self._lock:  # process-global blend singletons -> serialize
            with tempfile.TemporaryDirectory(prefix="qcfuse_") as d:
                ssd = {"c": os.path.join(d, "chunk"), "q": os.path.join(d, "query")}
                os.makedirs(ssd["c"], exist_ok=True); os.makedirs(ssd["q"], exist_ok=True)
                t = {}
                t0 = time.perf_counter()
                # Phase-I: KVCOMPUTE (build per-chunk PIC cache + KVzip anchors)
                self._drain(prompt, {"temperature": 0, "max_new_tokens": 1},
                            ssd, **self._blend_args("KVCOMPUTE", 0.0, save_query_cache=True))
                t["kvcompute_ms"] = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                # Phase-II a: QCOMPUTE (query probe -> select P = topk(rho*N))
                self._drain(prompt, {"temperature": 0, "max_new_tokens": 0},
                            ssd, **self._blend_args("QCOMPUTE", RATIO))
                t["qcompute_ms"] = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                # Phase-II b: DO_BLEND_FINISH (sparse recompute of P across layers + decode)
                out_text = ""
                for chunk in self.llm.generate(
                        prompt, {"temperature": temperature, "max_new_tokens": max_new_tokens},
                        stream=True, ssd_cache_path_chunk=ssd["c"],
                        ssd_cache_path_query=ssd["q"],
                        **self._blend_args("DO_BLEND_FINISH", RATIO)):
                    out_text = chunk.get("text", out_text) if isinstance(chunk, dict) else out_text
                t["blend_finish_ms"] = (time.perf_counter() - t0) * 1000
                return out_text, t


ENGINE: QCFuseProxyEngine = None
app = FastAPI(title="QCFuse proxy")
# Dedicated single-thread pool: sgl.Engine shuts down asyncio's default executor at
# init, so to_thread breaks; a single worker also enforces the single-flight blend.
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
_POOL = ThreadPoolExecutor(max_workers=1)


def _content(m):
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c)


def _parse_tool_calls(text):
    calls = []
    for i, mt in enumerate(TOOLCALL_RE.finditer(text)):
        try:
            obj = json.loads(mt.group(1))
        except Exception:
            continue
        a = obj.get("arguments", {})
        calls.append({"id": "call_%d" % i, "type": "function",
                      "function": {"name": obj.get("name", ""),
                                   "arguments": a if isinstance(a, str) else json.dumps(a)}})
    content = TOOLCALL_RE.sub("", text).strip() or None
    return content, (calls or None)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": ENGINE.model_name, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat(body: dict):
    # Sync def -> FastAPI runs it in its own threadpool (like the plain-sync runner),
    # so sgl.Engine.generate never fights uvicorn's event loop.
    messages = body.get("messages", [])
    tools = body.get("tools")
    max_new = body.get("max_tokens") or 1024
    temperature = body.get("temperature", 0.7)
    prompt = ENGINE.build_prompt(messages, tools)
    try:
        text, timings = ENGINE.run(prompt, max_new, temperature)
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "tb": traceback.format_exc()[-800:]})
    content, tool_calls = _parse_tool_calls(text)
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    # blend_check: PROOF the real recompute ran (rho + phase timings land in results).
    return {
        "id": "chatcmpl-%s" % uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "model": ENGINE.model_name,
        "choices": [{"index": 0, "message": msg,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "blend_check": {"method": "qcfuse", "rho": RATIO,
                        "critical_layers": ENGINE.critical_layers,
                        "digest_ratio": DIGEST_RATIO, **timings},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/model-cache")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    ENGINE = QCFuseProxyEngine(args.model_path)
    uvicorn.run(app, host=args.host, port=args.port)
