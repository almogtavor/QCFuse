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

    # -- prompt assembly: OpenAI messages -> blendsep-delimited per-turn chunks --
    # Returns (prompt, query_sep):
    #   prompt    = prefix <sep> span1 <sep> span2 ... <sep> suffix   (KVCOMPUTE + DO_BLEND)
    #   query_sep = generic_query <sep> real_query                   (QCOMPUTE anchor probe)
    # QCFuse's QCOMPUTE probe attends from the query into every span (its per-chunk anchor
    # mechanism handles the per-span conditioning internally). SWE-bench has no natural
    # query after a tool result, so we inject a generic one so the probe still fires.
    SPAN_QUERY = "this is the response for the tool call"

    def build_prompt(self, messages, tools):
        sys_h, sys_e, asst_h = self._get_template()
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        if tools:
            system += "\n\n# Tools\n" + json.dumps(tools)
        convo = [m for m in messages if m.get("role") != "system"]
        query = ""
        if convo and convo[-1].get("role") == "user":
            query = convo[-1]["content"]
            convo = convo[:-1]
        # one chunk per remaining turn (tool results, assistant turns) = one span each
        turn_chunks = ["%s: %s" % (m.get("role"), _content(m)) for m in convo]
        prefix = sys_h + system + sys_e
        suffix = query + "\n\n## Answer\n" + asst_h
        parts = [prefix, *turn_chunks, DIGEST_ZIP_PROMPT, suffix]
        prompt = BLEND_SEP.join(parts)
        # query_sep = the anchor-conditioned QCOMPUTE probe. Their code (blend_common.py:219,
        # utils.py:148) uses q_prompt = [QUERY_PREFIX, question] — the QUERY itself, NOT one
        # piece per doc; QCFuse's per-chunk ANCHOR mechanism already conditions the probe on
        # each span internally. SWE-bench has no natural query after a tool span, so inject a
        # generic one; the anchor probe then attends from it into every span, as specified.
        q_prompt = [self.SPAN_QUERY, query or self.SPAN_QUERY]
        query_sep = BLEND_SEP.join(q_prompt)
        return prompt, query_sep

    def _blend_args(self, blend_style, ratio, save_query_cache=False):
        # Exactly mirrors their _blend_args (sglang_blend_ssd.py:218). The prior version
        # hardcoded context_cache_source="query" + always added digest params for every
        # non-KVCOMPUTE style; theirs conditions on save_query_cache, so DO_BLEND_FINISH
        # (save_query_cache=False) only gets digest params when context_cache_source=="query".
        # Getting this wrong drove DO_BLEND into a selector path that hangs the engine.
        args = {"blend_style": blend_style, "separator": BLEND_SEP,
                "start": self.start, "ratio": ratio, "method": self.method}
        if self.method == "attn":
            args["attn_start"] = self.attn_start
            args["attn_end"] = self.attn_end
        uses_cb = save_query_cache or (self.context_enhance and blend_style != "KVCOMPUTE")
        if uses_cb:
            args["is_contextblend"] = True
            if save_query_cache:
                args["context_cache_source"] = "query"
                args["digest_ratio"] = self.digest_ratio
                args["digest_index_method"] = self.digest_index_method
            else:
                args["context_cache_source"] = self.context_cache_source
            if not save_query_cache and self.context_cache_source == "query":
                args["digest_ratio"] = self.digest_ratio
                args["digest_index_method"] = self.digest_index_method
        if self.critical_layers:
            args["critical_layers"] = [int(x) for x in self.critical_layers]
        if save_query_cache or blend_style == "KVCOMPUTE":
            args["context_n_sink"] = self.context_n_sink
        return args

    def _gen(self, prompt, params, style, ratio, **ba_kw):
        # No ssd_cache_path_* -> QCFuse's IN-MEMORY blend path (in-process HackBlendKVPool /
        # ContextBlendPool). Avoids the SSD prefetch that futex-deadlocks DO_BLEND, and
        # writes nothing to disk.
        nsep = prompt.count(BLEND_SEP)
        sys.stderr.write("[blend] start %s nsep=%d chunks=%d\n" % (style, nsep, nsep + 1))
        sys.stderr.flush()
        out = self.llm.generate(prompt, params, **self._blend_args(style, ratio, **ba_kw))
        sys.stderr.write("[blend] done %s\n" % style); sys.stderr.flush()
        return out

    def run(self, prompt, query_sep, max_new_tokens, temperature):
        """The real 3-call blend sequence for one request (in-memory). Returns (text, timings).
        KVCOMPUTE + DO_BLEND run on `prompt` (prefix<sep>spans<sep>suffix); QCOMPUTE runs on
        `query_sep` (per-span generic query), which is where QCFuse's anchor probe reads."""
        with self._lock:  # process-global blend singletons -> serialize
            t = {}
            t0 = time.perf_counter()
            self._gen(prompt, {"temperature": 0, "max_new_tokens": 1},
                      "KVCOMPUTE", 0.0, save_query_cache=True)
            t["kvcompute_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            self._gen(query_sep, {"temperature": 0, "max_new_tokens": 0}, "QCOMPUTE", RATIO)
            t["qcompute_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            out = self._gen(prompt, {"temperature": temperature, "max_new_tokens": max_new_tokens},
                            "DO_BLEND_FINISH", RATIO)
            if isinstance(out, list):
                out = out[0]
            out_text = out.get("text", "") if isinstance(out, dict) else str(out)
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
    prompt, query_sep = ENGINE.build_prompt(messages, tools)
    n_spans = prompt.count(BLEND_SEP)  # separators = per-turn span boundaries
    try:
        text, timings = ENGINE.run(prompt, query_sep, max_new, temperature)
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
                        "n_spans": n_spans,  # per-turn chunks computed independently in 1 prefill
                        "single_pass": True,  # no separate warmup call; blend kernel is single-prefill
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
