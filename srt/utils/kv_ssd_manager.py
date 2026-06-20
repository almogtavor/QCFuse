"""
KV Cache SSD Manager - SSD I/O, async prefetch threads, synchronization.

Provides offline save (KVCOMPUTE -> SSD) and online load (SSD -> GPU with layer prefetch).
"""

import os
import json
import threading
import torch

from sglang.srt.utils.cache_blender_info import (
    ContextBlendPool,
    DEFAULT_DIGEST_RATIO,
    HackBlendKVPool,
)
from sglang.srt.utils.digest_index_manager import DigestIndexManager

# Global synchronization primitives
hack_pool_lock = threading.Lock()
context_pool_lock = threading.Lock()
task_b_event = threading.Event()

PACKED_KV_FORMAT = "sgblend_kv_packed_v1"
QUERY_CACHE_FORMAT = "sgblend_query_cache_v1"
_PACKED_FORMAT = PACKED_KV_FORMAT
_PACKED_BIN_NAME = "kv_packed.bin"
_PACKED_META_NAME = "kv_packed_meta.json"
_QUERY_BIN_NAME = "query_packed.bin"
_QUERY_META_NAME = "query_packed_meta.json"

_PACKED_DTYPE_MAP = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
for _packed_name, _torch_name in (
    ("U16", "uint16"),
    ("U32", "uint32"),
    ("U64", "uint64"),
):
    if hasattr(torch, _torch_name):
        _PACKED_DTYPE_MAP[_packed_name] = getattr(torch, _torch_name)

_PACKED_DTYPE_NAMES = {v: k for k, v in _PACKED_DTYPE_MAP.items()}


class KVSSDManager:
    # Mode flags
    _online = False
    _offline = False

    # Paths (set via configure())
    _sample_dir_chunk = ""
    _sample_dir_query = ""

    # Parameters
    _num_layers = 0
    _device = "cuda:0"
    # Thread state
    _layer_ready_events = []
    _loader_error = None
    _loader_error_lock = threading.Lock()
    _query_meta_cache = None
    _query_meta_cache_dir = ""

    # Pre-allocated pinned host buffers and CUDA streams keyed by load kind.
    # Query and chunk prefetch can run concurrently, so they must not share a
    # mutable pinned source buffer.
    _transfer_pools = {}

    # ================================================================
    # Configuration
    # ================================================================

    @classmethod
    def configure(
        cls,
        online=False,
        offline=False,
        sample_dir_chunk="",
        sample_dir_query="",
        num_layers=0,
        device="cuda:0",
    ):
        cls._online = online
        cls._offline = offline
        cls._sample_dir_chunk = sample_dir_chunk
        cls._sample_dir_query = sample_dir_query
        cls._num_layers = num_layers
        cls._device = device
        cls._layer_ready_events = [threading.Event() for _ in range(num_layers)]
        cls._query_meta_cache = None
        cls._query_meta_cache_dir = ""
        task_b_event.clear()
        cls._clear_loader_error()
        if offline:
            DigestIndexManager.clear()
        cls._ensure_transfer_pool("chunk", device)
        cls._ensure_transfer_pool("query", device)
        cls._ensure_transfer_pool("query_digest", device)
        cls._ensure_transfer_pool("query_critical", device)

    @classmethod
    def is_online(cls):
        return cls._online

    @classmethod
    def is_offline(cls):
        return cls._offline

    @classmethod
    def reset(cls):
        """Full reset after DO_BLEND_FINISH: clear all state so next sample
        triggers fresh initialization via is_online() check."""
        cls._online = False
        cls._offline = False
        task_b_event.clear()
        cls._clear_loader_error()
        cls._query_meta_cache = None
        cls._query_meta_cache_dir = ""
        for evt in cls._layer_ready_events:
            evt.clear()

    @classmethod
    def cleanup_runtime_state(cls):
        """Release per-request Blend/SSD runtime state.

        The reusable pinned I/O buffers are intentionally kept to avoid repeated
        pinned allocations across SSD-backed requests.
        """
        with hack_pool_lock:
            HackBlendKVPool.clear()
        with context_pool_lock:
            ContextBlendPool.clear()
        DigestIndexManager.clear()
        cls.reset()

    @classmethod
    def reset_layer_events(cls):
        """Clear layer-ready events before DO_BLEND prefetch."""
        for evt in cls._layer_ready_events:
            evt.clear()

    @classmethod
    def _clear_loader_error(cls):
        with cls._loader_error_lock:
            cls._loader_error = None

    @classmethod
    def _record_loader_error(cls, exc):
        with cls._loader_error_lock:
            if cls._loader_error is None:
                cls._loader_error = exc

    @classmethod
    def _raise_loader_error(cls):
        with cls._loader_error_lock:
            error = cls._loader_error
        if error is not None:
            raise RuntimeError("SSD KV loader failed") from error

    @classmethod
    def wait_layer_ready(cls, layer_id):
        cls._layer_ready_events[layer_id].wait()
        cls._raise_loader_error()

    @classmethod
    def wait_task_b(cls):
        task_b_event.wait()
        cls._raise_loader_error()

    @classmethod
    def _set_layer_ready(cls, layer_id):
        cls._layer_ready_events[layer_id].set()

    @classmethod
    def _set_layers_ready(cls, layer_ids):
        for layer_id in layer_ids:
            if 0 <= layer_id < len(cls._layer_ready_events):
                cls._set_layer_ready(layer_id)

    @classmethod
    def _start_loader_thread(
        cls,
        name,
        load_fn,
        release_layer_ids=(),
        release_task_b=False,
    ):
        release_layer_ids = list(release_layer_ids)

        def _worker():
            try:
                load_fn()
            except Exception as exc:
                cls._record_loader_error(exc)
                cls._set_layers_ready(release_layer_ids)
                if release_task_b:
                    task_b_event.set()
            else:
                if release_task_b:
                    task_b_event.set()

        return threading.Thread(target=_worker, name=name, daemon=True)

    # ================================================================
    # Offline: Save to SSD
    # ================================================================

    @staticmethod
    def _dtype_to_packed_name(dtype):
        if dtype not in _PACKED_DTYPE_NAMES:
            raise ValueError(f"Unsupported packed KV dtype: {dtype}")
        return _PACKED_DTYPE_NAMES[dtype]

    @staticmethod
    def _packed_name_to_dtype(name):
        if name not in _PACKED_DTYPE_MAP:
            raise ValueError(f"Unsupported packed KV dtype name: {name}")
        return _PACKED_DTYPE_MAP[name]

    @classmethod
    def _tensor_nbytes(cls, shape, dtype):
        return cls._shape_numel(shape) * torch.empty((), dtype=dtype).element_size()

    @staticmethod
    def _write_tensor_bytes(f, tensor):
        byte_view = tensor.view(torch.uint8).reshape(-1)
        if byte_view.numel() == 0:
            return
        f.write(memoryview(byte_view.numpy()))

    @classmethod
    def _save_packed_kv(
        cls,
        sample_dir,
        num_layers,
        get_layer_kv,
        token_indices=None,
        token_indices_by_layer=None,
    ):
        os.makedirs(sample_dir, exist_ok=True)
        bin_path = os.path.join(sample_dir, _PACKED_BIN_NAME)
        meta_path = os.path.join(sample_dir, _PACKED_META_NAME)
        meta = {
            "format": _PACKED_FORMAT,
            "num_layers": int(num_layers),
            "layers": {},
        }

        offset = 0
        token_indices_t = None
        with open(bin_path, "wb") as f:
            for layer_id in range(num_layers):
                full_k, full_v = get_layer_kv(layer_id)
                if full_k is None or full_v is None:
                    raise ValueError(f"Missing KV tensor for layer {layer_id}")

                layer_token_indices = None
                if token_indices_by_layer is not None:
                    layer_token_indices = token_indices_by_layer[layer_id]
                elif token_indices is not None:
                    layer_token_indices = token_indices

                if layer_token_indices is not None:
                    if token_indices_by_layer is not None:
                        token_indices_t = torch.as_tensor(
                            layer_token_indices, dtype=torch.long, device=full_k.device
                        )
                    elif token_indices_t is None or token_indices_t.device != full_k.device:
                        token_indices_t = torch.as_tensor(
                            layer_token_indices, dtype=torch.long, device=full_k.device
                        )
                    full_k = full_k.index_select(0, token_indices_t)
                    full_v = full_v.index_select(0, token_indices_t)

                layer_meta = {}
                for name, tensor in (("k", full_k), ("v", full_v)):
                    tensor_cpu = tensor.detach().contiguous().cpu()
                    shape = list(tensor_cpu.shape)
                    dtype_name = cls._dtype_to_packed_name(tensor_cpu.dtype)
                    nbytes = cls._tensor_nbytes(shape, tensor_cpu.dtype)
                    layer_meta[name] = {
                        "offset": offset,
                        "nbytes": nbytes,
                        "shape": shape,
                        "dtype": dtype_name,
                    }
                    cls._write_tensor_bytes(f, tensor_cpu)
                    offset += nbytes
                    del tensor_cpu
                meta["layers"][str(layer_id)] = layer_meta

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def save_all_chunk_cache(cls, sample_dir, num_layers, token_indices=None):
        """Save HackBlendKVPool to SSD as one packed file per sample."""
        cls._save_packed_kv(
            sample_dir, num_layers, HackBlendKVPool.get_kv, token_indices=token_indices
        )

    @classmethod
    def _write_query_layer(cls, f, offset, layer_id, full_k, full_v, token_indices=None):
        if full_k is None or full_v is None:
            raise ValueError(f"Missing KV tensor for query cache layer {layer_id}")
        if token_indices is not None:
            token_indices_t = torch.as_tensor(
                token_indices, dtype=torch.long, device=full_k.device
            )
            full_k = full_k.index_select(0, token_indices_t)
            full_v = full_v.index_select(0, token_indices_t)

        layer_meta = {}
        for name, tensor in (("k", full_k), ("v", full_v)):
            tensor_cpu = tensor.detach().contiguous().cpu()
            shape = list(tensor_cpu.shape)
            dtype_name = cls._dtype_to_packed_name(tensor_cpu.dtype)
            nbytes = cls._tensor_nbytes(shape, tensor_cpu.dtype)
            layer_meta[name] = {
                "offset": offset,
                "nbytes": nbytes,
                "shape": shape,
                "dtype": dtype_name,
            }
            cls._write_tensor_bytes(f, tensor_cpu)
            offset += nbytes
            del tensor_cpu
        return layer_meta, offset

    @classmethod
    def save_all_query_cache(
        cls,
        sample_dir,
        num_digest_layers,
        digest_token_indices_by_layer,
        critical_layers,
        critical_token_indices=None,
    ):
        """Save QCOMPUTE cache as one packed file.

        The query cache contains materialized digest KV for QCOMPUTE context
        plus the raw critical chunk KV needed by ATTN selection.
        """
        os.makedirs(sample_dir, exist_ok=True)
        bin_path = os.path.join(sample_dir, _QUERY_BIN_NAME)
        meta_path = os.path.join(sample_dir, _QUERY_META_NAME)

        digest_meta, indices_by_method = DigestIndexManager.export_payload()
        method = DigestIndexManager.normalize_method(
            digest_meta.get("digest_index_method")
        )
        critical_layers = [int(x) for x in (critical_layers or [])]
        query_meta = {
            "format": QUERY_CACHE_FORMAT,
            "digest_index_version": digest_meta.get("digest_index_version"),
            "available_methods": digest_meta.get("available_methods", [method]),
            "num_layers": int(num_digest_layers),
            "digest_ratio": digest_meta.get("digest_ratio"),
            "digest_index_method": method,
            "critical_layers": critical_layers,
            "qcompute_end": int(num_digest_layers),
            "materialized_digest": True,
            "metadata": digest_meta,
            "indices_by_method": indices_by_method,
            "digest": {
                "num_layers": int(num_digest_layers),
                "layers": {},
            },
            "critical": {
                "layer_ids": critical_layers,
                "layers": {},
            },
        }

        offset = 0
        with open(bin_path, "wb") as f:
            for layer_id in range(int(num_digest_layers)):
                token_indices = digest_token_indices_by_layer[layer_id]
                full_k, full_v = HackBlendKVPool.get_kv(layer_id)
                layer_meta, offset = cls._write_query_layer(
                    f, offset, layer_id, full_k, full_v, token_indices=token_indices
                )
                query_meta["digest"]["layers"][str(layer_id)] = layer_meta

            for layer_id in critical_layers:
                full_k, full_v = HackBlendKVPool.get_kv(layer_id)
                layer_meta, offset = cls._write_query_layer(
                    f,
                    offset,
                    layer_id,
                    full_k,
                    full_v,
                    token_indices=critical_token_indices,
                )
                query_meta["critical"]["layers"][str(layer_id)] = layer_meta

        with open(meta_path, "w") as f:
            json.dump(query_meta, f, indent=2)

    # ================================================================
    # Online: Load from SSD
    # ================================================================

    @classmethod
    def _ensure_transfer_pool(cls, pool_key, device):
        pool = cls._transfer_pools.get(pool_key)
        if pool is not None and pool["device"] == device:
            return pool

        pool = {
            "device": device,
            "lock": threading.Lock(),
            "stream": torch.cuda.Stream(device=device),
            "slot": {},
        }
        cls._transfer_pools[pool_key] = pool
        return pool

    @classmethod
    def _load_layer_to_gpu(cls, sample_dir, layer_id, device, pool_key):
        """Load a single packed layer and transfer to GPU via reusable pinned buffers."""
        meta = cls._load_packed_meta(sample_dir, cls._num_layers or None)
        return cls._load_packed_layer_to_gpu(
            sample_dir, meta, layer_id, device, pool_key
        )

    @classmethod
    def _load_packed_layer_to_gpu(cls, sample_dir, meta, layer_id, device, pool_key):
        pool = cls._ensure_transfer_pool(pool_key, device)
        with pool["lock"]:
            slot = pool["slot"]
            cls._clear_slot_metadata(slot)
            cls._read_packed_layer_to_slot(sample_dir, meta, layer_id, slot)
            return cls._copy_packed_layer_from_block_to_gpu(
                slot, layer_id, device, pool["stream"]
            )

    @staticmethod
    def _shape_numel(shape):
        numel = 1
        for dim in shape:
            numel *= dim
        return numel

    @staticmethod
    def _read_exact_into(f, tensor, nbytes):
        view = memoryview(tensor[:nbytes].numpy())
        total = 0
        while total < nbytes:
            n = f.readinto(view[total:])
            if n is None:
                continue
            if n == 0:
                raise EOFError("Unexpected EOF while reading packed KV payload")
            total += n

    @classmethod
    def _load_packed_meta(cls, sample_dir, expected_num_layers=None):
        meta_path = os.path.join(sample_dir, _PACKED_META_NAME)
        bin_path = os.path.join(sample_dir, _PACKED_BIN_NAME)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing packed KV metadata: {meta_path}")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Missing packed KV data: {bin_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        if meta.get("format") != _PACKED_FORMAT:
            raise ValueError(
                f"Unsupported packed KV format in {meta_path}: {meta.get('format')}"
            )

        num_layers = meta.get("num_layers")
        if not isinstance(num_layers, int) or num_layers < 0:
            raise ValueError(f"Bad packed KV num_layers in {meta_path}: {num_layers}")
        if expected_num_layers is not None and num_layers != expected_num_layers:
            raise ValueError(
                f"Packed KV num_layers mismatch in {meta_path}: "
                f"{num_layers} != {expected_num_layers}"
            )

        layers = meta.get("layers")
        if not isinstance(layers, dict):
            raise ValueError(f"Bad packed KV layers metadata in {meta_path}")

        file_size = os.path.getsize(bin_path)
        expected_offset = 0
        for layer_id in range(num_layers):
            layer = layers.get(str(layer_id))
            if not isinstance(layer, dict):
                raise ValueError(f"Missing packed KV metadata for layer {layer_id}")
            for name in ("k", "v"):
                item = layer.get(name)
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Missing packed KV metadata for layer {layer_id}.{name}"
                    )
                offset = item.get("offset")
                nbytes = item.get("nbytes")
                shape = item.get("shape")
                dtype_name = item.get("dtype")
                if (
                    not isinstance(offset, int)
                    or not isinstance(nbytes, int)
                    or offset < 0
                    or nbytes < 0
                    or offset + nbytes > file_size
                ):
                    raise ValueError(
                        f"Bad packed KV byte range for layer {layer_id}.{name}"
                    )
                if offset != expected_offset:
                    raise ValueError(
                        f"Non-contiguous packed KV offset for layer {layer_id}.{name}: "
                        f"{offset} != {expected_offset}"
                    )
                if not isinstance(shape, list) or not all(
                    isinstance(dim, int) and dim >= 0 for dim in shape
                ):
                    raise ValueError(
                        f"Bad packed KV shape for layer {layer_id}.{name}: {shape}"
                    )
                dtype = cls._packed_name_to_dtype(dtype_name)
                expected_nbytes = cls._tensor_nbytes(shape, dtype)
                if nbytes != expected_nbytes:
                    raise ValueError(
                        f"Bad packed KV nbytes for layer {layer_id}.{name}: "
                        f"{nbytes} != {expected_nbytes}"
                    )
                expected_offset += nbytes
        if expected_offset != file_size:
            raise ValueError(
                f"Packed KV size mismatch in {bin_path}: "
                f"{file_size} != {expected_offset}"
            )
        return meta

    @classmethod
    def _validate_tensor_item(cls, item, file_size, meta_path, label, expected_offset):
        if not isinstance(item, dict):
            raise ValueError(f"Missing packed KV metadata for {label}")
        offset = item.get("offset")
        nbytes = item.get("nbytes")
        shape = item.get("shape")
        dtype_name = item.get("dtype")
        if (
            not isinstance(offset, int)
            or not isinstance(nbytes, int)
            or offset < 0
            or nbytes < 0
            or offset + nbytes > file_size
        ):
            raise ValueError(f"Bad query cache byte range for {label}")
        if offset != expected_offset:
            raise ValueError(
                f"Non-contiguous query cache offset for {label}: "
                f"{offset} != {expected_offset}"
            )
        if not isinstance(shape, list) or not all(
            isinstance(dim, int) and dim >= 0 for dim in shape
        ):
            raise ValueError(f"Bad query cache shape for {label}: {shape}")
        dtype = cls._packed_name_to_dtype(dtype_name)
        expected_nbytes = cls._tensor_nbytes(shape, dtype)
        if nbytes != expected_nbytes:
            raise ValueError(
                f"Bad query cache nbytes for {label} in {meta_path}: "
                f"{nbytes} != {expected_nbytes}"
            )
        return expected_offset + nbytes

    @classmethod
    def _validate_query_layer(cls, section, layer_id, file_size, meta_path, expected_offset):
        layers = section.get("layers")
        if not isinstance(layers, dict):
            raise ValueError(f"Bad query cache layers metadata in {meta_path}")
        layer = layers.get(str(layer_id))
        if not isinstance(layer, dict):
            raise ValueError(f"Missing query cache metadata for layer {layer_id}")
        for name in ("k", "v"):
            expected_offset = cls._validate_tensor_item(
                layer.get(name),
                file_size,
                meta_path,
                f"layer {layer_id}.{name}",
                expected_offset,
            )
        return expected_offset

    @classmethod
    def _load_query_meta(cls, sample_dir):
        if cls._query_meta_cache is not None and cls._query_meta_cache_dir == sample_dir:
            return cls._query_meta_cache
        meta_path = os.path.join(sample_dir, _QUERY_META_NAME)
        bin_path = os.path.join(sample_dir, _QUERY_BIN_NAME)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing query cache metadata: {meta_path}")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Missing query cache data: {bin_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("format") != QUERY_CACHE_FORMAT:
            raise ValueError(
                f"Unsupported query cache format in {meta_path}: "
                f"{meta.get('format')}"
            )

        digest = meta.get("digest")
        critical = meta.get("critical")
        if not isinstance(digest, dict) or not isinstance(critical, dict):
            raise ValueError(f"Bad query cache sections in {meta_path}")
        num_digest_layers = digest.get("num_layers")
        if not isinstance(num_digest_layers, int) or num_digest_layers < 0:
            raise ValueError(
                f"Bad query cache digest num_layers in {meta_path}: "
                f"{num_digest_layers}"
            )
        critical_layer_ids = critical.get("layer_ids", [])
        if not isinstance(critical_layer_ids, list):
            raise ValueError(f"Bad query cache critical layer_ids in {meta_path}")

        file_size = os.path.getsize(bin_path)
        expected_offset = 0
        for layer_id in range(num_digest_layers):
            expected_offset = cls._validate_query_layer(
                digest, layer_id, file_size, meta_path, expected_offset
            )
        for layer_id in critical_layer_ids:
            expected_offset = cls._validate_query_layer(
                critical, int(layer_id), file_size, meta_path, expected_offset
            )
        if expected_offset != file_size:
            raise ValueError(
                f"Query cache size mismatch in {bin_path}: "
                f"{file_size} != {expected_offset}"
            )
        cls._query_meta_cache = meta
        cls._query_meta_cache_dir = sample_dir
        return meta

    @classmethod
    def _query_section_layers_as_packed_meta(cls, query_meta, section_name, layer_ids):
        section = query_meta[section_name]
        layers = {
            str(layer_id): section["layers"][str(layer_id)]
            for layer_id in layer_ids
        }
        return {"layers": layers}

    @classmethod
    def _read_query_layers_to_slot(
        cls, sample_dir, query_meta, section_name, layer_ids, slot
    ):
        if not layer_ids:
            raise ValueError("Cannot read empty query cache layer block")

        packed_meta = cls._query_section_layers_as_packed_meta(
            query_meta, section_name, layer_ids
        )
        first_k = cls._packed_tensor_meta(packed_meta, layer_ids[0], "k")
        last_v = cls._packed_tensor_meta(packed_meta, layer_ids[-1], "v")
        block_offset = first_k["offset"]
        block_end = last_v["end"]
        block_nbytes = block_end - block_offset
        if block_nbytes < 0:
            raise ValueError(
                f"Bad query cache block range for {section_name}: {layer_ids}"
            )

        for layer_id in layer_ids:
            for name in ("k", "v"):
                item = cls._packed_tensor_meta(packed_meta, layer_id, name)
                if item["offset"] < block_offset or item["end"] > block_end:
                    raise ValueError(
                        f"Query cache layer {layer_id}.{name} is outside block range"
                    )

        cls._ensure_slot_byte_buf(slot, "block", block_nbytes)
        bin_path = os.path.join(sample_dir, _QUERY_BIN_NAME)
        with open(bin_path, "rb", buffering=0) as f:
            f.seek(block_offset)
            cls._read_exact_into(f, slot["pin_buf_block"], block_nbytes)
        slot["layer_ids"] = list(layer_ids)
        slot["block_offset"] = block_offset
        slot["block_nbytes"] = block_nbytes
        slot["meta"] = packed_meta

    @classmethod
    def _read_query_layer_from_open_file(
        cls, f, query_meta, section_name, layer_id, slot
    ):
        packed_meta = cls._query_section_layers_as_packed_meta(
            query_meta, section_name, [layer_id]
        )
        first_k = cls._packed_tensor_meta(packed_meta, layer_id, "k")
        last_v = cls._packed_tensor_meta(packed_meta, layer_id, "v")
        block_offset = first_k["offset"]
        block_end = last_v["end"]
        block_nbytes = block_end - block_offset
        cls._ensure_slot_byte_buf(slot, "block", block_nbytes)
        f.seek(block_offset)
        cls._read_exact_into(f, slot["pin_buf_block"], block_nbytes)
        slot["layer_ids"] = [layer_id]
        slot["block_offset"] = block_offset
        slot["block_nbytes"] = block_nbytes
        slot["meta"] = packed_meta

    @classmethod
    def _load_query_layers_progressively(
        cls,
        query_meta,
        section_name,
        layer_ids,
        device,
        on_layer,
        pool_key="query",
    ):
        layer_ids = [int(x) for x in layer_ids]
        if not layer_ids:
            return

        pool = cls._ensure_transfer_pool(pool_key, device)
        bin_path = os.path.join(cls._sample_dir_query, _QUERY_BIN_NAME)
        with pool["lock"]:
            slot = pool["slot"]
            with open(bin_path, "rb", buffering=0) as f:
                for layer_id in layer_ids:
                    cls._clear_slot_metadata(slot)
                    cls._read_query_layer_from_open_file(
                        f, query_meta, section_name, layer_id, slot
                    )
                    k_gpu, v_gpu = cls._copy_packed_layer_from_block_to_gpu(
                        slot, layer_id, device, pool["stream"]
                    )
                    on_layer(layer_id, k_gpu, v_gpu)

    @classmethod
    def _load_query_layers_to_gpu(
        cls, query_meta, section_name, layer_ids, device, pool_key="query"
    ):
        layer_ids = [int(x) for x in layer_ids]
        if not layer_ids:
            return {}

        pool = cls._ensure_transfer_pool(pool_key, device)
        with pool["lock"]:
            slot = pool["slot"]
            cls._clear_slot_metadata(slot)
            cls._read_query_layers_to_slot(
                cls._sample_dir_query, query_meta, section_name, layer_ids, slot
            )
            return {
                layer_id: cls._copy_packed_layer_from_block_to_gpu(
                    slot, layer_id, device, pool["stream"]
                )
                for layer_id in layer_ids
            }

    @classmethod
    def _load_query_layer_to_gpu(cls, query_meta, section_name, layer_id, device):
        return cls._load_query_layers_to_gpu(
            query_meta, section_name, [layer_id], device
        )[int(layer_id)]

    @classmethod
    def _packed_tensor_meta(cls, meta, layer_id, name):
        item = meta["layers"][str(layer_id)][name]
        offset = item["offset"]
        nbytes = item["nbytes"]
        return {
            "offset": offset,
            "end": offset + nbytes,
            "nbytes": nbytes,
            "shape": tuple(item["shape"]),
            "dtype": cls._packed_name_to_dtype(item["dtype"]),
        }

    @classmethod
    def _ensure_slot_byte_buf(cls, slot, name, nbytes):
        buf_name = f"pin_buf_{name}"
        cap_name = f"byte_capacity_{name}"
        if slot.get(buf_name) is not None and slot.get(cap_name, 0) >= nbytes:
            return
        capacity = max(nbytes, 16384)
        slot[buf_name] = torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
        slot[cap_name] = capacity

    @staticmethod
    def _clear_slot_metadata(slot):
        for key in list(slot.keys()):
            if key.startswith("pin_buf_") or key.startswith("byte_capacity_"):
                continue
            del slot[key]

    @classmethod
    def _read_packed_layer_to_slot(cls, sample_dir, meta, layer_id, slot):
        cls._read_packed_block_to_slot(sample_dir, meta, [layer_id], slot)

    @classmethod
    def _read_packed_block_to_slot(cls, sample_dir, meta, layer_ids, slot):
        """Read consecutive packed layers into one caller-owned pinned block slot."""
        if not layer_ids:
            raise ValueError("Cannot read empty packed KV layer block")

        first_k = cls._packed_tensor_meta(meta, layer_ids[0], "k")
        last_v = cls._packed_tensor_meta(meta, layer_ids[-1], "v")
        block_offset = first_k["offset"]
        block_end = last_v["end"]
        block_nbytes = block_end - block_offset
        if block_nbytes < 0:
            raise ValueError(f"Bad packed KV block range for layers {layer_ids}")

        for layer_id in layer_ids:
            for name in ("k", "v"):
                item = cls._packed_tensor_meta(meta, layer_id, name)
                if item["offset"] < block_offset or item["end"] > block_end:
                    raise ValueError(
                        f"Packed KV layer {layer_id}.{name} is outside block range"
                    )

        cls._ensure_slot_byte_buf(slot, "block", block_nbytes)
        bin_path = os.path.join(sample_dir, _PACKED_BIN_NAME)
        with open(bin_path, "rb", buffering=0) as f:
            f.seek(block_offset)
            cls._read_exact_into(f, slot["pin_buf_block"], block_nbytes)

        slot["layer_ids"] = list(layer_ids)
        slot["block_offset"] = block_offset
        slot["block_nbytes"] = block_nbytes
        slot["meta"] = meta

    @classmethod
    def _copy_packed_layer_from_block_to_gpu(cls, slot, layer_id, device, stream):
        meta = slot["meta"]
        block_offset = slot["block_offset"]

        def _view_from_block(name):
            item = cls._packed_tensor_meta(meta, layer_id, name)
            rel = item["offset"] - block_offset
            if rel < 0 or rel + item["nbytes"] > slot["block_nbytes"]:
                raise ValueError(
                    f"Packed KV layer {layer_id}.{name} is outside pinned block"
                )
            return (
                slot["pin_buf_block"][rel : rel + item["nbytes"]]
                .view(item["dtype"])
                .view(item["shape"])
            ), item

        k_pin, k_item = _view_from_block("k")
        v_pin, v_item = _view_from_block("v")

        with torch.cuda.stream(stream):
            k_gpu = torch.empty(k_item["shape"], dtype=k_item["dtype"], device=device)
            v_gpu = torch.empty(v_item["shape"], dtype=v_item["dtype"], device=device)
            k_gpu.copy_(k_pin, non_blocking=True)
            v_gpu.copy_(v_pin, non_blocking=True)
        stream.synchronize()
        return k_gpu, v_gpu

    @classmethod
    def load_layer_all_chunks(cls, sample_dir, layer_id, device):
        return cls._load_layer_to_gpu(sample_dir, layer_id, device, "chunk")

    # ================================================================
    # Online: Context metadata restore
    # ================================================================

    @classmethod
    def _restore_context_pool_metadata(cls, meta, index, digest_ratio=None):
        ranked_by_chunk = [
            [int(x) for x in chunk]
            for chunk in index.get("ranked_indices_by_chunk", [])
        ]
        ranked_by_layer_chunk = [
            [[int(x) for x in chunk] for chunk in layer]
            for layer in index.get("ranked_indices_by_layer_chunk", [])
        ]
        total_tokens = int(meta.get("total_tokens", 0))
        num_layers = int(meta.get("num_layers", cls._num_layers or 0) or 0)
        orig_ranges = [
            tuple(r)
            for r in meta.get("orig_chunk_ranges", [])
            if (
                len(r) == 2
                and int(r[0]) < total_tokens
                and int(r[1]) <= total_tokens
            )
        ]
        ContextBlendPool.set_index_metadata(
            ranked_indices_by_chunk=ranked_by_chunk,
            ranked_indices_by_layer_chunk=ranked_by_layer_chunk,
            orig_chunk_ranges=orig_ranges,
            total_tokens=total_tokens,
            num_layers=num_layers,
        )
        if meta.get("materialized_digest") and meta.get("context_positions_by_layer"):
            ContextBlendPool.set_materialized_positions(
                meta.get("context_positions_by_layer", [])
            )
        else:
            fallback_digest_ratio = (
                DEFAULT_DIGEST_RATIO if digest_ratio is None else digest_ratio
            )
            ContextBlendPool.build_context_positions(
                digest_ratio=fallback_digest_ratio
            )
        return meta

    @classmethod
    def restore_query_metadata(
        cls, sample_dir=None, digest_index_method=None, digest_ratio=None
    ):
        sample_dir = sample_dir or cls._sample_dir_query
        query_meta = cls._load_query_meta(sample_dir)
        method = DigestIndexManager.normalize_method(digest_index_method)
        meta = query_meta.get("metadata", {})
        indices_by_method = query_meta.get("indices_by_method", {})
        if method not in indices_by_method:
            available = sorted(indices_by_method)
            raise ValueError(
                f"Digest method {method!r} is not available in query cache; "
                f"available={available}"
            )
        cls._restore_context_pool_metadata(
            meta, indices_by_method[method], digest_ratio=digest_ratio
        )
        return query_meta

    @classmethod
    def start_query_cache(cls, num_digest_layers, critical_layers, query_meta=None):
        """Load query_cache digest layers and critical raw chunk layers."""
        digest_layer_ids = list(range(int(num_digest_layers)))
        critical_layer_ids = [int(x) for x in (critical_layers or [])]
        release_layer_ids = digest_layer_ids + critical_layer_ids

        def _query_worker():
            try:
                meta = query_meta if query_meta is not None else cls._load_query_meta(
                    cls._sample_dir_query
                )
                digest = meta.get("digest", {})
                critical = meta.get("critical", {})
                if int(digest.get("num_layers", 0)) < len(digest_layer_ids):
                    raise ValueError(
                        "Query cache has fewer digest layers than requested: "
                        f"{digest.get('num_layers')} < {len(digest_layer_ids)}"
                    )
                available_critical = {
                    int(x) for x in critical.get("layer_ids", [])
                }
                missing = [
                    x for x in critical_layer_ids if x not in available_critical
                ]
                if missing:
                    raise ValueError(
                        f"Query cache missing critical layers: {missing}"
                    )
            except Exception as exc:
                cls._record_loader_error(exc)
                cls._set_layers_ready(release_layer_ids)
                task_b_event.set()
                return

            def _put_digest_layer(layer_id, k_gpu, v_gpu):
                with context_pool_lock:
                    ContextBlendPool.k_buffer[layer_id] = k_gpu
                    ContextBlendPool.v_buffer[layer_id] = v_gpu
                cls._set_layer_ready(layer_id)

            def _load_digest_layers():
                cls._load_query_layers_progressively(
                    meta,
                    "digest",
                    digest_layer_ids,
                    cls._device,
                    _put_digest_layer,
                    pool_key="query_digest",
                )

            def _load_critical_layers():
                if digest_layer_ids:
                    cls.wait_layer_ready(digest_layer_ids[0])
                critical_kv_by_layer = cls._load_query_layers_to_gpu(
                    meta,
                    "critical",
                    critical_layer_ids,
                    cls._device,
                    pool_key="query_critical",
                )
                for layer_id in critical_layer_ids:
                    k_gpu, v_gpu = critical_kv_by_layer[layer_id]
                    with hack_pool_lock:
                        HackBlendKVPool.k_buffer[layer_id] = k_gpu
                        HackBlendKVPool.v_buffer[layer_id] = v_gpu

            digest_task = cls._start_loader_thread(
                "ssd-query-digest-prefetch",
                _load_digest_layers,
                release_layer_ids=digest_layer_ids,
            )
            critical_task = cls._start_loader_thread(
                "ssd-query-critical-prefetch",
                _load_critical_layers,
                release_layer_ids=critical_layer_ids,
                release_task_b=True,
            )
            digest_task.start()
            critical_task.start()

        return threading.Thread(
            target=_query_worker, name="ssd-query-cache-prefetch", daemon=True
        )

    # ================================================================
    # Online: Task B - Chunk cache load (attn layers)
    # ================================================================

    @classmethod
    def start_task_b(cls, layer_list):
        """Load chunk_cache layers into HackBlendKVPool."""
        layer_ids = list(layer_list)

        def _task_b_worker():
            meta = cls._load_packed_meta(
                cls._sample_dir_chunk, cls._num_layers or None
            )
            for layer_id in layer_ids:
                k_gpu, v_gpu = cls._load_packed_layer_to_gpu(
                    cls._sample_dir_chunk, meta, layer_id, cls._device, "chunk"
                )
                with hack_pool_lock:
                    HackBlendKVPool.k_buffer[layer_id] = k_gpu
                    HackBlendKVPool.v_buffer[layer_id] = v_gpu

        return cls._start_loader_thread(
            "ssd-task-b-prefetch",
            _task_b_worker,
            release_layer_ids=layer_ids,
            release_task_b=True,
        )

    # ================================================================
    # Online: DO_BLEND layer-by-layer prefetch
    # ================================================================

    @classmethod
    def start_do_blend_prefetch(cls, prefetch_from, num_layers):
        """Return a thread that layer-prefetches DO_BLEND layers into HackBlendKVPool.

        Layers already loaded by Task B (checked via has_kv) are skipped.
        Each layer sets its _layer_ready_event when data is available.
        """
        layer_ids = list(range(prefetch_from, num_layers))

        def _layerwise_worker():
            meta = cls._load_packed_meta(
                cls._sample_dir_chunk, cls._num_layers or num_layers
            )

            for layer_id in layer_ids:
                with hack_pool_lock:
                    if HackBlendKVPool.has_kv(layer_id):
                        cls._set_layer_ready(layer_id)
                        continue
                k_gpu, v_gpu = cls._load_packed_layer_to_gpu(
                    cls._sample_dir_chunk, meta, layer_id, cls._device, "chunk"
                )
                with hack_pool_lock:
                    HackBlendKVPool.k_buffer[layer_id] = k_gpu
                    HackBlendKVPool.v_buffer[layer_id] = v_gpu
                cls._set_layer_ready(layer_id)

        return cls._start_loader_thread(
            "ssd-do-blend-prefetch", _layerwise_worker, release_layer_ids=layer_ids
        )

    @classmethod
    def clear_do_blend_layers(cls, num_layers, keep_layers_set):
        """Clear HackKVPool layers loaded by DO_BLEND prefetch, keeping Task B layers."""
        for layer_id in range(num_layers):
            if layer_id not in keep_layers_set:
                HackBlendKVPool.clear_layer(layer_id)
        # Clear layer ready events for next DO_BLEND round
        for evt in cls._layer_ready_events:
            evt.clear()
