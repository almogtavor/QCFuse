import json
import math
import os
import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.model_executor.forward_batch_info import compute_position
from sglang.srt.utils.cache_blender_info import (
    BlendStyle,
    ContextBlendPool,
    DEFAULT_DIGEST_RATIO,
    HackBlendKVPool,
)


DIGEST_INDEX_VERSION = 8
DIGEST_INDEX_METHOD = "kvzip"


class DigestIndexManager:
    """Build, save, and load context digest token rankings."""

    _metadata: Dict = {}
    _indices_by_method: Dict[str, Dict] = {}
    _kvzip_scores_by_layer_doc_chunk: Dict[int, List[Optional[torch.Tensor]]] = {}

    @classmethod
    def clear(cls):
        cls._metadata = {}
        cls._indices_by_method = {}
        cls._kvzip_scores_by_layer_doc_chunk = {}

    @staticmethod
    def normalize_method(method: Optional[str]) -> str:
        method = (method or "kvzip").lower()
        if method != DIGEST_INDEX_METHOD:
            raise ValueError(
                f"Unsupported digest_index_method={method!r}; "
                f"expected {DIGEST_INDEX_METHOD!r}"
            )
        return method

    @staticmethod
    def index_filename(method: str) -> str:
        return f"index_{DigestIndexManager.normalize_method(method)}.json"

    @staticmethod
    def _cumsum_lens(lengths: Sequence[int]) -> List[int]:
        out = [0]
        total = 0
        for length in lengths:
            total += int(length)
            out.append(total)
        return out

    @staticmethod
    def prepare_augmented_locs_for_request(
        raw_locs: Sequence[int],
    ) -> Optional[Dict]:
        """Detect sys/doc/zip/.../query layout and build forward/original locs.

        raw_locs come from CacheBlender.split_tokens() after separator removal.
        For an augmented offline prompt they describe:
            sys, doc1, zipprompt1, doc2, zipprompt2, ..., query
        The model forward uses:
            sys, doc1+zipprompt1, doc2+zipprompt2, ..., query
        Metadata and packed SSD cache use:
            sys, doc1, doc2, ..., query
        """

        raw_locs = [int(x) for x in raw_locs]
        num_raw_chunks = len(raw_locs) - 1
        if num_raw_chunks < 4 or num_raw_chunks % 2 != 0:
            return None

        doc_chunk_indices = list(range(1, num_raw_chunks - 1, 2))
        zip_chunk_indices = list(range(2, num_raw_chunks - 1, 2))
        if len(doc_chunk_indices) != len(zip_chunk_indices):
            return None

        sys_start, sys_end = raw_locs[0], raw_locs[1]
        query_start = raw_locs[-2]
        query_end = raw_locs[-1]

        forward_lens = [sys_end - sys_start]
        original_lens = [sys_end - sys_start]
        keep_indices = list(range(sys_start, sys_end))
        aug_doc_ranges: List[Tuple[int, int]] = []
        aug_zip_ranges: List[Tuple[int, int]] = []

        for doc_idx, zip_idx in zip(doc_chunk_indices, zip_chunk_indices):
            doc_start = raw_locs[doc_idx]
            doc_end = raw_locs[doc_idx + 1]
            zip_start = raw_locs[zip_idx]
            zip_end = raw_locs[zip_idx + 1]

            forward_lens.append(zip_end - doc_start)
            original_lens.append(doc_end - doc_start)
            keep_indices.extend(range(doc_start, doc_end))
            aug_doc_ranges.append((doc_start, doc_end))
            aug_zip_ranges.append((zip_start, zip_end))

        forward_lens.append(query_end - query_start)
        original_lens.append(query_end - query_start)
        keep_indices.extend(range(query_start, query_end))

        return {
            "forward_locs": DigestIndexManager._cumsum_lens(forward_lens),
            "original_locs": DigestIndexManager._cumsum_lens(original_lens),
            "keep_indices": keep_indices,
            "aug_sys_range": (sys_start, sys_end),
            "aug_doc_ranges": aug_doc_ranges,
            "aug_zip_ranges": aug_zip_ranges,
        }

    @staticmethod
    def ensure_forward_positions(blend_info):
        positions = getattr(blend_info, "positions", None)
        if positions is not None:
            return positions

        extend_prefix_lens = torch.zeros_like(blend_info.chunk_lens)
        positions, _ = compute_position(
            "flashinfer",
            extend_prefix_lens,
            blend_info.chunk_lens,
            blend_info.chunk_loc_list[-1],
        )
        blend_info.positions = positions
        return positions

    @staticmethod
    def _chunk_abs_ranges(chunk_loc_list) -> List[Tuple[int, int]]:
        locs = [int(x) for x in chunk_loc_list]
        return [(locs[i], locs[i + 1]) for i in range(len(locs) - 1)]

    @staticmethod
    def _rank_from_scores(
        scores: torch.Tensor, chunk_len: int, n_sink: int
    ) -> List[int]:
        if chunk_len <= 0:
            return []

        scores = scores.detach()
        sink_count = min(max(int(n_sink), 0), int(chunk_len))
        sink_indices = torch.arange(sink_count, device=scores.device)
        if sink_count < chunk_len:
            tail_scores = scores[sink_count:]
            tail_order = torch.argsort(tail_scores, descending=True) + sink_count
            ranked = torch.cat([sink_indices, tail_order])
        else:
            ranked = sink_indices
        return [int(x) for x in ranked.cpu().tolist()]

    @classmethod
    def _rank_scores_by_layer(
        cls,
        scores_by_layer_chunk: Sequence[Sequence[torch.Tensor]],
        doc_chunk_lengths: Sequence[int],
        n_sink: int,
    ) -> List[List[List[int]]]:
        ranked_by_layer = []
        for layer_scores in scores_by_layer_chunk:
            ranked = [[]]
            for idx, chunk_len in enumerate(doc_chunk_lengths):
                chunk_len = int(chunk_len)
                if chunk_len <= 0:
                    ranked.append([])
                    continue
                if idx < len(layer_scores) and layer_scores[idx].numel() == chunk_len:
                    scores = layer_scores[idx]
                else:
                    scores = torch.zeros(chunk_len)
                ranked.append(cls._rank_from_scores(scores, chunk_len, n_sink))
            ranked_by_layer.append(ranked)
        return ranked_by_layer

    @classmethod
    def _scores_from_kvzip_dict(
        cls,
        doc_chunk_lengths: Sequence[int],
        scores_by_layer_chunk: Dict[int, List[Optional[torch.Tensor]]],
        num_layers: int,
    ) -> List[List[torch.Tensor]]:
        out = []
        for layer_id in range(max(0, int(num_layers))):
            raw_layer_scores = scores_by_layer_chunk.get(layer_id, [])
            layer_scores = []
            for idx, chunk_len in enumerate(doc_chunk_lengths):
                chunk_len = int(chunk_len)
                if (
                    chunk_len > 0
                    and idx < len(raw_layer_scores)
                    and raw_layer_scores[idx] is not None
                    and raw_layer_scores[idx].numel() == chunk_len
                ):
                    layer_scores.append(raw_layer_scores[idx].detach().float())
                else:
                    layer_scores.append(torch.zeros(max(0, chunk_len)))
            out.append(layer_scores)
        return out

    @classmethod
    def build_all_indices(cls, blend_info, rotary_emb):
        full_num_layers = (
            int(blend_info.att_params.num_layers)
            if getattr(blend_info, "att_params", None) is not None
            else len(HackBlendKVPool.k_buffer)
        )
        qcompute_end = int(
            getattr(blend_info, "qcompute_end", None) or full_num_layers
        )
        num_layers = max(0, min(qcompute_end, full_num_layers))
        raw_digest_ratio = getattr(blend_info, "digest_ratio", None)
        digest_ratio = float(
            DEFAULT_DIGEST_RATIO if raw_digest_ratio is None else raw_digest_ratio
        )
        digest_method = cls.normalize_method(
            getattr(blend_info, "digest_index_method", "kvzip")
        )
        critical_layers = [
            int(x) for x in (getattr(blend_info, "critical_layers", None) or [])
        ]
        original_locs = getattr(
            blend_info, "digest_original_chunk_loc_list", None
        )
        if original_locs is None:
            original_locs = blend_info.chunk_loc_list

        original_ranges = cls._chunk_abs_ranges(original_locs)
        num_chunks = len(original_ranges)
        if num_chunks == 0:
            cls._metadata = {
                "digest_index_version": DIGEST_INDEX_VERSION,
                "orig_chunk_ranges": [],
                "total_tokens": 0,
                "available_methods": [digest_method],
                "layer_wise": True,
                "num_layers": num_layers,
                "digest_ratio": digest_ratio,
                "digest_index_method": digest_method,
                "critical_layers": critical_layers,
                "qcompute_end": num_layers,
                "materialized_digest": True,
                "context_positions_by_layer": [],
            }
            cls._indices_by_method = {
                digest_method: {
                    "digest_index_version": DIGEST_INDEX_VERSION,
                    "method": digest_method,
                    "ranked_indices_by_chunk": [],
                    "ranked_indices_by_layer_chunk": [],
                }
            }
            ContextBlendPool.set_index_metadata(
                ranked_indices_by_chunk=[],
                ranked_indices_by_layer_chunk=[],
                orig_chunk_ranges=[],
                total_tokens=0,
                num_layers=num_layers,
            )
            return

        doc_original_ranges = original_ranges[1:-1]
        doc_chunk_lengths = [end - start for start, end in doc_original_ranges]

        total_tokens = int(original_ranges[-1][0])
        n_sink = max(0, int(getattr(blend_info, "context_n_sink", 0) or 0))

        kvzip_scores = cls._scores_from_kvzip_dict(
            doc_chunk_lengths,
            cls._kvzip_scores_by_layer_doc_chunk,
            num_layers,
        )
        selected_index = cls._rank_scores_by_layer(
            kvzip_scores, doc_chunk_lengths, n_sink
        )
        ranked_shared = selected_index[0] if selected_index else []
        selected_payload = {
            "context_n_sink": n_sink,
            "score_reduce": "max_head_query",
        }

        cls._metadata = {
            "digest_index_version": DIGEST_INDEX_VERSION,
            "orig_chunk_ranges": [
                [int(start), int(end)] for start, end in original_ranges
            ],
            "total_tokens": total_tokens,
            "available_methods": [digest_method],
            "layer_wise": True,
            "num_layers": num_layers,
            "digest_ratio": digest_ratio,
            "digest_index_method": digest_method,
            "critical_layers": critical_layers,
            "qcompute_end": num_layers,
            "materialized_digest": True,
        }
        cls._indices_by_method = {
            digest_method: {
                "digest_index_version": DIGEST_INDEX_VERSION,
                "method": digest_method,
                "ranked_indices_by_chunk": ranked_shared,
                "ranked_indices_by_layer_chunk": selected_index,
                "layer_wise": True,
                **selected_payload,
            },
        }

        ContextBlendPool.set_index_metadata(
            ranked_indices_by_chunk=ranked_shared,
            ranked_indices_by_layer_chunk=selected_index,
            orig_chunk_ranges=original_ranges[:-1],
            total_tokens=total_tokens,
            num_layers=num_layers,
        )
        ContextBlendPool.build_context_positions(digest_ratio=digest_ratio)
        cls._metadata["context_positions_by_layer"] = [
            [int(x) for x in layer_positions]
            for layer_positions in ContextBlendPool.context_positions_by_layer
        ]
        cls._kvzip_scores_by_layer_doc_chunk = {}

    @classmethod
    def save(cls, sample_dir: str):
        if not cls._metadata or not cls._indices_by_method:
            raise ValueError("Digest indices have not been built")

        os.makedirs(sample_dir, exist_ok=True)
        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(cls._metadata, f, indent=2)
        for method, payload in cls._indices_by_method.items():
            with open(os.path.join(sample_dir, cls.index_filename(method)), "w") as f:
                json.dump(payload, f, indent=2)

    @classmethod
    def export_payload(cls, method: Optional[str] = None):
        if not cls._metadata or not cls._indices_by_method:
            raise ValueError("Digest indices have not been built")
        method = cls.normalize_method(method or cls._metadata.get("digest_index_method"))
        if method not in cls._indices_by_method:
            available = sorted(cls._indices_by_method)
            raise ValueError(
                f"Digest method {method!r} is not available; available={available}"
            )
        return (
            copy.deepcopy(cls._metadata),
            {method: copy.deepcopy(cls._indices_by_method[method])},
        )

    @classmethod
    def load(cls, sample_dir: str, method: Optional[str] = None):
        method = cls.normalize_method(method)
        meta_path = os.path.join(sample_dir, "metadata.json")
        index_path = os.path.join(sample_dir, cls.index_filename(method))
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("digest_index_version") != DIGEST_INDEX_VERSION:
            raise ValueError(
                f"Unsupported digest metadata version in {meta_path}: "
                f"{meta.get('digest_index_version')}"
            )
        available = meta.get("available_methods", [])
        if method not in available:
            raise ValueError(
                f"Digest method {method!r} is not available in {meta_path}; "
                f"available={available}"
            )
        with open(index_path, "r") as f:
            index = json.load(f)
        if index.get("method") != method:
            raise ValueError(
                f"Digest index method mismatch in {index_path}: "
                f"{index.get('method')} != {method}"
            )
        return meta, index

    @classmethod
    @torch.no_grad()
    def accumulate_kvzip_layer_score(
        cls,
        blend_info,
        layer_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        rotary_emb,
    ):
        if (
            blend_info is None
            or blend_info.blend_style != BlendStyle.KVCOMPUTE
            or not getattr(blend_info, "is_contextblend", False)
        ):
            return
        qcompute_end = getattr(blend_info, "qcompute_end", None)
        if qcompute_end is not None and int(layer_id) >= int(qcompute_end):
            return

        doc_ranges = getattr(blend_info, "digest_aug_doc_ranges", None)
        zip_ranges = getattr(blend_info, "digest_aug_zip_ranges", None)
        if not doc_ranges or not zip_ranges:
            return

        params = getattr(blend_info, "att_params", None)
        if params is None:
            return

        head_dim = int(params.head_dim)
        num_heads = int(params.num_heads)
        num_kv_heads = int(params.num_kv_heads)
        if q.shape[-1] != num_heads * head_dim:
            num_heads = q.shape[-1] // head_dim
        if k.shape[-1] != num_kv_heads * head_dim:
            num_kv_heads = k.shape[-1] // head_dim
        if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            return

        kvzip_layer_scores = cls._kvzip_scores_by_layer_doc_chunk.setdefault(
            int(layer_id), [None] * len(doc_ranges)
        )
        if len(kvzip_layer_scores) != len(doc_ranges):
            kvzip_layer_scores = [None] * len(doc_ranges)
            cls._kvzip_scores_by_layer_doc_chunk[int(layer_id)] = kvzip_layer_scores

        positions = cls.ensure_forward_positions(blend_info).to(device=q.device)
        if rotary_emb is not None:
            # The CUDA rotary kernel mutates q/k in-place. KVzip scoring must not
            # change the tensors used by the real attention path.
            q_for_score, k_for_score = rotary_emb(positions, q.clone(), k.clone())
        else:
            q_for_score, k_for_score = q, k

        sys_start, sys_end = getattr(blend_info, "digest_aug_sys_range", (0, 0))
        n_sink = max(0, int(getattr(blend_info, "context_n_sink", 0) or 0))
        sink_end = min(int(sys_end), int(sys_start) + n_sink)
        sink_indices = list(range(int(sys_start), sink_end))
        scale = 1.0 / math.sqrt(float(head_dim))
        num_groups = num_heads // num_kv_heads

        for chunk_idx, ((doc_start, doc_end), (zip_start, zip_end)) in enumerate(
            zip(doc_ranges, zip_ranges)
        ):
            doc_start = int(doc_start)
            doc_end = int(doc_end)
            zip_start = int(zip_start)
            zip_end = int(zip_end)
            doc_len = doc_end - doc_start
            zip_len = zip_end - zip_start
            if doc_len <= 0 or zip_len <= 0:
                continue

            key_indices = sink_indices + list(range(doc_start, doc_end)) + list(
                range(zip_start, zip_end)
            )
            key_index_t = torch.tensor(key_indices, dtype=torch.long, device=q.device)
            q_zip = q_for_score[zip_start:zip_end].view(
                zip_len, num_heads, head_dim
            )
            k_sub = k_for_score.index_select(0, key_index_t).view(
                len(key_indices), num_kv_heads, head_dim
            )

            q_grouped = (
                q_zip.permute(1, 0, 2)
                .contiguous()
                .view(num_kv_heads, num_groups, zip_len, head_dim)
            )
            k_grouped = k_sub.permute(1, 0, 2).contiguous()
            logits = torch.matmul(
                q_grouped, k_grouped.unsqueeze(1).transpose(-1, -2)
            ) * scale

            sink_len = len(sink_indices)
            zip_col_start = sink_len + doc_len
            if zip_len > 1:
                causal_mask = torch.ones(
                    zip_len, zip_len, dtype=torch.bool, device=q.device
                ).triu(1)
                logits[..., zip_col_start:] = logits[..., zip_col_start:].masked_fill(
                    causal_mask.view(1, 1, zip_len, zip_len),
                    torch.finfo(logits.dtype).min,
                )

            attn = torch.softmax(logits.float(), dim=-1)
            attn_doc = attn[..., sink_len : sink_len + doc_len]
            kvzip_score_cpu = attn_doc.amax(dim=(0, 1, 2)).detach().float().cpu()

            prev = kvzip_layer_scores[chunk_idx]
            if prev is None:
                kvzip_layer_scores[chunk_idx] = kvzip_score_cpu
            else:
                kvzip_layer_scores[chunk_idx] = torch.maximum(
                    prev, kvzip_score_cpu
                )
