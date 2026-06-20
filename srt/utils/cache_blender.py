from typing import List, Tuple, Optional
import torch
import numpy as np

from sglang.srt.utils.cache_blender_info import (
    HackBlendKVPool,
    BlendStyle,
    SelectMode,
    ContextBlendPool,
    DEFAULT_DIGEST_RATIO,
)
from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.indice_select import IndiceSelector
from sglang.srt.utils.digest_index_manager import DigestIndexManager
from sglang.srt.utils.kv_ssd_manager import (
    KVSSDManager,
    hack_pool_lock,
    context_pool_lock,
)


def _apply_rotary(rotary_emb, positions, q, k):
    """Apply rotary embedding if available, returning (q, k)."""
    if rotary_emb is not None:
        return rotary_emb(positions, q, k)
    return q, k


class CacheBlender:

    @staticmethod
    def _find_pattern_matches_numpy(
        input_ids: np.ndarray, sep_token: np.ndarray
    ) -> np.ndarray:
        """Find separator token matches with vectorized NumPy operations."""
        sep_len = len(sep_token)
        n = len(input_ids)
        if sep_len > n:
            return np.array([], dtype=np.int64)

        if sep_len == 1:
            return np.where(input_ids == sep_token[0])[0]

        windows = np.lib.stride_tricks.sliding_window_view(input_ids, sep_len)
        matches = np.all(windows == sep_token, axis=1)
        return np.where(matches)[0]

    @staticmethod
    def split_tokens(
        input_text: Optional[str],
        input_ids: List[int],
        separator: str,
        sep_token: List[int],
    ) -> Tuple[Optional[str], List[int], Optional[List[int]]]:
        """
        Split tokens by separator with vectorized NumPy operations.
        """
        sep_len = len(sep_token)
        n = len(input_ids)

        if sep_len == 0 or sep_len > n:
            return input_text, input_ids, None

        # Convert to NumPy arrays.
        input_arr = np.asarray(input_ids, dtype=np.int64)
        sep_arr = np.asarray(sep_token, dtype=np.int64)

        # Find all separator matches.
        matches = CacheBlender._find_pattern_matches_numpy(input_arr, sep_arr)
        num_matches = len(matches)
        if num_matches == 0:
            return input_text, input_ids, [0, len(input_ids)]

        # Build the keep mask.
        keep_mask = np.ones(n, dtype=np.bool_)
        for match_idx in matches:
            keep_mask[match_idx : match_idx + sep_len] = False

        new_input_ids = input_arr[keep_mask].tolist()

        blend_loc_list = []
        current_new_pos = 0
        prev_end = 0

        for i, match_idx in enumerate(matches):
            # Length of the current segment without the separator.
            segment_len = match_idx - prev_end
            blend_loc_list.append(current_new_pos)
            current_new_pos += segment_len
            prev_end = match_idx + sep_len

        blend_loc_list.append(current_new_pos)
        blend_loc_list.append(len(new_input_ids))

        new_text = input_text.replace(separator, "") if input_text is not None else None
        return new_text, new_input_ids, blend_loc_list

    @staticmethod
    def _tokenize_segment(tokenizer, text: str) -> List[int]:
        if hasattr(tokenizer, "encode"):
            return tokenizer.encode(text, add_special_tokens=False)
        encoded = tokenizer(text, add_special_tokens=False)
        return encoded["input_ids"]

    @staticmethod
    def split_text_tokens(
        input_text: str,
        separator: str,
        tokenizer,
    ) -> Tuple[str, List[int], List[int]]:
        """Split at text level, then tokenize each segment independently.

        Searching for tokenized separators inside a fully tokenized prompt is
        brittle because BPE/SentencePiece can merge the separator with adjacent
        text. This path makes the separator a pure frontend delimiter.
        """
        parts = input_text.split(separator)
        input_ids = []
        blend_loc_list = [0]
        for part in parts:
            part_ids = CacheBlender._tokenize_segment(tokenizer, part)
            input_ids.extend(part_ids)
            blend_loc_list.append(len(input_ids))
        return "".join(parts), input_ids, blend_loc_list

    @staticmethod
    def build_context_pool(blend_info, rotary_emb):
        """
        Build ContextBlendPool after KVCOMPUTE completes.

        Materialized digest path:
        - Build per-layer kvzip rankings and context positions.
        - Offline save writes selected K/V directly to query_cache.
        - Online QCOMPUTE loads query_cache without indexed reads from chunk_cache.
        """
        ContextBlendPool.clear()
        DigestIndexManager.build_all_indices(blend_info, rotary_emb)

    @staticmethod
    def blend(
        layer_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        rotary_emb: RotaryEmbedding,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the cache blending path for one attention layer."""
        blend_info = forward_batch.blend_info
        if blend_info is None:
            q, k = _apply_rotary(rotary_emb, positions, q, k)
            return q, k, v

        blend_info.rotary_emb = rotary_emb

        # Precompute KV cache.
        if blend_info.blend_style == BlendStyle.KVCOMPUTE:
            if blend_info.positions is None:
                DigestIndexManager.ensure_forward_positions(blend_info)
            q, k = _apply_rotary(rotary_emb, blend_info.positions, q, k)

            # ContextBlend stores KV during KVCOMPUTE and builds the digest
            # after the final layer.
            if blend_info.is_contextblend:
                num_layers = (
                    blend_info.att_params.num_layers if blend_info.att_params
                    else len(HackBlendKVPool.k_buffer)
                )
                if layer_id == num_layers - 1:
                    CacheBlender.build_context_pool(blend_info, rotary_emb)

            # Offline SSD save: one KVCOMPUTE writes chunk cache; contextblend
            # additionally writes query cache with digest + critical KV.
            if KVSSDManager.is_offline():
                num_layers = (
                    blend_info.att_params.num_layers if blend_info.att_params
                    else len(HackBlendKVPool.k_buffer)
                )
                if layer_id == num_layers - 1:
                    KVSSDManager.save_all_chunk_cache(
                        KVSSDManager._sample_dir_chunk,
                        num_layers,
                        token_indices=getattr(
                            blend_info, "digest_keep_indices", None
                        ),
                    )
                    if blend_info.is_contextblend:
                        qcompute_end = int(
                            getattr(blend_info, "qcompute_end", None) or num_layers
                        )
                        qcompute_end = max(0, min(qcompute_end, num_layers))
                        compact_positions_by_layer = (
                            ContextBlendPool.context_positions_by_layer[:qcompute_end]
                        )
                        keep_indices = getattr(blend_info, "digest_keep_indices", None)
                        if keep_indices is not None:
                            digest_token_indices_by_layer = [
                                [int(keep_indices[int(pos)]) for pos in positions]
                                for positions in compact_positions_by_layer
                            ]
                        else:
                            digest_token_indices_by_layer = [
                                [int(pos) for pos in positions]
                                for positions in compact_positions_by_layer
                            ]
                        KVSSDManager.save_all_query_cache(
                            KVSSDManager._sample_dir_query,
                            qcompute_end,
                            digest_token_indices_by_layer,
                            getattr(blend_info, "critical_layers", None) or [],
                            critical_token_indices=keep_indices,
                        )

            return q, k, v

        # QCOMPUTE with contextblend.
        if blend_info.blend_style == BlendStyle.QCOMPUTE and blend_info.is_contextblend:
            is_online = KVSSDManager.is_online()
            digest_ratio = getattr(blend_info, "digest_ratio", None)
            if digest_ratio is None:
                digest_ratio = DEFAULT_DIGEST_RATIO
            if is_online:
                KVSSDManager.wait_layer_ready(layer_id)
                with context_pool_lock:
                    ctx_k, ctx_v = ContextBlendPool.get(layer_id)
                    context_positions = ContextBlendPool.get_context_positions(layer_id)
            else:
                context_positions = ContextBlendPool.get_context_positions(layer_id)
                if not context_positions:
                    context_positions = ContextBlendPool.build_context_positions(
                        digest_ratio=digest_ratio,
                    )
                    context_positions = ContextBlendPool.get_context_positions(layer_id)
                old_k_ref, old_v_ref = HackBlendKVPool.get_kv(layer_id)
                context_pos_t_by_layer = getattr(
                    blend_info, "_context_positions_t_by_layer", None
                )
                if context_pos_t_by_layer is None:
                    context_pos_t_by_layer = {}
                    blend_info._context_positions_t_by_layer = context_pos_t_by_layer
                context_pos_t = context_pos_t_by_layer.get(layer_id)
                if (
                    context_pos_t is None
                    or context_pos_t.device != old_k_ref.device
                    or int(context_pos_t.numel()) != len(context_positions)
                ):
                    context_pos_t = torch.tensor(
                        context_positions,
                        dtype=torch.long,
                        device=old_k_ref.device,
                    )
                    context_pos_t_by_layer[layer_id] = context_pos_t
                ctx_k = old_k_ref.index_select(0, context_pos_t)
                ctx_v = old_v_ref.index_select(0, context_pos_t)

            if not context_positions:
                context_positions = ContextBlendPool.build_context_positions(
                    digest_ratio=digest_ratio,
                )
                context_positions = ContextBlendPool.get_context_positions(layer_id)
            if len(context_positions) != int(ctx_k.shape[0]):
                raise ValueError(
                    "ContextBlend QCOMPUTE position/KV mismatch: "
                    f"positions={len(context_positions)}, ctx_k={int(ctx_k.shape[0])}, "
                    f"layer={layer_id}"
                )

            context_pos_t_by_layer = getattr(
                blend_info, "_context_positions_t_by_layer", None
            )
            if context_pos_t_by_layer is None:
                context_pos_t_by_layer = {}
                blend_info._context_positions_t_by_layer = context_pos_t_by_layer
            context_pos_t = context_pos_t_by_layer.get(layer_id)
            if (
                context_pos_t is None
                or context_pos_t.device != ctx_k.device
                or int(context_pos_t.numel()) != len(context_positions)
            ):
                context_pos_t = torch.tensor(
                    context_positions,
                    dtype=torch.long,
                    device=ctx_k.device,
                )
                context_pos_t_by_layer[layer_id] = context_pos_t

            if rotary_emb is not None and ctx_k.shape[0] > 0:
                fake_q = getattr(blend_info, "_context_fake_q", None)
                if (
                    fake_q is None
                    or fake_q.shape != ctx_k.shape
                    or fake_q.device != ctx_k.device
                    or fake_q.dtype != ctx_k.dtype
                ):
                    fake_q = torch.empty_like(ctx_k)
                    blend_info._context_fake_q = fake_q
                if is_online:
                    # Query cache digest K is consumed only by this QCOMPUTE
                    # context attention, so mutating it avoids a per-layer clone.
                    _, ctx_k = rotary_emb(context_pos_t, fake_q, ctx_k)
                else:
                    # Other paths may share raw prefix K with later selection.
                    _, ctx_k = rotary_emb(context_pos_t, fake_q, ctx_k.clone())
            ctx_total = ContextBlendPool.total_tokens
            query_len = q.shape[0]

            # Cache metadata on the first call so later layers allocate nothing.
            if blend_info.positions is None:
                dev = q.device
                kv_total = ctx_k.shape[0] + query_len
                blend_info.positions = torch.arange(
                    ctx_total, ctx_total + query_len, device=dev
                )
                # Metadata required by the Triton backend.
                blend_info._context_q_len = query_len
                blend_info._context_kv_len = kv_total
                blend_info._context_q_positions = torch.arange(
                    ctx_k.shape[0], ctx_k.shape[0] + query_len, device=dev
                )
                blend_info._ctx_q_start_loc = torch.zeros(
                    1, dtype=torch.int32, device=dev
                )
                blend_info._ctx_q_lens_t = torch.tensor(
                    [query_len], dtype=torch.int32, device=dev
                )
                blend_info._ctx_kv_start_loc = torch.zeros(
                    1, dtype=torch.int32, device=dev
                )
                blend_info._ctx_kv_lens_t = torch.tensor(
                    [kv_total], dtype=torch.int32, device=dev
                )

            q, k = _apply_rotary(rotary_emb, blend_info.positions, q, k)
            # Concatenate K/V as [context, query].
            k_cat = torch.cat([ctx_k, k], dim=0)
            v_cat = torch.cat([ctx_v, v], dim=0)
            return q, k_cat, v_cat

        # Blend mode.
        if blend_info.blend_style in (BlendStyle.DO_BLEND, BlendStyle.DO_BLEND_FINISH):
            if layer_id < blend_info.start:
                q, k = _apply_rotary(rotary_emb, positions, q, k)
            elif layer_id == blend_info.start:
                # Initialize metadata and run selection.
                blend_info.init_attmeta = True

                if blend_info.ratio <= 0:
                    indices, lens = IndiceSelector.select(blend_info)
                elif blend_info.select_mode == SelectMode.ATTN:
                    if KVSSDManager.is_online():
                        KVSSDManager.wait_task_b()
                    critical_layers = getattr(blend_info, "critical_layers", None)
                    if critical_layers:
                        old_k, _ = HackBlendKVPool.get_kv_layers(critical_layers)
                        old_q = HackBlendKVPool.get_q_layers(critical_layers)
                    else:
                        old_k, _ = HackBlendKVPool.get_all_kv(
                            blend_info.attn_start, blend_info.attn_end
                        )
                        old_q = HackBlendKVPool.get_all_q(
                            blend_info.attn_start, blend_info.attn_end
                        )
                    indices, lens = IndiceSelector.select(
                        blend_info, old_k=old_k, old_q=old_q, positions=positions
                    )
                else:
                    raise ValueError(
                        f"Unsupported select mode for release: "
                        f"{blend_info.select_mode}"
                    )

                blend_info.blend_top_indices = indices
                blend_info.blend_top_lens = lens
                # Align positions as 1D index for downstream usage
                blend_info.positions = positions[indices]
                blend_info.fake_q = torch.zeros_like(q)

                q, k = _apply_rotary(rotary_emb, positions, q, k)
                q = q[indices]
            else:
                # Later layers wait for prefetch, then use the loaded cache.
                if KVSSDManager.is_online():
                    KVSSDManager.wait_layer_ready(layer_id)
                    with hack_pool_lock:
                        old_k_ref, old_v_ref = HackBlendKVPool.get_kv(layer_id)
                else:
                    old_k_ref, old_v_ref = HackBlendKVPool.get_kv(layer_id)
                if blend_info.blend_style == BlendStyle.DO_BLEND_FINISH:
                    # Last ratio: direct reference, will be cleared after
                    old_k = old_k_ref
                    old_v = old_v_ref
                elif KVSSDManager.is_online() and layer_id not in blend_info.keep_layers_set:
                    # SSD non-Task-B layer: re-loaded from SSD each round, safe to mutate
                    old_k = old_k_ref
                    old_v = old_v_ref
                else:
                    # Task B layer or non-SSD: clone to preserve buffer for next ratio
                    old_k = old_k_ref.clone()
                    old_v = old_v_ref.clone()
                indices = blend_info.blend_top_indices

                old_k[indices] = k
                old_v[indices] = v

                q, k = _apply_rotary(rotary_emb, blend_info.positions, q, k)
                if rotary_emb is not None:
                    _, old_k = rotary_emb(positions, blend_info.fake_q, old_k)

                return q, old_k, old_v
        else:
            q, k = _apply_rotary(rotary_emb, positions, q, k)

        return q, k, v
