from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache

PastLayerKV = Tuple[torch.Tensor, torch.Tensor]
PastKV = List[PastLayerKV]


class LegacyCache(list):
    """Minimal wrapper providing HF-style cache API."""

    def __init__(self, layers: Iterable[PastLayerKV]):
        super().__init__(layers)

    def get_seq_length(self) -> int:
        if not self:
            return 0
        return self[0][0].size(2)


PastKVLike = Optional[Union["KVCacheItem", PastKV, LegacyCache, Cache]]


@dataclass
class KVCacheItem:
    past_kv: Optional[PastKV] = None


def to_legacy_cache(past_kv: Optional[PastKV]) -> Optional[LegacyCache]:
    if past_kv is None:
        return None
    return LegacyCache([(k, v) for k, v in past_kv])


def concat_past_kv(base: Optional[PastKV], extras: List[PastKV]) -> Optional[LegacyCache]:
    if base is None and not extras:
        return None

    if base is None:
        combined = [(k.clone(), v.clone()) for k, v in extras[0]]
        return LegacyCache(combined)

    merged: PastKV = []
    for idx, (k_base, v_base) in enumerate(base):
        k_parts = [k_base]
        v_parts = [v_base]
        for ext in extras:
            k_ext, v_ext = ext[idx]
            k_parts.append(k_ext)
            v_parts.append(v_ext)
        merged.append((torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2)))
    return LegacyCache(merged)


def extend_attention_mask(mask: torch.Tensor, added_len: int) -> torch.Tensor:
    if added_len <= 0:
        return mask

    if mask.dim() == 2:
        pad = mask.new_ones(mask.size(0), added_len)
        return torch.cat([pad, mask], dim=1)

    if mask.dim() == 4:
        pad = mask.new_ones(mask.size(0), 1, 1, added_len)
        return torch.cat([pad, mask], dim=-1)

    raise ValueError(f"Unsupported mask shape {mask.shape}")


def extend_position_ids(position_ids: torch.Tensor, added_len: int) -> torch.Tensor:
    if added_len <= 0:
        return position_ids
    return position_ids + added_len


def kv_seq_len(past_kv: PastKVLike) -> int:
    if past_kv is None:
        return 0

    if isinstance(past_kv, LegacyCache):
        return past_kv.get_seq_length()
    if isinstance(past_kv, Cache):
        return past_kv.get_seq_length()

    if isinstance(past_kv, KVCacheItem):
        return kv_seq_len(past_kv.past_kv)

    if isinstance(past_kv, list):
        for layer in past_kv:
            if layer is None:
                continue
            if isinstance(layer, KVCacheItem):
                return kv_seq_len(layer.past_kv)
            if isinstance(layer, (list, tuple)) and layer:
                key = layer[0]
                if isinstance(key, torch.Tensor):
                    return key.size(2)
        return 0

    return 0
