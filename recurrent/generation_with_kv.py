from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache

from recurrent.kvcache_utils import (
    KVCacheItem,
    LegacyCache,
    extend_attention_mask,
    extend_position_ids,
    kv_seq_len,
    to_legacy_cache,
)


def _resolve_device(model: PreTrainedModel, device_hint: Optional[str]) -> torch.device:
    if device_hint is not None:
        return torch.device(device_hint)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _prepare_inputs(
    messages: Sequence[torch.Tensor],
    tokenizer,
    pad_to: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not messages:
        raise ValueError("messages 不能为空，至少需要一个输入样本")

    input_ids = pad_sequence(
        messages,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    current_len = input_ids.size(1)
    if current_len > pad_to:
        raise ValueError(
            f"input_pad_to={pad_to} 小于当前序列长度 {current_len}，请增大 pad_to"
        )
    if current_len < pad_to:
        pad_len = pad_to - current_len
        pad_tail = input_ids.new_full(
            (input_ids.size(0), pad_len),
            tokenizer.pad_token_id,
        )
        input_ids = torch.cat([input_ids, pad_tail], dim=1)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    position_ids = torch.arange(
        input_ids.size(1),
        dtype=torch.long,
        device=input_ids.device,
    ).unsqueeze(0).expand(input_ids.size(0), -1)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)

    return input_ids, attention_mask, position_ids


def _legacy_to_cache(legacy: Optional[LegacyCache]) -> Optional[Cache]:
    if legacy is None:
        return None
    legacy_tuple = tuple((k, v) for k, v in legacy)
    return DynamicCache.from_legacy_cache(legacy_tuple)


def _normalize_kv_cache(kv_cache_in) -> Optional[Cache]:
    if kv_cache_in is None:
        return None
    if isinstance(kv_cache_in, KVCacheItem):
        return _legacy_to_cache(to_legacy_cache(kv_cache_in.past_kv))
    if isinstance(kv_cache_in, LegacyCache):
        return _legacy_to_cache(kv_cache_in)
    if isinstance(kv_cache_in, (list, tuple)):
        if len(kv_cache_in) == 0:
            return None
        normalized: List[Optional[LegacyCache]] = []
        for item in kv_cache_in:
            if item is None:
                normalized.append(None)
            elif isinstance(item, LegacyCache):
                normalized.append(item)
            elif isinstance(item, KVCacheItem):
                normalized.append(to_legacy_cache(item.past_kv))
            else:
                normalized.append(to_legacy_cache(item))
        first_valid = next((kv for kv in normalized if kv is not None), None)
        if first_valid is None:
            return None
        num_layers = len(first_valid)
        merged: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            keys: List[torch.Tensor] = []
            values: List[torch.Tensor] = []
            for kv in normalized:
                if kv is None:
                    continue
                key, value = kv[layer_idx]
                keys.append(key)
                values.append(value)
            if not keys:
                return None
            merged.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        return _legacy_to_cache(LegacyCache(merged))
    if isinstance(kv_cache_in, Cache):
        return kv_cache_in
    return _legacy_to_cache(to_legacy_cache(kv_cache_in))


def _split_past_kv(past_kv) -> List[LegacyCache]:
    if past_kv is None:
        return []
    if isinstance(past_kv, KVCacheItem):
        layers = past_kv.past_kv or []
    elif isinstance(past_kv, LegacyCache):
        layers = list(past_kv)
    elif isinstance(past_kv, Cache):
        if hasattr(past_kv, "to_legacy_cache"):
            layers = past_kv.to_legacy_cache()
        else:
            raise AttributeError(
                f"{type(past_kv).__name__} does not expose to_legacy_cache for conversion"
            )
    else:
        layers = past_kv

    if not layers:
        return []

    batch = layers[0][0].size(0)
    per_sample: List[LegacyCache] = []
    for b in range(batch):
        sample_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for key, value in layers:
            sample_layers.append((key[b : b + 1].clone(), value[b : b + 1].clone()))
        per_sample.append(LegacyCache(sample_layers))
    return per_sample


@torch.inference_mode()
def generate_with_kv(
    model: PreTrainedModel,
    tokenizer,
    messages: Sequence[torch.Tensor],
    meta_info: Dict,
    generation_kwargs: Dict,
) -> Tuple[Dict, Optional[tuple]]:
    """单步采样：将消息拼接成 batch，并注入外部 KV 缓存."""

    pad_to = meta_info.get("input_pad_to")
    assert pad_to is not None, "agent.meta_info 必须提供 input_pad_to 用于 batch pad"

    device = _resolve_device(model, meta_info.get("device"))
    input_ids, attention_mask, position_ids = _prepare_inputs(
        messages=messages,
        tokenizer=tokenizer,
        pad_to=pad_to,
        device=device,
    )

    kv_cache_in = meta_info.get("kv_cache_in")
    past_kv = _normalize_kv_cache(kv_cache_in)

    added_len = kv_seq_len(past_kv)
    if added_len > 0:
        attention_mask = extend_attention_mask(attention_mask, added_len)
        position_ids = extend_position_ids(position_ids, added_len)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_kv,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )

    logits = outputs.logits[:, -1, :]
    temperature = float(generation_kwargs.get("temperature", 1.0))
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0.0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        probs = torch.softmax(logits / temperature, dim=-1)
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

    result = {
        "next_token": next_token,
        "logits": logits,
    }
    kv_cache_out_raw = outputs.past_key_values
    kv_cache_out = _split_past_kv(kv_cache_out_raw)
    return result, kv_cache_out
