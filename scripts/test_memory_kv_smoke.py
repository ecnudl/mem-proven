import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recurrent.impls.memory_kvcache import MemoryKVCacheAgent, MemoryConfig
from recurrent.kvcache_utils import KVCacheItem
from recurrent.generation_with_kv import generate_with_kv

# =============== 模型加载 ===============
model_name_or_path = "/mnt/ssd2/models/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

# =============== 构造 agent ===============
config = MemoryConfig(
    context_key="context",
    max_prompt_length=1024,
    chunk_size=128,
    max_memorization_length=1,
    max_chunks=2,
    max_final_response_length=256,
)
agent = MemoryKVCacheAgent(tokenizer=tokenizer, config=config)

# =============== 构造伪 DataProto (batch=1) ===============
from verl.protocol import DataProto
prompt = [{"role": "user", "content": "What is the capital of France?"}]
context = "France is a country in Western Europe. Its capital city is Paris."

batch_tensors = {
    "context_length": torch.tensor([len(tokenizer.encode(context))], dtype=torch.long),
    "context_ids": tokenizer(context, return_tensors="pt", add_special_tokens=False)["input_ids"],
}
non_tensor_batch = {
    "prompt_ids": np.array(
        [tokenizer.encode(prompt[0]["content"], add_special_tokens=False)],
        dtype=object,
    )
}

gen_batch = DataProto.from_dict(tensors=batch_tensors, non_tensors=non_tensor_batch, meta_info={})
timing_raw = {}

# =============== Step0 吸收阶段 ===============
agent.start(gen_batch, timing_raw)
messages, meta_info = agent.action()

print(f"Step0 messages tokens: {len(messages[0])}")

result, kv_cache_out = generate_with_kv(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    meta_info=meta_info,
    generation_kwargs={"temperature": 0.0},
)
kv_meta = {"kv_cache_out": kv_cache_out if kv_cache_out is not None else []}
gen_output = DataProto.from_dict(
    tensors={"responses": result["next_token"]},
    meta_info=kv_meta,
)
agent.update(gen_output)

# =============== Step1 最终回答阶段 ===============
messages, meta_info = agent.action()
print(f"Step1 messages tokens: {len(messages[0])}")

result, kv_cache_out = generate_with_kv(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    meta_info=meta_info,
    generation_kwargs={"temperature": 0.0},
)
decoded = tokenizer.decode(result["next_token"][0])
print(f"Model output token: {decoded}")

agent.end()
