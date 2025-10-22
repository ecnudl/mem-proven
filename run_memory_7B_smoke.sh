#!/bin/bash
set -euo pipefail

# 简易 smoke 测试脚本，使用本地数据与 KV-cache 记忆实现。

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

MODEL_PATH="/mnt/ssd2/models/Qwen2.5-0.5B-Instruct"
DATA_ROOT="$(pwd)/data/ruler"
TRAIN_PATH="${DATA_ROOT}/train_100.parquet"
VAL_PATH="${DATA_ROOT}/dev_100.parquet"

if [[ ! -f "${TRAIN_PATH}" ]] || [[ ! -f "${VAL_PATH}" ]]; then
  echo "[ERROR] 找不到 smoke 测试数据：${TRAIN_PATH} 或 ${VAL_PATH}"
  echo "       请确认已按照说明生成 data/ruler/train_100.parquet 和 dev_100.parquet。"
  exit 1
fi

python3 -m verl.trainer.main_ppo \
  recurrent.enable=memory \
  recurrent.memory.path="recurrent/impls/memory_kvcache.py" \
  recurrent.memory.config.chunk_size=2048 \
  data.train_batch_size=16 \
  data.train_files="${TRAIN_PATH}" \
  data.val_files="${VAL_PATH}" \
  data.shuffle=False \
  data.filter_overlong_prompts=True \
  data.truncation='center' \
  +data.context_key='context' \
  data.max_prompt_length=4096 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.actor.fsdp_config.model_dtype='bfloat16' \
  actor_rollout_ref.actor.train_batch_size=16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.name='hf' \
  actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  critic.model.path="${MODEL_PATH}" \
  critic.model.tokenizer_path="${MODEL_PATH}" \
  +critic.model.fsdp_config.model_dtype='bfloat16' \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  trainer.logger=['console'] \
  trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
  trainer.nnodes=${NNODES} \
  trainer.project_name='verl-smoke' \
  trainer.experiment_name='memory_agent_smoke' \
  reward_model.enable=False
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
