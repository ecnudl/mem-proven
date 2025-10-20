# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from contextlib import contextmanager
from typing import Dict, List, Tuple, Type

import torch
from codetiming import Timer

from verl import DataProto

from .generation_with_kv import generate_with_kv
from .interface import RAgent, RConfig
from .utils import chat_template, graceful_padding, indexing_proto

logger = logging.getLogger(__file__)
logger.setLevel('INFO')



@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timing_raw.get(name, 0.) + timer.last




class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: RConfig,
        agent_cls: Type[RAgent]
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.world_size = actor_rollout_wg.world_size
        self.agent = agent_cls(tokenizer, config)
        self.chat_template = chat_template(tokenizer)
        self.PADDING_WORD_TOKENS = tokenizer.encode(self.chat_template.format(message="Hello."), add_special_tokens=False)
        self._rollout_model = None
        self._use_remote_kv = False


    from functools import lru_cache
    @lru_cache(maxsize=3)
    def get_paddings(self, shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return padding_token_ids, padding_attention_masks, padding_position_ids
        """
        pad_shape = shape[1:]
        padding_word_ids = self.PADDING_WORD_TOKENS
        padding_token_ids = torch.full(pad_shape, fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        padding_attention_masks = torch.zeros(pad_shape, dtype=torch.long)
        padding_position_ids = torch.zeros(pad_shape, dtype=torch.long)
        # token_ids <pad> <pad> <pad> <tok> <tok> <tok>
        # attn_mask 0     0     0     1     1     1
        # posit_ids 0     0     0     0     1     2
        padding_token_ids[-len(padding_word_ids):] = torch.tensor(padding_word_ids, dtype=torch.long)
        padding_attention_masks[-len(padding_word_ids):] = 1
        padding_position_ids[-len(padding_word_ids):] = torch.arange(0, len(padding_word_ids))
        return padding_token_ids, padding_attention_masks, padding_position_ids

    def _get_rollout_model(self):
        if self._rollout_model is not None:
            return self._rollout_model

        candidate_attrs = ("module", "model")
        for attr in candidate_attrs:
            model = getattr(self.actor_rollout_wg, attr, None)
            if model is not None:
                self._rollout_model = model
                self._use_remote_kv = False
                return model

        remote_generate = getattr(self.actor_rollout_wg, "generate_with_kv", None)
        if callable(remote_generate):
            self._use_remote_kv = True
            return None

        raise AttributeError(
            "actor_rollout_wg 必须提供可访问的模型(module/model)以便单步 KV 推理。"
        )
    
    def generate_with_graceful_padding(self, input_ids: torch.Tensor,
                                    attention_masks: torch.Tensor,
                                    position_ids: torch.Tensor,
                                    meta_info: dict):

        """
        batch may not be divisible by wordsize.
        Use "Hello" as padding, insert padding data into batch so that data 
        """
        bsz = input_ids.shape[0]

        group_nums = self.world_size
        remainder = bsz % group_nums
        if remainder:
            # Example pattern for bsz=7, group_nums=3:
            # no_padding_mask: [1, 1, 1, 0, 1, 1, 0, 1, 1]
            # padding_index:   [0, 1, 2, -1, 3, 4, -1, 5, 6]
            padding_index, no_padding_mask = graceful_padding(bsz, group_nums)
            padding_token_ids, padding_attention_masks, padding_position_ids = self.get_paddings(input_ids.shape)
            def padding_by_index(tensor, padding, padding_index):
                if not len(padding.shape) == 2:
                    padding = padding.unsqueeze(0)
                # 2. prepare data for padding, concat padding to the end of batch
                tensor_for_indexing = torch.cat([tensor, padding], dim=0)
                # 3. index, -1 will select padding, else select the corresponding original data 
                return tensor_for_indexing[padding_index]
            
            input_ids = padding_by_index(input_ids, padding_token_ids, padding_index)
            attention_masks = padding_by_index(attention_masks, padding_attention_masks, padding_index)
            position_ids = padding_by_index(position_ids, padding_position_ids, padding_index)

        batch = DataProto.from_dict(tensors={
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_masks
        }, meta_info=meta_info)
        output_batch = self.actor_rollout_wg.generate_sequences(batch)
        if remainder:
            # 4. remove padding
            output_batch = indexing_proto(output_batch, no_padding_mask)
        return output_batch

    def run_llm_loop(self, gen_batch, timing_raw) -> Tuple[DataProto, torch.BoolTensor, torch.LongTensor]:
        """Run main LLM generation loop.
        genbatch: 'context_ids','context_length','prompt_ids'
        timing_raw: timing dict used in ray_trainer, note that we will accumulate the time cost in this loop, instead of override each time as in ray_trainer.
        see `_timer` implementation at the top of this file for more details.
        """
        active_num_list = [] # trace the active number of sample in each turn
        gen_output_list = [] # store I/O batch in each turn, used for policy optimization
        meta_info = gen_batch.meta_info #  do_sample, is_validate, eos/pad are stored in here.
        model = self._get_rollout_model()
        self.agent.start(gen_batch, timing_raw)
        # Main generation loop, agent should indicate whether to stop
        while not self.agent.done():
            with _timer('mt_prepare', timing_raw):
                messages, meta_info_gen = self.agent.action()
                meta_info_gen.update(meta_info)
                active_num_list.append(len(messages))
            with _timer('mt_gen', timing_raw):
                if self._use_remote_kv:
                    result, kv_cache_out = self.actor_rollout_wg.generate_with_kv(
                        tokenizer=self.tokenizer,
                        messages=messages,
                        meta_info=meta_info_gen,
                        generation_kwargs=meta_info_gen.get("generation_kwargs", {}),
                    )
                else:
                    result, kv_cache_out = generate_with_kv(
                        model=model,
                        tokenizer=self.tokenizer,
                        messages=messages,
                        meta_info=meta_info_gen,
                        generation_kwargs=meta_info_gen.get("generation_kwargs", {}),
                    )
                kv_cache_out_meta = list(kv_cache_out) if kv_cache_out is not None else []
                meta_for_agent = {"kv_cache_out": kv_cache_out_meta}
                gen_output = DataProto.from_dict(
                    tensors={
                        "responses": result["next_token"],
                        "logits": result["logits"],
                    },
                    meta_info=meta_for_agent,
                )
                logger.info('generation done')
            with _timer('mt_update', timing_raw):
                gen_output = self.agent.update(gen_output)
                gen_output_list.append(gen_output)
                logger.info('agent update done')
        final_mask, sample_index = self.agent.end()
        
        # OK, now we've got all we need in gen_output_list, and the final_mask indicates which one is final answer.
        assert len(sample_index) == sum(active_num_list)
        assert sum(final_mask) == len(gen_batch)
        logger.info(f"ACTIVE_TRAJ_NUM: {active_num_list}")
        return DataProto.concat(gen_output_list), final_mask, sample_index # pyright: ignore
