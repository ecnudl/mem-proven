import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import TokenTemplate, chat_template
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel('INFO')


@dataclass
class MemoryConfig(RConfig):
    context_key: str
    max_prompt_length: int
    chunk_size: int
    max_memorization_length: int
    max_chunks: int
    max_final_response_length: int

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + self.chunk_size + self.max_memorization_length

    @property
    def gen_max_tokens_memorization(self):
        return self.max_memorization_length

    @property
    def gen_max_tokens_final_response(self):
        return self.max_final_response_length

    @property
    def gen_pad_to(self):
        return max(self.max_prompt_length, self.max_final_response_length)


class MemoryDataset(RDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        recurrent_config: MemoryConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if data_config.truncation != 'center':
            raise ValueError('MemoryDataset only support center truncation')
        data_config.max_prompt_length = recurrent_config.max_chunks * recurrent_config.chunk_size
        self.context_key = recurrent_config.context_key
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )

    @override
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)
        context = row_dict.pop(self.context_key)

        model_inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)

        context_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        context_ids, attention_mask = verl_F.postprocess_data(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,  # pyright: ignore
            left_pad=False,
            truncation=self.truncation,
        )

        row_dict["context_ids"] = context_ids[0]
        lengths = attention_mask.sum(dim=-1)
        row_dict["context_length"] = lengths[0]
        row_dict["prompt_ids"] = self.tokenizer.encode(
            chat[0]["content"], add_special_tokens=False
        )
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
        # tensor can use 2-deminsional index for chunking.
        # while prompt_ids will not be indexed, so keep it as list.
        return ["context_ids", "context_length"], ["prompt_ids"]


TEMPLATE_ABSORB = """You are solving a problem and will be shown the context in chunks. Carefully read the chunk and update your internal memory. Do not produce any response and wait for further instructions.

<problem>
{prompt}
</problem>

<chunk>
{chunk}
</chunk>

Absorb silently."""

TEMPLATE_FINAL_RESPONSE = """You have read all relevant context chunks. Using the information stored in memory, answer the problem below.

<problem>
{prompt}
</problem>

Final answer:"""


class MemoryKVCacheAgent(RAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: MemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        self.chat_template = chat_template(tokenizer)
        self.token_absorb_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_ABSORB), tokenizer)
        self.token_final_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_FINAL_RESPONSE), tokenizer)
        template_length = max(self.token_absorb_template.length, self.token_final_template.length)
        self.max_input_length = self.config.max_raw_input_length + template_length
        logger.info(
            f'\n[RECURRENT] max_input_length: {self.config.max_raw_input_length}(raw) '
            f'+ {template_length}(template) = {self.max_input_length}\n'
        )

    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.step = 0
        self.final_mask_list: List[torch.Tensor] = []
        self.sample_index_list: List[torch.Tensor] = []

        self.ctx_length = gen_batch.batch['context_length']
        self.bsz = len(self.ctx_length)
        self.memory = np.empty(self.bsz, dtype=object)
        for i in range(self.bsz):
            self.memory[i] = {"kv": None}
        self.is_final = False

    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        active_mask = self.ctx_length > self.step * self.config.chunk_size
        self.active_mask = active_mask
        np_active_mask = active_mask.detach().cpu().numpy().astype(bool)
        gen_batch = self.gen_batch

        if active_mask.sum().item() == 0:
            self.is_final = True
            prompts = []
            for prompt in gen_batch.non_tensor_batch['prompt_ids']:
                if isinstance(prompt, np.ndarray):
                    if prompt.dtype == np.object_:
                        prompt = np.array(prompt.tolist(), dtype=np.int64)
                    else:
                        prompt = prompt.astype(np.int64, copy=False)
                elif isinstance(prompt, list):
                    prompt = np.asarray(prompt, dtype=np.int64)
                prompt_tensor = torch.as_tensor(prompt, dtype=torch.long)
                prompts.append(
                    self.token_final_template.format(
                        prompt=prompt_tensor,
                    )
                )
            self.messages = prompts
            sample_index = torch.arange(self.bsz, dtype=torch.int)
            final_mask = torch.full(sample_index.shape, True, dtype=torch.bool)
            kv_cache_in = [mem["kv"] for mem in self.memory]
            self.meta_info = {
                'input_pad_to': self.max_input_length,
                'pad_to': self.config.gen_pad_to,
                'generation_kwargs': {
                    'max_tokens': self.config.gen_max_tokens_final_response,
                    'n': 1,
                },
                'kv_cache_in': kv_cache_in,
            }
            logger.info('FINAL TURN: MemoryKVCacheAgent.action() done')
        else:
            prompt_i = gen_batch.non_tensor_batch['prompt_ids'][np_active_mask]
            chunk_i = gen_batch.batch['context_ids'][
                active_mask,
                self.config.chunk_size * self.step: self.config.chunk_size * (self.step + 1)
            ]
            kv_cache_in = []
            indices = torch.nonzero(active_mask, as_tuple=False).flatten()
            self.messages = []
            for idx_tensor, prompt, chunk in zip(indices, prompt_i, chunk_i):
                if isinstance(prompt, np.ndarray):
                    if prompt.dtype == np.object_:
                        prompt = np.array(prompt.tolist(), dtype=np.int64)
                    else:
                        prompt = prompt.astype(np.int64, copy=False)
                elif isinstance(prompt, list):
                    prompt = np.asarray(prompt, dtype=np.int64)
                prompt_tokens = torch.as_tensor(prompt, dtype=torch.long)
                chunk_tokens = chunk[chunk != self.tokenizer.pad_token_id]
                self.messages.append(
                    self.token_absorb_template.format(
                        prompt=prompt_tokens,
                        chunk=chunk_tokens,
                    )
                )
                kv_cache_in.append(self.memory[int(idx_tensor)]["kv"])

            sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask]
            final_mask = torch.full(sample_index.shape, False, dtype=torch.bool)
            self.meta_info = {
                'input_pad_to': self.max_input_length,
                'pad_to': self.config.gen_pad_to,
                'generation_kwargs': {
                    'max_tokens': self.config.gen_max_tokens_memorization,
                    'n': 1,
                },
                'kv_cache_in': kv_cache_in,
            }
            logger.info('MemoryKVCacheAgent.action() done')

        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        return self.messages, self.meta_info

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        if not self.is_final:
            kv_cache_out = gen_output.meta_info.get("kv_cache_out", [])
            active_indices = torch.nonzero(self.active_mask, as_tuple=False).flatten().tolist()
            if len(kv_cache_out) != len(active_indices):
                logger.warning(
                    "kv_cache_out size %s does not match active batch size %s",
                    len(kv_cache_out),
                    len(active_indices),
                )
            for idx, kv in zip(active_indices, kv_cache_out):
                self.memory[idx] = {"kv": kv}
        self.log_step(gen_output)
        self.step += 1
        return gen_output

    @override
    def done(self):
        return self.is_final

    @override
    def end(self):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.memory
        del self.messages
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index

    def log_step(self, gen_output):
        """Log multi-turn conversation details in a single consolidated function."""

        def clip_long_string(string, max_length=2000):
            """Clip long string to a maximum length."""
            if not len(string) > max_length:
                return string
            return string[:max_length // 2] + '\n\n...(ignored)\n\n' + string[-max_length // 2:]

        step = self.step if not self.is_final else "FINAL"
        logger.info(f"\n{'=' * 30}[RECURRENT] STEP{step}{'=' * 30}")

        if len(self.messages) > 0:
            decoded_message = self.tokenizer.decode(self.messages[0])
            responses = gen_output.batch.get('responses')
            if responses is not None and responses.shape[0] > 0:
                rsp0 = responses[0]
                decoded_response = self.tokenizer.decode(
                    rsp0[rsp0 != self.tokenizer.pad_token_id]
                )
            else:
                decoded_response = "<no response>"
            logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            logger.info(f"{' ' * 10}{'-' * 20}prompt end{'-' * 20}{' ' * 10}")
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' ' * 10}{'-' * 20}response end{'-' * 20}{' ' * 10}")
        else:
            logger.info("MESSAGE and RESPONSE are empty.")


REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryKVCacheAgent)
