# Copyright 2025 Bytedance Ltd. and/or its affiliates
# 版权所有 2025 字节跳动有限公司及其关联公司
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 2.0 许可发布
# you may not use this file except in compliance with the License.
# 使用本文件必须遵守 Apache 2.0 许可协议
# You may obtain a copy of the License at
# 可访问以下链接获取许可文本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律另有规定或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,
# 本软件按“原样”分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何形式的明示或暗示担保
# See the License for the specific language governing permissions and
# 请参阅许可以了解权限与限制
# limitations under the License.
# 许可范围以原文为准
from abc import ABC, abstractmethod  # 从 abc 模块导入抽象基类与抽象方法
from dataclasses import dataclass  # 导入 dataclass 装饰器以简化数据结构定义
from typing import Any, Optional, Type, List, Union, Dict, Tuple  # 导入常用类型注解工具
from uuid import uuid4  # 导入 uuid4 以生成唯一标识
import numpy as np  # 导入 numpy 用于数值计算

import torch  # 导入 PyTorch 库进行张量操作
from tensordict import TensorDict  # 导入 TensorDict 用于字典式张量管理
from omegaconf import DictConfig  # 导入 DictConfig 处理配置
from transformers import PreTrainedTokenizer, ProcessorMixin  # 导入分词器与处理器混入

from verl.protocol import DataProto, DataProtoItem  # 导入协议数据结构
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn  # 导入 RLHF 数据集基类与拼接函数


@dataclass  # 数据类装饰器定义配置结构
class RConfig:
    # 多轮策略优化的配置接口
    """
    Configuration for Multi-turn Policy Optimization.
    Just an interface. Add anything you need in a subclass of it.
    """
    pass  # 默认实现为空，实际配置由子类扩展

class RDataset(RLHFDataset):
    # 多轮策略优化数据集实现，继承 RLHFDataset
    """
    Dataset for Multi-turn Policy Optimization.
    This class can be used directly as a subclass of RLHFDataset for RecurrentRL
    (if you do not need any new features)

    Overwritten Method:
        - __getitem__: get a single sample
        - get_batch_keys: tensor keys and non-tensor keys, should be contained in the batch.
        - get_collate_fn: collate function for dataloader, default to the same as RLHFDataset.
    
    The inherited methods are hdfs/parquet related methods. 
    Make sure to call super().__init__() in your subclass to reuse RLHFDataset's initializer.
    """
    def __init__(
        self,
        recurrent_config: RConfig,
        data_files: Union[str, list[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=data_config, processor=processor)  # 调用父类初始化复用通用逻辑

    def __getitem__(self, item) -> dict:
        # 获取单条样本，可以在子类中重写
        """
        Enforce subclass to override this method by declaring it as an abstract method.
        If you don't want to change its behavior, just return super().__getitem__(item).
        """
        row_dict = super().__getitem__(item)  # 调用父类方法获取基础样本字典
        # used in validation metrics reduce
        row_dict["sample_uuid"] = str(uuid4())  # 为样本添加唯一标识用于验证统计
        return row_dict  # 返回增强后的样本字典


    def get_bactch_keys(self) -> tuple[list[str], list[str]]:
        return ["input_ids", "attention_mask", "position_ids"], []  # 指定张量键与非张量键

    @staticmethod
    def get_collate_fn():
        return collate_fn  # 返回默认的批处理函数

from .async_utils import ChatCompletionProxy  # 导入异步聊天代理工具

class AsyncOutput(ABC):
    def __init__(self, 
                 conversations: List[List[Dict[str, str]]], 
                 sample_index: int, 
                 final_mask: bool,
                 timing_raw: dict,
                 metrics: dict = None):
        self.conversations = conversations  # 保存多轮对话内容
        self.sample_index = sample_index  # 保存样本索引
        self.final_mask = final_mask  # 标记是否为最终轮
        self.timing_raw = timing_raw  # 记录时间开销信息
        if metrics is None:  # 若未提供指标字典则创建空字典
            metrics = {}
        self.metrics = metrics  # 保存评估指标
        if "workflow/num_conv" not in metrics:  # 若缺少对话次数指标则补充
            metrics["workflow/num_conv"] = len(conversations)  # 记录对话轮数
    
class AsyncRAgent(ABC):
    """
    An async recurrent agent interface.

    1. Any const variable that can be created in advance? (__init__)
    2. How to start a new generation? (start)
    3. How to prompt LLM / How to process generated response / When to stop (rollout)
    > note that you should focus on a SINGLE sample instead of a group or a batch.
    """
    def __init__(self, proxy: ChatCompletionProxy, tokenizer:PreTrainedTokenizer, config: RConfig, rollout_config: DictConfig):
        self.proxy = proxy  # 保存异步代理实例
        self.tokenizer = tokenizer  # 保存分词器对象
        self.config = config  # 保存自定义配置
        self.rollout_config = rollout_config  # 保存抽样配置
        self.timing_raw = {}  # 初始化计时信息容器

    # If you need to initialize/clean up some resource, override this two methods.
    def start(self, gen_batch: DataProto, timing_raw: dict):
        pass  # 可按需在子类中重写启动逻辑
    def end(self):
        pass  # 可按需在子类中重写收尾逻辑
        

    @abstractmethod
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
        """
        Rollout a single sample, returns conversations/sample_index/final_mask + timing/metrics...
        """
        pass  # 子类实现单样本异步推理流程
    
    def sampling_params(self, meta_info):
        """
        Adapted from works/rollout/vllm_spmd_rollout, returns topp/temperature/n for generation
        Notice that you should specify max_completion_tokens manually, since it can be different for different agents
        Also notice that top_k is not supported in async mode
        """
        kwargs = dict(
                n=1,
                temperature=self.rollout_config.temperature,
                top_p=self.rollout_config.top_p,
            )
        do_sample = meta_info.get("do_sample", True)  # 读取是否进行采样的标志
        is_validate = meta_info.get("validate", False)  # 读取是否为验证模式
        if not do_sample:  # 不采样时使用贪心策略
                # logger.info(f"original {kwargs=}, updating becase do_sample is False")
            kwargs.update({
                    'best_of': 1,
                    'top_p': 1.0,
                    'min_p': 0.0,
                    'temperature': 0,
                    'n': 1  # if greedy, only 1 response
                })
        elif is_validate:  # 验证模式下使用验证配置
                # logger.info(f"original {kwargs=}, updating because is_validate is True")
                # TODO: try **
            kwargs.update({
                    'top_p': self.rollout_config.val_kwargs.top_p,
                    'temperature': self.rollout_config.val_kwargs.temperature,
                    'n': 1,  # if validate, already repeat in ray_trainer
                })
            
        return kwargs  # 返回采样参数字典


    def reduce_timings(self, timing_raws: list[dict]) -> dict:
        """
        Reduce timing_raw of multiple agents.
        Make sure to follow the naming convention of timing_raw: "async" should be contained in the key,
        if and only if the timed code is an `await` statement.
        """
        reduced = {}  # 初始化汇总字典
        for k in timing_raws[0]:  # 遍历首个字典的键
            if "async" in k:  # 异步计时代码可并行执行
                # async method can be executed parallelly
                reduced[k] = sum([timing_raw[k] for timing_raw in timing_raws]) / len(timing_raws)  # 使用均值
            else:  # 同步部分需要累加时长
                # sync method is executed sequentially
                reduced[k] = sum([timing_raw[k] for timing_raw in timing_raws])  # 求和得到总耗时
        return reduced  # 返回汇总结果

    def reduce_metrics(self, metrics: list[dict]) -> dict:
        reduced = {}  # 初始化指标结果字典
        for k in metrics[0]:  # 遍历首个指标键
            reduced[k + "_mean"] = np.mean([m[k] for m in metrics])  # 计算均值
            reduced[k + "_min"] = np.min([m[k] for m in metrics])  # 计算最小值
            reduced[k + "_max"] = np.max([m[k] for m in metrics])  # 计算最大值
        return reduced  # 返回汇总指标

class RAgent(ABC):
    """
    A recurrent agent interface, you should focus on:

    1. Any const variable that can be created in advance? (__init__)
    2. How to start a new generation? (start)
    3. How to prompt LLM? (action)
    4. How to process generated response? (update)
    5. When to stop? (done)
    6. Any resource cleanup? (end)

    All methods are marked as abstract, they WILL NOT be called by default and are just a hint
    about how it should be implemented.
    """
    @abstractmethod
    def __init__(self, tokenizer:PreTrainedTokenizer, config: RConfig):
        pass  # 子类需实现初始化逻辑
    @abstractmethod
    def start(self, gen_batch: DataProto, timing_raw: dict):
        """
        Called once at the beginning of generation loop.
        Initialize agent state, store gen_batch and timing_raw.
        """
        self.gen_batch = gen_batch  # 保存当前批次数据
        self.timing_raw = timing_raw  # 保存计时信息
        self.step = 0  # 重置步数计数器
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        # final_mask_list 保存每步是否为最终轮
        self.sample_index_list = [] # map each turn to the sample id in the original batch
        # sample_index_list 保存每步对应的样本索引
        pass  # 子类可在此基础上扩展
    @abstractmethod
    def action(self) -> tuple[list[torch.Tensor], dict]:
        """
        Called once for each rollout step.
        Return (input_ids(list[IntTensor]), meta_info).
        Remember to add sample_index to internal state.
        If the agent can decide if the sample is the final turn, also remember to add final_mask,
        else, you can decide in `update`.

        e.g. MemoryAgent will terminate the generation loop after all context is consumed, so it can
        compute a final_mask here
        """
        sample_index = torch.arange(len(self.gen_batch), dtype=torch.long)  # 构建样本索引序列
        self.sample_index_list.append(sample_index)  # 将索引保存到列表中
        self.final_mask_list.append(torch.full(sample_index.shape, False, dtype=torch.bool))  # 假定当前步非最终轮
        pass  # 子类需返回输入张量与元信息
    @abstractmethod
    def update(self, gen_output: DataProto) -> DataProto:
        """
        Called once after rollout, agent can execute tool calling or other custom action, and update agent state.
        
        e.g. CodeAgnet will terminate the generation loop if there is no code within ```python```.
        """
        pass  # 子类实现对模型输出的处理
    @abstractmethod
    def done(self):
        """
        Whether the generation loop should stop.
        """
        return False  # 默认不终止，子类可重写
    @abstractmethod
    def end(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Called once after done() returns True.
        `del` the previouly saved data here, `gen_batch` for example.
        Can save some cpu memory(this batch will not be deleted until the next iteration).

        Returns final_mask(bool) and sample_index(long)
        """
        del self.gen_batch  # 删除批次数据释放内存
        del self.timing_raw  # 删除计时信息
        self.step = 0  # 重置步骤计数
        sample_index = torch.cat(self.sample_index_list)  # 拼接样本索引
        final_mask = torch.cat(self.final_mask_list)  # 拼接终止标志
        del self.final_mask_list  # 清理终止标志缓存
        del self.sample_index_list  # 清理样本索引缓存
        return final_mask, sample_index  # 返回终止掩码与索引

@dataclass
class RRegister:
    """Register your custom recurrent implementation with this class. The register object will be used to create these classes.
    """
    config_cls: Type[RConfig]
    dataset_cls: Type[RDataset]
    agent_cls: Type[RAgent]

    @classmethod
    def from_filename(cls, file_path: str, obj_name: str) -> 'RRegister':
        import importlib.util
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Recurrent implementation file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        if not spec:
            raise FileNotFoundError(f"Failed to create model spec for '{file_path}'.")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}")

        if not hasattr(module, obj_name):
            raise AttributeError(f"Register object '{obj_name}' not found in '{file_path}'.")

        obj = getattr(module, obj_name)
        if not isinstance(obj, cls):
            raise TypeError(f"Object '{obj_name}' in '{file_path}' is not an instance of {cls}.")
        print(f"[RECURRENT] recurrent enabled, using register '{obj_name}' from '{file_path}'.")
        return obj
