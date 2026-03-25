"""
IPO Trainer
基于DPO的IPO训练实现
论文Section 4: 使用偏好学习在干预轨迹对上训练
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

from .data_builder import IPOSample


@dataclass
class IPOConfig:
    """IPO训练配置
    
    论文Appendix A.2:
    - 第一阶段: 纯DPO训练（无SFT）
    - 第二阶段: DPO + 辅助SFT loss (系数0.2)
    """
    # 模型配置
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    use_4bit: bool = True
    use_lora: bool = True
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # DPO配置
    beta: float = 0.1  # DPO的KL惩罚系数
    
    # 训练配置
    output_dir: str = "./ipo_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    max_length: int = 2048
    max_prompt_length: int = 512
    weight_decay: float = 0.0  # 论文设置weight decay为0
    
    # 第二阶段: 辅助SFT loss（论文Appendix A.2）
    sft_loss_coef: float = 0.0  # 第一阶段设为0，第二阶段设为0.2
    stage: str = "stage1"  # "stage1" 或 "stage2"
    
    # 其他
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3


class IPOTrainer:
    """IPO训练器"""
    
    def __init__(self, config: Optional[IPOConfig] = None):
        """
        初始化训练器
        
        Args:
            config: IPO训练配置
        """
        self.config = config or IPOConfig()
        self.model = None
        self.tokenizer = None
        self.ref_model = None
    
    def load_model(self):
        """加载模型和tokenizer"""
        print(f"Loading model: {self.config.model_name}")
        
        # 量化配置
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 应用LoRA
        if self.config.use_lora:
            if self.config.use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        print("Model loaded successfully!")
        return self.model, self.tokenizer
    
    def prepare_dataset(
        self,
        samples: List[IPOSample],
    ) -> Dataset:
        """
        准备训练数据集
        
        Args:
            samples: IPOSample列表
            
        Returns:
            HuggingFace Dataset
        """
        data = []
        for sample in samples:
            # 转换为DPO格式
            dpo_format = sample.to_dpo_format()
            data.append(dpo_format)
        
        dataset = Dataset.from_list(data)
        return dataset
    
    def prepare_dataset_from_json(self, json_path: str) -> Dataset:
        """从JSON文件加载数据集"""
        import json
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return Dataset.from_list(data)
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        """
        执行IPO训练
        
        论文核心：使用DPO在干预轨迹对上训练
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
        """
        if self.model is None:
            self.load_model()
        
        # DPO训练配置
        training_args = DPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            seed=self.config.seed,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit" if self.config.use_4bit else "adamw_torch",
            # DPO特定参数
            beta=self.config.beta,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            remove_unused_columns=False,
        )
        
        # 创建DPO训练器
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # 使用LoRA时不需要ref_model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        print("Starting IPO training...")
        trainer.train()
        
        # 保存模型
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"Model saved to {self.config.output_dir}")
        
        return trainer
    
    def merge_and_save(self, output_path: str):
        """
        合并LoRA权重并保存完整模型
        
        Args:
            output_path: 输出路径
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.config.use_lora:
            print("Merging LoRA weights...")
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(output_path)
        else:
            self.model.save_pretrained(output_path)
        
        self.tokenizer.save_pretrained(output_path)
        print(f"Merged model saved to {output_path}")


def create_ipo_trainer(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    output_dir: str = "./ipo_output",
    **kwargs
) -> IPOTrainer:
    """
    创建IPO训练器的便捷函数
    
    Args:
        model_name: 模型名称
        output_dir: 输出目录
        **kwargs: 其他配置参数
        
    Returns:
        IPOTrainer实例
    """
    config = IPOConfig(
        model_name=model_name,
        output_dir=output_dir,
        **kwargs
    )
    return IPOTrainer(config)
