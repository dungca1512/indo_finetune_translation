"""
Configuration module for fine-tuning
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Data configuration"""
    data_path: str = "/content/"
    file_en: str = "WikiMatrix.en-id.en"
    file_id: str = "WikiMatrix.en-id.id"
    num_samples: int = 50000
    test_size: float = 0.05
    seed: int = 42
    
    @property
    def file_en_path(self) -> str:
        return os.path.join(self.data_path, self.file_en)
    
    @property
    def file_id_path(self) -> str:
        return os.path.join(self.data_path, self.file_id)


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "SeaLLMs/SeaLLMs-v3-1.5B-Chat"
    output_dir: str = "/kaggle/working/results_finetune"
    load_in_4bit: bool = True
    compute_dtype: str = "float16"
    trust_remote_code: bool = True


@dataclass
class LoraConfig:
    """LoRA configuration"""
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    r: int = 16
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    fp16: bool = True
    bf16: bool = True
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "constant"
    max_seq_length: int = 512
    report_to: str = "none"


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
