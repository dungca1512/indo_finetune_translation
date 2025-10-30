"""
Trainer setup module for fine-tuning
"""
from transformers import TrainingArguments
from peft import LoraConfig as PeftLoraConfig
from trl import SFTTrainer


class TrainerSetup:
    """Setup and configure trainer for fine-tuning"""
    
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.trainer = None
    
    def setup_trainer(self):
        """Setup the SFTTrainer with LoRA configuration"""
        print("Đang cấu hình LoRA (PEFT) và Training Arguments...")
        
        peft_config = PeftLoraConfig(
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            r=self.config.lora.r,
            bias=self.config.lora.bias,
            task_type=self.config.lora.task_type,
            target_modules=self.config.lora.target_modules,
        )
        
        training_args = TrainingArguments(
            output_dir=self.config.model.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            eval_steps=self.config.training.eval_steps,
            save_steps=self.config.training.save_steps,
            logging_steps=self.config.training.logging_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            max_grad_norm=self.config.training.max_grad_norm,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            report_to=self.config.training.report_to,
            eval_strategy="steps",
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=peft_config,
            args=training_args
        )
        
        print("=== TRAINER ĐÃ SẴN SÀNG ===")
        return self.trainer
    
    def train(self):
        """Start training"""
        if self.trainer is None:
            self.setup_trainer()
        
        print("Bắt đầu huấn luyện model...")
        self.trainer.train()
        print("Huấn luyện hoàn tất.")
        return self.trainer
    
    def save_model(self, output_path=None):
        """Save the trained model"""
        if output_path is None:
            output_path = self.config.model.output_dir
        
        print(f"Đang lưu model vào {output_path}...")
        self.trainer.save_model(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Đã lưu model thành công!")
