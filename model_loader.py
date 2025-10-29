"""
Model loading and initialization module
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class ModelLoader:
    """Load and configure model for fine-tuning"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load model with quantization"""
        print(f"Đang tải model {self.config.model_name} (chế độ 4-bit)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, self.config.compute_dtype),
            bnb_4bit_use_double_quant=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            torch_dtype=getattr(torch, self.config.compute_dtype),
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code
        )
        self.model.config.use_cache = False
        
        print("Model đã được tải và lượng tử hóa 4-bit.")
        return self.model
    
    def load_tokenizer(self):
        """Load and configure tokenizer"""
        print("Đang tải tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = "right"
        
        print("Tokenizer đã được tải và cấu hình.")
        return self.tokenizer
    
    def load_all(self):
        """Load both model and tokenizer"""
        self.load_model()
        self.load_tokenizer()
        print("=== MODEL VÀ TOKENIZER SẴN SÀNG =====")
        return self.model, self.tokenizer
