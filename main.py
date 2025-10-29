"""
Main script for fine-tuning
Usage: python main.py
"""
import os
from config import Config
from data_processor import DataProcessor
from model_loader import ModelLoader
from trainer_setup import TrainerSetup


def setup_environment():
    """Setup environment variables"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("Đã cài đặt biến môi trường PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")


def main():
    """Main training pipeline"""
    setup_environment()
    
    config = Config()
    
    # Customize config here if needed
    # config.data.num_samples = 100000
    # config.model.model_name = "your-model-name"
    # config.training.num_train_epochs = 2
    
    print("\n=== BƯỚC 1: XỬ LÝ DỮ LIỆU ===")
    data_processor = DataProcessor(config.data)
    train_dataset, eval_dataset = data_processor.load_and_process()
    
    print("\n=== BƯỚC 2: TẢI MODEL VÀ TOKENIZER ===")
    model_loader = ModelLoader(config.model)
    model, tokenizer = model_loader.load_all()
    
    print("\n=== BƯỚC 3: THIẾT LẬP TRAINER ===")
    trainer_setup = TrainerSetup(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config
    )
    
    print("\n=== BƯỚC 4: BẮT ĐẦU HUẤN LUYỆN ===")
    trainer = trainer_setup.train()
    
    print("\n=== BƯỚC 5: LƯU MODEL ===")
    trainer_setup.save_model()
    
    print("\n=== HOÀN TẤT ===")
    print(f"Model đã được lưu tại: {config.model.output_dir}")


if __name__ == "__main__":
    main()
