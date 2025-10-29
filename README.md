# Fine-tuning Module for Translation Models

Module hóa code fine-tune model dịch thuật English-Indonesian.

## 🚀 Quick Start

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Chạy training
python main.py
```

## 📁 Cấu trúc

```
finetune_project/
├── config.py              # Cấu hình
├── data_processor.py      # Xử lý dữ liệu  
├── model_loader.py        # Load model
├── trainer_setup.py       # Setup trainer
├── main.py               # Script chính
├── requirements.txt       # Dependencies
└── README.md             # Docs
```

## ⚙️ Tùy chỉnh

```python
from config import Config

config = Config()
config.data.num_samples = 100000
config.training.num_train_epochs = 2
config.training.learning_rate = 1e-4
```

## 📝 Các file chính

- **config.py**: Quản lý tất cả cấu hình
- **data_processor.py**: Xử lý dữ liệu song song
- **model_loader.py**: Load model với quantization
- **trainer_setup.py**: Setup LoRA trainer
- **main.py**: Pipeline huấn luyện hoàn chỉnh
