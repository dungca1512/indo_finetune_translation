"""
Data processing module for translation fine-tuning
"""
from datasets import load_dataset, concatenate_datasets
from typing import Dict, Any


class DataProcessor:
    """Process parallel text data for translation fine-tuning"""
    
    def __init__(self, config):
        self.config = config
        self.train_dataset = None
        self.eval_dataset = None
    
    def load_and_process(self):
        """Load and process the parallel text data"""
        print("Bắt đầu tải và xử lý dữ liệu...")
        
        print(f"Đang tải file English: {self.config.file_en_path}")
        ds_en = load_dataset("text", data_files={"train": self.config.file_en_path})['train']
        
        print(f"Đang tải file Indonesian: {self.config.file_id_path}")
        ds_id = load_dataset("text", data_files={"train": self.config.file_id_path})['train']
        
        ds_en = ds_en.rename_column("text", "en_text")
        ds_id = ds_id.rename_column("text", "id_text")
        
        if len(ds_en) != len(ds_id):
            raise ValueError(
                f"Hai tệp không có cùng số dòng! "
                f"Tiếng Anh: {len(ds_en)}, Tiếng Indo: {len(ds_id)}"
            )
        
        dataset = concatenate_datasets([ds_en, ds_id], axis=1)
        print(f"Đã ghép thành công! Tổng số cặp câu: {len(dataset)}")
        
        print(f"Đang lấy mẫu ngẫu nhiên {self.config.num_samples} cặp câu...")
        dataset_sampled = dataset.shuffle(seed=self.config.seed).select(
            range(min(self.config.num_samples, len(dataset)))
        )
        
        print("Đang định dạng dữ liệu sang dạng Chat...")
        dataset_formatted = dataset_sampled.map(
            self._format_data_for_chat,
            remove_columns=['en_text', 'id_text']
        )
        
        dataset_formatted = dataset_formatted.filter(lambda x: x['messages'] is not None)
        
        dataset_split = dataset_formatted.train_test_split(
            test_size=self.config.test_size,
            seed=self.config.seed
        )
        self.train_dataset = dataset_split['train']
        self.eval_dataset = dataset_split['test']
        
        print("\n=== XỬ LÝ DỮ LIỆU HOÀN TẤT ===")
        print(f"Số lượng mẫu huấn luyện: {len(self.train_dataset)}")
        print(f"Số lượng mẫu đánh giá: {len(self.eval_dataset)}")
        print("\n--- Mẫu dữ liệu đã định dạng ---")
        print(self.train_dataset[0]['messages'])
        
        return self.train_dataset, self.eval_dataset
    
    @staticmethod
    def _format_data_for_chat(example: Dict[str, str]) -> Dict[str, Any]:
        """Format parallel text into chat format"""
        eng_text = example['en_text'].strip()
        ind_text = example['id_text'].strip()
        
        if eng_text and ind_text:
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Dịch câu sau từ tiếng Anh sang tiếng Indonesia: '{eng_text}'"
                    },
                    {
                        "role": "assistant",
                        "content": ind_text
                    }
                ]
            }
        else:
            return {"messages": None}
