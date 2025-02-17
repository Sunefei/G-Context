from transformers import DataCollatorWithPadding
import torch

class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)  # 继承 DataCollatorWithPadding 的初始化

    def __call__(self, features):
        # 1. 先用父类方法处理 input_ids, attention_mask 等
        batch = super().__call__(features)

        # 2. 处理 label (支持多个正确答案)
        if "clf_label" in features[0]:  
            labels = [f["clf_label"] for f in features]  # 获取所有 label
            max_label_len = max(len(label) if isinstance(label, list) else 1 for label in labels)

            # 3. 统一 label 形状，并转换为 tensor
            padded_labels = [
                label + [-100] * (max_label_len - len(label)) if isinstance(label, list) else [label] + [-100] * (max_label_len - 1)
                for label in labels
            ]
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch