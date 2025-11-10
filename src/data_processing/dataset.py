"""
数据集类
"""

import json
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class KnowledgeGraphDataset(Dataset):
    """知识图谱三元组抽取数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 512,
        max_target_length: int = 256,
        prompt_template: Optional[str] = None
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径 (JSON格式)
            tokenizer: 分词器
            max_source_length: 输入最大长度
            max_target_length: 输出最大长度
            prompt_template: 提示模板 (可选)
        """
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prompt_template = prompt_template
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"加载数据集: {len(self.data)} 条样本")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        item = self.data[idx]
        
        # 获取prompt和response
        prompt = item['prompt']
        response = item['response']
        
        # 如果提供了prompt模板，则应用
        if self.prompt_template:
            prompt = self.prompt_template.format(input_text=item.get('text', ''))
        
        # 构建完整输入 (ChatGLM格式)
        # 格式: [Round 0]\n问：{prompt}\n答：{response}
        full_input = f"[Round 0]\n问：{prompt}\n答：{response}"
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            prompt,
            max_length=self.max_source_length,
            truncation=True,
            padding=False
        )
        
        label_ids = self.tokenizer.encode(
            response,
            max_length=self.max_target_length,
            truncation=True,
            padding=False
        )
        
        # 拼接input和label (用于训练)
        # ChatGLM使用特殊格式
        combined_ids = input_ids + [self.tokenizer.eos_token_id] + label_ids
        
        # 创建labels (-100表示不计算loss的位置)
        labels = [-100] * len(input_ids) + [-100] + label_ids
        
        # 转换为tensor
        input_ids_tensor = torch.tensor(combined_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': input_ids_tensor,
            'labels': labels_tensor
        }


class DataCollatorForKG:
    """知识图谱数据的collator，用于batch处理"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, padding: str = 'longest'):
        self.tokenizer = tokenizer
        self.padding = padding
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        将多个样本组合成batch
        
        Args:
            features: 样本列表
            
        Returns:
            批次数据字典
        """
        # 提取input_ids和labels
        input_ids = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        
        # 获取最大长度
        if self.padding == 'longest':
            max_length = max(len(ids) for ids in input_ids)
        else:
            max_length = self.padding
        
        # Padding
        padded_input_ids = []
        padded_labels = []
        attention_mask = []
        
        for ids, lbls in zip(input_ids, labels):
            # 计算需要padding的长度
            padding_length = max_length - len(ids)
            
            # Pad input_ids
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_input_ids.append(padded_ids)
            
            # Pad labels (-100表示padding位置)
            padded_lbls = lbls + [-100] * padding_length
            padded_labels.append(padded_lbls)
            
            # Attention mask (1表示真实token，0表示padding)
            mask = [1] * len(ids) + [0] * padding_length
            attention_mask.append(mask)
        
        # 转换为tensor
        batch = {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
        
        return batch


def load_prompt_template(template_path: str, template_type: str = 'full') -> str:
    """
    加载提示模板
    
    Args:
        template_path: 模板配置文件路径
        template_type: 模板类型 (full/ablation)
        
    Returns:
        格式化的提示模板字符串
    """
    import yaml
    
    with open(template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if template_type == 'full':
        template_config = config['full_template']
        instruction = config['instruction_template']
        
        # 组合完整模板
        full_template = instruction.format(
            system_role=template_config['system_role'],
            triple_definition=template_config['triple_definition'],
            output_format=template_config['output_format'],
            cot_reasoning=template_config['cot_reasoning'],
            input_text='{input_text}'  # 保留占位符
        )
        return full_template
    else:
        # 消融实验模板
        raise NotImplementedError(f"Template type '{template_type}' not implemented")


if __name__ == "__main__":
    # 测试数据集加载
    from transformers import AutoTokenizer
    
    # 注意: 需要先下载ChatGLM模型
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b",
        trust_remote_code=True
    )
    
    # 创建数据集
    dataset = KnowledgeGraphDataset(
        data_path="data/splits/train.json",
        tokenizer=tokenizer,
        max_source_length=512,
        max_target_length=256
    )
    
    # 测试获取样本
    sample = dataset[0]
    print("样本示例:")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    
    # 测试DataCollator
    from torch.utils.data import DataLoader
    
    collator = DataCollatorForKG(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator
    )
    
    batch = next(iter(dataloader))
    print("\nBatch示例:")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")

