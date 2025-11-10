"""
数据加载器
"""

import json
import random
from typing import List, Dict, Tuple
from pathlib import Path


class DataLoader:
    """数据加载器 - 从现有文件加载数据"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
    
    def load_from_file(self, file_path: str) -> List[Dict]:
        """
        从JSON文件加载数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            数据列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        print(f"加载数据文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持多种格式
        if isinstance(data, list):
            # 直接是列表格式
            loaded_data = data
        elif isinstance(data, dict):
            # 字典格式，尝试提取数据
            if 'data' in data:
                loaded_data = data['data']
            elif 'samples' in data:
                loaded_data = data['samples']
            elif 'items' in data:
                loaded_data = data['items']
            else:
                # 尝试将字典值作为数据
                loaded_data = list(data.values())
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
        
        print(f"  加载了 {len(loaded_data)} 条样本")
        
        return loaded_data
    
    def normalize_data_format(self, item: Dict) -> Dict:
        """
        标准化数据格式
        
        将不同格式的数据统一为标准格式:
        {
            "prompt": "...",
            "response": "...",
            "text": "...",
            "triples": [...]
        }
        """
        normalized = {}
        
        # 提取text
        if 'text' in item:
            normalized['text'] = item['text']
        elif 'sentence' in item:
            normalized['text'] = item['sentence']
        elif 'content' in item:
            normalized['text'] = item['content']
        elif 'input' in item:
            normalized['text'] = item['input']
        else:
            # 尝试从prompt中提取
            if 'prompt' in item:
                prompt = item['prompt']
                # 尝试提取"给定句子："后的内容
                if "给定句子：" in prompt:
                    normalized['text'] = prompt.split("给定句子：")[-1].strip()
                else:
                    normalized['text'] = prompt
            else:
                raise ValueError(f"无法找到文本字段: {item.keys()}")
        
        # 提取triples
        if 'triples' in item:
            triples = item['triples']
        elif 'triplets' in item:
            triples = item['triplets']
        elif 'relations' in item:
            triples = item['relations']
        elif 'response' in item:
            # 尝试从response中解析
            response = item['response']
            try:
                # 尝试eval解析
                triples = eval(response) if isinstance(response, str) else response
            except:
                # 如果解析失败，设为空列表
                triples = []
        else:
            triples = []
        
        # 确保triples是列表格式
        if isinstance(triples, str):
            try:
                triples = eval(triples)
            except:
                triples = []
        
        # 确保每个triple是tuple或list
        normalized_triples = []
        for triple in triples:
            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                normalized_triples.append(tuple(triple))
            else:
                print(f"警告: 跳过无效的三元组: {triple}")
        
        normalized['triples'] = normalized_triples
        
        # 构建prompt
        if 'prompt' in item:
            normalized['prompt'] = item['prompt']
        else:
            normalized['prompt'] = f"请从给定句子中抽取所有三元组。给定句子：{normalized['text']}"
        
        # 构建response
        if 'response' in item:
            normalized['response'] = item['response']
        else:
            normalized['response'] = str(normalized['triples'])
        
        return normalized
    
    def load_and_merge(
        self,
        unstructured_file: str = "data/raw/unstructured_web_data.json",
        clinical_file: str = "data/raw/clinical_data.json",
        structured_file: str = "data/raw/public_graph_data.json"
    ) -> List[Dict]:
        """
        加载并合并三种类型的数据
        
        Args:
            unstructured_file: 非结构化网页数据文件路径
            clinical_file: 半结构化临床数据文件路径
            structured_file: 结构化公开图谱数据文件路径
            
        Returns:
            合并后的数据列表
        """
        all_data = []
        
        # 加载非结构化数据
        if Path(unstructured_file).exists():
            unstructured_data = self.load_from_file(unstructured_file)
            print(f"  非结构化数据: {len(unstructured_data)} 条")
            all_data.extend(unstructured_data)
        else:
            print(f"  警告: 未找到非结构化数据文件: {unstructured_file}")
        
        # 加载临床数据
        if Path(clinical_file).exists():
            clinical_data = self.load_from_file(clinical_file)
            print(f"  半结构化临床数据: {len(clinical_data)} 条")
            all_data.extend(clinical_data)
        else:
            print(f"  警告: 未找到临床数据文件: {clinical_file}")
        
        # 加载结构化数据
        if Path(structured_file).exists():
            structured_data = self.load_from_file(structured_file)
            print(f"  结构化公开图谱数据: {len(structured_data)} 条")
            all_data.extend(structured_data)
        else:
            print(f"  警告: 未找到结构化数据文件: {structured_file}")
        
        # 标准化格式
        print("\n标准化数据格式...")
        normalized_data = []
        for i, item in enumerate(all_data):
            try:
                normalized_item = self.normalize_data_format(item)
                normalized_data.append(normalized_item)
            except Exception as e:
                print(f"  警告: 跳过第 {i+1} 条数据，错误: {e}")
                continue
        
        print(f"  标准化后: {len(normalized_data)} 条有效数据")
        
        # 打乱数据
        random.shuffle(normalized_data)
        
        return normalized_data
    
    def load_from_directory(
        self,
        data_dir: str = "data/raw",
        file_patterns: List[str] = None
    ) -> List[Dict]:
        """
        从目录中加载所有数据文件
        
        Args:
            data_dir: 数据目录
            file_patterns: 文件名模式列表，如果为None则自动查找
            
        Returns:
            合并后的数据列表
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_path}")
        
        if file_patterns is None:
            # 自动查找常见的文件名
            file_patterns = [
                "*unstructured*.json",
                "*web*.json",
                "*clinical*.json",
                "*structured*.json",
                "*graph*.json",
                "*.json"
            ]
        
        all_data = []
        
        for pattern in file_patterns:
            files = list(data_path.glob(pattern))
            for file_path in files:
                if file_path.is_file():
                    try:
                        data = self.load_from_file(str(file_path))
                        all_data.extend(data)
                    except Exception as e:
                        print(f"  警告: 加载文件 {file_path} 失败: {e}")
        
        # 标准化格式
        print("\n标准化数据格式...")
        normalized_data = []
        for i, item in enumerate(all_data):
            try:
                normalized_item = self.normalize_data_format(item)
                normalized_data.append(normalized_item)
            except Exception as e:
                print(f"  警告: 跳过第 {i+1} 条数据，错误: {e}")
                continue
        
        # 打乱数据
        random.shuffle(normalized_data)
        
        return normalized_data


def split_dataset(
    dataset: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    output_dir: str = "data/splits"
):
    """
    划分数据集为训练集/验证集/测试集
    默认设置: 80% / 10% / 10%
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    # 保存各个数据集
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
    
    for split_name, split_data in splits.items():
        split_path = output_path / f"{split_name}.json"
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"{split_name}集: {len(split_data)} 样本 -> {split_path}")
    
    # 保存统计信息
    stats = {
        "total": total,
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1 - train_ratio - val_ratio
    }
    
    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据集统计信息已保存到: {stats_path}")
    return splits


if __name__ == "__main__":
    # 测试数据加载
    loader = DataLoader(seed=42)
    
    # 方式1: 从指定文件加载
    try:
        dataset = loader.load_and_merge(
            unstructured_file="data/raw/unstructured_web_data.json",
            clinical_file="data/raw/clinical_data.json",
            structured_file="data/raw/public_graph_data.json"
        )
    except FileNotFoundError:
        # 方式2: 从目录自动查找
        print("尝试从目录自动查找数据文件...")
        dataset = loader.load_from_directory("data/raw")
    
    print(f"\n总共加载了 {len(dataset)} 条数据")
    
    # 划分数据集
    if len(dataset) > 0:
        print("\n" + "="*50)
        print("划分数据集...")
        print("="*50 + "\n")
        
        splits = split_dataset(
            dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            output_dir="data/splits"
        )
        
        print("\n数据准备完成！")

