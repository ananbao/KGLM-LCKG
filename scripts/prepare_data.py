"""
数据准备脚本 - 从现有数据文件加载和准备数据集
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from src.data_processing.data_loader import DataLoader, split_dataset


def main():
    parser = argparse.ArgumentParser(description="准备数据集")
    parser.add_argument(
        '--unstructured_file',
        type=str,
        default='data/raw/unstructured_web_data.json',
        help='非结构化网页数据文件路径'
    )
    parser.add_argument(
        '--clinical_file',
        type=str,
        default='data/raw/clinical_data.json',
        help='半结构化临床数据文件路径'
    )
    parser.add_argument(
        '--structured_file',
        type=str,
        default='data/raw/public_graph_data.json',
        help='结构化公开图谱数据文件路径'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='数据目录 (如果指定，将从目录自动查找所有JSON文件)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='输出目录'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='训练集比例'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='验证集比例'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("肺癌知识图谱数据准备")
    print("="*60)
    
    # 创建数据加载器
    loader = DataLoader(seed=42)
    
    # 加载数据
    if args.data_dir:
        # 从目录自动加载
        print(f"\n从目录加载数据: {args.data_dir}")
        dataset = loader.load_from_directory(args.data_dir)
    else:
        # 从指定文件加载
        print("\n从指定文件加载数据:")
        dataset = loader.load_and_merge(
            unstructured_file=args.unstructured_file,
            clinical_file=args.clinical_file,
            structured_file=args.structured_file
        )
    
    if len(dataset) == 0:
        print("\n错误: 未加载到任何数据！")
        print("请检查数据文件路径或数据格式。")
        return
    
    # 保存完整数据集
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    full_dataset_path = output_path / "full_dataset.json"
    import json
    with open(full_dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n完整数据集已保存到: {full_dataset_path}")
    
    # 划分数据集 (80/10/10)
    print("\n" + "="*60)
    print("划分训练集/验证集/测试集")
    print("="*60)
    
    splits = split_dataset(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        output_dir="data/splits"
    )
    
    print("\n" + "="*60)
    print("数据准备完成！")
    print("="*60)
    print(f"总样本数: {len(dataset)}")
    print(f"训练集: {len(splits['train'])} 样本 ({args.train_ratio*100:.1f}%)")
    print(f"验证集: {len(splits['val'])} 样本 ({args.val_ratio*100:.1f}%)")
    print(f"测试集: {len(splits['test'])} 样本 ({(1-args.train_ratio-args.val_ratio)*100:.1f}%)")
    print("\n可以开始训练模型了！")


if __name__ == "__main__":
    main()

