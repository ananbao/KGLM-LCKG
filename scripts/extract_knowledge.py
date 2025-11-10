"""
知识抽取脚本
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from src.knowledge_extraction.extractor import KnowledgeExtractor


def main():
    parser = argparse.ArgumentParser(description="使用KGLM抽取知识三元组")
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/kglm/final_model',
        help='KGLM模型路径'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='THUDM/chatglm-6b',
        help='基础模型路径'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='输入文件 (JSON格式)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='输出文件'
    )
    parser.add_argument(
        '--use_prompt',
        action='store_true',
        help='是否使用提示模板'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='批次大小'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("知识抽取")
    print("="*60)
    print(f"模型路径: {args.model_path}")
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {args.output_file}")
    print(f"使用提示: {args.use_prompt}")
    print("="*60 + "\n")
    
    # 创建抽取器
    extractor = KnowledgeExtractor(
        model_path=args.model_path,
        base_model_path=args.base_model,
        use_prompt=args.use_prompt
    )
    
    # 抽取知识
    results = extractor.extract_from_file(
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*60)
    print("抽取完成！")
    print("="*60)


if __name__ == "__main__":
    main()

