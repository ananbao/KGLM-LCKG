"""
实体对齐脚本
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from src.entity_alignment.alignment import EntityAligner


def main():
    parser = argparse.ArgumentParser(description="实体对齐和知识融合")
    parser.add_argument(
        '--input_files',
        type=str,
        nargs='+',
        required=True,
        help='输入文件列表 (多个数据源)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='outputs/aligned_knowledge.json',
        help='输出文件'
    )
    parser.add_argument(
        '--jaccard_threshold',
        type=float,
        default=0.85,
        help='Jaccard相似度阈值'
    )
    parser.add_argument(
        '--sbert_threshold',
        type=float,
        default=0.85,
        help='SBERT相似度阈值'
    )
    parser.add_argument(
        '--sbert_model',
        type=str,
        default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        help='SBERT模型'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("实体对齐和知识融合")
    print("="*60)
    print(f"输入文件: {len(args.input_files)} 个数据源")
    for i, f in enumerate(args.input_files, 1):
        print(f"  {i}. {f}")
    print(f"输出文件: {args.output_file}")
    print(f"Jaccard阈值: {args.jaccard_threshold}")
    print(f"SBERT阈值: {args.sbert_threshold}")
    print("="*60 + "\n")
    
    # 创建对齐器
    aligner = EntityAligner(
        sbert_model=args.sbert_model,
        jaccard_threshold=args.jaccard_threshold,
        sbert_threshold=args.sbert_threshold
    )
    
    # 融合知识
    kg_data = aligner.merge_knowledge_sources(
        source_files=args.input_files,
        output_file=args.output_file
    )
    
    print("\n融合统计:")
    print(f"  原始三元组: {kg_data['statistics']['original_triples']}")
    print(f"  对齐后三元组: {kg_data['statistics']['aligned_triples']}")
    print(f"  原始实体: {kg_data['statistics']['original_entities']}")
    print(f"  对齐后实体: {kg_data['statistics']['aligned_entities']}")


if __name__ == "__main__":
    main()

