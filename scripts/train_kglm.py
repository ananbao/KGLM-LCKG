"""
KGLM模型训练脚本
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from src.model_training.train_kglm import KGLMTrainer


def main():
    parser = argparse.ArgumentParser(description="训练KGLM模型")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/kglm_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/splits',
        help='数据目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/kglm',
        help='模型输出目录'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("KGLM模型训练")
    print("="*60)
    print(f"配置文件: {args.config}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print("="*60 + "\n")
    
    # 创建训练器
    trainer = KGLMTrainer(config_path=args.config)
    
    # 加载模型
    trainer.load_model_and_tokenizer()
    
    # 准备数据
    trainer.prepare_datasets()
    
    # 设置训练参数
    trainer.setup_training_args()
    
    # 训练
    train_metrics = trainer.train()
    
    # 评估
    eval_metrics = trainer.evaluate()
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"训练loss: {train_metrics.get('train_loss', 'N/A'):.4f}")
    print(f"验证loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"模型保存在: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

