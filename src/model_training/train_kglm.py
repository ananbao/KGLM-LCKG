"""
KGLM模型训练 - 使用QLoRA微调ChatGLM-6B
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# 添加项目路径
sys.path.append(str(Path(__file__).parents[2]))
from src.data_processing.dataset import KnowledgeGraphDataset, DataCollatorForKG


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(
        default="THUDM/chatglm-6b",
        metadata={"help": "ChatGLM模型路径"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "是否使用LoRA"}
    )
    quantization_bit: int = field(
        default=8,
        metadata={"help": "量化位数 (4/8)"}
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    train_file: str = field(
        default="data/splits/train.json",
        metadata={"help": "训练数据路径"}
    )
    validation_file: str = field(
        default="data/splits/val.json",
        metadata={"help": "验证数据路径"}
    )
    max_source_length: int = field(
        default=512,
        metadata={"help": "输入最大长度"}
    )
    max_target_length: int = field(
        default=256,
        metadata={"help": "输出最大长度"}
    )


class KGLMTrainer:
    """KGLM模型训练器"""
    
    def __init__(self, config_path: str):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("="*50)
        print("初始化KGLM训练器")
        print("="*50)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def load_model_and_tokenizer(self):
        """
        加载模型和分词器，应用量化和LoRA
        """
        print("\n加载模型和分词器...")
        model_config = self.config['model']
        quant_config = self.config['quantization']
        lora_config = self.config['lora']
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['model_path'],
            trust_remote_code=True
        )
        print(f"✓ 分词器加载完成")
        
        # 加载模型 (带量化)
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto"
        }
        
        # 配置8-bit量化
        if quant_config['load_in_8bit']:
            load_kwargs['load_in_8bit'] = True
            print(f"✓ 启用8-bit量化")
        
        self.model = AutoModel.from_pretrained(
            model_config['model_path'],
            **load_kwargs
        )
        
        print(f"✓ 模型加载完成")
        
        # 准备模型用于k-bit训练
        self.model = prepare_model_for_kbit_training(self.model)
        
        # 配置LoRA
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            target_modules=lora_config['target_modules'],
            bias=lora_config['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        print(f"✓ LoRA配置完成")
    
    def prepare_datasets(self):
        """准备训练和验证数据集"""
        print("\n准备数据集...")
        data_config = self.config['data']
        
        # 创建训练集
        self.train_dataset = KnowledgeGraphDataset(
            data_path=data_config['train_file'],
            tokenizer=self.tokenizer,
            max_source_length=data_config['max_source_length'],
            max_target_length=data_config['max_target_length']
        )
        
        # 创建验证集
        self.eval_dataset = KnowledgeGraphDataset(
            data_path=data_config['validation_file'],
            tokenizer=self.tokenizer,
            max_source_length=data_config['max_source_length'],
            max_target_length=data_config['max_target_length']
        )
        
        # 创建data collator
        self.data_collator = DataCollatorForKG(
            tokenizer=self.tokenizer,
            padding='longest'
        )
        
        print(f"✓ 训练集: {len(self.train_dataset)} 样本")
        print(f"✓ 验证集: {len(self.eval_dataset)} 样本")
    
    def setup_training_args(self):
        """设置训练参数"""
        print("\n配置训练参数...")
        train_config = self.config['training']
        
        self.training_args = TrainingArguments(
            # 输出目录
            output_dir=train_config['output_dir'],
            
            # 训练参数
            num_train_epochs=train_config['num_train_epochs'],
            max_steps=train_config['max_steps'],  # 3000步
            per_device_train_batch_size=train_config['per_device_train_batch_size'],  # 64
            per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            
            # 优化器配置
            learning_rate=train_config['learning_rate'],
            lr_scheduler_type=train_config['lr_scheduler_type'],
            warmup_steps=train_config['warmup_steps'],
            weight_decay=train_config['weight_decay'],
            optim=train_config['optim'],
            
            # 混合精度 (节省显存)
            fp16=train_config['fp16'],
            
            # 梯度检查点 (节省显存)
            gradient_checkpointing=train_config['gradient_checkpointing'],
            
            # 日志和保存
            logging_dir=self.config['logging']['log_dir'],
            logging_steps=train_config['logging_steps'],
            save_steps=train_config['save_steps'],
            eval_steps=train_config['eval_steps'],
            save_total_limit=train_config['save_total_limit'],
            
            # 评估策略
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # 其他
            remove_unused_columns=False,
            report_to="tensorboard" if self.config['logging']['tensorboard'] else "none",
            seed=self.config['hardware']['seed']
        )
        
        print("✓ 训练参数配置完成")
        print(f"  - 训练步数: {train_config['max_steps']}")
        print(f"  - Batch size: {train_config['per_device_train_batch_size']}")
        print(f"  - 学习率: {train_config['learning_rate']}")
        print(f"  - 优化器: {train_config['optim']}")
    
    def train(self):
        """执行训练"""
        print("\n" + "="*50)
        print("开始训练 KGLM 模型")
        print("="*50 + "\n")
        
        # 配置早停
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config['early_stopping']['patience'],
            early_stopping_threshold=0.0
        )
        
        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            callbacks=[early_stopping]
        )
        
        # 开始训练
        train_result = self.trainer.train()
        
        # 保存模型
        print("\n保存模型...")
        self.trainer.save_model()
        self.trainer.save_state()
        
        # 保存训练指标
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        print("\n" + "="*50)
        print("训练完成！")
        print("="*50)
        print(f"最终loss: {metrics.get('train_loss', 'N/A'):.4f}")
        print(f"训练步数: {metrics.get('train_steps', 'N/A')}")
        print(f"训练时间: {metrics.get('train_runtime', 0) / 3600:.2f} 小时")
        
        return metrics
    
    def evaluate(self):
        """评估模型"""
        print("\n评估模型...")
        eval_metrics = self.trainer.evaluate()
        
        self.trainer.log_metrics("eval", eval_metrics)
        self.trainer.save_metrics("eval", eval_metrics)
        
        print(f"验证loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
        
        return eval_metrics


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练KGLM模型")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/kglm_config.yaml',
        help='配置文件路径'
    )
    args = parser.parse_args()
    
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
    
    print("\n所有训练任务完成！")
    print(f"模型保存在: {trainer.config['training']['output_dir']}")


if __name__ == "__main__":
    main()

