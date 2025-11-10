"""
完整流程自动运行脚本
一键运行所有实验步骤
"""

import os
import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parents[1]
os.chdir(project_root)


def run_command(cmd, description):
    """运行命令并显示进度"""
    print("\n" + "="*60)
    print(f"执行: {description}")
    print("="*60)
    print(f"命令: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n错误: {description} 失败")
        return False
    
    print(f"\n✓ {description} 完成")
    return True


def main():
    print("="*60)
    print("肺癌知识图谱完整流程自动运行")
    print("="*60)
    print("本脚本将依次执行:")
    print("1. 数据准备")
    print("2. 模型训练")
    print("3. 知识抽取")
    print("4. 实体对齐")
    print("5. 图谱构建")
    print("6. 模型评估")
    print("="*60)
    
    response = input("\n是否继续? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 步骤1: 数据准备
    if not run_command(
        "python scripts/prepare_data.py",
        "数据准备"
    ):
        return
    
    # 步骤2: 模型训练
    print("\n注意: 模型训练需要约5小时")
    response = input("是否跳过训练并使用预训练模型? (y/n): ")
    
    if response.lower() != 'y':
        if not run_command(
            "python scripts/train_kglm.py --config configs/kglm_config.yaml",
            "模型训练"
        ):
            return
    else:
        print("跳过模型训练")
    
    # 步骤3: 知识抽取
    # 首先需要准备输入文件
    print("\n准备抽取输入数据...")
    
    # 使用测试集的前100条作为示例
    import json
    with open("data/splits/test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    sample_data = test_data[:100]
    
    with open("data/raw/sample_input.json", 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    if not run_command(
        "python scripts/extract_knowledge.py "
        "--model_path models/kglm/final_model "
        "--input_file data/raw/sample_input.json "
        "--output_file outputs/extracted_triples.json "
        "--use_prompt",
        "知识抽取"
    ):
        return
    
    # 步骤4: 实体对齐
    if not run_command(
        "python scripts/entity_alignment.py "
        "--input_files outputs/extracted_triples.json "
        "--output_file outputs/aligned_knowledge.json",
        "实体对齐"
    ):
        return
    
    # 步骤5: 图谱构建 (可选)
    print("\n注意: 需要Neo4j服务运行")
    response = input("是否构建Neo4j图谱? (y/n): ")
    
    if response.lower() == 'y':
        password = input("请输入Neo4j密码: ")
        
        run_command(
            f"python scripts/build_kg.py "
            f"--input_file outputs/aligned_knowledge.json "
            f"--neo4j_password {password} "
            f"--clear_existing",
            "图谱构建"
        )
    
    # 步骤6: 评估
    if not run_command(
        "python scripts/evaluate.py "
        "--test_file data/splits/test.json "
        "--model_path models/kglm/final_model "
        "--eval_type full",
        "模型评估"
    ):
        return
    
    print("\n" + "="*60)
    print("完整流程执行完成！")
    print("="*60)
    print("\n结果文件:")
    print("  - 抽取的三元组: outputs/extracted_triples.json")
    print("  - 对齐后知识: outputs/aligned_knowledge.json")
    print("  - 评估结果: outputs/evaluation_results.json")
    print("\n查看详细指南: EXPERIMENT_GUIDE.md")
    print("="*60)


if __name__ == "__main__":
    main()

