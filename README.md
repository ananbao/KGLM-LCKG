# 基于大语言模型的肺癌知识图谱构建

## 项目简介

本项目是基于ChatGLM-6B的QLoRA微调、提示工程、知识抽取和图谱构建。

## 实验环境

- **硬件**: RTX 4090 24GB (或同等GPU)
- **操作系统**: Windows/Linux
- **Python**: 3.9+
- **CUDA**: 11.8+

## 项目结构

```
LCKG-Experiment/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 预处理后数据
│   └── splits/                    # 训练/验证/测试集
├── models/                        # 模型目录
│   ├── chatglm-6b/               # ChatGLM基础模型
│   ├── kglm/                     # 微调后的KGLM模型
│   └── baseline_models/          # 基线模型(BERT/CNN)
├── src/                          # 源代码
│   ├── data_processing/          # 数据处理
│   ├── model_training/           # 模型训练
│   ├── knowledge_extraction/     # 知识抽取
│   ├── entity_alignment/         # 实体对齐
│   ├── evaluation/               # 评估代码
│   └── visualization/            # 可视化
├── configs/                      # 配置文件
├── scripts/                      # 运行脚本
├── outputs/                      # 输出结果
├── requirements.txt              # 依赖包
└── README.md                     # 本文件
```

## 快速开始

### 1. 环境安装

```bash
# 克隆项目
cd KGLM-LCKG

# 创建虚拟环境
conda create -n lckg python=3.9
conda activate lckg

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

如果您已有三种类型的数据文件，请将它们放在 `data/raw/` 目录下：

```bash
# 数据文件命名 (任选一种方式)
# 方式A: 指定文件名
data/raw/unstructured_web_data.json    # 非结构化网页数据
data/raw/clinical_data.json            # 半结构化临床数据
data/raw/public_graph_data.json        # 结构化公开图谱数据

# 方式B: 任意文件名，脚本会自动查找
data/raw/*.json
```

然后运行：
```bash
# 从指定文件加载
python scripts/prepare_data.py

# 或从目录自动查找所有JSON文件
python scripts/prepare_data.py --data_dir data/raw
```

**数据格式要求**: 每个JSON文件应包含以下格式的数据：
```json
[
  {
    "text": "患者诊断为肺腺癌。主要症状为咳嗽、胸痛。",
    "triples": [
      ["肺腺癌", "症状", "咳嗽"],
      ["肺腺癌", "症状", "胸痛"]
    ]
  }
]
```

### 3. 模型训练

```bash
# 微调KGLM模型 (QLoRA)
python scripts/train_kglm.py --config configs/kglm_config.yaml

# 训练基线模型 (可选)
python scripts/train_baseline.py --model bert
python scripts/train_baseline.py --model cnn
```

### 4. 知识抽取

```bash
# 使用KGLM抽取三元组
python scripts/extract_knowledge.py \
    --model_path models/kglm/final_model \
    --input_file data/raw/unstructured_text.json \
    --output_file outputs/extracted_triples.json
```

### 5. 实体对齐

```bash
# 执行实体对齐和知识融合
python scripts/entity_alignment.py \
    --input_files outputs/extracted_triples.json \
    --output_file outputs/aligned_knowledge.json
```

### 6. 知识图谱构建

```bash
# 导入Neo4j数据库
python scripts/build_kg.py \
    --input_file outputs/aligned_knowledge.json \
    --neo4j_uri bolt://localhost:7687
```

### 7. 评估

```bash
# 运行完整评估
python scripts/evaluate.py --config configs/eval_config.yaml
```
