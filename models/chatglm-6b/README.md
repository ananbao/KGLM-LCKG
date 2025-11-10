# ChatGLM-6B 模型目录

此目录用于存放 ChatGLM-6B 预训练模型文件。

## 目录说明

本文件夹应包含 ChatGLM-6B 模型的相关文件：
- 模型权重文件（pytorch_model*.bin）
- 模型配置文件（config.json）
- 分词器文件（tokenizer相关文件）
- 其他必要的模型文件

## 模型下载

请从以下渠道下载 ChatGLM-6B 模型：

### 方法 1：从 Hugging Face 下载
```bash
git lfs install
git clone https://huggingface.co/THUDM/chatglm-6b models/chatglm-6b
```

### 方法 2：从清华云盘下载
访问 [ChatGLM-6B 官方仓库](https://github.com/THUDM/ChatGLM-6B) 获取下载链接

### 方法 3：使用 Hugging Face 镜像站
```bash
git clone https://hf-mirror.com/THUDM/chatglm-6b models/chatglm-6b
```

## 使用方法

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("models/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("models/chatglm-6b", trust_remote_code=True).half().cuda()
```

