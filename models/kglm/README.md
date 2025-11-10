# KGLM 模型目录

此目录用于存放 KGLM (Knowledge Graph Language Model) 模型的训练结果和权重文件。


## 使用方法

运行以下命令训练 KGLM 模型：
```bash
python scripts/train_kglm.py
```

加载已训练的模型：
```python
from src.model_training.train_kglm import load_model
model = load_model('models/kglm/best_model.pt')
```


