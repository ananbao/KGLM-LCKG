"""
基线模型 - BERT和CNN用于关系抽取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from typing import List, Tuple, Dict
import json


class RelationExtractionDataset(Dataset):
    """关系抽取数据集"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 128,
        relation_labels: List[str] = None
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 构建关系标签映射
        if relation_labels is None:
            all_relations = set()
            for item in data:
                for _, rel, _ in item['triples']:
                    all_relations.add(rel)
            self.relation_labels = sorted(list(all_relations))
        else:
            self.relation_labels = relation_labels
        
        self.rel2id = {rel: i for i, rel in enumerate(self.relation_labels)}
        self.id2rel = {i: rel for rel, i in self.rel2id.items()}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 简化: 只取第一个三元组作为标签
        if item['triples']:
            _, rel, _ = item['triples'][0]
            label = self.rel2id.get(rel, 0)
        else:
            label = 0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTRelationExtractor(nn.Module):
    """
    基于BERT的关系抽取模型
    """
    
    def __init__(
        self,
        bert_model: str = "bert-base-chinese",
        num_relations: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # BERT编码器
        self.bert = BertModel.from_pretrained(bert_model)
        hidden_size = self.bert.config.hidden_size
        
        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_relations)
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_relations]
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 取[CLS]的表示
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class BERTWithAttention(nn.Module):
    """
    BERT + Attention
    """
    
    def __init__(
        self,
        bert_model: str = "bert-base-chinese",
        num_relations: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model)
        hidden_size = self.bert.config.hidden_size
        
        # Attention层
        self.attention = nn.Linear(hidden_size, 1)
        
        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_relations)
    
    def forward(self, input_ids, attention_mask):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Self-attention
        attention_scores = self.attention(sequence_output)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Mask padding
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0,
            float('-inf')
        )
        
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len]
        
        # Weighted sum
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        weighted_output = torch.sum(
            sequence_output * attention_weights,
            dim=1
        )  # [batch_size, hidden_size]
        
        # 分类
        weighted_output = self.dropout(weighted_output)
        logits = self.classifier(weighted_output)
        
        return logits


class CNNRelationExtractor(nn.Module):
    """
    基于CNN的关系抽取模型
    """
    
    def __init__(
        self,
        vocab_size: int = 21128,
        embedding_dim: int = 300,
        num_filters: int = 128,
        filter_sizes: List[int] = [3, 4, 5],
        num_relations: int = 20,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=fs
            )
            for fs in filter_sizes
        ])
        
        # 全连接层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_relations)
    
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_relations]
        """
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        # 卷积 + ReLU + MaxPooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, seq_len - fs + 1]
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]
        
        # 拼接
        cat_output = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        
        # 全连接
        output = self.dropout(cat_output)
        logits = self.fc(output)
        
        return logits


class CNNWithAttention(nn.Module):
    """CNN + Attention"""
    
    def __init__(
        self,
        vocab_size: int = 21128,
        embedding_dim: int = 300,
        num_filters: int = 128,
        filter_sizes: List[int] = [3, 4, 5],
        num_relations: int = 20,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs)
            for fs in filter_sizes
        ])
        
        # Attention
        total_filters = len(filter_sizes) * num_filters
        self.attention = nn.Linear(total_filters, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(total_filters, num_relations)
    
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1)
        
        # 卷积
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            # 不做max pooling，保留序列
            conv_out = conv_out.permute(0, 2, 1)  # [batch, seq, filters]
            conv_outputs.append(conv_out)
        
        # 拼接所有卷积输出
        # 这里简化处理，只取第一个
        conv_cat = conv_outputs[0]
        
        # Attention
        attention_scores = self.attention(conv_cat).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
        
        weighted = torch.sum(conv_cat * attention_weights, dim=1)
        
        # 分类
        output = self.dropout(weighted)
        logits = self.fc(output)
        
        return logits


def train_baseline_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    device: str = "cuda"
):
    """
    训练基线模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
    
    Returns:
        训练好的模型
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    print(f"\n开始训练 {model.__class__.__name__}")
    print("="*50)
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            
            if hasattr(model, 'bert'):
                logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids)
            
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                if hasattr(model, 'bert'):
                    logits = model(input_ids, attention_mask)
                else:
                    logits = model(input_ids)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {accuracy:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # torch.save(model.state_dict(), 'best_model.pt')
    
    print("="*50)
    print("训练完成！\n")
    
    return model


if __name__ == "__main__":
    # 测试基线模型
    
    # 创建BERT模型
    print("测试BERT模型:")
    bert_model = BERTRelationExtractor(num_relations=10)
    
    # 测试输入
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    logits = bert_model(input_ids, attention_mask)
    print(f"输出形状: {logits.shape}")  # [4, 10]
    
    # 创建CNN模型
    print("\n测试CNN模型:")
    cnn_model = CNNRelationExtractor(num_relations=10)
    logits = cnn_model(input_ids)
    print(f"输出形状: {logits.shape}")  # [4, 10]

