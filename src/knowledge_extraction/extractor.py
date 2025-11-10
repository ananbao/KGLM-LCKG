"""
知识抽取器 - 使用KGLM模型和提示模板抽取三元组
"""

import json
import re
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from tqdm import tqdm


class KnowledgeExtractor:
    """知识抽取器 - 基于KGLM和提示工程"""
    
    def __init__(
        self,
        model_path: str,
        base_model_path: str = "THUDM/chatglm-6b",
        prompt_config_path: str = "configs/prompt_template.yaml",
        use_prompt: bool = True,
        device: str = "cuda"
    ):
        """
        初始化抽取器
        
        Args:
            model_path: KGLM微调模型路径
            base_model_path: 基础模型路径
            prompt_config_path: 提示模板配置路径
            use_prompt: 是否使用提示模板
            device: 设备
        """
        self.device = device
        self.use_prompt = use_prompt
        
        print("="*50)
        print("初始化知识抽取器")
        print("="*50)
        
        # 加载提示模板
        if use_prompt:
            self.load_prompt_template(prompt_config_path)
            print("✓ 提示模板加载完成")
        
        # 加载模型和分词器
        self.load_model(model_path, base_model_path)
        print("✓ 模型加载完成")
    
    def load_prompt_template(self, config_path: str):
        """加载提示模板 (四组件模板)"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        template_config = config['full_template']
        
        # 构建完整提示模板
        self.prompt_template = f"""
{template_config['system_role']}

{template_config['triple_definition']}

{template_config['output_format']}

{template_config['cot_reasoning']}

请从以下句子中抽取所有知识三元组:

给定句子: {{input_text}}

抽取的三元组:
""".strip()
    
    def load_model(self, model_path: str, base_model_path: str):
        """加载KGLM模型"""
        print(f"加载基础模型: {base_model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # 加载基础模型
        self.model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # 如果提供了微调模型路径，加载LoRA权重
        if Path(model_path).exists():
            print(f"加载LoRA权重: {model_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                model_path,
                device_map="auto"
            )
            # 合并权重以加速推理
            self.model = self.model.merge_and_unload()
        else:
            print(f"警告: 未找到微调模型，使用基础模型")
        
        self.model.eval()
        self.model.to(self.device)
    
    def extract_from_text(
        self,
        text: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[Tuple[str, str, str]]:
        """
        从文本中抽取知识三元组
        
        Args:
            text: 输入文本
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            
        Returns:
            三元组列表 [(head, relation, tail), ...]
        """
        # 构建输入
        if self.use_prompt:
            prompt = self.prompt_template.format(input_text=text)
        else:
            prompt = f"请从给定句子中抽取所有三元组。给定句子：{text}"
        
        # 生成
        with torch.no_grad():
            response, history = self.model.chat(
                self.tokenizer,
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
        
        # 解析三元组
        triples = self.parse_triples(response)
        
        return triples
    
    def parse_triples(self, response: str) -> List[Tuple[str, str, str]]:
        """
        解析模型输出的三元组
        
        支持多种格式:
        - [(头, 关系, 尾), ...]
        - (头, 关系, 尾)
        - 头 - 关系 - 尾
        """
        triples = []
        
        try:
            # 尝试直接eval (如果是Python列表格式)
            if response.strip().startswith('['):
                parsed = eval(response.strip())
                if isinstance(parsed, list):
                    triples = parsed
            else:
                # 使用正则表达式提取
                # 匹配 (头, 关系, 尾) 格式
                pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
                matches = re.findall(pattern, response)
                triples = [(h.strip(), r.strip(), t.strip()) for h, r, t in matches]
        
        except Exception as e:
            print(f"解析错误: {e}")
            print(f"原始输出: {response}")
        
        # 过滤和清理
        cleaned_triples = []
        for triple in triples:
            if len(triple) == 3:
                head, rel, tail = triple
                # 移除引号
                head = head.strip('"\'')
                rel = rel.strip('"\'')
                tail = tail.strip('"\'')
                
                # 过滤过长的实体 (避免冗长短语)
                if len(head) < 50 and len(tail) < 50 and len(rel) < 30:
                    cleaned_triples.append((head, rel, tail))
        
        return cleaned_triples
    
    def extract_from_file(
        self,
        input_file: str,
        output_file: str,
        batch_size: int = 1
    ):
        """
        从文件批量抽取知识
        
        Args:
            input_file: 输入文件 (JSON格式，每行一个文本)
            output_file: 输出文件
            batch_size: 批次大小
        """
        print(f"\n从文件抽取知识: {input_file}")
        
        # 加载输入数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # 如果是字典格式，提取文本字段
            texts = [item.get('text', item.get('prompt', '')) for item in data]
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError("不支持的输入格式")
        
        print(f"共 {len(texts)} 条文本")
        
        # 批量抽取
        all_results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="抽取中"):
            batch_texts = texts[i:i+batch_size]
            
            for text in batch_texts:
                triples = self.extract_from_text(text)
                all_results.append({
                    'text': text,
                    'triples': triples
                })
        
        # 保存结果
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n抽取完成！")
        print(f"结果已保存到: {output_file}")
        print(f"共抽取 {sum(len(r['triples']) for r in all_results)} 个三元组")
        
        return all_results
    
    def extract_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[Tuple[str, str, str]]]:
        """
        批量抽取
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
            
        Returns:
            三元组列表的列表
        """
        results = []
        
        iterator = tqdm(texts, desc="抽取中") if show_progress else texts
        
        for text in iterator:
            triples = self.extract_from_text(text)
            results.append(triples)
        
        return results


class PromptAblationExtractor:
    """提示模板消融实验抽取器"""
    
    def __init__(self, base_extractor: KnowledgeExtractor, ablation_type: str):
        """
        初始化消融实验抽取器
        
        Args:
            base_extractor: 基础抽取器
            ablation_type: 消融类型 (without_system_role/without_triple_schema/without_cot/free_generation)
        """
        self.extractor = base_extractor
        self.ablation_type = ablation_type
        
        # 加载消融配置
        with open('configs/prompt_template.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.build_ablation_prompt(config)
    
    def build_ablation_prompt(self, config: Dict):
        """构建消融实验的提示模板"""
        template_config = config['full_template']
        ablation_config = config['ablation_templates'][self.ablation_type]
        
        components = []
        
        if ablation_config.get('use_system_role', False):
            components.append(template_config['system_role'])
        
        if ablation_config.get('use_triple_definition', False):
            components.append(template_config['triple_definition'])
        
        if ablation_config.get('use_output_format', False):
            components.append(template_config['output_format'])
        
        if ablation_config.get('use_cot_reasoning', False):
            components.append(template_config['cot_reasoning'])
        
        # 添加任务指令
        components.append("请从以下句子中抽取所有知识三元组:")
        components.append("给定句子: {input_text}")
        components.append("抽取的三元组:")
        
        self.extractor.prompt_template = "\n\n".join(components)
        print(f"消融实验: {self.ablation_type}")
    
    def extract_from_text(self, text: str) -> List[Tuple[str, str, str]]:
        """抽取三元组"""
        return self.extractor.extract_from_text(text)


if __name__ == "__main__":
    # 测试知识抽取
    
    # 创建抽取器
    extractor = KnowledgeExtractor(
        model_path="models/kglm/final_model",
        base_model_path="THUDM/chatglm-6b",
        use_prompt=True
    )
    
    # 测试文本
    test_text = """
    患者男性，65岁，诊断为肺腺癌。主要症状为咳嗽、咳痰、胸痛。
    CT扫描发现右下肺磨玻璃结节，大小约15mm。
    采用化疗治疗，使用紫杉醇，剂量为175mg/m²，计划4个疗程。
    """
    
    print("\n测试文本:")
    print(test_text)
    
    print("\n抽取结果:")
    triples = extractor.extract_from_text(test_text)
    for i, (h, r, t) in enumerate(triples, 1):
        print(f"{i}. ({h}, {r}, {t})")

