"""
实体对齐 - 基于Jaccard相似度和SBERT的混合方法
"""

import json
from typing import List, Dict, Set, Tuple
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


class EntityAligner:
    """
    实体对齐器 - 混合Jaccard和SBERT方法
    
    算法流程:
    1. 首先使用Jaccard相似度快速过滤
    2. 如果Jaccard > t1，再计算SBERT语义相似度
    3. 如果SBERT > t2，则判定为重复实体
    
    推荐阈值: t1=0.85, t2=0.85
    """
    
    def __init__(
        self,
        sbert_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        jaccard_threshold: float = 0.85,
        sbert_threshold: float = 0.85,
        device: str = "cuda"
    ):
        """
        初始化对齐器
        
        Args:
            sbert_model: SBERT模型名称 (支持中文)
            jaccard_threshold: Jaccard阈值 (t1)
            sbert_threshold: SBERT阈值 (t2)
            device: 计算设备
        """
        print("="*50)
        print("初始化实体对齐器")
        print("="*50)
        
        self.jaccard_threshold = jaccard_threshold
        self.sbert_threshold = sbert_threshold
        
        # 加载SBERT模型
        print(f"加载SBERT模型: {sbert_model}")
        self.sbert_model = SentenceTransformer(sbert_model, device=device)
        print("✓ SBERT模型加载完成")
        
        print(f"✓ Jaccard阈值: {jaccard_threshold}")
        print(f"✓ SBERT阈值: {sbert_threshold}")
    
    def jaccard_similarity(self, str1: str, str2: str) -> float:
        """
        计算Jaccard相似度 (字符级)
        
        J(A,B) = |A ∩ B| / |A ∪ B|
        
        Args:
            str1, str2: 待比较的字符串
            
        Returns:
            Jaccard相似度 [0, 1]
        """
        set1 = set(str1)
        set2 = set(str2)
        
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def sbert_similarity(self, str1: str, str2: str) -> float:
        """
        计算SBERT语义相似度
        
        使用余弦相似度:
        cos(A, B) = A·B / (|A| × |B|)
        
        Args:
            str1, str2: 待比较的字符串
            
        Returns:
            余弦相似度 [-1, 1]
        """
        # 编码
        emb1 = self.sbert_model.encode(str1, convert_to_tensor=True)
        emb2 = self.sbert_model.encode(str2, convert_to_tensor=True)
        
        # 计算余弦相似度
        similarity = util.cos_sim(emb1, emb2).item()
        
        return similarity
    
    def find_duplicate_entities(
        self,
        target_entity: str,
        candidate_entities: List[str]
    ) -> List[str]:
        """
        为目标实体找到所有重复实体
                
        Args:
            target_entity: 目标实体
            candidate_entities: 候选实体列表
            
        Returns:
            重复实体列表
        """
        duplicates = []
        
        for candidate in candidate_entities:
            # 跳过相同实体
            if candidate == target_entity:
                continue
            
            # 步骤1: 计算Jaccard相似度
            jaccard_sim = self.jaccard_similarity(target_entity, candidate)
            
            # 步骤2: 如果Jaccard超过阈值，计算SBERT相似度
            if jaccard_sim >= self.jaccard_threshold:
                sbert_sim = self.sbert_similarity(target_entity, candidate)
                
                # 步骤3: 如果SBERT也超过阈值，判定为重复
                if sbert_sim >= self.sbert_threshold:
                    duplicates.append(candidate)
        
        return duplicates
    
    def align_entities(
        self,
        entities: List[str],
        show_progress: bool = True
    ) -> Dict[str, List[str]]:
        """
        对实体列表进行对齐，找出所有重复实体组
        
        Args:
            entities: 实体列表
            show_progress: 是否显示进度条
            
        Returns:
            实体组字典 {canonical_entity: [duplicates]}
        """
        print(f"\n对齐 {len(entities)} 个实体...")
        
        # 去重
        unique_entities = list(set(entities))
        print(f"去重后: {len(unique_entities)} 个唯一实体")
        
        # 存储已处理的实体
        processed = set()
        entity_groups = {}
        
        iterator = tqdm(unique_entities) if show_progress else unique_entities
        
        for entity in iterator:
            # 如果已处理，跳过
            if entity in processed:
                continue
            
            # 找到所有重复实体
            duplicates = self.find_duplicate_entities(entity, unique_entities)
            
            if duplicates:
                # 选择最短的作为规范名称 
                all_variants = [entity] + duplicates
                canonical = min(all_variants, key=len)
                
                entity_groups[canonical] = all_variants
                
                # 标记所有变体为已处理
                processed.update(all_variants)
            else:
                # 没有重复，单独成组
                entity_groups[entity] = [entity]
                processed.add(entity)
        
        print(f"✓ 对齐完成，共 {len(entity_groups)} 个实体组")
        
        return entity_groups
    
    def align_triples(
        self,
        triples: List[Tuple[str, str, str]],
        show_progress: bool = True
    ) -> Tuple[List[Tuple[str, str, str]], Dict[str, str]]:
        """
        对三元组进行实体对齐
        
        Args:
            triples: 三元组列表 [(head, rel, tail), ...]
            show_progress: 是否显示进度条
            
        Returns:
            (对齐后的三元组列表, 实体映射字典)
        """
        print(f"\n对齐三元组中的实体...")
        print(f"原始三元组数量: {len(triples)}")
        
        # 提取所有实体
        all_entities = set()
        for head, rel, tail in triples:
            all_entities.add(head)
            all_entities.add(tail)
        
        print(f"唯一实体数量: {len(all_entities)}")
        
        # 对齐实体
        entity_groups = self.align_entities(list(all_entities), show_progress)
        
        # 构建实体映射: entity -> canonical_entity
        entity_mapping = {}
        for canonical, variants in entity_groups.items():
            for variant in variants:
                entity_mapping[variant] = canonical
        
        # 应用映射到三元组
        aligned_triples = []
        for head, rel, tail in triples:
            aligned_head = entity_mapping.get(head, head)
            aligned_tail = entity_mapping.get(tail, tail)
            aligned_triples.append((aligned_head, rel, aligned_tail))
        
        # 去重
        aligned_triples = list(set(aligned_triples))
        
        print(f"✓ 对齐后三元组数量: {len(aligned_triples)}")
        print(f"✓ 实体数量变化: {len(all_entities)} -> {len(entity_groups)}")
        
        return aligned_triples, entity_mapping
    
    def merge_knowledge_sources(
        self,
        source_files: List[str],
        output_file: str
    ) -> Dict:
        """
        融合多个知识源
        
        融合三个数据源:
        1. 非结构化网页数据
        2. 半结构化临床数据
        3. 结构化公开图谱
        
        Args:
            source_files: 源文件路径列表
            output_file: 输出文件路径
            
        Returns:
            融合后的知识字典
        """
        print("\n" + "="*50)
        print("知识融合")
        print("="*50)
        
        # 加载所有数据源
        all_triples = []
        
        for i, source_file in enumerate(source_files, 1):
            print(f"\n加载数据源 {i}: {source_file}")
            
            with open(source_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取三元组
            if isinstance(data, list):
                for item in data:
                    if 'triples' in item:
                        all_triples.extend(item['triples'])
                    elif isinstance(item, (list, tuple)) and len(item) == 3:
                        all_triples.append(tuple(item))
            
            print(f"  加载了 {len(all_triples)} 个三元组")
        
        print(f"\n总计: {len(all_triples)} 个三元组")
        
        # 实体对齐
        aligned_triples, entity_mapping = self.align_triples(all_triples)
        
        # 构建知识图谱数据
        kg_data = {
            'triples': aligned_triples,
            'entity_mapping': entity_mapping,
            'statistics': {
                'original_triples': len(all_triples),
                'aligned_triples': len(aligned_triples),
                'original_entities': len(set(
                    [e for t in all_triples for e in [t[0], t[2]]]
                )),
                'aligned_entities': len(set(
                    [e for t in aligned_triples for e in [t[0], t[2]]]
                )),
                'entity_groups': len(set(entity_mapping.values()))
            }
        }
        
        # 保存结果
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*50)
        print("融合完成！")
        print("="*50)
        print(f"三元组: {kg_data['statistics']['original_triples']} -> {kg_data['statistics']['aligned_triples']}")
        print(f"实体: {kg_data['statistics']['original_entities']} -> {kg_data['statistics']['aligned_entities']}")
        print(f"保存到: {output_file}")
        
        return kg_data


class ThresholdOptimizer:
    """阈值优化器 - 寻找最佳t1和t2"""
    
    def __init__(self, aligner: EntityAligner):
        self.aligner = aligner
    
    def evaluate_thresholds(
        self,
        validation_pairs: List[Tuple[str, str, int]],
        t1_range: Tuple[float, float] = (0.70, 0.95),
        t2_range: Tuple[float, float] = (0.70, 0.95),
        step: float = 0.05
    ):
        """
        评估不同阈值组合的性能
        
        Args:
            validation_pairs: 验证对 [(entity1, entity2, is_duplicate), ...]
            t1_range: Jaccard阈值范围
            t2_range: SBERT阈值范围
            step: 步长
            
        Returns:
            最佳阈值和性能指标
        """
        print("优化阈值...")
        
        best_f1 = 0
        best_t1 = 0.85
        best_t2 = 0.85
        
        t1_values = np.arange(t1_range[0], t1_range[1], step)
        t2_values = np.arange(t2_range[0], t2_range[1], step)
        
        results = []
        
        for t1 in t1_values:
            for t2 in t2_values:
                # 设置阈值
                self.aligner.jaccard_threshold = t1
                self.aligner.sbert_threshold = t2
                
                # 评估
                tp = fp = fn = 0
                
                for entity1, entity2, is_duplicate in validation_pairs:
                    # 预测
                    jaccard_sim = self.aligner.jaccard_similarity(entity1, entity2)
                    
                    predicted_duplicate = False
                    if jaccard_sim >= t1:
                        sbert_sim = self.aligner.sbert_similarity(entity1, entity2)
                        if sbert_sim >= t2:
                            predicted_duplicate = True
                    
                    # 统计
                    if predicted_duplicate and is_duplicate:
                        tp += 1
                    elif predicted_duplicate and not is_duplicate:
                        fp += 1
                    elif not predicted_duplicate and is_duplicate:
                        fn += 1
                
                # 计算指标
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                results.append({
                    't1': t1,
                    't2': t2,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
                
                # 更新最佳
                if f1 > best_f1:
                    best_f1 = f1
                    best_t1 = t1
                    best_t2 = t2
        
        print(f"\n最佳阈值:")
        print(f"  t1 (Jaccard): {best_t1:.2f}")
        print(f"  t2 (SBERT): {best_t2:.2f}")
        print(f"  F1 Score: {best_f1:.4f}")
        
        return best_t1, best_t2, results


if __name__ == "__main__":
    # 测试实体对齐
    
    # 创建对齐器
    aligner = EntityAligner(
        jaccard_threshold=0.85,
        sbert_threshold=0.85
    )
    
    # 测试实体
    test_entities = [
        "肺癌",
        "肺部肿瘤",
        "肺腺癌",
        "小细胞肺癌",
        "慢性阻塞性肺疾病",
        "慢阻肺",
        "COPD",
        "咳嗽",
        "咳嗽症状"
    ]
    
    print("\n测试实体对齐:")
    entity_groups = aligner.align_entities(test_entities, show_progress=False)
    
    for canonical, variants in entity_groups.items():
        if len(variants) > 1:
            print(f"\n规范名: {canonical}")
            print(f"变体: {', '.join(variants)}")

