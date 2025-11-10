"""
评估指标 - Precision, Recall, F1 Score
"""

from typing import List, Tuple, Dict
from collections import defaultdict


def extract_entities_and_relations(triples: List[Tuple[str, str, str]]):
    """从三元组中提取实体和关系"""
    entities = set()
    relations = set()
    
    for head, rel, tail in triples:
        entities.add(head)
        entities.add(tail)
        relations.add(rel)
    
    return entities, relations


def calculate_metrics(
    predicted_triples: List[Tuple[str, str, str]],
    gold_triples: List[Tuple[str, str, str]],
    mode: str = "strict"
) -> Dict[str, float]:
    """
    计算关系抽取的评估指标
    
    Args:
        predicted_triples: 预测的三元组
        gold_triples: 标准答案三元组
        mode: 评估模式
            - "strict": 严格匹配 (头实体、关系、尾实体都要完全匹配)
            - "boundaries": 边界匹配 (只要实体边界正确)
            - "relaxed": 宽松匹配 (允许实体部分重叠)
    
    Returns:
        包含Precision, Recall, F1的字典
    """
    # 转换为集合
    pred_set = set(predicted_triples)
    gold_set = set(gold_triples)
    
    # 计算交集
    if mode == "strict":
        correct = pred_set & gold_set
    else:
        # 其他模式的实现
        correct = set()
        for pred in pred_set:
            for gold in gold_set:
                if is_match(pred, gold, mode):
                    correct.add(pred)
                    break
    
    # 计算TP, FP, FN
    tp = len(correct)
    fp = len(pred_set) - tp
    fn = len(gold_set) - tp
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def is_match(pred_triple: Tuple, gold_triple: Tuple, mode: str) -> bool:
    """判断两个三元组是否匹配"""
    pred_h, pred_r, pred_t = pred_triple
    gold_h, gold_r, gold_t = gold_triple
    
    if mode == "strict":
        return pred_h == gold_h and pred_r == gold_r and pred_t == gold_t
    
    elif mode == "boundaries":
        # 实体边界匹配
        return (
            pred_h in gold_h or gold_h in pred_h
        ) and pred_r == gold_r and (
            pred_t in gold_t or gold_t in pred_t
        )
    
    elif mode == "relaxed":
        # 宽松匹配: 实体有重叠即可
        h_match = any(c in gold_h for c in pred_h) or any(c in pred_h for c in gold_h)
        t_match = any(c in gold_t for c in pred_t) or any(c in pred_t for c in gold_t)
        r_match = pred_r == gold_r
        
        return h_match and r_match and t_match
    
    return False


def calculate_metrics_by_relation(
    predicted_triples: List[Tuple[str, str, str]],
    gold_triples: List[Tuple[str, str, str]]
) -> Dict[str, Dict[str, float]]:
    """按关系类型分别计算指标"""
    # 按关系分组
    pred_by_rel = defaultdict(list)
    gold_by_rel = defaultdict(list)
    
    for h, r, t in predicted_triples:
        pred_by_rel[r].append((h, r, t))
    
    for h, r, t in gold_triples:
        gold_by_rel[r].append((h, r, t))
    
    # 计算每个关系的指标
    all_relations = set(pred_by_rel.keys()) | set(gold_by_rel.keys())
    
    results = {}
    for rel in all_relations:
        pred = pred_by_rel.get(rel, [])
        gold = gold_by_rel.get(rel, [])
        
        metrics = calculate_metrics(pred, gold)
        results[rel] = metrics
    
    # 计算宏平均
    macro_precision = sum(m['precision'] for m in results.values()) / len(results)
    macro_recall = sum(m['recall'] for m in results.values()) / len(results)
    macro_f1 = sum(m['f1'] for m in results.values()) / len(results)
    
    results['macro_avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1
    }
    
    return results


def evaluate_model(
    model,
    test_data: List[Dict],
    extract_fn
) -> Dict[str, float]:
    """
    评估模型在测试集上的性能
    
    Args:
        model: 模型对象
        test_data: 测试数据 (包含text和triples)
        extract_fn: 抽取函数 (接受model和text，返回三元组列表)
    
    Returns:
        评估指标字典
    """
    all_pred = []
    all_gold = []
    
    for item in test_data:
        text = item['text']
        gold_triples = item['triples']
        
        # 预测
        pred_triples = extract_fn(model, text)
        
        all_pred.extend(pred_triples)
        all_gold.extend(gold_triples)
    
    # 计算指标
    metrics = calculate_metrics(all_pred, all_gold)
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "模型"):
    """打印评估指标"""
    print(f"\n{model_name} 评估结果:")
    print("-" * 40)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print("-" * 40)


def compare_models(results: Dict[str, Dict[str, float]]):
    """
    对比多个模型的结果
    
    Args:
        results: {model_name: metrics_dict}
    """
    print("\n" + "="*60)
    print("模型对比结果")
    print("="*60)
    
    # 表头
    print(f"{'模型':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-"*60)
    
    # 按F1排序
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['f1'],
        reverse=True
    )
    
    for model_name, metrics in sorted_models:
        print(
            f"{model_name:<20} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1']:<12.4f}"
        )
    
    print("="*60)


if __name__ == "__main__":
    # 测试评估指标
    
    # 示例数据
    predicted = [
        ("肺癌", "症状", "咳嗽"),
        ("肺癌", "症状", "胸痛"),
        ("肺癌", "检查方法", "CT扫描"),
        ("肺癌", "治疗方法", "手术"),  # FP
    ]
    
    gold = [
        ("肺癌", "症状", "咳嗽"),
        ("肺癌", "症状", "胸痛"),
        ("肺癌", "检查方法", "CT扫描"),
        ("肺癌", "治疗方法", "化疗"),  # FN
    ]
    
    # 计算指标
    metrics = calculate_metrics(predicted, gold)
    print_metrics(metrics, "测试模型")
    
    # 按关系计算
    print("\n按关系类型评估:")
    rel_metrics = calculate_metrics_by_relation(predicted, gold)
    for rel, m in rel_metrics.items():
        if rel != 'macro_avg':
            print(f"\n关系: {rel}")
            print(f"  P: {m['precision']:.4f}, R: {m['recall']:.4f}, F1: {m['f1']:.4f}")

