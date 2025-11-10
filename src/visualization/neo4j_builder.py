"""
Neo4j知识图谱构建和可视化
"""

import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm


class Neo4jKnowledgeGraph:
    """Neo4j知识图谱构建器"""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password"
    ):
        """
        初始化Neo4j连接
        
        Args:
            uri: Neo4j服务器地址
            user: 用户名
            password: 密码
        """
        print("="*50)
        print("连接Neo4j数据库")
        print("="*50)
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            print(f"✓ 成功连接到 {uri}")
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            print("请确保Neo4j服务正在运行")
            raise
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            print("Neo4j连接已关闭")
    
    def clear_database(self):
        """清空数据库"""
        print("\n清空数据库...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ 数据库已清空")
    
    def create_entity_node(
        self,
        entity_name: str,
        entity_type: str = "Entity",
        properties: Optional[Dict] = None
    ):
        """
        创建实体节点
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型 (作为Neo4j标签)
            properties: 额外属性
        """
        if properties is None:
            properties = {}
        
        properties['name'] = entity_name
        
        with self.driver.session() as session:
            query = f"""
            MERGE (e:{entity_type} {{name: $name}})
            SET e += $properties
            RETURN e
            """
            session.run(query, name=entity_name, properties=properties)
    
    def create_relationship(
        self,
        head_entity: str,
        relation: str,
        tail_entity: str,
        head_type: str = "Entity",
        tail_type: str = "Entity",
        properties: Optional[Dict] = None
    ):
        """
        创建关系
        
        Args:
            head_entity: 头实体
            relation: 关系类型
            tail_entity: 尾实体
            head_type: 头实体类型
            tail_type: 尾实体类型
            properties: 关系属性
        """
        if properties is None:
            properties = {}
        
        # Neo4j关系类型不能包含某些特殊字符，进行清理
        clean_relation = relation.replace(" ", "_").replace("-", "_")
        
        with self.driver.session() as session:
            query = f"""
            MERGE (h:{head_type} {{name: $head}})
            MERGE (t:{tail_type} {{name: $tail}})
            MERGE (h)-[r:{clean_relation}]->(t)
            SET r += $properties
            RETURN r
            """
            session.run(
                query,
                head=head_entity,
                tail=tail_entity,
                properties=properties
            )
    
    def build_from_triples(
        self,
        triples: List[Tuple[str, str, str]],
        clear_existing: bool = True,
        batch_size: int = 1000
    ):
        """
        从三元组列表构建知识图谱
        
        Args:
            triples: 三元组列表 [(head, relation, tail), ...]
            clear_existing: 是否清空现有数据
            batch_size: 批次大小
        """
        print(f"\n构建知识图谱: {len(triples)} 个三元组")
        
        if clear_existing:
            self.clear_database()
        
        # 批量导入
        with self.driver.session() as session:
            for i in tqdm(range(0, len(triples), batch_size), desc="导入中"):
                batch = triples[i:i+batch_size]
                
                # 构建批量查询
                query = """
                UNWIND $triples AS triple
                MERGE (h:Entity {name: triple.head})
                MERGE (t:Entity {name: triple.tail})
                MERGE (h)-[r:RELATION {type: triple.relation}]->(t)
                """
                
                batch_data = [
                    {'head': h, 'relation': r, 'tail': t}
                    for h, r, t in batch
                ]
                
                session.run(query, triples=batch_data)
        
        print("✓ 知识图谱构建完成")
        
        # 统计信息
        self.print_statistics()
    
    def build_from_file(
        self,
        input_file: str,
        clear_existing: bool = True
    ):
        """
        从文件构建知识图谱
        
        Args:
            input_file: 输入文件 (JSON格式)
            clear_existing: 是否清空现有数据
        """
        print(f"\n从文件构建知识图谱: {input_file}")
        
        # 加载数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取三元组
        if isinstance(data, dict) and 'triples' in data:
            triples = [tuple(t) for t in data['triples']]
        elif isinstance(data, list):
            triples = []
            for item in data:
                if 'triples' in item:
                    triples.extend([tuple(t) for t in item['triples']])
                elif isinstance(item, (list, tuple)) and len(item) == 3:
                    triples.append(tuple(item))
        else:
            raise ValueError("不支持的文件格式")
        
        print(f"加载了 {len(triples)} 个三元组")
        
        # 构建图谱
        self.build_from_triples(triples, clear_existing)
    
    def print_statistics(self):
        """打印知识图谱统计信息"""
        with self.driver.session() as session:
            # 节点数
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()['count']
            
            # 关系数
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
            
            # 实体类型
            entity_types = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as type, count(*) as count
                ORDER BY count DESC
            """).data()
            
            # 关系类型
            relation_types = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """).data()
        
        print("\n" + "="*50)
        print("知识图谱统计信息")
        print("="*50)
        print(f"节点数: {node_count}")
        print(f"关系数: {rel_count}")
        
        print(f"\n实体类型 (Top 5):")
        for item in entity_types[:5]:
            print(f"  {item['type']}: {item['count']}")
        
        print(f"\n关系类型 (Top 10):")
        for item in relation_types:
            print(f"  {item['type']}: {item['count']}")
        
        print("="*50)
    
    def query_entity(self, entity_name: str, max_depth: int = 2):
        """
        查询实体的所有关系
        
        Args:
            entity_name: 实体名称
            max_depth: 最大深度
        
        Returns:
            查询结果
        """
        with self.driver.session() as session:
            query = f"""
            MATCH path = (s:Entity {{name: $name}})-[*1..{max_depth}]-(o)
            RETURN path
            LIMIT 100
            """
            result = session.run(query, name=entity_name)
            return result.data()
    
    def export_to_cypher(self, output_file: str):
        """
        导出为Cypher语句
        
        Args:
            output_file: 输出文件
        """
        print(f"\n导出Cypher语句到: {output_file}")
        
        with self.driver.session() as session:
            # 获取所有节点和关系
            nodes = session.run("MATCH (n) RETURN n").data()
            relationships = session.run("MATCH ()-[r]->() RETURN r, startNode(r) as start, endNode(r) as end").data()
        
        # 生成Cypher语句
        cypher_statements = []
        
        # 创建节点
        for node in nodes:
            n = node['n']
            name = n.get('name', '')
            labels = ':'.join(n.labels)
            cypher_statements.append(f'CREATE (:{labels} {{name: "{name}"}})')
        
        # 创建关系
        for rel in relationships:
            r = rel['r']
            start = rel['start']['name']
            end = rel['end']['name']
            rel_type = r.type
            cypher_statements.append(
                f'MATCH (s {{name: "{start}"}}), (e {{name: "{end}"}}) '
                f'CREATE (s)-[:{rel_type}]->(e)'
            )
        
        # 保存到文件
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cypher_statements))
        
        print(f"✓ 已导出 {len(cypher_statements)} 条Cypher语句")


def visualize_subgraph(
    kg: Neo4jKnowledgeGraph,
    entity_name: str,
    output_html: str = "outputs/subgraph.html"
):
    """
    可视化子图 (使用pyvis)
    
    Args:
        kg: 知识图谱对象
        entity_name: 中心实体
        output_html: 输出HTML文件
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("需要安装pyvis: pip install pyvis")
        return
    
    print(f"\n可视化实体 '{entity_name}' 的子图...")
    
    # 查询子图
    paths = kg.query_entity(entity_name, max_depth=2)
    
    # 创建网络图
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.barnes_hut()
    
    # 添加节点和边
    nodes_added = set()
    
    for path_data in paths:
        path = path_data['path']
        
        for node in path.nodes:
            node_name = node.get('name', '')
            if node_name not in nodes_added:
                net.add_node(
                    node_name,
                    label=node_name,
                    title=node_name,
                    color='#00ff1e' if node_name == entity_name else '#97c2fc'
                )
                nodes_added.add(node_name)
        
        for rel in path.relationships:
            start_node = rel.start_node.get('name', '')
            end_node = rel.end_node.get('name', '')
            rel_type = rel.type
            
            net.add_edge(start_node, end_node, label=rel_type, title=rel_type)
    
    # 保存
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.show(str(output_path))
    
    print(f"✓ 子图已保存到: {output_html}")


if __name__ == "__main__":
    # 测试Neo4j构建
    
    # 创建知识图谱
    kg = Neo4jKnowledgeGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="your_password"  # 修改为实际密码
    )
    
    # 测试三元组
    test_triples = [
        ("肺癌", "症状", "咳嗽"),
        ("肺癌", "症状", "胸痛"),
        ("肺癌", "检查方法", "CT扫描"),
        ("CT扫描", "检查结果", "磨玻璃结节"),
        ("肺癌", "治疗方法", "化疗"),
        ("化疗", "药物", "紫杉醇"),
        ("紫杉醇", "剂量", "175mg/m²"),
    ]
    
    # 构建图谱
    kg.build_from_triples(test_triples)
    
    # 查询
    print("\n查询 '肺癌' 实体:")
    results = kg.query_entity("肺癌")
    print(f"找到 {len(results)} 条路径")
    
    # 关闭连接
    kg.close()

