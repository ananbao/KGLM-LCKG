"""
构建Neo4j知识图谱脚本
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from src.visualization.neo4j_builder import Neo4jKnowledgeGraph


def main():
    parser = argparse.ArgumentParser(description="构建Neo4j知识图谱")
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='对齐后的知识数据文件'
    )
    parser.add_argument(
        '--neo4j_uri',
        type=str,
        default='bolt://localhost:7687',
        help='Neo4j服务器地址'
    )
    parser.add_argument(
        '--neo4j_user',
        type=str,
        default='neo4j',
        help='Neo4j用户名'
    )
    parser.add_argument(
        '--neo4j_password',
        type=str,
        default='password',
        help='Neo4j密码'
    )
    parser.add_argument(
        '--clear_existing',
        action='store_true',
        help='清空现有数据'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("构建Neo4j知识图谱")
    print("="*60)
    print(f"输入文件: {args.input_file}")
    print(f"Neo4j URI: {args.neo4j_uri}")
    print(f"清空现有数据: {args.clear_existing}")
    print("="*60 + "\n")
    
    # 创建知识图谱
    kg = Neo4jKnowledgeGraph(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password
    )
    
    try:
        # 构建图谱
        kg.build_from_file(
            input_file=args.input_file,
            clear_existing=args.clear_existing
        )
        
        print("\n" + "="*60)
        print("知识图谱构建完成！")
        print("="*60)
        print(f"可以通过Neo4j Browser访问: http://localhost:7474")
        print("示例查询: MATCH (n:Entity {name:'肺癌'})-[r]->(m) RETURN n,r,m")
        
    finally:
        kg.close()


if __name__ == "__main__":
    main()

