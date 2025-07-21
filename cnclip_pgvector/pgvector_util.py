import numpy as np
from pgvector.psycopg2 import register_vector
import torch
import psycopg2
from psycopg2.extensions import connection as _connection

TEXT_VECTOR_COLUMN = "text_feature_vector"
IMAGE_VECTOR_COLUMN = "image_feature_vector"


def vector_to_pg_str(vector: np.ndarray | torch.Tensor) -> str:
    """
    将向量转为 pgvector 需要的 SQL 字符串格式
    """
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().cpu().numpy()
    return ','.join(f"{x:.6f}" for x in vector.tolist())


def query_similar_features(
    query_vector: np.ndarray | torch.Tensor,
    table_name: str,
    conn: _connection,
    offset: int = 0,
    limit: int = 20,
    probes: int = 10,
    vector_column: str = "embedding"
):
    """
    用 pgvector 加速从 PostgreSQL 检索最相似向量

    参数说明：
    - query_vector: 查询向量（np 或 torch.Tensor）
    - table_name: pgvector 表名（如 'text_features', 'image_features'）
    - conn: psycopg2 的连接对象（从连接池 db_pool.getconn() 获取）
    - limit: 返回前 K 个最近邻
    - probes: 搜索 probes 个簇（用于 IVFFlat 索引）
    - vector_column: 向量列名，默认是 embedding
    """
    cur = conn.cursor()

    # 1. 设置 pgvector 索引使用的 probes 数量
    cur.execute(f"SET ivfflat.probes = {probes};")

    # 2. 构造向量字符串
    vector_str = vector_to_pg_str(query_vector)

    # 3. 构造 SQL 查询语句（使用 <=> 进行加速）
    sql = f"""
        SELECT record_id, 1 - ({vector_column} <=> '[{vector_str}]') AS similarity
        FROM {table_name}
        ORDER BY {vector_column} <=> '[{vector_str}]'
        OFFSET {offset}
        LIMIT {limit};
    """

    # 4. 执行查询
    cur.execute(sql)
    rows = cur.fetchall()

    cur.close()
    return rows  # [(record_id, similarity_score), ...]

def save_features_to_db(features, record_ids, table_name, connection):
    register_vector(connection)
    cur = connection.cursor()
    
    # 批量插入特征向量
    for feature, record_id in zip(features, record_ids):
        cur.execute(
            f"INSERT INTO {table_name} (record_id, feature_vector) VALUES (%s, %s)",
            (record_id, feature.numpy())
        )
    connection.commit()





