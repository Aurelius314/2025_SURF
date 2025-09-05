# 在bash中，通过以下两行命令进入retrieval_db
# export PATH=$HOME/postgresql-16.2/bin:$PATH
# psql -d retrieval_db

from psycopg2 import pool
import numpy as np
from pgvector.psycopg2 import register_vector
import torch
import psycopg2
from psycopg2.extensions import connection as _connection
from psycopg2.extras import RealDictCursor
from config import get_probes_for_task

# ---基础数据库操作---
db_pool = None 

def init_db():
    global db_pool
    db_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        host='localhost',
        port='5432',
        database='retrieval_db'
    )

    with db_pool.getconn() as conn:
        register_vector(conn)
        db_pool.putconn(conn)

def get_conn():
    conn = db_pool.getconn()
    register_vector(conn)
    return conn

def put_conn(conn):
    db_pool.putconn(conn)

def close_db():
    if db_pool:
        db_pool.closeall()

# ---动态获取所有子表名（在corpus version中未启用）---
def get_all_image_tables() -> list[str]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM image_tables")
            rows = cur.fetchall()
            return [row[0] for row in rows]
    finally:
        put_conn(conn)

def get_all_text_tables() -> list[str]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM text_tables")
            rows = cur.fetchall()
            return [row[0] for row in rows]
    finally:
        put_conn(conn)

# --- 在多个 pgvector 表中检索最相似的记录，返回全局 top-K ---
# lists and probes 在该函数中未生效，在corpus version中未启用
def query_similar_features(
    query_vector: np.ndarray | torch.Tensor,
    table_names: list[str],
    record_column_name: str,
    vector_column: str,
    conn: _connection,
    offset: int = 0,
    limit: int = 20,
    probes: int = 18  # 默认使用优化后的probes值
):

    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")

    # 拼接子查询
    query_vector_str = ','.join([f"{x:.6f}" for x in query_vector.tolist()])
    subqueries = []
    for table in table_names:
        sub_sql = f"""
            SELECT 
                '{table}' AS table_name, 
                {record_column_name}, 
                1 - ({vector_column} <=> '[{query_vector_str}]') AS similarity
            FROM {table}
        """
        subqueries.append(sub_sql)

    union_sql = "\nUNION ALL\n".join(subqueries)

    final_sql = f"""
        SELECT * FROM (
            {union_sql}
        ) AS all_results
        ORDER BY similarity DESC
        OFFSET {offset}
        LIMIT {limit};
    """

    cur.execute(final_sql)
    rows = cur.fetchall()
    cur.close()

    # 返回格式: [(table_name, record_id, similarity), ...]
    return rows

#-----由id构造完整数据-----
def get_image_record_by_id(table, image_id):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT *, 
                       substring(image_feature::text, 1, 20) || '...' AS image_feature_preview 
                FROM {table} 
                WHERE image_id = %s
            """, (str(image_id),))
            row = cur.fetchone()
            # print(f"[DEBUG] 查询记录 {table}.{image_id} → {row}")
            record = dict(row) if row else None 
            if record:
                if "image_feature" in record:
                    del record["image_feature"]
            return record
    finally:
        put_conn(conn)

def get_text_record_by_id(table, text_id):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT *, 
                       substring(text_feature::text, 1, 20) || '...' AS text_feature_preview 
                FROM {table} 
                WHERE text_id = %s
            """, (str(text_id),))
            row = cur.fetchone()
            # print(f"[DEBUG] 查询记录 {table}.{text_id} → {row}")
            record = dict(row) if row else None 
            if "text_feature" in record:
                    del record["text_feature"]
            return record
    finally:
        put_conn(conn)

#-----将record（dict）拆解成动态json_data字段-----
def get_record_element_by_id(table: str, id: str, record: dict) -> dict:
    if not record:
        return {
            "table": table,
            "id": id,
            "fields": None 
        }

    json_data = {
        "table": table,
        "id": id,
        "fields": {}
    }

    # 遍历 record 的所有字段（已排除 feature）
    for key, value in record.items():
        json_data["fields"][key] = value

    return json_data

# 直接在image_corpus和text_corpus中进行向量检索 (corpus version)

def query_similar_features_from_corpus(
    query_vector: np.ndarray,
    corpus_table: str,  # 'image_corpus' 或 'text_corpus'
    conn: _connection,
    offset: int = 0,
    limit: int = 20,
    probes: int = None
):
    """
    直接在corpus表中检索相似特征
    
    Args:
        query_vector: 查询向量
        corpus_table: corpus表名 ('image_corpus' 或 'text_corpus')
        conn: 数据库连接
        offset: 偏移量
        limit: 返回数量
        probes: probes参数，如果为None则自动选择
    
    Returns:
        [(feature_id, similarity), ...]
    """
    
    # 自动选择probes参数
    if probes is None:
        if corpus_table == 'image_corpus':
            probes = get_probes_for_task('text_to_image')  # 图片corpus用于text→image
        else:
            probes = get_probes_for_task('image_to_text')  # 文本corpus用于image→text
    
    # 确定向量列名 - corpus表结构: src, id, embedding
    vector_column = "embedding"  
    id_column = "id"           
    
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")
    
    # 构建查询向量字符串
    query_vector_str = ','.join([f"{x:.6f}" for x in query_vector.tolist()])
    
    # 直接在corpus表中检索
    sql = f"""
    SELECT 
        {id_column},
        1 - ({vector_column} <=> '[{query_vector_str}]') AS similarity
    FROM {corpus_table}
    ORDER BY {vector_column} <=> '[{query_vector_str}]'
    OFFSET {offset}
    LIMIT {limit};
    """
    
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    
    # 返回格式: [(id, similarity), ...]
    return rows

def find_records_by_features(
    feature_ids: list,
    corpus_table: str,
    conn: _connection
):
    """
    根据feature_id查找对应的记录信息
    
    Args:
        feature_ids: feature_id列表
        corpus_table: corpus表名
        conn: 数据库连接
    
    Returns:
        [{'id': id, 'src': src, 'table': table_name}, ...]
    """
    
    if not feature_ids:
        return []
    
    # 确定ID列名 - corpus表结构: src, id, embedding
    id_column = "id"      # 第2列
    src_column = "src"    # 第1列
    
    # 构建IN查询
    id_list = ','.join([f"'{id}'" for id in feature_ids])
    
    sql = f"""
    SELECT {id_column}, {src_column}
    FROM {corpus_table}
    WHERE {id_column} IN ({id_list})
    ORDER BY array_position(ARRAY[{id_list}], {id_column}::text);
    """
    
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    
    # 解析src字段，提取表名和记录ID
    results = []
    for row in rows:
        feature_id, src = row
        # src格式: "surf_image:1" 或 "surf_text:1"
        if ':' in src:
            table_name, record_id = src.split(':', 1)
            results.append({
                'feature_id': feature_id,
                'table': table_name,
                'record_id': record_id,
                'src': src
            })
        else:
            # 如果src格式不同，直接使用
            results.append({
                'feature_id': feature_id,
                'table': src,  # 整个src作为表名
                'record_id': feature_id,  # 使用feature_id作为记录ID
                'src': src
            })
    
    return results

def query_similar_features_corpus_optimized(
    query_vector: np.ndarray,
    corpus_table: str,
    conn: _connection,
    offset: int = 0,
    limit: int = 20,
    probes: int = None
):
    """
    优化的corpus检索函数
    返回完整的记录信息
    """
    
    # 1. 在corpus中检索相似特征
    feature_results = query_similar_features_from_corpus(
        query_vector, corpus_table, conn, offset, limit, probes
    )
    
    if not feature_results:
        return []
    
    # 2. 提取feature_id
    feature_ids = [row[0] for row in feature_results]
    
    # 3. 查找对应的记录信息
    record_info = find_records_by_features(feature_ids, corpus_table, conn)
    
    # 4. 合并结果
    final_results = []
    for i, (feature_id, similarity) in enumerate(feature_results):
        if i < len(record_info):
            record = record_info[i]
            final_results.append({
                'table': record['table'],
                'record_id': record['record_id'],
                'feature_id': feature_id,
                'similarity': similarity
            })
    
    return final_results


