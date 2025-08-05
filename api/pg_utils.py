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

# 基础数据库操作
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

# -----mapping-----
# 动态获取所有子表名
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
# score/similarity = 1 - distance
# probes是会话级别的设置，只影响当前连接的后续查询
# limit = 检索结果数量(LIMIT_PER_PAGE)
def query_similar_features(
    query_vector: np.ndarray | torch.Tensor,
    table_names: list[str],
    record_column_name: str,
    vector_column: str,
    conn: _connection,
    offset: int = 0,
    limit: int = 20,
    probes: int = 10
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
            print(f"[DEBUG] 查询记录 {table}.{image_id} → {row}")
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
            print(f"[DEBUG] 查询记录 {table}.{text_id} → {row}")
            record = dict(row) if row else None 
            if "text_feature" in record:
                    del record["text_feature"]
            return record
    finally:
        put_conn(conn)

def get_record_element_by_id(table: str, id: str, record: dict) -> dict:
    """
    将数据库中返回的 record（dict）拆解成 json_data 格式，字段完全动态。
    """
    if not record:
        return {
            "table": table,
            "id": id,
            "fields": None  # 或者返回空字典 {}
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


