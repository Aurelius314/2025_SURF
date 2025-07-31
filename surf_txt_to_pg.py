'''
-----text dataset in pgvector-----
table name: surf_text
column_1: text_id 
column_2: text_feature
column_3: original_text
column_4: NLD_text
column_10: source (SURF敦煌检索数据集)
'''

import os
import json
import torch
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from pathlib import Path

# 路径设置
FEATURE_PATH = "/home/hsh/surf/cache/surf/surf_text_features.pt"
NLD_ROOT = "/home/hsh/surf/cache/surf/surf_nld.jsonl"
ORIGINAL_ROOT = "/home/hsh/surf/cache/surf/surf_original.jsonl"
SOURCE = "SURF敦煌检索数据集"

# ====== 初始化数据库连接池 ======
db_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=5,
    host='localhost',
    port='5432',
    database='retrieval_db'
)

# ====== 建表 ======
def create_table():
    conn = db_pool.getconn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS surf_text (
            text_id TEXT PRIMARY KEY,
            text_feature VECTOR(1024),
            original_text TEXT,
            NLD_text TEXT,
            source TEXT
        );
    """)
    conn.commit()
    cur.close()
    db_pool.putconn(conn)

# ====== 读取特征文件 ======
print("加载 pt 特征文件中...")
data = torch.load(FEATURE_PATH)
text_ids_tensor = data["text_ids"]           # Tensor, shape: (N,)
features_tensor = data["text_features"]      # Tensor, shape: (N, 1024)

text_ids = [str(i.item()) for i in text_ids_tensor]
features = features_tensor.tolist()

# ====== 读取 jsonl 文本 ======
def load_texts(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip())["text"] for line in f]

original_texts = load_texts(ORIGINAL_ROOT)
nld_texts = load_texts(NLD_ROOT)

# ====== 长度一致性检查 ======
assert len(text_ids) == len(original_texts) == len(nld_texts), "数据长度不一致"

# ====== 创建表 ======
create_table()

# ====== 批量插入数据 ======
print("开始批量入库...")
conn = db_pool.getconn()
cur = conn.cursor()

insert_sql = """
INSERT INTO surf_text (text_id, text_feature, original_text, NLD_text, source)
VALUES %s
ON CONFLICT (text_id) DO NOTHING;
"""

records = [
    (text_ids[i], features[i], original_texts[i], nld_texts[i], SOURCE)
    for i in range(len(text_ids))
]

execute_values(cur, insert_sql, records, page_size=1000)
conn.commit()
cur.close()
db_pool.putconn(conn)

print(f"✅ 成功写入 {len(records)} 条文本记录到 surf_text 表中。")
