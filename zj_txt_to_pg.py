"""
-----text dataset in pgvector-----
table name: zj_text
column_1: text_id <---> 注意与image_id的逻辑关联
column_2: text_feature (from "language_description")
column_3: language_description
column_4: item_name
column_5: era
column_6: item_level
column_7: size
column_8: item_type
column_9: summary
column_10: source (浙江省博物院)
"""
import os
import json
import torch
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path

FEATURE_PATH = "/home/hsh/surf/cache/zj_text_features.pt"
JSON_ROOT = "/mnt/disk20T1/surf2025/Retrieval/jsonresult/result"
SOURCE = "浙江省博物院"

# === PostgreSQL 连接信息 ===
DB_CONFIG = {
    'dbname': 'retrieval_db',
    'host': 'localhost',
    'port': '5432'
}

# === 创建表 SQL ===
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS zj_text (
    text_id TEXT PRIMARY KEY,
    text_feature VECTOR(1024),
    language_description TEXT,
    item_name TEXT,
    era TEXT,
    item_level TEXT,
    size TEXT,
    item_type TEXT,
    summary TEXT,
    source TEXT
);
"""

def load_json_data(json_root):
    """加载所有 JSON 文件内容，构建一个 dict，key 为 text_id，value 为字段 dict"""
    data_dict = {}
    for json_file in Path(json_root).glob("*.json"):
        file_num = json_file.stem.split("_")[0]  # 例如 1_result.json -> "1"
        try:
            with json_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                for idx, item in enumerate(data):
                    text_id = f"{file_num}-{idx}"
                    record = {
                        "language_description": item.get("language_description", ""),
                        "item_name": item.get("名称", ""),
                        "era": item.get("藏品年代", ""),
                        "item_level": item.get("藏品级别", ""),
                        "size": item.get("尺寸(cm)", ""),
                        "item_type": item.get("馆藏类型", ""),
                        "summary": item.get("简介", ""),
                        "source": SOURCE
                    }
                    data_dict[text_id] = record
        except Exception as e:
            print(f"[ERROR] Failed to process {json_file}: {e}")
    return data_dict

def insert_data_to_pg(data, db_config):
    """批量插入数据到 PostgreSQL"""
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)

    records = []
    for row in data:
        records.append((
            row["text_id"],
            row["text_feature"],
            row["language_description"],
            row["item_name"],
            row["era"],
            row["item_level"],
            row["size"],
            row["item_type"],
            row["summary"],
            row["source"]
        ))

    insert_sql = """
    INSERT INTO zj_text (
        text_id, text_feature, language_description, item_name, era,
        item_level, size, item_type, summary, source
    )
    VALUES %s
    ON CONFLICT (text_id) DO NOTHING;
    """

    execute_values(cur, insert_sql, records)
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ 插入完成，共插入 {len(records)} 条记录")

def main():
    # 加载 text_features.pt
    print("🔍 正在加载特征文件...")
    pt_data = torch.load(FEATURE_PATH, map_location="cpu")
    text_ids = pt_data["text_ids"]
    text_features = pt_data["text_features"]

    print(f"✅ 成功加载 {len(text_ids)} 条特征")

    # 加载 JSON 原始信息
    print("🔍 正在加载原始 JSON 文本数据...")
    text_info_map = load_json_data(JSON_ROOT)

    # 构造用于插入的数据列表
    insert_data = []
    for i, text_id in enumerate(text_ids):
        if text_id not in text_info_map:
            print(f"[WARN] text_id {text_id} 在 JSON 文件中未找到，跳过")
            continue
        info = text_info_map[text_id]
        insert_data.append({
            "text_id": text_id,
            "text_feature": text_features[i].tolist(),  # 转为 list 存入 pgvector
            **info
        })

    print(f"📦 准备插入 {len(insert_data)} 条记录到 PostgreSQL")
    insert_data_to_pg(insert_data, DB_CONFIG)

if __name__ == "__main__":
    main()


