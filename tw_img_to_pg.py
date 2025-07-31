"""
image dataset in pgvector
table name: tw_image
column_1: image_id
column_2: image_feature
column_3: local_path
column_4: url
column_5: source (台湾国立故宫博物院)
"""
import os
import csv
import torch
import psycopg2
from psycopg2 import pool
from tqdm import tqdm
from glob import glob

# ====== 配置路径 ======
FEATURE_PATH = "/home/hsh/surf/cache/tw/tw_image_features.pt"
FOUND_CSV = "/home/hsh/surf/cache/tw/tw_found_images.csv"
URL_CSV_ROOT = "/mnt/disk9T/shared/datasets/博物院/台湾国立故宫博物院/典藏资料"
SOURCE = "台湾国立故宫博物院"

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
        CREATE TABLE IF NOT EXISTS tw_image (
            image_id TEXT PRIMARY KEY,
            image_feature VECTOR(1024),
            local_path TEXT,
            url TEXT,
            source TEXT
        );
    """)
    conn.commit()
    cur.close()
    db_pool.putconn(conn)

# ====== 加载特征 ======
def load_features(path):
    print("Loading features...")
    raw = torch.load(path)
    return {k: v.tolist() for k, v in raw}

# ====== 加载 tw_found_images.csv：image_id → local_path ======
def load_local_paths(csv_path):
    print("Loading local paths...")
    mapping = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['image_id']] = row['local_path']
    return mapping

# ====== 加载 URL CSVs：image_id → url ======
def load_urls(root):
    print("Loading URLs from", root)
    mapping = {}
    for csv_file in glob(os.path.join(root, "imageid-*.csv")):
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row['image_id']
                url = row['url']
                if url:
                    mapping[image_id] = url
    return mapping

# ====== 插入数据 ======
def insert_data(features, local_paths, urls):
    conn = db_pool.getconn()
    cur = conn.cursor()

    inserted = 0
    skipped = 0
    for image_id, feat in tqdm(features.items(), desc="Inserting into tw_image"):
        if image_id not in local_paths:
            skipped += 1
            continue
        if image_id not in urls:
            skipped += 1
            continue

        local_path = local_paths[image_id]
        url = urls[image_id]

        try:
            cur.execute("""
                INSERT INTO tw_image (image_id, image_feature, local_path, url, source)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (image_id) DO NOTHING;
            """, (image_id, feat, local_path, url, SOURCE))
            inserted += 1
        except Exception as e:
            print(f"[Error] image_id: {image_id}, reason: {e}")
            skipped += 1

    conn.commit()
    cur.close()
    db_pool.putconn(conn)
    print(f"✅ Inserted: {inserted}, Skipped: {skipped}")

# ====== 主流程 ======
if __name__ == "__main__":
    create_table()
    features = load_features(FEATURE_PATH)
    local_paths = load_local_paths(FOUND_CSV)
    urls = load_urls(URL_CSV_ROOT)
    insert_data(features, local_paths, urls)

