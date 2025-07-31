"""
image dataset in pgvector
table name: zj_image
column_1: image_id
column_2: image_feature
column_3: local_path
column_4: url
column_5: source (浙江省博物院)
"""
import os
import csv
import torch
import psycopg2
from psycopg2 import pool
from tqdm import tqdm

# ====== 路径配置 ======
FEATURE_PATH = "/home/hsh/surf/cache/zj_image_features.pt"
FOUND_CSV = "/home/hsh/surf/cache/found_images.csv"
URL_CSV_ROOT = "/mnt/disk9T/shared/datasets/博物院/浙江省博物馆/馆藏精品"
SOURCE = "浙江省博物院"

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
        CREATE TABLE IF NOT EXISTS zj_image (
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

# ====== 加载所有 CSV 文件中的 URL（按 image_id 索引）======
def build_url_map():
    url_map = {}
    for file in os.listdir(URL_CSV_ROOT):
        if file.startswith("imageid-") and file.endswith(".csv"):
            path = os.path.join(URL_CSV_ROOT, file)
            with open(path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_id = row.get("image_id", "").strip()
                    url = row.get("url", "").strip()
                    if image_id and url:
                        url_map[image_id] = url
    return url_map

# ====== 插入数据到数据库 ======
def insert_features(feature_dict, records, url_map):
    conn = db_pool.getconn()
    cur = conn.cursor()

    # ==== ✅ 预读取已存在的 image_id 列表 ====
    print("Checking existing image_ids in zj_image...")
    cur.execute("SELECT image_id FROM zj_image;")
    existing_ids = set(row[0] for row in cur.fetchall())

    inserted = 0
    for record in tqdm(records, desc="Inserting records"):
        image_id = record["image_id"]

        # ==== ✅ 如果已经存在，就跳过 ====
        if image_id in existing_ids:
            continue

        feature_tensor = feature_dict.get(image_id)
        if feature_tensor is None:
            continue

        feature_list = feature_tensor.cpu().tolist()
        local_path = record["local_path"]
        url = url_map.get(image_id, "")

        try:
            cur.execute("""
                INSERT INTO zj_image (image_id, image_feature, local_path, url, source)
                VALUES (%s, %s, %s, %s, %s);
            """, (image_id, feature_list, local_path, url, SOURCE))
            inserted += 1
        except Exception as e:
            print(f"[⚠️ Error] {image_id}: {e}")

    conn.commit()
    cur.close()
    db_pool.putconn(conn)
    print(f"✅ Inserted {inserted} new records.")


# ====== 主程序 ======
def main():
    print("Loading image features...")
    loaded_data = torch.load(FEATURE_PATH, map_location="cpu")
    feature_dict = {image_id: feature for image_id, feature in loaded_data}

    print("Reading found_images.csv...")
    records = []
    with open(FOUND_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "image_id": row["image_id"].strip(),
                "local_path": row["local_path"].strip()
            })

    print("Building image_id → URL map...")
    url_map = build_url_map()

    print("Creating table if not exists...")
    create_table()

    print("Inserting into PostgreSQL + pgvector...")
    insert_features(feature_dict, records, url_map)

    print("Done.")

if __name__ == "__main__":
    main()
