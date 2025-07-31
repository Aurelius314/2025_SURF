import os
import torch
import psycopg2
from psycopg2 import pool
from tqdm import tqdm
from glob import glob

# ====== 配置路径 ======
FEATURE_PATH = "/home/hsh/surf/cache/surf/surf_image_features.pt"
IMAGE_ROOTS = [
    "/mnt/disk9T/shared/datasets/敦煌检索/images_split/split1",
    "/mnt/disk9T/shared/datasets/敦煌检索/images_split/split2"
]
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
        CREATE TABLE IF NOT EXISTS surf_image (
            image_id TEXT PRIMARY KEY,
            image_feature VECTOR(1024),
            local_path TEXT,
            source TEXT
        );
    """)
    conn.commit()
    cur.close()
    db_pool.putconn(conn)

# ====== 加载本地路径映射 ======
def build_local_path_map():
    path_map = {}
    for root in IMAGE_ROOTS:
        image_files = glob(os.path.join(root, "*.png"))
        for path in image_files:
            filename = os.path.basename(path)
            image_id = os.path.splitext(filename)[0]
            path_map[int(image_id)] = path  # 注意 image_id 是 int 类型
    return path_map

# ====== 插入数据到数据库 ======
def insert_features():
    print("🔄 正在加载特征文件...")
    data = torch.load(FEATURE_PATH, map_location="cpu")
    image_ids = data["image_ids"].tolist()
    features = data["image_features"]

    print("📦 构建 image_id → local_path 映射...")
    id_to_path = build_local_path_map()

    print("📝 写入数据库...")
    conn = db_pool.getconn()
    cur = conn.cursor()

    inserted = 0
    for i in tqdm(range(len(image_ids))):
        image_id = int(image_ids[i])
        feature = features[i].tolist()
        local_path = id_to_path.get(image_id)

        if local_path is None:
            continue  # 跳过无路径文件

        try:
            cur.execute(
                "INSERT INTO surf_image (image_id, image_feature, local_path, source) VALUES (%s, %s, %s, %s)",
                (str(image_id), feature, local_path, SOURCE)
            )
            inserted += 1
        except psycopg2.errors.UniqueViolation:
            conn.rollback()
            continue
        except Exception as e:
            print(f"❌ 插入失败 image_id={image_id}: {e}")
            conn.rollback()
            continue

    conn.commit()
    cur.close()
    db_pool.putconn(conn)
    print(f"✅ 成功插入 {inserted} 条记录。")

# ====== 主流程 ======
if __name__ == "__main__":
    create_table()
    insert_features()
