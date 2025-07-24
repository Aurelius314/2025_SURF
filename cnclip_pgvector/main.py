# 启动命令
# cd E:\surf\cnclip_pgvector
# uvicorn main:app --reload

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from PIL import Image
import sys, os, io, gc, base64
from pgvector_util import query_similar_features
from database import init_db, close_db, get_conn, put_conn
import cn_clip.clip as clip
import threading
import time
sys.path.append(r"E:\Chinese-CLIP-master")
sys.path.append(r"E:/surf")
from utils import load_surf_checkpoint_model_from_base
from lazy_load_utils import get_lmdb_record_by_text_id, get_lmdb_record_by_image_id
from pgvector_util import TEXT_VECTOR_COLUMN, IMAGE_VECTOR_COLUMN, TEXT_RECORD_ID_COLUMN, IMAGE_RECORD_ID_COLUMN

# ---------------- 初始化 ----------------
app = FastAPI()
model_lock = threading.Lock()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
preprocess = None
text_ids = []
original_texts = []
nld_texts = []
image_ids_list = []
img_rel_path = []
image_paths_list = []

IMAGE_ROOT = "E:/SURF2025/检索文化数据集/images/images"
LMDB_PATH = r"E:\surf\dataset"
API_KEYS = {"demo": "surf_demo_api_key"}

def verify_api_key(api_key: str):
    if api_key not in API_KEYS.values():
        raise HTTPException(status_code=403, detail="Unauthorized")

def fix_base64_padding(b64_string: str) -> str:
    return b64_string + '=' * ((4 - len(b64_string) % 4) % 4)

@app.on_event("startup")
async def startup_event():
    global model, preprocess, text_ids, original_texts, nld_texts
    global image_ids_list, img_rel_path, image_paths_list

    print("初始化数据库连接...")
    init_db()

    print("加载模型...")
    model, preprocess = load_surf_checkpoint_model_from_base(
        ckpt_path="E:/SURF2025/检索文化数据集/checkpoint/epoch_latest.pt"
    )

    model.eval()
    print("服务器初始化完成！")

@app.on_event("shutdown")
def shutdown_event():
    close_db()
    print("服务器关闭，无需手动释放连接池连接。")

@app.get("/")
def read_root():
    return {"message": "Welcome to the CN-CLIP API"}

# ---------------- 图搜文 ----------------
@app.post("/image-to-text/")
async def image_to_text(
    query_img_base64: str = Body(...),
    api_key: str = Body(...),
    offset: int = Body(0),
    limit: int = Body(20)
):
    global model, preprocess, text_ids, original_texts, nld_texts, img_rel_path

    LMDB_PATH = r"E:\surf\dataset"
    if model is None or preprocess is None:
        raise HTTPException(status_code=503, detail="Server is still initializing. Please try again later.")
    
    verify_api_key(api_key)

    total_start = time.time()


    # 1. 图像预处理
    stage1_start = time.time()
    try:
        img_bytes = base64.b64decode(fix_base64_padding(query_img_base64))
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    image_tensor = preprocess(image)
    image_input = image_tensor.unsqueeze(0).to(device)
    stage1_end = time.time()

    # 2. 模型编码
    stage2_start = time.time()
    with model_lock:
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features.float()
            image_features /= image_features.norm(dim=1, keepdim=True)
            image_features = image_features.cpu()
        torch.cuda.empty_cache()
        gc.collect()
    stage2_end = time.time()

    # 3. 向量检索
    stage3_start = time.time()
    conn = get_conn()
    # print("db_pool in handler:", conn)

    try:
        topk_records = query_similar_features(
            image_features.squeeze(0),
            'text_features',
            record_column_name=TEXT_RECORD_ID_COLUMN,
            conn=conn,
            offset=offset,
            limit=limit,
            probes=10,
            vector_column=TEXT_VECTOR_COLUMN
        )
        # print("图搜文耗时：", end_query - start_query)
    finally:
        conn = put_conn(conn)
    stage3_end = time.time()

    # 4. 结果构造（包括加载LMDB + 图像base64）
    stage4_start = time.time()
    # temperature缩放
    temperature = 0.05
    similarities = np.array([sim for _, sim in topk_records])
    exp_scores = np.exp(similarities / temperature)
    probs = exp_scores / np.sum(exp_scores)

    results = []
    for i, (rec_id, _) in enumerate(topk_records):
        rec_id_search_in_lmdb = int(rec_id)
        record = get_lmdb_record_by_text_id(LMDB_PATH, rec_id_search_in_lmdb)
        if not record:
            print(f"找不到 text_id {rec_id} 对应的记录")
            continue

        # image_basename = os.path.basename(record["image_paths"][0])
        image_basename = rec_id + ".png"
        image_abs_path = os.path.join(IMAGE_ROOT, image_basename).replace("\\", "/")
        print(f"[DEBUG] 尝试加载图像: rec_id={rec_id}, 路径={image_abs_path}")

        # 加载图像并转 base64
        image_base64 = ""
        if os.path.isfile(image_abs_path):
            with open(image_abs_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

        results.append({
            "rank": i + 1,
            "text_id": record["text_id"],
            "original_text": record["original_text"],
            "NLD_text": record["NLD_text"],
            "image_base64": image_base64,
            "score": round(probs[i] * 100, 3)
        })
    stage4_end = time.time()

    total_end = time.time()

    print("------FastAPI 后端耗时统计（图搜文）------")
    print(f"  预处理耗时：{stage1_end - stage1_start:.4f} 秒")
    print(f"  编码耗时：{stage2_end - stage2_start:.4f} 秒")
    print(f"  检索耗时：{stage3_end - stage3_start:.4f} 秒")
    print(f"  构造结果耗时：{stage4_end - stage4_start:.4f} 秒")
    print(f"  总耗时：{total_end - total_start:.4f} 秒")

    return{
        "query": query_img_base64[:30],
        "offset": offset,
        "limit": limit,
        "top_k_results": results
    }
    

# ---------------- 文搜图 ----------------
@app.post("/text-to-image/")
async def text_to_image(
    query_text: str = Body(...),
    api_key: str = Body(...),
    offset: int = Body(0),
    limit: int = Body(20)
):
    global model, text_ids, original_texts, nld_texts, image_ids_list, image_paths_list

    if model is None:
        raise HTTPException(status_code=503, detail="Server is still initializing. Please try again later.")
    verify_api_key(api_key)

    total_start = time.time()

    # 1. 文本预处理
    stage1_start = time.time()
    try:
        tokens = clip.tokenize([query_text]).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid text input: {e}")
    stage1_end = time.time()

    # 2. 模型编码
    stage2_start = time.time()
    with model_lock:
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features.float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu()
        torch.cuda.empty_cache()
        gc.collect()
    stage2_end = time.time()

    # 3. 向量检索
    stage3_start = time.time()
    conn = get_conn()
    print("db_pool in handler:", conn)
    try:
        # probes=5更快响应（可能略牺牲 recall）：
        topk_records = query_similar_features(
            text_features.squeeze(0),
            'image_features',
            record_column_name=IMAGE_RECORD_ID_COLUMN,
            conn=conn,
            offset=offset,
            limit=limit,
            probes=10,
            vector_column=IMAGE_VECTOR_COLUMN
        )
        # print("文搜图耗时：", end_query - start_query)
    finally:
        conn = put_conn(conn)
    stage3_end = time.time()

    # 4. 构造结果（包括LMDB + 图像base64）
    stage4_start = time.time()
    # temperature缩放
    temperature = 0.05
    similarities = np.array([sim for _, sim in topk_records])
    exp_scores = np.exp(similarities / temperature)
    probs = exp_scores / np.sum(exp_scores)

    results = []
    for i, (rec_id, _) in enumerate(topk_records):
        rec_id_int = int(rec_id)
        rec_id_search_in_lmdb = [rec_id_int]
        record = get_lmdb_record_by_image_id(LMDB_PATH, rec_id_search_in_lmdb)

        if not record:
            print(f"找不到 image_id {rec_id} 对应的记录")
            continue

        image_basename = rec_id + ".png"
        # image_basename = os.path.basename(record["image_paths"][0]) str 1.png
        image_abs_path = os.path.join(IMAGE_ROOT, image_basename).replace("\\", "/")
        # print(f"[DEBUG] 图像路径: rec_id={rec_id}, image_basename={image_basename}, image_abs_path={image_abs_path}")

        image_base64 = ""
        if os.path.isfile(image_abs_path):
            with open(image_abs_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
        else:
            print(f"⚠️ 缺失图像文件: {image_abs_path}")
            continue

        results.append({
            "rank": i + 1,
            "image_id": record["image_id"][0] if isinstance(record["image_id"], list) else record["image_id"],
            "text_id": record["text_id"],
            "original_text": record["original_text"],
            "NLD_text": record["NLD_text"],
            "image_path": image_abs_path,
            "image_base64": image_base64,
            "score": round(probs[i] * 100, 3)
        })

    stage4_end = time.time()
    total_end = time.time()

    # 打印耗时日志
    print("------FastAPI 后端耗时统计（文搜图）------")
    print(f"  预处理耗时：{stage1_end - stage1_start:.4f} 秒")
    print(f"  编码耗时：{stage2_end - stage2_start:.4f} 秒")
    print(f"  检索耗时：{stage3_end - stage3_start:.4f} 秒")
    print(f"  构造结果耗时：{stage4_end - stage4_start:.4f} 秒")
    print(f"  总耗时：{total_end - total_start:.4f} 秒")

    return {
        "query": query_text,
        "offset": offset,
        "limit": limit,
        "top_k_results": results
    }
