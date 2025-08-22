# uvicorn main:app --reload --host 127.0.0.1 --port 8002
# uvicorn main:app  --host 127.0.0.1 --port 8002 

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from PIL import Image
import base64, io, gc
import os
from typing import List, Optional
from uuid import uuid4
import time
from utils import load_surf_checkpoint_model_from_base, verify_api_key, fix_base64_padding
import cn_clip.clip as clip
from log import log_timing
from prometheus_client import start_http_server
from pg_utils import (
    init_db, close_db, get_all_image_tables, get_all_text_tables, query_similar_features,
    get_text_record_by_id, get_image_record_by_id, get_record_element_by_id, get_conn, put_conn
)

app = FastAPI()
model = None
preprocess = None
model_lock = torch.multiprocessing.Lock()
device = "cuda" if torch.cuda.is_available() else "cpu"
API_KEYS = {"demo": "surf_demo_api_key"}

# ---- 简易会话缓存（用于分页复用首轮检索结果，减少重复计算） ----
CACHE_TTL_SECONDS = 300
PREFETCH_LIMIT = 200
SESSION_CACHE = {}

@app.on_event("startup")
async def startup_event():
    global model, preprocess
    start_http_server(8001)
    init_db()
    model, preprocess = load_surf_checkpoint_model_from_base(
        ckpt_path="/home/hsh/surf/pgvector69/epoch_latest.pt"
    )
    model.eval()
    print("CN-CLIP 模型与数据库初始化完成")

@app.on_event("shutdown")
def shutdown_event():
    close_db()
    print("服务器关闭，数据库连接释放")

@app.get("/")
async def root():
    return {"message": "Multimodal Retrieval API is running."}

# ---------- 图搜文 ----------
@app.post("/image-to-text/")
@log_timing("图搜文")
async def image_to_text(
    query_image: str = Body(...),
    offset: int = Body(0),
    limit: int = Body(20)
):

    try:
        img_bytes = base64.b64decode(fix_base64_padding(query_image))
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    with model_lock:
        with torch.no_grad():
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(image_tensor).float()
            image_features /= image_features.norm(dim=1, keepdim=True)
            image_features = image_features.cpu()

    torch.cuda.empty_cache(); gc.collect()

    conn = get_conn()
    try:
        tables = get_all_text_tables()
        topk = query_similar_features(
            query_vector=image_features.squeeze(0),
            table_names=tables,
            record_column_name="text_id",
            vector_column="text_feature",
            conn=conn,
            offset=offset,
            limit=limit
        )
    finally:
        put_conn(conn)

    score = np.array([sim for _, _, sim in topk])
    # 相对分数归一化的参考值（Top-1 分数）。若无结果则设为极小值避免除零
    top_score = float(score.max()) if score.size > 0 else 1e-12


    results = []
    for i, (table, text_id, score) in enumerate(topk):
        record = get_text_record_by_id(table, text_id)
        if record is not None:
            result_item = {
                "rank": i + 1,
                # "score": round(score * 100, 3),
                "score": round((score / top_score) * 100.0, 3),
                "table": table,
                "record": get_record_element_by_id(table, text_id, record)
            }
            results.append(result_item)
        
    return {
        "query": "image",
        "offset": offset,
        "limit": limit,
        "results": results
    }


# ---------- 文搜图 ----------
@app.post("/text-to-image/")
@log_timing("文搜图")
async def text_to_image(
    query_text: str = Body(...),
    offset: int = Body(0),
    limit: int = Body(20),
    session_id: Optional[str] = Body(None)
):

    now_ts = time.time()

    # 命中缓存：相同 session 且 TTL 内且 query_text 一致
    full_topk = None
    if session_id is not None:
        entry = SESSION_CACHE.get(session_id)
        if entry and entry.get("query_text") == query_text and (now_ts - entry.get("ts", 0)) < CACHE_TTL_SECONDS:
            full_topk = entry.get("topk")

    if full_topk is None:
        # 首次或缓存失效：计算一次，并预取较多候选
        text = clip.tokenize([query_text]).to("cuda")
        with torch.no_grad():
            text_feature = model.encode_text(text)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature.cpu().numpy()[0]

        conn = get_conn()
        try:
            tables = get_all_image_tables()
            full_topk = query_similar_features(
                query_vector=text_feature,
                table_names=tables,
                record_column_name="image_id",
                vector_column="image_feature",
                conn=conn,
                offset=0,
                limit=PREFETCH_LIMIT
            )
        finally:
            put_conn(conn)

        # 写入缓存
        if session_id is None:
            session_id = str(uuid4())
        SESSION_CACHE[session_id] = {
            "ts": now_ts,
            "query_text": query_text,
            "topk": full_topk,
        }

    # 当前页切片 + 使用全局 Top-1 归一化，确保只有全局 rank1 为 100
    page_slice = full_topk[offset: offset + limit]
    global_sims = np.array([sim for _, _, sim in full_topk])
    global_top = float(global_sims.max()) if global_sims.size > 0 else 1e-12

    results = []
    for i, (table, image_id, sim) in enumerate(page_slice):
        record = get_image_record_by_id(table, image_id)
        if record is None:
            continue
        results.append({
            "rank": offset + i + 1,
            "score": round((sim / global_top) * 100.0, 3),
            "table": table,
            "record": get_record_element_by_id(table, image_id, record)
        })

    return {
        "query": query_text,
        "offset": offset,
        "limit": limit,
        "results": results,
        "session_id": session_id,
    }
