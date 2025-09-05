# uvicorn main_corpus:app --reload --host 127.0.0.1 --port 8003
# 使用corpus表进行查询的优化版本

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
from pg_utils import init_db, close_db, get_conn, put_conn, get_text_record_by_id, get_image_record_by_id, get_record_element_by_id, query_similar_features_corpus_optimized
from config import get_probes_for_task, RETRIEVAL_CONFIG, get_model_config, validate_config

app = FastAPI()
model = None
preprocess = None
model_lock = torch.multiprocessing.Lock()
device = "cuda" if torch.cuda.is_available() else "cpu"
API_KEYS = {"demo": "surf_demo_api_key"}

# ---- 简易会话缓存（用于分页复用首轮检索结果，减少重复计算） ----
CACHE_TTL_SECONDS = RETRIEVAL_CONFIG['cache_ttl_seconds']
PREFETCH_LIMIT = RETRIEVAL_CONFIG['prefetch_limit']
SESSION_CACHE = {}

@app.on_event("startup")
async def startup_event():
    global model, preprocess
    start_http_server(8001)
    
    # 验证配置
    validate_config()
    
    # 初始化数据库
    init_db() 
    
    # 获取模型配置并加载模型
    model_config = get_model_config()
    model, preprocess = load_surf_checkpoint_model_from_base(
        ckpt_path=model_config['checkpoint_path'],
        download_root=model_config['download_root']
    )
    model.eval()
    print("CN-CLIP 模型与数据库初始化完成 (corpus版本)")

@app.get("/")
async def root():
    return {"message": "Multimodal Retrieval API is running (corpus version)."}

@app.on_event("shutdown")
def shutdown_event():
    close_db()
    print("服务器关闭，数据库连接释放")

# ---------- 图搜文 ----------
@app.post("/image-to-text/")
@log_timing("图搜文(corpus)")
async def image_to_text(
    query_image: str = Body(...),
    offset: int = Body(0),
    limit: int = Body(RETRIEVAL_CONFIG['default_limit']),
    session_id: Optional[str] = Body(None)
):
    if limit <= 0 or limit > RETRIEVAL_CONFIG['max_limit']:
        raise HTTPException(
            status_code=400, 
            detail=f"Limit must be between 1 and {RETRIEVAL_CONFIG['max_limit']}"
        )
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset must be non-negative")

    now_ts = time.time()

    full_topk = None
    if session_id is not None:
        entry = SESSION_CACHE.get(session_id)
        if entry and entry.get("query_image") == query_image and (now_ts - entry.get("ts", 0)) < CACHE_TTL_SECONDS:
            full_topk = entry.get("topk")

    if full_topk is None:
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
            full_topk = query_similar_features_corpus_optimized(
                query_vector=image_features.squeeze(0),
                corpus_table="text_corpus",
                conn=conn,
                offset=0,
                limit=PREFETCH_LIMIT,
                probes=get_probes_for_task('image_to_text')
            )
        finally:
            put_conn(conn)

        if session_id is None:
            session_id = str(uuid4())
        SESSION_CACHE[session_id] = {
            "ts": now_ts,
            "query_image": query_image,
            "topk": full_topk,
        }

    # 当前页切片 + 使用全局 Top-1 归一化
    page_slice = full_topk[offset: offset + limit]
    global_sims = np.array([item['similarity'] for item in full_topk])
    global_top = float(global_sims.max()) if global_sims.size > 0 else 1e-12

    results = []
    for i, item in enumerate(page_slice):
        table = item['table']
        record_id = item['record_id']
        sim = item['similarity']
        
        if table and record_id:
            record = get_text_record_by_id(table, record_id)
            if record is not None:
                result_item = {
                    "rank": offset + i + 1,
                    "score": round((sim / global_top) * 100.0, 3),
                    "table": table,
                    "record": get_record_element_by_id(table, record_id, record)
                }
                results.append(result_item)
        
    return {
        "query": "image",
        "offset": offset,
        "limit": limit,
        "results": results,
        "session_id": session_id,
    }

# ---------- 文搜图 ----------
@app.post("/text-to-image/")
@log_timing("文搜图(corpus)")
async def text_to_image(
    query_text: str = Body(...),
    offset: int = Body(0),
    limit: int = Body(RETRIEVAL_CONFIG['default_limit']),
    session_id: Optional[str] = Body(None)
):
    if limit <= 0 or limit > RETRIEVAL_CONFIG['max_limit']:
        raise HTTPException(
            status_code=400, 
            detail=f"Limit must be between 1 and {RETRIEVAL_CONFIG['max_limit']}"
        )
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset must be non-negative")

    now_ts = time.time()

    full_topk = None
    if session_id is not None:
        entry = SESSION_CACHE.get(session_id)
        if entry and entry.get("query_text") == query_text and (now_ts - entry.get("ts", 0)) < CACHE_TTL_SECONDS:
            full_topk = entry.get("topk")

    if full_topk is None:
        text = clip.tokenize([query_text]).to("cuda")
        with torch.no_grad():
            text_feature = model.encode_text(text)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature.cpu().numpy()[0]

        conn = get_conn()
        try:
            # 直接在image_corpus中查询
            full_topk = query_similar_features_corpus_optimized(
                query_vector=text_feature,
                corpus_table="image_corpus",
                conn=conn,
                offset=0,
                limit=PREFETCH_LIMIT,
                probes=get_probes_for_task('text_to_image')
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

    # 当前页切片 + 使用全局 Top-1 归一化
    page_slice = full_topk[offset: offset + limit]
    global_sims = np.array([item['similarity'] for item in full_topk])
    global_top = float(global_sims.max()) if global_sims.size > 0 else 1e-12

    results = []
    for i, item in enumerate(page_slice):
        table = item['table']
        record_id = item['record_id']
        sim = item['similarity']
        
        if table and record_id:
            record = get_image_record_by_id(table, record_id)
            if record is not None:
                results.append({
                    "rank": offset + i + 1,
                    "score": round((sim / global_top) * 100.0, 3),
                    "table": table,
                    "record": get_record_element_by_id(table, record_id, record)
                })

    return {
        "query": query_text,
        "offset": offset,
        "limit": limit,
        "results": results,
        "session_id": session_id,
    }
