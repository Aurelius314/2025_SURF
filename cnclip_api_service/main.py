'''

这是一个由 FastAPI 搭建的示例后端，您可以通过在终端运行 `uvicorn main:app --reload` 启动并访问。

'''

import torch
from fastapi import FastAPI, HTTPException, Body
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
# import base64, io, os, numpy as np, lmdb, ast, pickle
import io, os, lmdb, pickle, base64
from lmdb import open as open_lmdb
# import pickle as pkl
from typing import List
from tqdm import tqdm
import sys  
sys.path.append(r'E:/surf')  # 添加该路径  
from utils import load_surf_checkpoint_model_from_base, load_text_data_from_lmdb, load_images_from_paths  
from cache_manager import cache_image_features, cache_text_features

# -------初始化-----------
app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_surf_checkpoint_model_from_base(
    ckpt_path="E:/SURF2025/检索文化数据集/checkpoint/epoch_latest.pt"
)
model.eval()

CACHE_DIR = "E:/surf/cache"
LMDB_PATH = "E:/surf/valid_output_lmdb"    # mock
FEATURES_CACHE = os.path.join(CACHE_DIR, "image_features.pt")
META_CACHE = os.path.join(CACHE_DIR, "image_meta.pkl")


# API Key 模拟
API_KEYS = {"demo": "your_demo_api_key"}

def verify_api_key(api_key: str):
    if api_key not in API_KEYS.values():
        raise HTTPException(status_code=403, detail="Unauthorized")
    
def fix_base64_padding(b64_string: str) -> str:
    return b64_string + '=' * ((4 - len(b64_string) % 4) % 4)

@app.get("/")
def read_root():
    return {"message": "Welcome to the CN-CLIP API"}

# -------------加载数据------------------------
text_ids, original_texts, nld_texts, image_ids_list, image_paths_list, splits = load_text_data_from_lmdb(LMDB_PATH)
image_list = load_images_from_paths(image_paths_list)

image_records = [
    {
        "image": image_list[i],
        "image_id": image_ids_list[i],
        "text_id": text_ids[i],
        "original_text": original_texts[i],
        "NLD_text": nld_texts[i],
        "image_path": image_paths_list[i][0] if image_paths_list[i] else ""
    }
    for i in range(len(image_list))
]

image_features_tensor, meta = cache_image_features(model, preprocess, image_records, CACHE_DIR)
text_ids = meta["text_ids"]
original_texts_list = meta["original_texts"]
nld_texts_list = meta["nld_texts"]
image_paths_list = meta["image_paths"]

# 缓存文本特征（基于 original_texts）
text_features_tensor = cache_text_features(model, meta["original_texts"], CACHE_DIR)



# --------加载缓存或重新提取
if os.path.exists(FEATURES_CACHE) and os.path.exists(META_CACHE):
    print("Loading cached image features and metadata...")
    image_features_tensor = torch.load(FEATURES_CACHE).to(device)
    # with open(META_CACHE, "rb") as f:
    #     image_ids_list, caption_list, tags_list = pickle.load(f)
    with open(META_CACHE, "rb") as f:
        meta = pickle.load(f)
    image_ids_list = meta["image_ids"]
    caption_list = meta["original_texts"]
    nld_texts_list = meta["nld_texts"]
    text_ids_list = meta["text_ids"]
    image_paths_list = meta["image_paths"]
    print(f"Loaded {len(image_ids_list)} image features.")
else:
    print("Loading LMDB records and image files...")


    # 提取图像路径列表
    env = open_lmdb(LMDB_PATH, readonly=True, lock=False)
    image_paths_list = []
    with env.begin() as txn:
        for key, value in txn.cursor():
            data = pickle.loads(value)
            image_paths_list.append(data["image_paths"])
    env.close()


    # 创建 'original_texts', 'nld_texts', no tags
    original_texts_list = original_texts
    nld_texts_list = nld_texts

    print("Calculating image features...")
    image_features_list = []
    for img in tqdm(image_list, desc="🔍 Extracting image features"):
        if img is None:
            image_features_list.append(torch.zeros(model.visual.output_dim))  # 占位向量
            continue
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
            feat /= feat.norm(dim=-1, keepdim=True) + 1e-7
            image_features_list.append(feat.cpu())

    image_features_tensor = torch.stack(image_features_list, dim=0)

    # 缓存保存
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(image_features_tensor, FEATURES_CACHE)
    with open(META_CACHE, "wb") as f:
        pickle.dump((image_ids_list, original_texts_list, nld_texts_list), f)
    print(f"Saved {len(image_ids_list)} features and metadata to cache.")


# -------------------- 图搜文 --------------------

@app.post("/image-to-text/")
async def image_to_text(img_base64: str = Body(...), api_key: str = Body(...)):
    try:
        verify_api_key(api_key)

        # 构建结构化记录
        text_records = [
            {
                "text_id": tid,
                "original_text": ori,
                "NLD_text": nld,
                "image_paths": path
            }
            for tid, ori, nld, path in zip(text_ids, original_texts_list, nld_texts_list, image_paths_list)
        ]

        # 去重原文
        seen = set()
        text_candidates, record_mapping = [], []
        for i, rec in enumerate(text_records):
            if rec["original_text"] not in seen:
                seen.add(rec["original_text"])
                text_candidates.append(rec["original_text"])
                record_mapping.append(i)

        print("使用的 text_candidates 数量:", len(text_candidates))

        img = Image.open(io.BytesIO(base64.b64decode(img_base64))).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True) + 1e-7
            text_tokens = clip.tokenize(text_candidates).to(device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True) + 1e-7
            logits_per_image, _ = model.get_similarity(img_tensor, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            # # 计算相似度（图 -> 文）点积
            # similarity = image_features @ text_features.T
            # probs = similarity.softmax(dim=-1).cpu().numpy()[0]

        # Top-K 匹配
        topk_indices = probs.argsort()[-5:][::-1]
        results = []
        for i, idx in enumerate(topk_indices):
            rec = text_records[record_mapping[idx]]
            results.append({
                "rank": i + 1,
                "text_id": rec["text_id"],
                "original_text": rec["original_text"],
                "NLD_text": rec["NLD_text"],
                "image_path": rec["image_paths"][0] if rec["image_paths"] else None,
                "score": round(probs[idx] * 100, 3)
            })
        return {"top_k_results": results}

    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        return {"error": str(e)}


# -------------------- 文搜图 --------------------

@app.post("/text-to-image/")
async def text_to_image(query_text: str = Body(...), api_key: str = Body(...)):
    verify_api_key(api_key)

    with torch.no_grad():
        # 文本向量化
        text_tokens = clip.tokenize([query_text]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True) + 1e-7
        logits = model.logit_scale.exp() * (text_features @ image_features_tensor.T)
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    topk_indices = probs.argsort()[-5:][::-1]
    results = []
    for i, idx in enumerate(topk_indices):
        img = image_list[idx]
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        results.append({
            "rank": i + 1,
            "image_id": image_ids_list[idx],
            "text_id": text_ids[idx],
            "original_text": original_texts_list[idx],
            "NLD_text": nld_texts_list[idx],
            "image_path": image_paths_list[idx][0] if image_paths_list[idx] else None,
            "image_base64": img_base64,
            "score": round(probs[idx] * 100, 3)
        })

    return {"top_k_results": results}


