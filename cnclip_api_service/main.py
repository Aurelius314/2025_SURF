'''

这是一个由 FastAPI 搭建的示例后端，您可以通过在终端运行 `uvicorn main:app --reload` 启动并访问。

'''

# uvicorn main:app --reload
# curl http://127.0.0.1:8000/

import torch
from fastapi import FastAPI, HTTPException, Body
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
import base64, io, os, numpy as np, lmdb, ast, pickle
from typing import List
from tqdm import tqdm

# FastAPI 初始化
app = FastAPI()

# 设置设备和加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-H-14", device=device, download_root='./')
model.eval()

# API Key 模拟
API_KEYS = {"demo": "your_demo_api_key"}

def verify_api_key(api_key: str):
    if api_key not in API_KEYS.values():
        raise HTTPException(status_code=403, detail="Unauthorized")
    
def fix_base64_padding(b64_string: str) -> str:
    return b64_string + '=' * ((4 - len(b64_string) % 4) % 4)

# 根路由
@app.get("/")
def read_root():
    return {"message": "Welcome to the CN-CLIP API"}

# LMDB 数据加载函数
def load_images_from_lmdb(lmdb_path: str):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    images, image_ids, captions, tags_list = [], [], [], []
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                data = ast.literal_eval(value.decode("utf-8"))
                image_data = base64.b64decode(fix_base64_padding(data["image_base64"]))
                img = Image.open(io.BytesIO(image_data)).convert("RGB")
                images.append(img)
                image_ids.append(data["image_id"])
                captions.append(data["caption"])
                tags_list.append(data.get("tags", []))  # 提取 tags 字段
            except Exception as e:
                print(f"[WARN] Failed to load record {key}: {e}")
                print(f"Offending base64 string (truncated): {data.get('image_base64', '')[:100]}...")
    env.close()
    return image_ids, images, captions, tags_list


# 缓存路径
CACHE_DIR = "E:/surf/cache"
LMDB_PATH = "E:/surf/coco_lmdb_database_small"
FEATURES_CACHE = os.path.join(CACHE_DIR, "image_features.pt")
META_CACHE = os.path.join(CACHE_DIR, "image_meta.pkl")

# 加载缓存或重新提取
if os.path.exists(FEATURES_CACHE) and os.path.exists(META_CACHE):
    print("Loading cached image features and metadata...")
    image_features_tensor = torch.load(FEATURES_CACHE).to(device)
    with open(META_CACHE, "rb") as f:
        image_ids_list, caption_list, tags_list = pickle.load(f)
    _, image_list, _, _ = load_images_from_lmdb(LMDB_PATH)
    print(f"Loaded {len(image_ids_list)} image features.")
else:
    print("Loading LMDB records...")
    image_ids_list, image_list, caption_list, tags_list = load_images_from_lmdb(LMDB_PATH)

    print("Calculating image features...")
    image_features_list = []
    for img in tqdm(image_list, desc="🔍 Extracting image features"):
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
            feat /= feat.norm(dim=-1, keepdim=True) + 1e-7
            image_features_list.append(feat.cpu())

    image_features_tensor = torch.cat(image_features_list, dim=0)

    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(image_features_tensor, FEATURES_CACHE)
    with open(META_CACHE, "wb") as f:
        pickle.dump((image_ids_list, caption_list, tags_list), f)
    print(f"✅ Saved {len(image_ids_list)} features and metadata to cache.")


# -------------------- 图搜文 --------------------

@app.post("/image-to-text/")
async def image_to_text(img_base64: str = Body(...), api_key: str = Body(...)):
    try:
        verify_api_key(api_key)
        text_candidates = list(set(caption_list))  # 去重后的 captions
        print("使用的 text_candidates 数量:", len(text_candidates))

        img = Image.open(io.BytesIO(base64.b64decode(img_base64))).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        image_features = model.encode_image(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True) + 1e-7

        text_tokens = clip.tokenize(text_candidates).to(device)
        with torch.no_grad():
            logits_per_image, _ = model.get_similarity(img_tensor, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        topk_indices = probs.argsort()[-5:][::-1]
        results = [
            {"rank": i + 1, "text": text_candidates[idx], "score": round(probs[idx] * 100, 3)}
            for i, idx in enumerate(topk_indices)
        ]
        return {"top_k_results": results}
    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        return {"error": str(e)}

# -------------------- 文搜图 --------------------

@app.post("/text-to-image/")
async def text_to_image(query_text: str = Body(...), api_key: str = Body(...)):
    verify_api_key(api_key)

    def get_image_tags(image_id: str) -> List[str]:
        if image_id in image_ids_list:
            idx = image_ids_list.index(image_id)
            return tags_list[idx]
        return []
  
    global image_list 
    if "image_list" not in globals() or not image_list:
        _, image_list, _ = load_images_from_lmdb(LMDB_PATH)

    with torch.no_grad():
        text_tokens = clip.tokenize([query_text]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True) + 1e-7

        logits = model.logit_scale.exp() * (text_features @ image_features_tensor.T)
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
        

    # # 慢
    # img_tensors = torch.stack([preprocess(img) for img in image_list]).to(device)

    # with torch.no_grad():
    #     text_tokens = clip.tokenize([query_text]).to(device)
    #     logits_per_text, _ = model.get_similarity(img_tensors, text_tokens)
    #     probs = logits_per_text.softmax(dim=-1).cpu().numpy()[0]


    topk_indices = probs.argsort()[-5:][::-1]
    results = []
    for i, idx in enumerate(topk_indices):

        # 从缓存图像列表读取并转为 base64
        img = image_list[idx]
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

         # 从 image_ids_list 中提取标签  
        tags = get_image_tags(image_ids_list[idx])  
        tag_string = ", ".join(tags)  

        results.append({
            "rank": i + 1,
            "image_id": image_ids_list[idx],
            "image_base64": img_base64,
            "score": round(probs[idx] * 100, 3),
            "tags": tag_string  
        })
    return {"top_k_results": results}


