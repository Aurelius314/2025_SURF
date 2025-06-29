'''

这是一个由 FastAPI 搭建的示例后端，您可以通过在终端运行 `uvicorn main:app --reload` 启动并访问。

'''

import torch
from fastapi import FastAPI, HTTPException, Body
import io, os, base64
import sys  
sys.path.append(r'E:/surf')  
from utils import load_surf_checkpoint_model_from_base, load_text_data_from_lmdb, load_images_from_paths, load_or_extract_image_features, load_or_extract_text_features, log  


# -------初始化-----------
app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_surf_checkpoint_model_from_base(
    ckpt_path="E:/SURF2025/检索文化数据集/checkpoint/epoch_latest.pt"
)
model.eval()

# 可能是数据造成了崩溃
LMDB_PATH = r"E:\surf\dataset"    # dataset!!! E:\surf\dataset

CACHE_DIR = "E:/surf/cache"
IMAGE_FEATURES_CACHE = os.path.join(CACHE_DIR, "image_features_fp16.pt")
TEXT_FEATURES_CACHE = os.path.join(CACHE_DIR, "text_features_fp16.pt")

# # 单张壁画image，用于测试
# # 暂时还没有test_text
# TEST_IMAGE_PATH = r"E:\surf\test examples\image1.png"
# TEST_IMAGE_FEATURE_CACHE = r"E:\surf\cached_test_img\test_image_features.pt"

# API Key 模拟
API_KEYS = {"demo": "your_demo_api_key"}

IMAGE_ROOT = "E:/SURF2025/检索文化数据集/images/images"  # 改成你本地图像根目录

def verify_api_key(api_key: str):
    if api_key not in API_KEYS.values():
        raise HTTPException(status_code=403, detail="Unauthorized")
    
def fix_base64_padding(b64_string: str) -> str:
    return b64_string + '=' * ((4 - len(b64_string) % 4) % 4)

@app.get("/")
def read_root():
    return {"message": "Welcome to the CN-CLIP API"}

# ------加载数据------
# 从lmdb中取出来，都是list
# suffix: ['images/1000.png']
text_ids, original_texts, nld_texts, image_ids_list, img_rel_path, splits = load_text_data_from_lmdb(LMDB_PATH)

# image本身
# image_paths_list: 两部分拼起来 ---→ 拿到图像路径list
image_list, image_paths_list = load_images_from_paths(img_rel_path)

# 全局record，可复用
data_records = [
    {
        "image": image_paths_list[i],
        "image_id": image_ids_list[i],
        "text_id": text_ids[i],
        "original_text": original_texts[i],
        "NLD_text": nld_texts[i],
        "image_path": image_paths_list[i][0] if image_paths_list[i] else ""
    }
    for i in range(len(image_paths_list))
]

# 可能是这部分导致了卡死
image_features_tensor = load_or_extract_image_features(model, preprocess, data_records, CACHE_DIR, batch_size=8, log_func=log)
text_features_tensor = load_or_extract_text_features(model, data_records, CACHE_DIR, batch_size=8, log_func=log)


# -------------------- 图搜文 --------------------

@app.post("/image-to-text/")
async def image_to_text(img_base64: str = Body(...), api_key: str = Body(...)):

    verify_api_key(api_key)

    # 构建"text"的结构化记录
    text_records = [
        {
            "text_id": tid,
            "original_text": ori,
            "NLD_text": nld,
        }
        for tid, ori, nld in zip(text_ids, original_texts, nld_texts)
    ]

    # 去重原文,seen已经见过的
    # record_mapping有用吗
    seen = set()
    text_candidates, record_mapping = [], []
    for i, rec in enumerate(text_records):
        if rec["original_text"] not in seen:
            seen.add(rec["original_text"])
            text_candidates.append(rec["original_text"])
            record_mapping.append(i)

    print("使用的 text_candidates 数量:", len(text_candidates))

    with torch.no_grad():
        # 已缓存好的特征张量
        image_features = image_features_tensor / image_features_tensor.norm(dim=1, keepdim=True)
        text_features = text_features_tensor / text_features_tensor.norm(dim=1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # Top-K 结果
    topk_indices = probs.argsort()[-9:][::-1]
    results = []
    for i, idx in enumerate(topk_indices):
        # 图搜文，先不要输出文本对应的图片
        rec = text_records[record_mapping[idx]]

        results.append({
            "rank": i + 1,
            "text_id": rec["text_id"],
            "original_text": rec["original_text"],
            "NLD_text": rec["NLD_text"],
            # "image_path": image_path_abs,
            # "image_base64": db_image_base64,
            "score": round(probs[idx] * 100, 3)
        })

    return {"top_k_results": results}



# -------------------- 文搜图 --------------------

@app.post("/text-to-image/")
async def text_to_image(query_text: str = Body(...), api_key: str = Body(...)):
    verify_api_key(api_key)

    with torch.no_grad():
        text_features = text_features_tensor / text_features_tensor.norm(dim=1, keepdim=True)
        image_features = image_features_tensor / image_features_tensor.norm(dim=1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits_per_text = logit_scale * text_features @ image_features.t()
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()[0]

    topk_indices = probs.argsort()[-9:][::-1]
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
            "original_text": original_texts[idx],
            "NLD_text": nld_texts[idx],
            "image_path": image_paths_list[idx] if idx < len(image_paths_list) else None,
            "image_base64": img_base64,
            "score": round(probs[idx] * 100, 3)
        })


    return {"top_k_results": results}
