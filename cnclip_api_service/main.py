'''

这是一个由 FastAPI 搭建的示例后端，您可以通过在终端运行 `uvicorn main:app --reload` 启动并访问。

'''

import torch
from fastapi import FastAPI, HTTPException, Body
from PIL import Image
import cn_clip.clip as clip
import io, os, pickle, base64
from lmdb import open as open_lmdb
from tqdm import tqdm
from pathlib import Path
import sys  
sys.path.append(r'E:/surf')  
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
LMDB_PATH = "E:/surf/valid_output_lmdb"    # dataset!!!
FEATURES_CACHE = os.path.join(CACHE_DIR, "image_features.pt")

# 单张壁画image，用于测试
TEST_IMAGE_PATH = r"E:\surf\test examples\image1.png"
TEST_IMAGE_FEATURE_CACHE = r"E:\surf\cached_test_img\test_image_features.pt"


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

# -------------加载数据------------------------
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

# 特征向量
image_features_tensor = cache_image_features(model, preprocess, data_records, CACHE_DIR)


# 缓存文本特征（基于 original_texts，因为较短，缓存较小，加载更快）
text_features_tensor = cache_text_features(model, data_records, CACHE_DIR)
# print(type(text_features_tensor)) <class 'torch.Tensor'>

# --------加载缓存  问题来了：缓存了哪些东西？
if os.path.exists(FEATURES_CACHE):
    print("Loading cached image features...")
    image_features_tensor = torch.load(FEATURES_CACHE).to(device)
    print(f"Loaded {len(image_features_tensor)} image features.")  # Loaded 577(576+1) image features. 

# -------或重新提取
else:
    print("Loading LMDB records and image files？...")


    # 提取图像路径列表
    env = open_lmdb(LMDB_PATH, readonly=True, lock=False)
    image_paths_list = []
    with env.begin() as txn:
        for key, value in txn.cursor():
            data = pickle.loads(value)
            image_paths_list.append(data["image_paths"])
    env.close()


    print("Calculating image features...")
    image_features_list = []


    for i, img in enumerate(tqdm(image_paths_list, desc="🔍 Extracting image features")):
        if img is None:
            image_features_list.append(torch.zeros(model.visual.output_dim))  # 占位向量
            continue

        # 如果当前图像是测试图像，则用缓存特征
        if Path(TEST_IMAGE_PATH).as_posix():
            img_filename = os.path.basename(TEST_IMAGE_PATH)
            print(f"[缓存命中] 使用 {img_filename} 的缓存特征")
            feat = torch.load(TEST_IMAGE_FEATURE_CACHE)
            image_features_list.append(feat.cpu())     # 把我们自己的测试图像特征也加进去了
            continue

        # 正常提取特征
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
            feat /= feat.norm(dim=-1, keepdim=True) + 1e-7
            image_features_list.append(feat.cpu())

    # 拼接为一个 tensor
    image_features_tensor = torch.stack(image_features_list, dim=0)

    # 缓存保存
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(image_features_tensor, FEATURES_CACHE)
    print(f"Saved {len(image_ids_list)} features to cache.")


# -------------------- 图搜文 --------------------

@app.post("/image-to-text/")
async def image_to_text(img_base64: str = Body(...), api_key: str = Body(...)):
    try:
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


    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        return {"error": str(e)}


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

        print(f"image_paths_list: {image_ids_list[idx]}")
        print(f"text_list: {original_texts[idx]}")
        print(f"nld_list: {nld_texts[idx]}")
        print(f"idx: {idx}")


    return {"top_k_results": results}


