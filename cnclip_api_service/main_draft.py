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
import torchvision.transforms as T
from typing import List, Dict, Any, Optional

sys.path.append(r'E:/surf')
from utils import load_surf_checkpoint_model_from_base, load_text_data_from_lmdb, load_images_from_paths
from cache.cache_manager import cache_image_features, cache_text_features

# -------初始化-----------
app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_surf_checkpoint_model_from_base(
    ckpt_path="E:/SURF2025/检索文化数据集/checkpoint/epoch_latest.pt"
)
model.eval()

CACHE_DIR = "E:/surf/cache"
LMDB_PATH = "E:/surf/valid_output_lmdb"

# FEATURES_CACHE = os.path.join(CACHE_DIR, "new_valid_image_features.pt")

# API Key 模拟
API_KEYS = {"demo": "your_demo_api_key"}

IMAGE_ROOT = "E:/SURF2025/检索文化数据集/images/images"

def verify_api_key(api_key: str):
    if api_key not in API_KEYS.values():
        raise HTTPException(status_code=403, detail="Unauthorized")

def fix_base64_padding(b64_string: str) -> str:
    return b64_string + '=' * ((4 - len(b64_string) % 4) % 4)

@app.get("/")
def read_root():
    return {"message": "Welcome to the CN-CLIP API"}

# -------------加载数据------------------------
# 元素的列表
text_ids, original_texts, nld_texts, image_ids_list, img_rel_path, splits = load_text_data_from_lmdb(LMDB_PATH)

image_list, image_paths_list = load_images_from_paths(img_rel_path)

# image list: [<PIL.Image.Image image mode=RGB size=71x72 at 0x1667C2F7FD0>,...]

# for i, paths in enumerate(image_paths_list[:10]):
#     print(f"[第{i}条] 类型: {type(paths)}，内容: {paths}")
# # [第0条] 类型: <class 'list'>，内容: ['images/1000.png']

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

# 加载特征并确保是float32类型
image_features_tensor = cache_image_features(model, preprocess, data_records, CACHE_DIR).float()
text_features_tensor = cache_text_features(model, data_records, CACHE_DIR).float()
# # 模型初始化阶段做一次
# image_features_tensor /= image_features_tensor.norm(dim=1, keepdim=True)

# -------------------- 图搜文 --------------------
@app.post("/image-to-text/")
async def image_to_text(img_base64: str = Body(...), api_key: str = Body(...)):
    verify_api_key(api_key)

    # 解码 base64 图片为 PIL.Image
    try:
        img_bytes = base64.b64decode(fix_base64_padding(img_base64))
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # 提取上传图片的特征
    image_tensor = preprocess(image)
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = T.ToTensor()(image)
    image_input = image_tensor.unsqueeze(0).to(device)  # shape: [1, 3, H, W]
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features.float()
        image_features /= image_features.norm(dim=1, keepdim=True)  # shape: [1, D]
        text_features = text_features_tensor / text_features_tensor.norm(dim=1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # Top-K 结果
    topk_indices = probs.argsort()[-20:][::-1].tolist() 

    text_records = [
        {
            "text_id": tid,
            "original_text": ori,
            "NLD_text": nld,
            "image_basename": os.path.basename(img_rel_path[i][0]) if (i < len(img_rel_path) and img_rel_path[i]) else None
        }
        for i, (tid, ori, nld) in enumerate(zip(text_ids, original_texts, nld_texts))
    ]

    seen = set()
    text_candidates, record_mapping = [], []
    for i, rec in enumerate(text_records):
        if rec["original_text"] not in seen:
            seen.add(rec["original_text"])
            text_candidates.append(rec["original_text"])
            record_mapping.append(i)

    results = []
    for i, idx in enumerate(topk_indices):
        rec = text_records[record_mapping[idx]]
        if rec["image_basename"]:
            image_abs_path = os.path.join(IMAGE_ROOT, rec["image_basename"]).replace('\\', '/')
            try:
                with open(image_abs_path, 'rb') as f:
                    image_base64 = base64.b64encode(f.read()).decode("utf-8")
            except FileNotFoundError:
                image_base64 = ""
        else:
            image_base64 = ""
        results.append({
            "rank": i + 1,
            "text_id": rec["text_id"],
            "original_text": rec["original_text"],
            "NLD_text": rec["NLD_text"],
            "image_base64": image_base64,
            "score": round(probs[idx] * 100, 3)
        })

    return {"top_k_results": results}



# -------------------- 文搜图 --------------------

@app.post("/text-to-image/")
async def text_to_image(query_text: str = Body(...), api_key: str = Body(...)):
    verify_api_key(api_key)

    with torch.no_grad():
        tokens = clip.tokenize([query_text]).to(device)
        text_features = model.encode_text(tokens)
        # 确保文本特征是 float32 类型
        text_features = text_features.float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 确保图像特征也是 float32 类型
        image_features = image_features_tensor.float()
        image_features /= image_features.norm(dim=1, keepdim=True)

        # Debug shape check
        assert text_features.ndim == 2, f"text_features wrong shape: {text_features.shape}"
        assert image_features.ndim == 2, f"image_features wrong shape: {image_features.shape}"

        logit_scale = model.logit_scale.exp()
        logits_per_text = logit_scale * text_features @ image_features.t()
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()[0]


    topk_indices = probs.argsort()[-20:][::-1]
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



# -------------------- 聊天功能示例 --------------------

# 模拟存储对话历史的简单数据库（实际应用中应使用真实数据库）
chat_history_db = {}

# 定义消息结构
class Message(dict):
    role: str
    content: str

# 聊天API接口
@app.post("/chat/")
async def chat(
    messages: List[Dict[str, str]] = Body(...),
    api_key: str = Body(...)
):
    verify_api_key(api_key)
    
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    # 获取最后一条用户消息
    last_message = messages[-1]
    if last_message.get("role") != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")
    
    user_input = last_message.get("content", "")
    
    # 简单的响应逻辑 - 实际应用中可以接入LLM或其他问答系统
    response = generate_response(user_input, messages[:-1])
    
'''
问答组代码....
'''