import os
import sys
import json
import torch
from tqdm import tqdm

# 设置路径
BASE_DIR = '/home/hsh/surf'
sys.path.append(BASE_DIR)
GH_DIR = os.path.join(BASE_DIR, 'fromGitHub')
sys.path.append(GH_DIR)
sys.path.append(os.path.join(GH_DIR, 'cn_clip'))

import cn_clip.clip as clip
from fromGitHub.utils import load_surf_checkpoint_model_from_base

# ---------- 参数配置 ----------
JSONL_PATH = "/home/hsh/surf/cache/surf/surf_nld.jsonl"
OUTPUT_PATH = "/home/hsh/surf/cache/surf/surf_text_features.pt"
CKPT_PATH = "/home/hsh/surf/pgvector69/epoch_latest.pt"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 加载模型 ----------
print("加载文本编码模型中...")
model, preprocess = load_surf_checkpoint_model_from_base(CKPT_PATH, device=DEVICE)
model.eval()
print("模型加载完成")

# ---------- 加载文本数据 ----------
print(f"加载 JSONL 文件：{JSONL_PATH}")
text_ids = []
texts = []

with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        if 'text_id' in item and 'text' in item:
            text_ids.append(item['text_id'])
            texts.append(item['text'])

assert len(text_ids) == len(texts), "text_id 和 text 数量不一致！"

# ---------- 提取文本特征 ----------
print(f"开始提取文本特征，总计 {len(texts)} 条...")
all_features = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        tokens = clip.tokenize(batch_texts).to(DEVICE)  # [B, 77]
        features = model.encode_text(tokens)                           # [B, D]
        features /= features.norm(dim=-1, keepdim=True)               # L2归一化
        all_features.append(features.cpu())

# 拼接特征
all_features_tensor = torch.cat(all_features, dim=0)  # [N, D]
text_id_tensor = torch.tensor(text_ids, dtype=torch.long)  # [N]

# ---------- 保存为 pt 文件 ----------
output_dict = {
    "text_ids": text_id_tensor,
    "text_features": all_features_tensor
}
torch.save(output_dict, OUTPUT_PATH)
print(f"✅ 文本特征提取完成，已保存至 {OUTPUT_PATH}")
