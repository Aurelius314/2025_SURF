import os
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# 添加模块路径
import sys
BASE_DIR = '/home/hsh/surf'
sys.path.append(BASE_DIR)
GH_DIR = os.path.join(BASE_DIR, 'fromGitHub')
sys.path.append(GH_DIR)
sys.path.append(os.path.join(GH_DIR, 'cn_clip'))
import cn_clip.clip as clip
from fromGitHub.utils import load_surf_checkpoint_model_from_base

# ---------------- 参数配置 ----------------
IMAGE_ROOTS = [
    "/mnt/disk9T/shared/datasets/敦煌检索/images_split/split1",
    "/mnt/disk9T/shared/datasets/敦煌检索/images_split/split2"
]
CACHE_FILE_PATH = "/home/hsh/surf/cache/surf/surf_image_features.pt"
BATCH_SIZE = 16
CKPT_PATH = "/home/hsh/surf/pgvector69/epoch_latest.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 加载模型 ----------------
print("加载模型中...")
model, preprocess = load_surf_checkpoint_model_from_base(CKPT_PATH, device=DEVICE)
model.eval()
print("模型加载完成")

# ---------------- 收集图像路径 ----------------
image_paths = []
for root in IMAGE_ROOTS:
    for file in os.listdir(root):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image_id = Path(file).stem  # 不带扩展名的文件名
            image_paths.append((int(image_id), os.path.join(root, file)))

# 按 image_id 升序排序
image_paths.sort(key=lambda x: x[0])

# ---------------- 图像批处理提取特征 ----------------
all_image_ids = []
all_features = []

print(f"开始提取图像特征，共 {len(image_paths)} 张图像...")
with torch.no_grad():
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch = image_paths[i:i+BATCH_SIZE]
        images = []
        batch_ids = []

        for image_id, path in batch:
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0)  # [1, 3, 224, 224]
                images.append(img_tensor)
                batch_ids.append(image_id)
            except Exception as e:
                print(f"跳过无效图像 {path}，错误: {e}")

        if not images:
            continue

        image_tensor_batch = torch.cat(images).to(DEVICE)  # [B, 3, 224, 224]
        features = model.encode_image(image_tensor_batch)
        features /= features.norm(dim=-1, keepdim=True)  # L2 归一化

        all_image_ids.extend(batch_ids)
        all_features.append(features.cpu())

# 拼接所有特征
all_features_tensor = torch.cat(all_features, dim=0)  # [N, D]

# ---------------- 保存为 pt 文件 ----------------
output_dict = {
    "image_ids": torch.tensor(all_image_ids),     # LongTensor [N]
    "image_features": all_features_tensor         # FloatTensor [N, D]
}
torch.save(output_dict, CACHE_FILE_PATH)
print(f"✅ 特征提取完毕，已保存至 {CACHE_FILE_PATH}")

