import torch
import json
from cn_clip.clip import load_from_name
import lmdb
import pickle
from PIL import Image
import os

def load_json_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_surf_checkpoint_model_from_base(
    ckpt_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # 加载模型底座
    model, preprocess = load_from_name("ViT-H-14", device=device, download_root='./')

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint["state_dict"]

    # 去掉module.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("[DEBUG] Missing keys:", missing_keys)
    print("[DEBUG] Unexpected keys:", unexpected_keys)

    model.to(device).eval()
    return model, preprocess

# LMDB 数据加载函数
'''
(
  text_ids,          # List[int]
  original_texts,    # List[str]
  nld_texts,         # List[str]
  image_ids_list,    # List[list]
  path
  splits             # List[str]
)
'''
def load_text_data_from_lmdb(lmdb_path: str):
    text_ids = []
    original_texts = []
    nld_texts = []
    image_ids_list = []
    image_paths_list = []
    splits = []

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                data = pickle.loads(value)

                # 提取字段
                text_ids.append(data["text_id"])
                original_texts.append(data["original_text"])
                nld_texts.append(data["NLD_text"])
                image_ids_list.append(data["image_id"])
                image_paths_list.append(data["image_paths"])
                splits.append(data["split"])

            except Exception as e:
                print(f"[WARN] Failed to load record {key}: {e}")

    env.close()
    return text_ids, original_texts, nld_texts, image_ids_list, image_paths_list, splits


# 加载图像辅助函数
def load_images_from_paths(image_paths_list, root_dir="E:/SURF2025/检索文化数据集/images/images"):
    images = []
    for paths in image_paths_list:
        if not paths:
            images.append(None)
            continue
        try:
            filename = os.path.basename(paths[0])  # 提取文件名（如 849.png）
            img_path = os.path.join(root_dir, filename)  # 拼接为干净路径
            img = Image.open(img_path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"⚠️ Failed to load image: {img_path}, error: {e}")
            images.append(None)
    return images


