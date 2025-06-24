import os
import pickle
import torch
import clip
from PIL import Image
from tqdm import tqdm
import cn_clip.clip as clip  



def cache_image_features(model, preprocess, image_list, cache_dir):
    features_path = os.path.join(cache_dir, "image_features.pt")
    meta_path = os.path.join(cache_dir, "image_meta.pkl")

    if os.path.exists(features_path) and os.path.exists(meta_path):
        print("Cached image features found.")
        image_features_tensor = torch.load(features_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return image_features_tensor, meta

    print("Extracting and caching image features...")
    image_features = []
    meta = {
        "image_ids": [],
        "text_ids": [],
        "original_texts": [],
        "nld_texts": [],
        "image_paths": []
    }

    for record in tqdm(image_list, desc="🔍 Processing images"):
        img = record["image"]
        if img is None:
            image_features.append(torch.zeros(model.visual.output_dim))
        else:
            img_tensor = preprocess(img).unsqueeze(0).to(model.visual.conv1.weight.device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
            feat /= feat.norm(dim=-1, keepdim=True) + 1e-7
        image_features.append(feat.squeeze(0).cpu())  


        meta["image_ids"].append(record["image_id"])
        meta["text_ids"].append(record["text_id"])
        meta["original_texts"].append(record["original_text"])
        meta["nld_texts"].append(record["NLD_text"])
        meta["image_paths"].append(record["image_path"])

    image_features_tensor = torch.stack(image_features, dim=0)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(image_features_tensor, features_path)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"Cached {len(image_features)} image features.")
    return image_features_tensor, meta


def cache_text_features(model, text_list, cache_dir):
    text_path = os.path.join(cache_dir, "text_features.pt")
    if os.path.exists(text_path):
        print("Cached text features found.")
        return torch.load(text_path)

    print("Encoding and caching text features...")
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    tokens = clip.tokenize(text_list).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True) + 1e-7
    torch.save(features.cpu(), text_path)
    print(f"Cached {features.shape[0]} text features.")
    return features
