import torch
from cn_clip.clip import load_from_name
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
import numpy as np

def load_surf_checkpoint_model_from_base(
    ckpt_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # 加载模型底座
    # model, preprocess = load_from_name("ViT-H-14", device=device, download_root='./')
    model_path = '/home/hsh/surf/pgvector69'
    model, preprocess = load_from_name("ViT-H-14", device=device, download_root=model_path)

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

def verify_api_key(api_key: str):
    API_KEYS = {"demo": "surf_demo_api_key"}
    if api_key not in API_KEYS.values():
        raise HTTPException(status_code=403, detail="Unauthorized")

def fix_base64_padding(b64_string: str) -> str:
    return b64_string + '=' * ((4 - len(b64_string) % 4) % 4)


