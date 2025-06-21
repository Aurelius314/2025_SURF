# -- Similarity calculation between Chinese text and images --

import numpy as np

import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name

device = "cuda" if torch.cuda.is_available() else "cpu"

# load ViT-H-14, set evaluation mode
model, preprocess = load_from_name("ViT-H-14", device=device, download_root='./')
model.eval()

# user input

image_path = input("请输入图像文件的路径：").strip('"')

# Add a dimension  -- unsqueeze(0)
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = clip.tokenize(["苹果","香蕉","梨子","芒果","桃子","葡萄","草莓"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # Normalize image and text features for downstream tasks
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    np.save("terminal_image_feat.npy", image_features.cpu().detach().numpy())


    # Calculate similarity
    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()



# 读取 vocab.txt 文件  
with open('E:/surf/data/vocab.txt', 'r', encoding='utf-8') as f:  
    text_labels = [line.strip() for line in f.readlines()]  

# 计算概率并打印结果  
percent_probs = [f"{prob * 100:.3f}%" for prob in probs[0]]  

for label, prob in zip(text_labels, percent_probs):  
    print(f"'{label}': {prob}")


