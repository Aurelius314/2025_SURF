import torch
from PIL import Image
import os
import cn_clip.clip as clip
from cn_clip.clip import load_from_name

device = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = load_from_name("ViT-H-14", device=device, download_root='./')
model.eval()


text_input = input("请输入描述文本：").strip()
text = clip.tokenize([text_input]).to(device)


image_folder = "E:\surf\data\images"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
               if f.lower().endswith((".jpg", ".jpeg", ".png", "webp"))]


images = [preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0) for img_path in image_paths]
images = torch.cat(images).to(device)


with torch.no_grad():
    image_features = model.encode_image(images)
    text_features = model.encode_text(text)

    # 特征归一化
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 计算相似度
    similarity = (text_features @ image_features.T)[0]  # shape: [num_images]
    probs = similarity.softmax(dim=0).cpu().numpy()

# 输出概率和最相似图像
for path, prob in zip(image_paths, probs):
    print(f"{path}: {prob*100:.2f}%")

# 输出最相似图像路径
best_match_index = probs.argmax()
print("\n最相似的图像是：", image_paths[best_match_index])
