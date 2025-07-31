# import os
# import csv
# from tqdm import tqdm

# # ---------------- 配置 ----------------
# CSV_ROOT = "/mnt/disk9T/shared/datasets/博物院/台湾国立故宫博物院/典藏资料"
# IMAGE_ROOT = "/mnt/disk9T/shared/datasets/博物院/台湾国立故宫博物院/典藏资料/images"
# CACHE_DIR = "/home/hsh/surf/cache/tw"
# MISSING_FILE = os.path.join(CACHE_DIR, "tw_missing_images.csv")
# FOUND_FILE = os.path.join(CACHE_DIR, "tw_found_images.csv")

# # ---------------- 函数 ----------------
# def get_all_csv_files(root_dir):
#     """获取所有 imageid-xxx.csv 文件路径"""
#     csv_files = []
#     for file in os.listdir(root_dir):
#         if file.startswith("imageid-") and file.endswith(".csv"):
#             csv_files.append(os.path.join(root_dir, file))
#     return csv_files


# def check_missing_images():
#     os.makedirs(CACHE_DIR, exist_ok=True)

#     csv_files = get_all_csv_files(CSV_ROOT)
#     print(f"Scanning CSVs: {len(csv_files)} files found.")

#     total_records = 0
#     found_count = 0
#     missing_count = 0

#     missing_list = []
#     found_list = []

#     # 遍历 CSV 文件
#     for csv_file in tqdm(csv_files, desc="Scanning CSVs"):
#         with open(csv_file, "r", encoding="utf-8-sig") as f:
#             reader = csv.DictReader(f)
#             for i, row in enumerate(reader):
#                 total_records += 1
#                 image_id = row.get("image_id", "").strip()

#                 # 用 category + image_id 构造 rel_path
#                 category = row.get("category", "").strip()
#                 rel_path = os.path.join(category, f"{image_id}.png") 

#                 local_path = os.path.join(IMAGE_ROOT, rel_path)
#                 row["local_path"] = local_path

#                 # 检查文件是否存在
#                 if os.path.exists(local_path):
#                     found_count += 1
#                     found_list.append({
#                         "csv_file": csv_file,
#                         "row_index": i,
#                         "image_id": image_id,
#                         "category": row.get("category", ""),
#                         "raw_path": row.get("raw_path", ""),
#                         "rel_path": rel_path,
#                         "local_path": local_path
#                     })
#                 else:
#                     missing_count += 1
#                     missing_list.append({
#                         "csv_file": csv_file,
#                         "row_index": i,
#                         "image_id": image_id,
#                         "category": row.get("category", ""),
#                         "raw_path": row.get("raw_path", ""),
#                         "rel_path": rel_path,
#                         "local_path": local_path
#                     })

#     # 保存 missing.csv
#     with open(MISSING_FILE, "w", encoding="utf-8-sig", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=["csv_file", "row_index", "image_id", "category", "raw_path", "rel_path", "local_path"])
#         writer.writeheader()
#         writer.writerows(missing_list)

#     # 保存 found.csv
#     with open(FOUND_FILE, "w", encoding="utf-8-sig", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=["csv_file", "row_index", "image_id", "category", "raw_path", "rel_path", "local_path"])
#         writer.writeheader()
#         writer.writerows(found_list)

#     # 终端输出统计信息
#     print(f"\n📄 Total records: {total_records}")
#     print(f"✅ Found: {found_count}")
#     print(f"❌ Missing: {missing_count} ({(missing_count / total_records * 100):.2f}% if total > 0 else 0)")
#     print(f"📝 Missing list saved to: {MISSING_FILE}")


# # ---------------- 主程序 ----------------
# if __name__ == "__main__":
#     check_missing_images()

import torch
import json

# ===== 1. 加载 surf_text_features.pt =====
feature_path = "/home/hsh/surf/cache/surf/surf_text_features.pt"
features_dict = torch.load(feature_path)
text_ids = features_dict["text_ids"]
text_features = features_dict["text_features"]

print(f"[text_features.pt] 总条数: {len(text_ids)}")
for i in range(min(3, len(text_ids))):
    print(f"  第{i+1}条 id: {text_ids[i].item()}, 特征前10字符: {str(text_features[i].tolist())[:10]}")

# ===== 2. 加载 surf_original.jsonl =====
original_path = "/home/hsh/surf/cache/surf/surf_original.jsonl"
with open(original_path, "r", encoding="utf-8") as f:
    original_lines = [json.loads(line.strip()) for line in f if line.strip()]
original_texts = [item["text"] for item in original_lines]

print(f"[surf_original.jsonl] 总条数: {len(original_texts)}")
for i in range(min(3, len(original_texts))):
    print(f"  第{i+1}条原文前10字符: {original_texts[i][:10]}")

# ===== 3. 加载 surf_nld.jsonl =====
nld_path = "/home/hsh/surf/cache/surf/surf_nld.jsonl"
with open(nld_path, "r", encoding="utf-8") as f:
    nld_lines = [json.loads(line.strip()) for line in f if line.strip()]
nld_texts = [item["text"] for item in nld_lines]

print(f"[surf_nld.jsonl] 总条数: {len(nld_texts)}")
for i in range(min(3, len(nld_texts))):
    print(f"  第{i+1}条 NLD 前10字符: {nld_texts[i][:10]}")

# ===== 4. 检查长度一致性 =====
if len(text_ids) == len(original_texts) == len(nld_texts):
    print("✅ 三者长度一致，可以继续入库。")
else:
    print("❌ 三者长度不一致，请检查数据源。")
