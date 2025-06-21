
import os  
import lmdb  
import base64  
from tqdm import tqdm  

# 输入路径  
image_folder = r"E:\coco-cn\train2014"  
caption_file = r"E:\coco-cn\coco-cn-version1805v1.1\imageid.human-written-caption.txt"  
tags_file = r"E:\coco-cn\coco-cn-version1805v1.1\imageid.human-written-tags.txt"  

# 输出路径（小数据版本）  
lmdb_output_path = r"E:\surf\coco_lmdb_database_small"
LIMIT = 100  # 只写入前100条

# 创建 LMDB 环境（1GB 空间）
env = lmdb.open(lmdb_output_path, map_size=1 * 1024 * 1024 * 1024)

def strip_hash(image_id):
    return image_id.split("#")[0]

# 读取tags
tags_mapping = {}  
with open(tags_file, 'r', encoding='utf-8') as f:  
    for line in f:  
        parts = line.strip().split()  
        image_id = parts[0]  
        tags = parts[1:]  
        tags_mapping[image_id] = tags  

# 读取 Caption  
caption_mapping = {}  
with open(caption_file, 'r', encoding='utf-8') as f:  
    for line in f:  
        parts = line.strip().split('\t')  
        image_id = strip_hash(parts[0])  
        caption = parts[1] if len(parts) > 1 else ""  
        caption_mapping[image_id] = caption  

# 写入数据
written_count = 0  
with env.begin(write=True) as txn:  
    for image_id in tqdm(caption_mapping.keys(), desc="Writing small LMDB"):  
        if "val2014" in image_id:
            continue

        image_path = os.path.join(image_folder, f"{image_id}.jpg")  
        if not os.path.exists(image_path):  
            continue

        with open(image_path, "rb") as image_file:  
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')  

        record = {  
            "image_id": image_id,  
            "image_base64": image_base64,  
            "caption": caption_mapping.get(image_id, ""),  
            "tags": tags_mapping.get(image_id, [])  
        }  

        txn.put(image_id.encode('utf-8'), str(record).encode('utf-8'))  
        written_count += 1  

        if written_count >= LIMIT:
            break

print(f"[INFO] Total records written to small LMDB: {written_count}")
env.close()



