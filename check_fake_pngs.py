import os
import csv
from tqdm import tqdm

# ---------------- 配置 ----------------
INPUT_CSV = "/home/hsh/surf/cache/tw/tw_found_images.csv"
OUTPUT_CSV = "/home/hsh/surf/cache/tw/tw_fake_images.csv"


# ---------------- 工具函数 ----------------
def is_html_disguised_as_image(file_path):  
    """判断一个文件是否是伪装成图片的 HTML 网页"""  
    try:  
        with open(file_path, 'rb') as f:  
            header = f.read(1024).lower()  
            return (b"<html" in header or  
                    b"<!doctype html" in header or  
                    b"<head" in header or  
                    b"<body" in header)  
    except Exception as e:  
        print(f"[读取失败] {file_path} 错误: {e}")  
        return False  


# ---------------- 主流程 ----------------
def scan_fake_images_from_csv(csv_path, output_path):  
    fake_files = []

    with open(csv_path, newline='', encoding='utf-8') as csvfile:  
        reader = csv.DictReader(csvfile)  
        for row in tqdm(reader, desc="检测伪图片"):  
            local_path = row.get("local_path")  
            if not local_path or not os.path.exists(local_path):  
                continue  
            if is_html_disguised_as_image(local_path):  
                fake_files.append(local_path)

    # 保存结果
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:  
        writer = csv.writer(outfile)  
        writer.writerow(["local_path"])  
        for path in fake_files:  
            writer.writerow([path])

    print(f"✅ 检测完毕，发现 {len(fake_files)} 个伪装的 HTML 图片，保存至 {output_path}")


# ---------------- 执行 ----------------
if __name__ == "__main__":  
    scan_fake_images_from_csv(INPUT_CSV, OUTPUT_CSV)


# /mnt/disk9T/shared/datasets/博物院/台湾国立故宫博物院/典藏资料/images/織品
# file "/mnt/disk9T/shared/datasets/博物院/台湾国立故宫博物院/典藏资料/images/織品/7250-13-12.png"

