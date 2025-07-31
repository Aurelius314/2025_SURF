import os
from pathlib import Path

def count_images_in_directory(directory):
    # 支持的图片文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    total_images = 0
    dir_stats = {}

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        image_count = 0
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_count += 1
        
        if image_count > 0:
            dir_stats[root] = image_count
            total_images += image_count

    return dir_stats, total_images

if __name__ == "__main__":
    target_directory = "/mnt/disk9T/shared/datasets/博物院/数字文物库_images"
    
    if not os.path.isdir(target_directory):
        print(f"错误：目录 '{target_directory}' 不存在或不可访问。")
    else:
        stats, total = count_images_in_directory(target_directory)
        
        print("各子目录中的图片文件数量统计：")
        for dir_path, count in stats.items():
            print(f"{dir_path}: {count} 个图片文件")
        
        print(f"\n总图片文件数量: {total} 个")