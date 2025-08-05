import requests
import json
import os
import base64
from datetime import datetime
from typing import Optional

# ------------------ 配置区 ------------------
API_URL = "http://127.0.0.1:8000/image-to-text/"
API_KEY = "surf_demo_api_key"
QUERY_IMAGE_PATH = "/home/hsh/surf/api/test/test_example.jpeg"
LIMIT_PER_PAGE = 10
SAVE_DIR = "/home/hsh/surf/api/test/i2t_log"
TEXT_TABLES = "text_tables"  # 可选：None 或逗号分隔表名
MAX_PAGES = 5  # 设置为 >1 可测试分页效果
# -------------------------------------------

def ensure_dir_exists(path: str):
    """确保保存目录存在"""
    os.makedirs(path, exist_ok=True)

def load_image_as_base64(path: str) -> str:
    """将图像编码为 base64 字符串"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def send_image_to_text_request(
    query_image_b64: str,
    offset: int,
    limit: int,
    text_tables: Optional[str] = None,
    api_key: Optional[str] = None
) -> dict:
    """发送 POST 请求"""
    payload = {
        "query_image": query_image_b64,
        "offset": offset,
        "limit": limit,
    }
    if api_key:
        payload["api_key"] = api_key
    if text_tables:
        payload["text_tables"] = text_tables

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()

def adjust_response(data: dict, offset: int):
    """
    修改 response 中的 rank 和字段结构
    - rank: 设置为全局 rank（offset + 当前页序号）
    - 删除 item 中的 "table" 字段（保留 record.table）
    """
    results = data.get("results", [])
    for i, item in enumerate(results):
        item["rank"] = offset + i + 1
        if "table" in item:
            del item["table"]

def save_json_response(json_data: dict, offset: int):
    """保存 JSON 响应到文件"""
    ensure_dir_exists(SAVE_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_to_text_offset{offset}_{timestamp}.json"
    file_path = os.path.join(SAVE_DIR, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"[✓] 已保存结果: {file_path}")

def main():
    print("[INFO] 正在加载图像并编码为 base64...")
    query_image_b64 = load_image_as_base64(QUERY_IMAGE_PATH)

    for page in range(MAX_PAGES):
        offset = page * LIMIT_PER_PAGE
        print(f"[→] 请求 offset={offset} ...")
        try:
            result = send_image_to_text_request(
                query_image_b64,
                offset,
                LIMIT_PER_PAGE,
                TEXT_TABLES,
                API_KEY
            )
            adjust_response(result, offset)
            save_json_response(result, offset)
        except requests.exceptions.RequestException as e:
            print(f"[×] 请求失败: {e}")
            break

if __name__ == "__main__":
    main()
