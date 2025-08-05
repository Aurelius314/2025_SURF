import requests
import json
import os
from datetime import datetime
from typing import Optional

# ------------------ 配置区 ------------------
API_URL = "http://127.0.0.1:8000/text-to-image/"
API_KEY = "surf_demo_api_key"
QUERY_TEXT = "九色鹿"
LIMIT_PER_PAGE = 10
SAVE_DIR = "/home/hsh/surf/api/test/t2i_log"
IMAGE_TABLES = "image_tables"  # 可选参数
MAX_PAGES = 5  # 设置为 >1 可测试分页效果
# -------------------------------------------


def ensure_dir_exists(path: str):
    """确保保存目录存在"""
    os.makedirs(path, exist_ok=True)


def adjust_response(data: dict, offset: int):
    """
    修改 response 中的 rank 和去除多余字段
    - rank → 全局 rank（非页内局部）
    - 删除 score 同级的 table 字段（保留 record.table）
    """
    results = data.get("results", [])
    for i, item in enumerate(results):
        global_rank = offset + i + 1
        item["rank"] = global_rank

        # 删除 score 同级 table 字段
        if "table" in item:
            del item["table"]


def save_json_to_file(data: dict, query_text: str, offset: int):
    """保存 JSON 响应到本地文件"""
    ensure_dir_exists(SAVE_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{query_text}_offset{offset}_{timestamp}.json"
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[✓] 保存结果到: {filepath}")


def send_text_to_image_request(
    query_text: str,
    api_key: str,
    offset: int,
    limit: int,
    image_tables: Optional[str] = None
) -> dict:
    """发送 POST 请求到 /text-to-image/ 接口"""
    payload = {
        "query_text": query_text,
        "api_key": api_key,
        "offset": offset,
        "limit": limit,
    }
    if image_tables:
        payload["image_tables"] = image_tables

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()


def run_test():
    """运行测试用例"""
    for page in range(MAX_PAGES):
        offset = page * LIMIT_PER_PAGE
        try:
            print(f"[→] 请求 offset={offset} ...")
            result_json = send_text_to_image_request(
                QUERY_TEXT, API_KEY, offset, LIMIT_PER_PAGE, IMAGE_TABLES
            )

            # 调整 rank 和字段结构
            adjust_response(result_json, offset)

            # 保存处理后的 JSON
            save_json_to_file(result_json, QUERY_TEXT, offset)
        except requests.exceptions.RequestException as e:
            print(f"[×] 请求失败: {e}")
            break


if __name__ == "__main__":
    run_test()
