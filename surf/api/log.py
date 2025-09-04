import time
import logging
import traceback
import csv
import os
from datetime import datetime
from functools import wraps

from prometheus_client import Histogram

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Prometheus histogram：记录每个 API 的耗时
API_TIME_HISTOGRAM = Histogram(
    "api_duration_seconds",
    "Time spent on each API call",
    ["api_name"]
)

# CSV日志文件路径
LOG_CSV_PATH = "log.csv"

def ensure_csv_header():
    """确保CSV文件存在并包含正确的表头"""
    if not os.path.exists(LOG_CSV_PATH):
        with open(LOG_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'api_name', 
                'query_type',
                'query_content',
                'offset',
                'limit',
                'session_id',
                'result_count',
                'duration_seconds',
                'status',
                'error_message'
            ])

def log_retrieval_to_csv(
    api_name: str,
    query_type: str,
    query_content: str,
    offset: int,
    limit: int,
    session_id: str,
    result_count: int,
    duration: float,
    status: str = "success",
    error_message: str = ""
):
    """将检索记录保存到CSV文件"""
    try:
        ensure_csv_header()
        
        # 处理查询内容，避免过长
        if len(query_content) > 100:
            query_content = query_content[:100] + "..."
        
        with open(LOG_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                api_name,
                query_type,
                query_content,
                offset,
                limit,
                session_id or "",
                result_count,
                round(duration, 3),
                status,
                error_message
            ])
    except Exception as e:
        logging.error(f"保存检索日志到CSV失败: {e}")

def extract_query_info_from_request_body(request_body: dict) -> tuple:
    """从请求体中提取查询信息"""
    query_type = "unknown"
    query_content = ""
    
    if 'query_image' in request_body:
        query_type = "image_to_text"
        query_content = "image_data"  # 图片数据太长，用标识符代替
    elif 'query_text' in request_body:
        query_type = "text_to_image"
        query_content = request_body.get('query_text', '')
    
    return query_type, query_content

def log_timing(api_name: str):
    """
    装饰器：记录函数耗时 + 捕捉异常 + Prometheus 监控 + CSV日志
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            logging.info(f"➡️ 开始处理: {api_name}")

            # 提取查询参数
            query_type = "unknown"
            query_content = ""
            offset = 0
            limit = 20
            session_id = ""
            result_count = 0
            status = "success"
            error_message = ""

            # 从kwargs中提取请求参数
            if kwargs:
                offset = kwargs.get('offset', 0)
                limit = kwargs.get('limit', 20)
                session_id = kwargs.get('session_id', "")
                
                # 提取查询类型和内容
                if 'query_image' in kwargs:
                    query_type = "image_to_text"
                    query_content = "image_data"
                elif 'query_text' in kwargs:
                    query_type = "text_to_image"
                    query_content = kwargs.get('query_text', '')

            try:
                result = await func(*args, **kwargs)
                
                # 从结果中提取信息
                if isinstance(result, dict):
                    result_count = len(result.get('results', []))
                    # 使用结果中的值覆盖默认值
                    offset = result.get('offset', offset)
                    limit = result.get('limit', limit)
                    session_id = result.get('session_id', session_id)
                
                return result
            except Exception as e:
                status = "error"
                error_message = str(e)
                logging.error(f"❌ {api_name} 执行出错: {e}")
                traceback_str = traceback.format_exc()
                logging.error(traceback_str)
                raise e
            finally:
                duration = time.time() - start_time
                logging.info(f"✅ 结束处理: {api_name}，耗时 {duration:.3f} 秒")
                API_TIME_HISTOGRAM.labels(api_name=api_name).observe(duration)
                
                # 保存到CSV
                log_retrieval_to_csv(
                    api_name=api_name,
                    query_type=query_type,
                    query_content=query_content,
                    offset=offset,
                    limit=limit,
                    session_id=session_id,
                    result_count=result_count,
                    duration=duration,
                    status=status,
                    error_message=error_message
                )
        return wrapper
    return decorator
