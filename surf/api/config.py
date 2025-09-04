# -*- coding: utf-8 -*-
"""
IVFFlat索引优化配置
基于交叉验证测试结果的最佳参数
"""

import os
from typing import Optional

def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """获取环境变量，支持默认值和必需检查"""
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value

# 模型配置（默认指向项目相对路径，容器中可通过环境变量覆盖）
_default_model_path = get_env_var('MODEL_PATH', './models')
MODEL_CONFIG = {
    'checkpoint_path': get_env_var('CHECKPOINT_PATH', os.path.join(_default_model_path, 'epoch_latest.pt')),
    'model_path': _default_model_path,
    # 如果未设置 MODEL_DOWNLOAD_ROOT，则默认与 model_path 一致
    'download_root': get_env_var('MODEL_DOWNLOAD_ROOT', _default_model_path),
}

# 优化后的IVFFlat参数配置
IVFFLAT_CONFIG = {
    # Text→Image 优化参数
    'text_to_image': {
        'lists': 170,
        'probes': 18,
        'expected_recall@10': 0.9544,
        'expected_avg_time_per_query': 0.102
    },
    
    # Image→Text 优化参数  
    'image_to_text': {
        'lists': 180,
        'probes': 18,
        'expected_recall@10': 0.9560,
        'expected_avg_time_per_query': 0.096
    }
}

# 数据库连接配置
DB_CONFIG = {
    'host': get_env_var('DB_HOST', 'localhost'),
    'port': get_env_var('DB_PORT', '5432'),
    'database': get_env_var('DB_NAME', 'retrieval_db'),
    'minconn': int(get_env_var('DB_MINCONN', '1')),
    'maxconn': int(get_env_var('DB_MAXCONN', '5'))
}

# 检索服务配置
RETRIEVAL_CONFIG = {
    'prefetch_limit': 200,  # 预取结果数量
    'cache_ttl_seconds': 300,  # 缓存过期时间
    'default_limit': 20,  # 默认返回结果数
    'max_limit': 100  # 最大返回结果数
}

def get_ivfflat_config(task_type: str) -> dict:
    """
    获取指定任务的IVFFlat配置
    
    Args:
        task_type: 'text_to_image' 或 'image_to_text'
    
    Returns:
        包含lists和probes的配置字典
    """
    if task_type not in IVFFLAT_CONFIG:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return IVFFLAT_CONFIG[task_type]

def get_probes_for_task(task_type: str) -> int:
    """
    获取指定任务的probes参数
    
    Args:
        task_type: 'text_to_image' 或 'image_to_text'
    
    Returns:
        probes值
    """
    config = get_ivfflat_config(task_type)
    return config['probes']

def get_model_config() -> dict:
    """
    获取模型配置
    
    Returns:
        模型配置字典
    """
    return MODEL_CONFIG

def get_db_config() -> dict:
    """
    获取数据库配置
    
    Returns:
        数据库配置字典
    """
    return DB_CONFIG

def validate_config() -> None:
    """
    验证配置的完整性
    """
    # 检查模型文件是否存在
    checkpoint_path = MODEL_CONFIG['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # 检查模型目录是否存在
    model_path = MODEL_CONFIG['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    print(f"✓ Model checkpoint: {checkpoint_path}")
    print(f"✓ Model directory: {model_path}")
    print(f"✓ Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
