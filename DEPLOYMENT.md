# CN-CLIP 检索服务部署指南

## 模型部署要求

### 1. 模型文件结构
```
/models/
├── epoch_latest.pt          # 训练好的模型权重文件
└── cn_clip/                 # CN-CLIP 预训练模型文件
    ├── ViT-H-14.pt         # Vision Transformer 模型
    ├── ViT-H-14.txt        # 模型配置信息
    └── ...                 # 其他预训练文件
```

### 2. 模型加载说明

#### 当前实现
- **模型架构**: ViT-H-14 (Vision Transformer Huge)
- **权重加载**: 从 `epoch_latest.pt` 加载自定义训练的权重
- **预训练底座**: 使用 CN-CLIP 的 ViT-H-14 预训练模型作为底座

#### 关键代码路径
```python
# utils.py 中的模型加载逻辑
def load_surf_checkpoint_model_from_base(ckpt_path: str):
    # 1. 加载预训练底座
    model, preprocess = load_from_name("ViT-H-14", device=device, download_root='./')
    
    # 2. 加载自定义权重
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    
    # 3. 处理权重键名（去掉 module. 前缀）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    
    # 4. 加载权重到模型
    model.load_state_dict(new_state_dict, strict=False)
    return model, preprocess
```

### 3. 部署环境要求

#### 硬件要求
- **GPU**: 推荐 NVIDIA GPU (CUDA 11.8+)
- **内存**: 至少 16GB RAM (模型加载需要 ~8GB)
- **存储**: 至少 10GB 可用空间 (模型文件 + 依赖)

#### 软件要求
- **Python**: 3.8-3.10
- **CUDA**: 11.8+ (如果使用 GPU)
- **PostgreSQL**: 13+ with pgvector extension

### 4. 模型文件部署

#### 方式一：直接文件部署
```bash
# 1. 创建模型目录
mkdir -p /opt/models

# 2. 复制模型文件
cp epoch_latest.pt /opt/models/
cp -r cn_clip/ /opt/models/

# 3. 设置权限
chmod -R 755 /opt/models
```

#### 方式二：容器化部署
```dockerfile
# 在 Dockerfile 中
COPY models/ /app/models/
ENV MODEL_PATH=/app/models
```

### 5. 环境变量配置

```bash
# 模型路径
MODEL_PATH=/opt/models
CHECKPOINT_PATH=/opt/models/epoch_latest.pt

# 数据库配置
DB_HOST=localhost
DB_PORT=5432
DB_NAME=retrieval_db
DB_USER=your_user
DB_PASSWORD=your_password

# 服务配置
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=1
```

### 6. 模型预热

#### 启动时预热
```python
@app.on_event("startup")
async def startup_event():
    global model, preprocess
    # 加载模型
    model, preprocess = load_surf_checkpoint_model_from_base(
        ckpt_path=CHECKPOINT_PATH
    )
    model.eval()
    
    # 预热模型（可选）
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        _ = model.encode_image(dummy_image)
    print("模型预热完成")
```

### 7. 性能优化建议

#### 模型优化
- **量化**: 使用 FP16 或 INT8 量化减少内存占用
- **批处理**: 支持批量图像处理提高吞吐量
- **缓存**: 实现模型输出缓存减少重复计算

#### 内存管理
```python
# 及时清理 GPU 内存
torch.cuda.empty_cache()
gc.collect()

# 使用 torch.no_grad() 减少内存占用
with torch.no_grad():
    features = model.encode_image(image_tensor)
```

### 8. 监控和日志

#### 模型性能监控
- **加载时间**: 记录模型加载耗时
- **推理时间**: 监控单次推理延迟
- **内存使用**: 监控 GPU/CPU 内存占用
- **错误率**: 统计模型推理失败率

#### 日志配置
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model.log'),
        logging.StreamHandler()
    ]
)
```

### 9. 故障排除

#### 常见问题
1. **CUDA 内存不足**
   - 减少批处理大小
   - 使用 CPU 推理
   - 清理 GPU 缓存

2. **模型加载失败**
   - 检查文件路径和权限
   - 验证模型文件完整性
   - 确认 PyTorch 版本兼容性

3. **推理速度慢**
   - 启用 GPU 加速
   - 使用模型量化
   - 优化图像预处理

#### 健康检查
```python
@app.get("/health")
async def health_check():
    try:
        # 测试模型推理
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model.encode_image(dummy_input)
        return {"status": "healthy", "model": "loaded"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### 10. 安全考虑

#### 模型文件安全
- 使用文件系统权限控制访问
- 考虑模型文件加密存储
- 定期备份模型文件

#### API 安全
- 实现 API 密钥认证
- 限制请求频率
- 验证输入数据格式

---

## 快速部署检查清单

- [ ] 模型文件已正确放置
- [ ] 依赖包已安装 (requirements.txt)
- [ ] 环境变量已配置
- [ ] 数据库连接正常
- [ ] GPU 驱动和 CUDA 已安装
- [ ] 服务启动成功
- [ ] 健康检查通过
- [ ] 监控和日志已配置
