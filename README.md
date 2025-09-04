
## 向量数据库

PostgreSQL数据库：`retrieval_db`

数据库结构：两大“物化视图”+三组“源表”

1. 两大“物化视图”

- `image_corpus: src, id, embedding`

- `text_corpus: src, id, embedding`

2. 三组“源表” （共约52万metadata，其中119,941文本数据，399,982图片数据，数据量共519,923）

- `zj_text`, `zj_image` 浙江省博物院
- `tw_text`, `tw_image` 台北故宫博物院
- `surf_text`, `surf_image` 敦煌数据集


## 模型

```
/models/
├── epoch_latest.pt          # checkpoints
└── cn_clip/                 # CN-CLIP 预训练底座
    ├── ViT-H-14.pt
    └── ...
```

后端推理用时（编码+归一化+similarity search，已经过ivfflat优化）
- `image_to_text`: ~0.17s
- `text_to_image`: ~0.13s

CN-CLIP 预训练底座部署：参考cnclip官方说明[OFA-Sys/Chinese-CLIP: Chinese version of CLIP which achieves Chinese cross-modal retrieval and representation generation.](https://github.com/OFA-Sys/Chinese-CLIP)

虚拟环境配置
- 确保 Python 环境有 torch、cn_clip、pgvector、psycopg2 等依赖
- 可参考requirements.txt

路径设置
- 请在启动前检查修改文件路径，如checkpoints路径

command
- 启动服务：`uvicorn main_corpus:app`

## demo

![荷花_demo](https://github.com/user-attachments/assets/544575c1-1455-4099-ae85-a359798f2418)
