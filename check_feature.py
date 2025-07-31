# # python check_feature.py

# """
# Item 0:
#   ('243-0-1', tensor([-0.0056, -0.0280, -0.0306,  ...,  0.0245,  0.0105, -0.0046],
#        dtype=torch.float16))
# --------------------
# Item 1:
#   ('243-1-1', tensor([ 0.0054, -0.0231, -0.0299,  ...,  0.0185,  0.0022,  0.0004],
#        dtype=torch.float16))
# --------------------
# """

# import torch  

# file_path = "/home/hsh/surf/cache/tw/tw_image_features.pt"  # 替换为你的文件路径  
# num_items_to_show = 3  # 你想显示多少个数据项  
# vector_preview_length = 10  # 向量显示多少个字符  

# try:  
#     data = torch.load(file_path)  

#     print(f"成功加载文件: {file_path}")  

#     if isinstance(data, dict):  
#         # 如果数据是字典  
#         for key, value in list(data.items())[:num_items_to_show]:  
#             print(f"Key: {key}")  
#             if isinstance(value, torch.Tensor):  
#                 # 如果值是 Tensor  
#                 vector_str = str(value.tolist())  
#                 if len(vector_str) > vector_preview_length:  
#                     print(f"Value: {vector_str[:vector_preview_length]}... (向量过长，已省略)")  
#                 else:  
#                     print(f"Value: {vector_str}")  
#             else:  
#                 print(f"Value: {value}")  # 如果不是 Tensor，直接打印  
#             print("-" * 20)  # 分隔符  
#     elif isinstance(data, list):  
#         # 如果数据是列表  
#         for i, item in enumerate(data[:num_items_to_show]):  
#             print(f"Item {i}:")  
#             if isinstance(item, torch.Tensor):  
#                 # 如果是 Tensor  
#                 vector_str = str(item.tolist())  
#                 if len(vector_str) > vector_preview_length:  
#                     print(f"  {vector_str[:vector_preview_length]}... (向量过长，已省略)")  
#                 else:  
#                     print(f"  {vector_str}")  
#             else:  
#                 print(f"  {item}")  
#             print("-" * 20)  # 分隔符  
#     else:  
#         print("数据类型无法识别，请检查数据结构。")  

# except FileNotFoundError:  
#     print(f"文件未找到: {file_path}")  
# except Exception as e:  
#     print(f"发生错误: {e}")
# --------------------------------------------------------

# import torch

# FEATURE_PATH = "/home/hsh/surf/cache/zj_text_features.pt"

# def main():
#     # 加载特征文件
#     data = torch.load(FEATURE_PATH, map_location="cpu")

#     text_ids = data.get("text_ids")
#     text_features = data.get("text_features")

#     if text_ids is None or text_features is None:
#         print("❌ 无法从缓存文件中读取 text_ids 或 text_features")
#         return

#     print(f"✅ 成功加载 {len(text_ids)} 条数据，特征维度: {text_features.shape[1]}")

#     print("\n📦 前5条样本（仅展示特征前10个值）:")
#     for i in range(min(5, len(text_ids))):
#         feature_short = text_features[i][:10].tolist()  # 转成 list 显示前10维
#         print(f"[{i}] text_id: {text_ids[i]} | feature[:10]: {feature_short}")

# if __name__ == "__main__":
#     main()
# ----------------------------------------------

# import psycopg2
# from psycopg2 import pool

# # ====== 初始化数据库连接池 ======
# db_pool = pool.SimpleConnectionPool(
#     minconn=1,
#     maxconn=5,
#     host='localhost',
#     port='5432',
#     database='retrieval_db'
# )

# def preview_tw_image(n=5):
#     conn = db_pool.getconn()
#     cur = conn.cursor()
#     cur.execute(f"SELECT image_id, LEFT(image_feature::text, 60) || '...', local_path, url, source FROM tw_image LIMIT {n}")
#     for row in cur.fetchall():
#         print(row)
#     cur.close()
#     db_pool.putconn(conn)

# preview_tw_image()

import torch

# 加载文件
data = torch.load("/home/hsh/surf/cache/surf/surf_image_features.pt", map_location="cpu")

# 打印数据类型
print("数据类型:", type(data))

# 如果是字典，打印 key 列表
if isinstance(data, dict):
    print("字典 keys:", list(data.keys()))
    for i, (k, v) in enumerate(data.items()):
        print(f"第{i+1}项: 键 = {k}, 值类型 = {type(v)}")
        if isinstance(v, torch.Tensor):
            print("值是 Tensor，前10个元素:", v.flatten()[:10])
        elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
            print("值是 Tensor 列表，第一个张量前10个元素:", v[0].flatten()[:10])
        if i >= 2:
            break

# 如果是列表，打印前几项
elif isinstance(data, list):
    print("列表长度:", len(data))
    for i, item in enumerate(data[:3]):
        print(f"第{i+1}项类型: {type(item)}")
        if isinstance(item, torch.Tensor):
            print("前10个元素:", item.flatten()[:10])

# 其他类型
else:
    print("未知结构，直接打印:", data)
'''
数据类型: <class 'dict'>
字典 keys: ['text_ids', 'text_features']
第1项: 键 = text_ids, 值类型 = <class 'torch.Tensor'>
值是 Tensor，前10个元素: tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
第2项: 键 = text_features, 值类型 = <class 'torch.Tensor'>
值是 Tensor，前10个元素: tensor([-0.0019, -0.0588, -0.0252, -0.0148, -0.0202, -0.0343, -0.0010,  0.0132,
        -0.0076,  0.0007], dtype=torch.float16)
hsh@star-R5300-G5 ~/s/pgvector69> python check_feature.py                                                                              (py310) 
数据类型: <class 'dict'>
字典 keys: ['image_ids', 'image_features']
第1项: 键 = image_ids, 值类型 = <class 'torch.Tensor'>
值是 Tensor，前10个元素: tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
第2项: 键 = image_features, 值类型 = <class 'torch.Tensor'>
值是 Tensor，前10个元素: tensor([ 0.0088, -0.0201,  0.0306, -0.0131, -0.0234, -0.0054,  0.0318,  0.0044,
         0.0011, -0.0077], dtype=torch.float16)
'''