import lmdb
import ast

lmdb_path = r"E:\surf\coco_lmdb_database"

# 打开只读环境
env = lmdb.open(lmdb_path, readonly=True, lock=False)

with env.begin() as txn:
    cursor = txn.cursor()
    count = 0

    print("\n[INFO] Previewing first 5 LMDB entries:\n")
    for key, value in cursor:
        image_id = key.decode("utf-8")

        # 由于写入时使用 str(record)，这里要反序列化字符串为字典
        try:
            record = ast.literal_eval(value.decode("utf-8"))
        except Exception as e:
            print(f"[ERROR] Failed to parse record for key {image_id}: {e}")
            continue

        print(f"Record #{count + 1}")
        print(f"Image ID      : {record.get('image_id')}")
        print(f"Caption       : {record.get('caption')}")
        # print(f"Boson Segment : {record.get('boson_segment')}")
        print(f"Tags          : {record.get('tags')}")
        print(f"Image Base64  : {record.get('image_base64')[:60]}...")  # 仅输出前60字符
        print("-" * 60)

        count += 1
        if count >= 5:
            break

env.close()
