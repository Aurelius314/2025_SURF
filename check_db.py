# import psycopg2
# from psycopg2 import pool

# try:
#     db_pool = pool.SimpleConnectionPool(
#         minconn=1,
#         maxconn=5,
#         host='localhost',
#         port='5432',
#         database='retrieval_db'
#     )

#     conn = db_pool.getconn()
#     cursor = conn.cursor()
#     cursor.execute("SELECT version();")
#     print("连接成功！PostgreSQL 版本:", cursor.fetchone())

#     cursor.close()
#     db_pool.putconn(conn)

# except Exception as e:
#     print("连接失败:", e)

# 连接成功！PostgreSQL 版本: ('PostgreSQL 17.5 on x86_64-windows, compiled by msvc-19.44.35209, 64-bit',)

# 连接成功！PostgreSQL 版本: ('PostgreSQL 16.2 on x86_64-pc-linux-gnu, compiled by gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0, 64-bit',)

# -----------------------------------------------

import psycopg2

# 替换成你自己的连接信息
conn = psycopg2.connect(
    dbname="retrieval_db",
    host="localhost",
    port=5432,
)
cursor = conn.cursor()

# 获取所有 table 名字
cursor.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
""")
tables = cursor.fetchall()
print("所有表：", [t[0] for t in tables])

# 查看 zj_text 前几条内容
if ('zj_text',) in tables:
    print("\n表 zj_text 示例数据：")
    cursor.execute("SELECT * FROM zj_text LIMIT 1")
    for row in cursor.fetchall():
        print(row)

# 查看 zj_image 前几条内容
if ('zj_image',) in tables:
    print("\n表 zj_image 示例数据：")
    cursor.execute("SELECT * FROM zj_image LIMIT 1")
    for row in cursor.fetchall():
        print(row)

cursor.close()
conn.close()
