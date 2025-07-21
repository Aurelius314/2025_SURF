from psycopg2 import pool
from pgvector.psycopg2 import register_vector

db_pool = None 

# 在 FastAPI 启动时调用
def init_db():
    global db_pool
    db_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        user='postgres',
        password='moxi2027aure',
        host='localhost',
        port='5432',
        database='cnclip_db'
    )

    conn = db_pool.getconn()
    register_vector(conn)
    db_pool.putconn(conn)