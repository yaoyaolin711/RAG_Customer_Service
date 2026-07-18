import sys
pkgs = [
    ('fastapi', 'fastapi'),
    ('uvicorn', 'uvicorn'),
    ('torch', 'torch'),
    ('transformers', 'transformers'),
    ('peft', 'peft'),
    ('FlagEmbedding', 'FlagEmbedding'),
    ('pymilvus', 'pymilvus'),
    ('langchain', 'langchain'),
    ('streamlit', 'streamlit'),
    ('redis', 'redis'),
    ('pymysql', 'pymysql'),
    ('jieba', 'jieba'),
]
ok = 0
fail = 0
for name, mod in pkgs:
    try:
        __import__(mod)
        print(f'  OK: {name}')
        ok += 1
    except ImportError as e:
        print(f'  MISS: {name} - {e}')
        fail += 1
print(f'\nTotal: {ok} OK, {fail} missing')
exit(0 if fail == 0 else 1)
