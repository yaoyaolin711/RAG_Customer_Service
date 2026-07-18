"""
FAQ 澶氱骇缂撳瓨鐏屽簱锛欵xcel 鈫?MySQL(rag_faq) + Redis 闂绱㈠紩

- MySQL锛氱嫭绔嬪簱 MYSQL_FAQ_DATABASE锛堥粯璁?rag_faq锛夛紝鍏ㄩ噺鏇挎崲鏈〃
- Redis锛氫粎娓呯悊 faq:qa:*锛岄鐑棶娉曠储寮?+ 绛旀锛堜笉杩囨湡绱㈠紩锛?
- 闂硶锛氱敱涓荤被/绫诲瀷/鍓被鐢熸垚 variants锛宻earch_text 涓嶅惈璇濇湳姝ｆ枃

鐢ㄦ硶:
  python scripts/ingest_faq_cache.py
  python scripts/ingest_faq_cache.py --file "path/to/瀹㈡湇鍥㈤槦_鍟嗗搧璇濇湳FAQ.xlsx"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
for _P in [os.path.join(_REPO_ROOT, "crm_agent", "crm_agent"), os.path.join(_REPO_ROOT, "RAG_mode", "mode")]:
    if os.path.isfile(os.path.join(_P, "settings.py")):
        sys.path.append(_P)
        break

import pandas as pd

from services.faq_store_mysql import get_mysql_faq_store
from services.faq_store_redis import get_redis_faq_store
from services.qa_cache import lookup_qa_cache, refresh_bm25_from_redis
from services.qa_normalize import normalize_question
from services.qa_variants import build_question_variants, build_search_text
from settings import MYSQL_FAQ_DATABASE

DEFAULT_FAQ = r"c:\Users\Administrator\Desktop\宸ヤ綔鍙癨寰呭垏鍒嗘枃妗瀹㈡湇鍥㈤槦_鍟嗗搧璇濇湳FAQ.xlsx"


def _norm_header(value: object) -> str:
    import re

    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip()
    return re.sub(
        "[" "\U0001F300-\U0001F9FF" "\U00002600-\U000027BF" "\U0000FE00-\U0000FE0F" "]+",
        "",
        text,
        flags=re.UNICODE,
    ).strip()


def _cell(row: pd.Series, *names: str) -> str:
    for name in names:
        if name not in row.index:
            continue
        value = row.get(name)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def excel_to_rows(path: str | Path) -> list[dict]:
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [_norm_header(c) for c in df.columns]
    source = Path(path).name
    rows: list[dict] = []
    for _, row in df.iterrows():
        answer = _cell(row, "蹇嵎璇濇湳")
        if not answer:
            continue
        main = _cell(row, "涓荤被") or "閫氱敤"
        qa_type = _cell(row, "绫诲瀷") or "閫氱敤"
        sub = _cell(row, "鍓被") or "璇濇湳"
        variants = build_question_variants(main, qa_type, sub)
        question_text = f"{main} {sub}"
        search_text = build_search_text(main, qa_type, sub, variants)
        rows.append(
            {
                "main_class": main,
                "qa_type": qa_type,
                "sub_class": sub,
                "question_text": question_text,
                "question_variants": variants,
                "search_text": search_text,
                "answer": answer,
                "source": source,
            }
        )
    # 鍚屼富棰樺璇濇湳锛歲uestion_text 杩藉姞 #n锛岄伩鍏嶅睍绀洪敭鍐茬獊锛泇ariants 鍏变韩浠嶅彲鍛戒腑鏈€鍚庡啓鍏ョ殑 exact
    seen: dict[str, int] = {}
    for row in rows:
        base = row["question_text"]
        n = seen.get(base, 0) + 1
        seen[base] = n
        if n > 1:
            row["question_text"] = f"{base} #{n}"
            # 涓哄璇濇湳琛ュ敮涓€闂硶锛岄伩鍏嶄粎涓婚閿簰鐩歌鐩?
            extra = f"{base} 璇濇湳{n}"
            variants = list(row["question_variants"])
            if extra not in variants:
                variants.append(extra)
            row["question_variants"] = variants
            row["search_text"] = build_search_text(
                row["main_class"], row["qa_type"], row["sub_class"], variants
            )
    return rows


def drop_legacy_faq_in_session_db() -> None:
    """鑻ユ棫浼氳瘽搴?rag_app 娈嬬暀 faq_qa_pairs锛屼粎鍒犺琛紝涓嶅姩 chat_sessions銆?""
    import pymysql
    from settings import MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER

    if MYSQL_DATABASE == MYSQL_FAQ_DATABASE:
        return
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset="utf8mb4",
            autocommit=True,
            connect_timeout=3,
        )
    except Exception as exc:
        print(f"璺宠繃娓呯悊鏃у簱 {MYSQL_DATABASE}: {exc}")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES LIKE 'faq_qa_pairs'")
            if cur.fetchone():
                cur.execute("DROP TABLE faq_qa_pairs")
                print(f"宸蹭粠浼氳瘽搴?`{MYSQL_DATABASE}` 鍒犻櫎閬楃暀琛?faq_qa_pairs锛堜繚鐣?chat_sessions锛?)
            else:
                print(f"浼氳瘽搴?`{MYSQL_DATABASE}` 鏃?faq_qa_pairs锛岃烦杩?)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="鐏屽叆 FAQ 鍒?MySQL(rag_faq) + Redis")
    parser.add_argument("--file", default=DEFAULT_FAQ, help="鍟嗗搧璇濇湳 FAQ xlsx")
    parser.add_argument(
        "--preload-answers",
        action="store_true",
        default=True,
        help="鍚屾椂棰勭儹绛旀鍒?Redis锛堥粯璁ゅ紑鍚級",
    )
    parser.add_argument("--no-preload-answers", action="store_true", help="鍙储寮曢棶棰橈紝绛旀闂埌鍐嶅啓")
    parser.add_argument(
        "--keep-legacy-table",
        action="store_true",
        help="淇濈暀浼氳瘽搴撲腑鐨勬棫 faq_qa_pairs 琛紙榛樿浼氬垹闄よ閬楃暀琛級",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        raise FileNotFoundError(f"鎵句笉鍒?FAQ 鏂囦欢: {args.file}")

    preload_answers = False if args.no_preload_answers else True
    rows = excel_to_rows(args.file)
    print(f"Excel 鏈夋晥琛? {len(rows)}")
    if not rows:
        raise ValueError("娌℃湁鍙亴鍏ョ殑 FAQ 琛?)
    avg_v = sum(len(r["question_variants"]) for r in rows) / len(rows)
    print(f"闂硶 variants 骞冲潎 {avg_v:.1f} 鏉?琛岋紱鐩爣搴?MYSQL_FAQ_DATABASE={MYSQL_FAQ_DATABASE}")

    if not args.keep_legacy_table:
        drop_legacy_faq_in_session_db()

    mysql_store = get_mysql_faq_store()
    n = mysql_store.replace_all(rows)
    print(f"MySQL `{MYSQL_FAQ_DATABASE}`.faq_qa_pairs: {n} 琛岋紙宸插叏閲忔浛鎹級")

    db_rows = mysql_store.list_all()
    redis_store = get_redis_faq_store()
    redis_store.ping()
    qn = redis_store.load_all_questions(db_rows, preload_answers=preload_answers)
    bm25_n = refresh_bm25_from_redis()
    print(f"Redis faq:qa:* 宸查噸寤? {qn}锛孊M25 璇枡: {bm25_n}锛宲reload_answers={preload_answers}")

    samples = [
        ("澶氫箙鍙戣揣", "鍙戣揣"),
        ("浣犲ソ", "闂€?),
        ("鏈夎繍璐归櫓鍚?, "杩愯垂闄?),
        ("鐢ㄤ粈涔堝揩閫?, "蹇€?),
        (rows[0]["question_text"], "涓婚閿?),
    ]
    for q, label in samples:
        hit = lookup_qa_cache(q)
        if hit is None:
            print(f"smoke [{label}] q={q!r} -> miss")
        else:
            snippet = hit.answer[:40].encode("unicode_escape").decode("ascii")
            print(
                f"smoke [{label}] q={q!r} -> {hit.match_type} id={hit.faq_id} "
                f"main={hit.main_class!r} sub={hit.sub_class!r} score={hit.score:.3f} "
                f"ans={snippet}"
            )
    print("normalize:", normalize_question("澶氫箙鍙戣揣鍛紵"))

if __name__ == "__main__":
    main()

