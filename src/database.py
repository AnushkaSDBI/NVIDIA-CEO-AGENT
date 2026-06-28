# ============================================================
#  src/database.py  —  TASK 2: SQLite repository
#                      (the dashboard's database endpoint)
#
#  Loads documents.json into a real, queryable SQLite database
#  and exposes a small query API the dashboard calls for the
#  Company Overview, source counts, document tables, and search.
#
#  Build:  python -m src.database
#  Use:    from src import database as db ; db.counts_by_source()
# ============================================================

import os
import json
import sqlite3
import hashlib
from datetime import datetime

import config as cfg


def _uid(d):
    """Stable id for de-duplication across daily runs."""
    key = d.get("url") or (d.get("source", "") + d.get("title", "") + (d.get("text", "") or "")[:200])
    return hashlib.md5(key.encode("utf-8", "ignore")).hexdigest()


def append_documents(docs):
    """Insert only NEW documents (dedup by uid). The table accumulates daily."""
    con = _conn()
    cur = con.cursor()
    # migrate any old-schema documents table (no uid column)
    cols = [r[1] for r in cur.execute("PRAGMA table_info(documents)").fetchall()]
    if cols and "uid" not in cols:
        cur.execute("DROP TABLE documents")
    cur.execute("""CREATE TABLE IF NOT EXISTS documents (
        uid TEXT PRIMARY KEY, source TEXT, section TEXT, title TEXT, text TEXT,
        url TEXT, published TEXT, sentiment TEXT, sentiment_score REAL, collected_at TEXT)""")
    now = datetime.now().isoformat(timespec="seconds")
    added = 0
    for d in docs:
        cur.execute("""INSERT OR IGNORE INTO documents
            (uid, source, section, title, text, url, published, sentiment, sentiment_score, collected_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (_uid(d), d.get("source"), d.get("section"), d.get("title"), d.get("text"),
             d.get("url"), d.get("published"), d.get("sentiment"), d.get("sentiment_score"), now))
        added += cur.rowcount
    con.commit()
    total = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    con.close()
    print(f"  DB: +{added} new documents (accumulated total {total})")
    return added


def export_documents_json(path=None):
    """Dump the full accumulated corpus to CLEAN_PATH so preprocess/repository see everything."""
    path = path or cfg.CLEAN_PATH
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT source, section, title, text, url, published, sentiment, sentiment_score "
                "FROM documents")
    cols = [c[0] for c in cur.description]
    docs = [dict(zip(cols, r)) for r in cur.fetchall()]
    con.close()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)
    print(f"  Exported {len(docs)} accumulated documents -> {path}")
    return docs


def build_db():
    """Legacy convenience: append whatever is in CLEAN_PATH, then raw + stock."""
    with open(cfg.CLEAN_PATH, encoding="utf-8") as f:
        append_documents(json.load(f))
    build_raw_table()
    build_stock_tables()


def _conn():
    os.makedirs(os.path.dirname(cfg.DB_PATH), exist_ok=True)
    con = sqlite3.connect(cfg.DB_PATH)
    con.row_factory = sqlite3.Row          # rows behave like dicts
    return con


# --- 1. Build the database from the cleaned corpus ----------
# --- Raw collected dump (pre-clean) -> raw_documents --------
def build_raw_table():
    if not os.path.exists(cfg.RAW_PATH):
        return
    with open(cfg.RAW_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    con = _conn(); cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS raw_documents")
    cur.execute("""CREATE TABLE raw_documents (
        id INTEGER PRIMARY KEY, source TEXT, section TEXT, title TEXT,
        text TEXT, url TEXT, published TEXT)""")
    cur.executemany("INSERT INTO raw_documents VALUES (?,?,?,?,?,?,?)",
        [(i + 1, d.get("source"), d.get("section"), d.get("title"),
          d.get("text"), d.get("url"), d.get("published")) for i, d in enumerate(raw)])
    con.commit(); con.close()
    print(f"  Loaded {len(raw)} raw documents -> raw_documents")


# --- Processed chunks -> chunks (called by preprocess) ------
def build_chunks_table():
    if not os.path.exists(cfg.CHUNKS_PATH):
        print("  (no chunks.json yet - run preprocess first)")
        return
    with open(cfg.CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    con = _conn(); cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS chunks")
    cur.execute("""CREATE TABLE chunks (
        chunk_id TEXT PRIMARY KEY, source TEXT, section TEXT, title TEXT,
        url TEXT, published TEXT, text TEXT, normalized TEXT)""")
    cur.executemany("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?)",
        [(c.get("chunk_id"), c.get("source"), c.get("section"), c.get("title"),
          c.get("url"), c.get("published"), c.get("text"), c.get("normalized"))
         for c in chunks])
    con.commit(); con.close()
    print(f"  Loaded {len(chunks)} processed chunks -> chunks table")


# --- Stock tables (prices + technical indicators) -----------
def build_stock_tables():
    path = "data/clean/stock.json"
    if not os.path.exists(path):
        print("  (no stock.json yet - run collect first)")
        return
    with open(path, encoding="utf-8") as f:
        stock = json.load(f)
    con = _conn(); cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS stock_prices")
    cur.execute("CREATE TABLE stock_prices (date TEXT, close REAL, volume INTEGER)")
    cur.executemany("INSERT INTO stock_prices VALUES (?,?,?)",
                    [(p["date"], p["close"], p["volume"]) for p in stock.get("history", [])])
    cur.execute("DROP TABLE IF EXISTS stock_indicators")
    cur.execute("CREATE TABLE stock_indicators (metric TEXT, value TEXT)")
    cur.executemany("INSERT INTO stock_indicators VALUES (?,?)",
                    [(k, str(v)) for k, v in stock.get("indicators", {}).items()])
    con.commit(); con.close()
    print(f"  Loaded {len(stock.get('history', []))} price rows + "
          f"{len(stock.get('indicators', {}))} indicators")


def stock_indicators():
    con = _conn(); cur = con.cursor()
    try:
        cur.execute("SELECT metric, value FROM stock_indicators")
        rows = {r["metric"]: r["value"] for r in cur.fetchall()}
    except Exception:
        rows = {}
    con.close(); return rows


def stock_history(limit=500):
    con = _conn(); cur = con.cursor()
    try:
        cur.execute("SELECT date, close, volume FROM stock_prices ORDER BY date LIMIT ?", (limit,))
        rows = [dict(r) for r in cur.fetchall()]
    except Exception:
        rows = []
    con.close(); return rows


# --- 2. Query API for the dashboard -------------------------
def overview():
    con = _conn(); cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS n, COUNT(DISTINCT source) AS s FROM documents")
    r = cur.fetchone(); con.close()
    return {"company": cfg.COMPANY["name"],
            "industry": cfg.COMPANY["industry"],
            "total_documents": r["n"], "sources": r["s"]}


def counts_by_source():
    con = _conn(); cur = con.cursor()
    cur.execute("SELECT source, COUNT(*) AS n FROM documents "
                "GROUP BY source ORDER BY n DESC")
    rows = cur.fetchall(); con.close()
    return {r["source"]: r["n"] for r in rows}


def latest_collected_at():
    """Most recent document-collection timestamp (used for the UI 'last refresh')."""
    con = _conn(); cur = con.cursor()
    cur.execute("SELECT MAX(collected_at) AS ts FROM documents")
    row = cur.fetchone(); con.close()
    return row["ts"] if row else None


def reference_summary(max_chars=650):
    """First clean paragraph of the collected Wikipedia (reference) article — used as a
    fallback company overview when the live Wikipedia summary is unavailable."""
    con = _conn(); cur = con.cursor()
    try:
        cur.execute("SELECT text FROM documents WHERE source='reference' AND text IS NOT NULL "
                    "ORDER BY LENGTH(text) DESC LIMIT 1")
        row = cur.fetchone()
    except Exception:
        row = None
    con.close()
    if not row or not row[0]:
        return ""
    text = " ".join(str(row[0]).split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    dot = cut.rfind(". ")
    return (cut[:dot + 1] if dot > 200 else cut + " …").strip()


def get_documents(source=None, limit=50):
    con = _conn(); cur = con.cursor()
    if source:
        cur.execute("SELECT source, section, title, url, published, sentiment "
                    "FROM documents WHERE source = ? LIMIT ?", (source, limit))
    else:
        cur.execute("SELECT source, section, title, url, published, sentiment "
                    "FROM documents LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]; con.close()
    return rows


def search_text(query, limit=20):
    con = _conn(); cur = con.cursor()
    cur.execute("SELECT source, title, url FROM documents "
                "WHERE text LIKE ? LIMIT ?", (f"%{query}%", limit))
    rows = [dict(r) for r in cur.fetchall()]; con.close()
    return rows


if __name__ == "__main__":
    build_db()
    print("overview :", overview())
    print("by source:", counts_by_source())