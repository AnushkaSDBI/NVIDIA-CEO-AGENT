import sqlite3, collections as c
con = sqlite3.connect("storage/ai_ceo.db")
rows = con.execute("SELECT source FROM documents WHERE substr(collected_at,1,10)='2026-06-18'")
print("today by source:", c.Counter(r[0] for r in rows))
