# ============================================================
#  src/daily_brief.py  —  "What changed since last collection"
#
#  Uses the accumulating documents table (collected_at) to produce a
#  daily briefing of NEW documents since the previous collection date,
#  grouped by source. Writes data/clean/daily_brief.json.
#
#  Run (after collect):  python -m src.daily_brief
# ============================================================

import json
import sqlite3

import config as cfg


def run():
    con = sqlite3.connect(cfg.DB_PATH)
    cur = con.cursor()
    dates = [r[0] for r in cur.execute(
        "SELECT DISTINCT substr(collected_at,1,10) d FROM documents WHERE collected_at IS NOT NULL "
        "ORDER BY d DESC")]
    if not dates:
        print("daily_brief: no dated documents yet.")
        return None

    latest = dates[0]
    prev = dates[1] if len(dates) > 1 else None
    rows = cur.execute(
        "SELECT source, title, url FROM documents WHERE substr(collected_at,1,10)=? ORDER BY source",
        (latest,)).fetchall()
    con.close()

    by_source = {}
    for s, t, u in rows:
        by_source.setdefault(s, []).append({"title": t, "url": u})

    brief = {
        "date": latest,
        "previous_date": prev,
        "new_total": len(rows),
        "by_source": {s: len(v) for s, v in by_source.items()},
        "items": {s: v[:10] for s, v in by_source.items()},   # cap per source
    }
    with open("data/clean/daily_brief.json", "w", encoding="utf-8") as f:
        json.dump(brief, f, indent=2)

    print(f"Daily brief for {latest} (prev: {prev}): {len(rows)} new documents")
    for s, n in sorted(brief["by_source"].items(), key=lambda x: -x[1]):
        print(f"  {s:12s} {n}")
    print("  -> data/clean/daily_brief.json")
    return brief


if __name__ == "__main__":
    run()