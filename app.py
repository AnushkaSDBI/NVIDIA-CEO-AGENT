# ============================================================
#  app.py  —  AI CEO: Strategic Intelligence Dashboard
#
#  Required 7 sections, all reading PRE-COMPUTED artifacts (fast):
#    1. Company Overview              (DB)
#    2. Market Intelligence           (DB: stock_prices + indicators)
#    3. Opportunity Monitor           (intelligence.json -> opportunities)
#    4. Risk Monitor                  (intelligence.json -> risks)
#    5. Sentiment Analysis            (sentiment.json)
#    6. Strategic Recommendations     (intelligence.json -> recommendations)
#    7. CEO Briefing                  (intelligence.json -> briefing)
#
#  Extra pages after the required 7:
#    8. Competitive Landscape         (entities.json)
#    9. Source Trust                  (entities.json -> NetworkX/pyvis)
#   10. Ask the Agent                 (live: intelligence.brief)
#
#  Run from the project root:   streamlit run app.py
#  (Ollama only needs to be running for section 7.)
# ============================================================

import os
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import config as cfg
from src import database as db

ACCENT = "#76B900"          # NVIDIA green — the single accent
POS, NEG, NEU = "#76B900", "#e23b3b", "#7a7a85"
SENT_PATH = "data/clean/sentiment.json"
ENT_PATH = "data/clean/entities.json"

st.set_page_config(page_title="AI CEO — Strategic Intelligence",
                   page_icon="🧠", layout="wide")

st.markdown(f"""
<style>
  .block-container {{padding-top: 2rem;}}
  .card {{border: 1px solid #2a2e37; border-left: 4px solid {ACCENT};
          border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;
          background: rgba(255,255,255,0.02);}}
  .badge {{display:inline-block; padding:2px 10px; border-radius:12px;
           font-size:0.72rem; font-weight:600; margin-right:6px;}}
  .b-ok  {{background:rgba(118,185,0,0.16);  color:{ACCENT};}}
  .b-no  {{background:rgba(226,59,59,0.16);  color:#e23b3b;}}
  .b-na  {{background:rgba(122,122,133,0.16);color:#9aa0aa;}}
  .b-hi  {{background:rgba(226,59,59,0.16);  color:#e23b3b;}}
  .b-md  {{background:rgba(232,168,56,0.16); color:#e8a838;}}
  .b-lo  {{background:rgba(118,185,0,0.16);  color:{ACCENT};}}
  .cite  {{font-size:0.82rem; color:#b8bcc4; border-left:2px solid #2a2e37;
           padding-left:10px; margin:6px 0;}}

  /* ---------------- NVIDIA theme ---------------- */
  .stApp {{background:#0b0d0f;}}
  [data-testid="stHeader"] {{background:rgba(0,0,0,0);}}
  [data-testid="stSidebar"] {{background:#0a0c0e; border-right:1px solid #1c2230;}}
  h1, h2, h3 {{color:#ffffff; font-weight:700;}}
  h2 {{border-bottom:2px solid {ACCENT}; padding-bottom:5px;}}
  [data-testid="stMetricValue"] {{color:{ACCENT};}}
  [data-testid="stMetricLabel"] {{color:#9aa0aa;}}
  .stRadio [aria-checked="true"] {{color:{ACCENT};}}
  a {{color:{ACCENT};}}
  hr {{border-color:rgba(118,185,0,0.25);}}
  .nv-wordmark {{font-weight:800; letter-spacing:.5px; color:{ACCENT};}}
</style>
""", unsafe_allow_html=True)

# NVIDIA logo — shown top-left and in the sidebar when present next to app.py.
_LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images\download.jpeg")
try:
    if os.path.exists(_LOGO):
        st.logo(_LOGO)
except Exception:
    pass


# ---------------- cached loaders (fast, graceful, auto-invalidating) ----------------
def _mtime(path):
    """Modification time of a file (0 if missing). Passed into the cached loaders
    so that regenerating an artifact busts the cache automatically — no manual
    'Clear cache' needed after `refresh.py` / `python -m src.intelligence`."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _load_json_cached(path, stamp):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_json(path):
    # mtime is part of the cache key -> file changes => cache miss => reload
    return _load_json_cached(path, _mtime(path))


@st.cache_data(show_spinner=False)
def _load_overview_cached(stamp):
    try:
        return db.overview(), db.counts_by_source()
    except Exception:
        return None, {}


def load_overview():
    return _load_overview_cached(_mtime(getattr(cfg, "DB_PATH", "")))


@st.cache_data(show_spinner=False)
def _load_stock_cached(stamp):
    try:
        return db.stock_history(800), db.stock_indicators()
    except Exception:
        return [], {}


def load_stock():
    return _load_stock_cached(_mtime(getattr(cfg, "DB_PATH", "")))


def _f(v, default=None):
    """Indicators are stored as strings; parse a float safely."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


@st.cache_data(show_spinner=False, ttl=86400)
def _wiki_summary_live(title):
    """Clean 1-3 sentence company summary from Wikipedia's REST summary endpoint.
    Cached for a day; returns '' on any failure (offline, rate-limited, etc.)."""
    import urllib.request, urllib.parse
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AI-CEO-Dashboard/1.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.loads(r.read()).get("extract", "") or ""
    except Exception:
        return ""


def company_summary():
    """Summarised overview: prefer the live Wikipedia extract, fall back to the
    Wikipedia text already collected in the corpus. Returns (text, source_label)."""
    title = cfg.COMPANY.get("wiki_title") or cfg.COMPANY.get("name", "")
    s = _wiki_summary_live(title)
    if s:
        return s, "Wikipedia"
    try:
        s = db.reference_summary()
    except Exception:
        s = ""
    return (s, "Wikipedia (collected)") if s else ("", "")


def _last_refresh():
    """Best 'last refresh' timestamp for the UI: prefer the analysis run time,
    fall back to the most recent collected document in the DB."""
    intel = load_json(cfg.INTEL_PATH)
    ts = (intel or {}).get("generated_at", "")
    if ts:
        return ts[:16].replace("T", " ")
    try:
        latest = db.latest_collected_at()
        if latest:
            return str(latest)[:16].replace("T", " ")
    except Exception:
        pass
    return "—"


def _badge_verified(verified, confidence):
    if verified is True:
        label = "✓ verified" + (f" {confidence:.0%}" if confidence is not None else "")
        return f'<span class="badge b-ok">{label}</span>'
    if verified is False:
        return '<span class="badge b-no">✗ unverified</span>'
    return '<span class="badge b-na">— not checked</span>'


def _badge_sources(n):
    """Badge for how many INDEPENDENT source types back a finding/recommendation."""
    if not n:
        return ""
    cls = "b-ok" if n >= 3 else "b-md" if n == 2 else "b-na"
    return f'<span class="badge {cls}">⛓ {n} source{"s" if n != 1 else ""}</span>'


def _badge_level(level):
    cls = {"high": "b-hi", "medium": "b-md", "low": "b-lo"}.get((level or "").lower(), "b-na")
    return f'<span class="badge {cls}">{(level or "n/a").lower()} impact/risk</span>'


def _citations_block(citations):
    if not citations:
        st.caption("No evidence retrieved for this finding.")
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            head = f"**{c.get('source','?')}** — {c.get('title','')}"
            if c.get("section"):
                head += f"  ·  _{c['section']}_"
            ent = c.get("entailment")
            if ent is not None:
                head += f"  ·  support {ent}"
            st.markdown(head)
            if c.get("snippet"):
                st.markdown(f"<div class='cite'>{c['snippet']}…</div>", unsafe_allow_html=True)
            if c.get("url"):
                st.markdown(f"[source link]({c['url']})")


def _sqlite_path():
    """Find the local SQLite knowledge base, regardless of the exact config variable name."""
    candidates = []
    for attr in ("DB_PATH", "DATABASE_PATH", "SQLITE_PATH", "DB_FILE"):
        value = getattr(cfg, attr, None)
        if value:
            candidates.append(value)

    candidates.extend([
        "storage/ai_ceo.db",
        "data/ai_ceo.db",
        "ai_ceo.db",
    ])

    for candidate in candidates:
        path = Path(str(candidate))
        if path.exists():
            return str(path)
    return None


def _quote_identifier(name):
    """Safely quote SQLite table/column names discovered from the database schema."""
    return '"' + str(name).replace('"', '""') + '"'


def _pick_col(columns, *names):
    """Return the first matching column name from a list of possible names."""
    lower = {c.lower(): c for c in columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _detect_document_table(conn):
    """Detect the table that stores collected documents/articles."""
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    preferred = ["documents", "articles", "docs", "items", "corpus"]

    for table in preferred + tables:
        if table not in tables:
            continue
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()]
        low = {c.lower() for c in cols}
        has_text = bool(low & {"text", "content", "body", "summary", "snippet"})
        has_title = "title" in low
        has_url = bool(low & {"url", "link"})
        if has_text and (has_title or has_url):
            return table, cols
    return None, []


COMPETITOR_ALIASES = {
    "amd": ["AMD", "Advanced Micro Devices", "Radeon", "Instinct", "MI300", "MI350"],
    "advanced micro devices": ["Advanced Micro Devices", "AMD", "Radeon", "Instinct", "MI300", "MI350"],
    "intel": ["Intel", "Xeon", "Gaudi", "Arc GPU", "Foundry"],
    "alphabet": ["Alphabet", "Google", "TPU", "Gemini", "Google Cloud"],
    "google": ["Google", "Alphabet", "TPU", "Gemini", "Google Cloud"],
    "microsoft": ["Microsoft", "Azure", "Maia", "Copilot"],
    "amazon": ["Amazon", "AWS", "Trainium", "Inferentia", "Bedrock"],
    "aws": ["AWS", "Amazon", "Trainium", "Inferentia", "Bedrock"],
    "meta": ["Meta", "Facebook", "Llama", "MTIA"],
    "apple": ["Apple", "M4", "M-series", "Apple Intelligence"],
    "broadcom": ["Broadcom", "VMware", "ASIC"],
    "qualcomm": ["Qualcomm", "Snapdragon", "Nuvia"],
    "tesla": ["Tesla", "Dojo", "FSD"],
    "openai": ["OpenAI", "ChatGPT", "GPT"],
}


def _competitor_aliases(name):
    aliases = COMPETITOR_ALIASES.get(str(name).lower(), [name])
    # Keep order but remove duplicates/empty aliases.
    seen, cleaned = set(), []
    for alias in aliases + [name]:
        alias = str(alias).strip()
        key = alias.lower()
        if alias and key not in seen:
            seen.add(key)
            cleaned.append(alias)
    return cleaned


def _safe_snippet(text, limit=240):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _activity_tag(text):
    """Simple explainable label for the competitor activity shown in the dashboard."""
    t = str(text or "").lower()
    if any(k in t for k in ["launch", "release", "announce", "unveil", "introduc"]):
        return "Product / announcement signal"
    if any(k in t for k in ["partner", "collaborat", "alliance", "deal", "customer"]):
        return "Partnership / customer signal"
    if any(k in t for k in ["invest", "capex", "fund", "acquir", "buy"]):
        return "Investment / acquisition signal"
    if any(k in t for k in ["gpu", "chip", "accelerator", "semiconductor", "ai", "data center", "cloud"]):
        return "AI / semiconductor signal"
    if any(k in t for k in ["revenue", "earnings", "margin", "forecast", "guidance", "stock"]):
        return "Financial / market signal"
    if any(k in t for k in ["regulat", "export", "antitrust", "lawsuit", "restriction"]):
        return "Regulatory / risk signal"
    return "Competitor activity signal"


@st.cache_data(show_spinner=False)
def load_competitor_activity(competitor_names, limit_per_competitor=4):
    """Return recent competitor-related documents from the local SQLite corpus.

    This does not browse the web live. It uses the project repository, so links update
    whenever `python refresh.py` collects fresh documents.
    """
    result = {name: [] for name in competitor_names}
    db_path = _sqlite_path()
    if not db_path or not competitor_names:
        return result

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        table, cols = _detect_document_table(conn)
        if not table:
            return result

        title_col = _pick_col(cols, "title", "headline", "name")
        text_col = _pick_col(cols, "text", "content", "body", "summary", "snippet")
        url_col = _pick_col(cols, "url", "link")
        source_col = _pick_col(cols, "source", "source_name", "provider", "feed")
        published_col = _pick_col(cols, "published", "published_at", "date", "created_at", "timestamp")

        searchable_cols = [c for c in [title_col, text_col] if c]
        if not searchable_cols:
            return result

        select_parts = []
        for alias, col in [
            ("title", title_col),
            ("text", text_col),
            ("url", url_col),
            ("source", source_col),
            ("published", published_col),
        ]:
            if col:
                select_parts.append(f"{_quote_identifier(col)} AS {_quote_identifier(alias)}")
            else:
                select_parts.append(f"'' AS {_quote_identifier(alias)}")

        order_clause = f" ORDER BY {_quote_identifier(published_col)} DESC" if published_col else ""

        for competitor in competitor_names:
            aliases = _competitor_aliases(competitor)
            where_parts, params = [], []
            for alias in aliases:
                for col in searchable_cols:
                    where_parts.append(f"LOWER({_quote_identifier(col)}) LIKE LOWER(?)")
                    params.append(f"%{alias}%")

            query = (
                f"SELECT {', '.join(select_parts)} "
                f"FROM {_quote_identifier(table)} "
                f"WHERE {' OR '.join(where_parts)}"
                f"{order_clause} LIMIT 120"
            )
            rows = conn.execute(query, params).fetchall()

            alias_patterns = [
                re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.I)
                for alias in aliases
            ]
            seen_urls, docs = set(), []
            for row in rows:
                title = row["title"] or "Untitled source"
                text = row["text"] or ""
                combined = f"{title} {text}"
                if not any(rx.search(combined) for rx in alias_patterns):
                    continue

                url = row["url"] or ""
                dedupe_key = url or title.lower()
                if dedupe_key in seen_urls:
                    continue
                seen_urls.add(dedupe_key)

                docs.append({
                    "title": _safe_snippet(title, 140),
                    "url": url,
                    "source": row["source"] or "source",
                    "published": row["published"] or "date n/a",
                    "snippet": _safe_snippet(text),
                    "tag": _activity_tag(combined),
                })
                if len(docs) >= limit_per_competitor:
                    break
            result[competitor] = docs
    except Exception:
        return result
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return result


# ======================= sections =======================
def section_overview():
    st.header("Company Overview")
    ov, counts = load_overview()
    if not ov:
        st.info("No data yet. Run `python refresh.py` to build the corpus.")
        return
    _, ind = load_stock()
    intel = load_json(cfg.INTEL_PATH)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Documents", f"{ov['total_documents']:,}")
    c2.metric("Sources", ov["sources"])
    c3.metric("Latest close", f"${_f(ind.get('latest_close')) or '—'}")
    c4.metric("RSI (14)", ind.get("rsi_14", "—"))
    trend = ind.get("trend", "")
    c5.metric("Trend", "Bullish" if "bullish" in trend else "Bearish" if "bearish" in trend else "—")

    # --- Company overview (between the metrics and the source breakdown) ---
    st.subheader("Overview")
    summary, src = company_summary()
    if summary:
        st.markdown(
            f"<div class='card'>{summary} "
            f"<span style='color:#9aa0aa;font-size:.8rem'>— {src}</span></div>",
            unsafe_allow_html=True)
    aliases = ", ".join(cfg.COMPANY.get("aliases", [])[:6])
    topics = " · ".join(cfg.COMPANY.get("topics", [])[1:6])
    st.markdown(
        "<div class='card'>"
        f"<b>{ov['company']} ({cfg.COMPANY['ticker']})</b> &nbsp;·&nbsp; {ov['industry']}<br>"
        f"<span style='color:#9aa0aa'>Key brand &amp; product terms:</span> {aliases}"
        + (f"<br><span style='color:#9aa0aa'>Focus areas tracked:</span> {topics}" if topics else "")
        + "</div>", unsafe_allow_html=True)
    if intel and intel.get("generated_at"):
        st.caption(f"Intelligence generated: {intel['generated_at']}  ·  "
                   f"{ov['company']} ({cfg.COMPANY['ticker']}) · {ov['industry']}")

    # --- Corpus by source (unchanged) ---
    st.subheader("Corpus by source")
    if counts:
        d = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1]), columns=["source", "documents"])
        fig = go.Figure(go.Bar(x=d["documents"], y=d["source"], orientation="h",
                               marker_color=ACCENT))
        fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')


def section_stock():
    st.header("Market Intelligence")
    hist, ind = load_stock()
    if not hist:
        st.info("No stock data. Run `python refresh.py`.")
        return
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    # SMAs are computed on the FULL history so they remain correct at the left edge of any window
    df["SMA50"] = df["close"].rolling(50).mean()
    df["SMA200"] = df["close"].rolling(200).mean()

    # date-range slider — default to the last 1 year
    win = st.select_slider("Date range", options=["1M", "3M", "6M", "1Y", "2Y"], value="1Y")
    days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 100000}[win]
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    dfv = df[df["date"] >= cutoff]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfv["date"], y=dfv["close"], name="Close", line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=dfv["date"], y=dfv["SMA50"], name="SMA 50", line=dict(color="#e8a838", width=1)))
    fig.add_trace(go.Scatter(x=dfv["date"], y=dfv["SMA200"], name="SMA 200", line=dict(color="#6aa9ff", width=1)))
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, width='stretch')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("52-week high", f"${ind.get('high_52w','—')}")
    c2.metric("52-week low", f"${ind.get('low_52w','—')}")
    c3.metric("From 52w high", f"{ind.get('pct_from_52w_high','—')}%")
    c4.metric("RSI (14)", ind.get("rsi_14", "—"))
    st.caption(f"Trend: {ind.get('trend','n/a')}  ·  SMA50 ${ind.get('sma_50','—')} · SMA200 ${ind.get('sma_200','—')}")


def section_sentiment():
    st.header("Sentiment Analysis")
    s = load_json(SENT_PATH)
    if not s:
        st.info("No sentiment yet. Run `python -m src.sentiment` (or `python refresh.py`).")
        return
    o = s.get("overall", {})
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        fig = go.Figure(go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[o.get("positive", 0), o.get("neutral", 0), o.get("negative", 0)],
            hole=0.55, marker_colors=[POS, NEU, NEG]))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=10, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, width='stretch')
    c2.metric("News sentiment (mean)", s.get("news_sentiment", {}).get("mean", "—"),
              help="Mean compound score over news + market sources")
    c3.metric("Public sentiment (mean)", s.get("public_sentiment", {}).get("mean", "—"),
              help="Reddit + Hacker News, with sarcastic posts down-weighted")
    flagged = s.get("public_sentiment", {}).get("flagged_sarcastic", 0)
    if flagged:
        st.caption(f"⚠ {flagged} public post(s) flagged as possibly sarcastic by the irony model "
                   "and down-weighted in the mean (polarity not trusted).")

    by = s.get("by_source", {})
    if by:
        st.subheader("Mean sentiment by source")
        rows = [(src, v.get("mean", 0)) for src, v in by.items() if v.get("count")]
        rows.sort(key=lambda x: x[1])
        d = pd.DataFrame(rows, columns=["source", "mean"])
        fig = go.Figure(go.Bar(x=d["mean"], y=d["source"], orientation="h",
                               marker_color=[POS if m >= 0 else NEG for m in d["mean"]]))
        fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')

    asp = s.get("aspects", {})
    asp = {k: v for k, v in asp.items() if v.get("mentions")}
    if asp:
        st.subheader("Aspect-based sentiment (by strategic theme)")
        st.caption("Sentiment of the sentences mentioning each theme — not whole documents.")
        rows = [(k, v["mean"], v["mentions"]) for k, v in asp.items()]
        rows.sort(key=lambda x: x[1])
        d = pd.DataFrame(rows, columns=["aspect", "mean", "mentions"])
        fig = go.Figure(go.Bar(x=d["mean"], y=d["aspect"], orientation="h",
                               marker_color=[POS if m >= 0 else NEG for m in d["mean"]],
                               text=[f"{m:+.2f}  ({n} mentions)" for m, n in zip(d["mean"], d["mentions"])],
                               textposition="auto"))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')


def section_competitors():
    st.header("Competitive Landscape")
    e = load_json(ENT_PATH)
    if not e:
        st.info("No entities yet. Run `python -m src.entities` (or `python refresh.py`).")
        return
    comp = e.get("competitors", [])
    orgs = e.get("all_orgs", [])

    if comp:
        st.subheader("Known competitors (by mentions)")
        d = pd.DataFrame(comp).sort_values("mentions")
        fig = go.Figure(go.Bar(x=d["mentions"], y=d["name"], orientation="h", marker_color="#e8a838"))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')
    else:
        st.caption("No known competitors matched in this corpus.")

    # New examiner-friendly section: competitor actions with clickable evidence links.
    st.subheader("What competitors are currently doing")
    st.caption(
        "These links come from the collected document repository. "
        "Run `python refresh.py` before the exam to refresh the latest competitor activity."
    )

    competitor_names = [c.get("name") for c in comp if c.get("name")]
    activities = load_competitor_activity(tuple(competitor_names), limit_per_competitor=4)

    if not competitor_names:
        st.info("No competitor names found in entities.json yet.")
    else:
        for competitor in competitor_names:
            docs = activities.get(competitor, [])
            mentions = next((c.get("mentions", 0) for c in comp if c.get("name") == competitor), 0)
            with st.expander(f"{competitor} — {mentions} mentions · {len(docs)} recent source link(s)", expanded=False):
                if not docs:
                    st.caption(
                        "No matching link was found in the local corpus for this competitor. "
                        "Refresh the pipeline or add competitor-specific RSS/news sources."
                    )
                    continue

                for doc in docs:
                    title = doc.get("title") or "Untitled source"
                    url = doc.get("url")
                    source = doc.get("source", "source")
                    published = doc.get("published", "date n/a")
                    tag = doc.get("tag", "Competitor activity signal")
                    snippet = doc.get("snippet", "")

                    if url:
                        st.markdown(f"- **[{title}]({url})**")
                    else:
                        st.markdown(f"- **{title}**")
                    st.caption(f"{tag} · {source} · {published}")
                    if snippet:
                        st.markdown(f"<div class='cite'>{snippet}</div>", unsafe_allow_html=True)

    if orgs:
        st.subheader("Most-mentioned organizations")
        d = pd.DataFrame(orgs[:15]).sort_values("mentions")
        fig = go.Figure(go.Bar(x=d["mentions"], y=d["name"], orientation="h", marker_color=ACCENT))
        fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')

def _badge_corroboration(corr):
    if not corr or not corr.get("score"):
        return ""
    cls = {"strong": "b-ok", "moderate": "b-md", "weak": "b-na"}.get(corr.get("level"), "b-na")
    srcs = ", ".join(corr.get("sources", []))
    return f'<span class="badge {cls}">⛓ {corr["level"]} · {srcs}</span>'


def _finding_cards(items):
    for f in items:
        contested = ('<span class="badge b-no">⚠ contested</span>' if f.get("contested") else "")
        st.markdown(
            f"<div class='card'><b>{f.get('title','')}</b> "
            f"{_badge_level(f.get('impact'))}{_badge_verified(f.get('verified'), f.get('confidence'))}"
            f"{_badge_corroboration(f.get('corroboration'))}{contested}"
            f"<br>{f.get('detail','')}</div>", unsafe_allow_html=True)
        if f.get("contested") and f.get("contradicting_evidence"):
            ce = f["contradicting_evidence"]
            st.markdown(f"<div class='cite'><b>Disputed by</b> ({ce.get('source')}): "
                        f"{ce.get('snippet','')}…</div>", unsafe_allow_html=True)
        sb = f.get("score_breakdown")
        if sb:
            ent = sb.get("entailment")
            st.caption(f"Confidence {f.get('confidence')} = "
                       f"entailment {ent if ent is not None else '—'} · "
                       f"corroboration {sb.get('corroboration')} · freshness {sb.get('freshness')}  "
                       f"(weights: {sb.get('weights')})")
        _citations_block(f.get("citations", []))



def _load_intel_or_stop():
    """Load precomputed intelligence output and stop the page gracefully if missing."""
    intel = load_json(cfg.INTEL_PATH)
    if not intel:
        st.info("No intelligence analysis yet. Run `python -m src.intelligence` or `python refresh.py`.")
        return None
    st.caption(f"Generated {intel.get('generated_at','')} · claims are grounded in retrieved evidence and checked by NLI verification.")
    return intel


def section_opportunities():
    st.header("Opportunity Monitor")
    intel = _load_intel_or_stop()
    if not intel:
        return
    opportunities = intel.get("opportunities", [])
    st.metric("Opportunities detected", len(opportunities))
    st.caption("Shows opportunity title, impact level, supporting evidence, verification status, and confidence score.")
    if not opportunities:
        st.info("No opportunities were found in the latest intelligence run.")
        return
    _finding_cards(opportunities)


def section_risks():
    st.header("Risk Monitor")
    intel = _load_intel_or_stop()
    if not intel:
        return
    risks = intel.get("risks", [])
    st.metric("Risks detected", len(risks))
    st.caption("Shows risk title, risk/severity level, supporting evidence, verification status, and confidence score.")
    if not risks:
        st.info("No risks were found in the latest intelligence run.")
        return
    _finding_cards(risks)


def section_trends():
    st.header("Trend Monitor")
    intel = _load_intel_or_stop()
    if not intel:
        return
    trends = intel.get("trends", [])
    st.metric("Trends detected", len(trends))
    st.caption("Emerging technology and market trends relevant to the company, each with supporting "
               "evidence, verification status, corroboration, and confidence score.")
    if not trends:
        st.info("No trends were found in the latest intelligence run.")
        return
    _finding_cards(trends)


def _recommendation_cards(recommendations):
    if not recommendations:
        st.info("No strategic recommendations were generated in the latest intelligence run.")
        return
    for r in recommendations:
        addr = r.get("addresses", {})
        pr = r.get("priority", "")
        pcls = {"high": "b-no", "medium": "b-md", "low": "b-ok"}.get(pr, "b-na")
        prio = f'<span class="badge {pcls}">priority: {pr}</span>' if pr else ""

        # Quality signals: verified + number of independent evidence sources.
        quality = _badge_verified(r.get("verified"), r.get("confidence")) + \
            _badge_sources(r.get("evidence_sources"))

        # Traceable link: which opportunity/risk/trend this recommendation responds to.
        link = ""
        if addr.get("title"):
            conf = addr.get("confidence")
            conf_s = f" · confidence {conf}" if conf is not None else ""
            corr = f" · {addr.get('corroboration')}" if addr.get("corroboration") else ""
            contested = " · ⚠ contested" if addr.get("contested") else ""
            link = (f"<br><span style='color:#9aa0aa'>Addresses {addr.get('type','')}: "
                    f"<b>{addr.get('title','')}</b>{conf_s}{corr}{contested}</span>")

        st.markdown(
            f"<div class='card'><b>{r.get('action','')}</b> "
            f"{prio}{_badge_level(r.get('risk_level'))}{quality}"
            f"{link}"
            f"<br><b>Why:</b> {r.get('rationale','')}"
            f"<br><b>Expected impact:</b> {r.get('expected_impact','')}</div>",
            unsafe_allow_html=True,
        )
        _citations_block(r.get("citations", []))


def section_recommendations():
    st.header("Strategic Recommendations")
    intel = _load_intel_or_stop()
    if not intel:
        return
    recommendations = intel.get("recommendations", [])
    st.metric("Recommendations", len(recommendations))
    st.caption("Shows recommendation, priority, supporting evidence, expected impact, and risk level. "
               "Badges show whether each recommendation is verified and how many independent sources back it.")
    _recommendation_cards(recommendations)


def section_briefing():
    st.header("CEO Briefing")
    intel = _load_intel_or_stop()
    if not intel:
        return
    st.caption("Executive summary answering: what happened, why it matters, and what management should do next.")
    briefing = intel.get("briefing")
    if not briefing:
        st.info("No CEO briefing was generated in the latest intelligence run.")
        return
    st.markdown(
        f"<div class='card' style='border-left:3px solid {ACCENT}'>"
        f"<b>CEO Briefing</b><br>{briefing.replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True,
    )

def section_intelligence():
    st.header("CEO Intelligence")
    intel = load_json(cfg.INTEL_PATH)
    if not intel:
        st.info("No analysis yet. Run `python -m src.intelligence` (Ollama must be running).")
        return
    st.caption(f"Generated {intel.get('generated_at','')}  ·  every claim is grounded in retrieved "
               "evidence and checked by an NLI entailment model.")

    if intel.get("briefing"):
        st.markdown(f"<div class='card' style='border-left:3px solid {ACCENT}'>"
                    f"<b>CEO Briefing</b><br>{intel['briefing'].replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True)

    tabs = st.tabs([f"Opportunities ({len(intel.get('opportunities',[]))})",
                    f"Risks ({len(intel.get('risks',[]))})",
                    f"Trends ({len(intel.get('trends',[]))})",
                    f"Recommendations ({len(intel.get('recommendations',[]))})"])
    with tabs[0]:
        _finding_cards(intel.get("opportunities", []))
    with tabs[1]:
        _finding_cards(intel.get("risks", []))
    with tabs[2]:
        _finding_cards(intel.get("trends", []))
    with tabs[3]:
        _recommendation_cards(intel.get("recommendations", []))


def build_graph_html(entities):
    """Knowledge graph: NVIDIA + organizations, linked by real co-occurrence edges
    (not a star). Entities that appear together in documents are connected."""
    from pyvis.network import Network
    net = Network(height="600px", width="100%", bgcolor="#0e1117",
                  font_color="#e6e6e6", cdn_resources="in_line")
    center = cfg.COMPANY["name"]
    net.add_node(center, label=center, color=ACCENT, size=42, shape="dot")
    comp_names = {c["name"] for c in entities.get("competitors", [])}

    nodes = {center}
    for o in entities.get("all_orgs", [])[:22]:
        is_comp = o["name"] in comp_names
        net.add_node(o["name"], label=o["name"],
                     color="#e8a838" if is_comp else "#5a6070",
                     size=(16 if is_comp else 10) + min(o["mentions"], 26),
                     title=f"{o['mentions']} mentions" + (" (competitor)" if is_comp else ""))
        nodes.add(o["name"])

    # inter-entity edges from co-occurrence (the actual network structure)
    linked = set()
    for e in entities.get("edges", []):
        a, b = e["source"], e["target"]
        if a in nodes and b in nodes:
            net.add_edge(a, b, color="#33373f", value=e.get("weight", 1))
            linked.add(a); linked.add(b)

    # competitors always tie to the company; orphans tie to center so nothing floats
    for n in nodes:
        if n == center:
            continue
        if n in comp_names:
            net.add_edge(center, n, color="#e8a838")
        elif n not in linked:
            net.add_edge(center, n, color="#262a31")

    net.set_options('{"physics":{"barnesHut":{"gravitationalConstant":-12000,'
                    '"springLength":140,"springConstant":0.04},"minVelocity":0.4}}')
    try:
        return net.generate_html()
    except Exception:
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "kg.html")
        net.save_graph(tmp)
        with open(tmp, encoding="utf-8") as f:
            return f.read()


def section_graph():
    st.header("Source Trust & Decision Weighting")
    st.caption("How much should the CEO trust each source when acting on it? "
               "Internal/first-party sources are weighted highest; external/crowd sources "
               "are sentiment signals to corroborate, not act on directly.")

    counts = {}
    try:
        counts = db.counts_by_source()
    except Exception:
        pass

    rows = []
    for src, (tier, weight, desc) in cfg.SOURCE_TRUST.items():
        rows.append({"source": src, "tier": tier, "weight": weight,
                     "docs": counts.get(src, 0), "desc": desc})
    d = pd.DataFrame(rows)

    # 1) Decision-weight ranking (the "ideal weighting" view)
    dd = d.sort_values("weight")
    colors = ["#76B900" if t == "internal" else "#6aa9ff" for t in dd["tier"]]
    fig = go.Figure(go.Bar(
        x=dd["weight"], y=dd["source"], orientation="h", marker_color=colors,
        text=[f"{w:.2f}  ·  {n} docs" for w, n in zip(dd["weight"], dd["docs"])],
        textposition="auto",
        customdata=dd["desc"], hovertemplate="%{y}: %{customdata}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="decision weight (how much to trust)")
    st.plotly_chart(fig, width='stretch')
    st.markdown("<span style='color:#76B900'>■</span> internal / first-party  "
                "&nbsp;&nbsp; <span style='color:#6aa9ff'>■</span> external / third-party",
                unsafe_allow_html=True)

    # 2) The decision strategy: volume vs trust quadrant
    st.subheader("Decision strategy — volume vs. trust")
    fig2 = go.Figure()
    for tier, col in [("internal", "#76B900"), ("external", "#6aa9ff")]:
        sub = d[d["tier"] == tier]
        fig2.add_trace(go.Scatter(
            x=sub["docs"], y=sub["weight"], mode="markers+text", text=sub["source"],
            textposition="top center", name=tier,
            marker=dict(size=[12 + min(n, 40) for n in sub["docs"]], color=col, opacity=0.8)))
    if len(d):
        fig2.add_hline(y=0.65, line_dash="dot", line_color="#555")
    fig2.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_title="volume (documents collected)",
                       yaxis_title="trust weight")
    st.plotly_chart(fig2, width='stretch')
    st.caption("Top band (high trust): act on directly — filings, official materials. "
               "Bottom band (high volume, low trust): treat as sentiment; require corroboration "
               "across sources before acting. This mirrors how findings are scored: "
               "confidence = entailment + cross-source corroboration + authority-aware freshness.")

    # 3) Entity relationship graph (kept as a secondary view)
    e = load_json(ENT_PATH)
    if e and (e.get("competitors") or e.get("all_orgs")):
        with st.expander("Organization relationship graph (co-mentions)"):
            try:
                import streamlit.components.v1 as components
                components.html(build_graph_html(e), height=560, scrolling=False)
            except Exception as ex:
                st.warning(f"Graph renderer unavailable ({ex}).")


def section_ask():
    st.header("Ask the CEO Agent")
    st.caption("An autonomous agent: it decides which tools to call (search, stock, sentiment, "
               "competitors), observes the results, and loops until it can answer — then cites its "
               "evidence. (Requires Ollama running.)")
    q = st.text_input("Question", placeholder="e.g. How exposed is NVIDIA to China export controls?")
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Agent is planning, calling tools, and reasoning…"):
            try:
                import importlib
                from src import agent as _agent_mod
                importlib.reload(_agent_mod)          # force the latest agent.py, bypassing stale cache
                res = _agent_mod.agent_answer(q.strip())
            except Exception as ex:
                import traceback
                st.error(f"Could not reach the agent: {ex}\n\nMake sure Ollama is running "
                         "(`ollama list`) and the index is built (`python -m src.repository`).")
                with st.expander("Error details (traceback)"):
                    st.code(traceback.format_exc())
                return
        st.markdown(res.get("answer", "_(no answer)_"))
        steps = res.get("reasoning", [])
        if steps:
            with st.expander(f"Agent reasoning ({len(steps)} steps)"):
                for t in steps:
                    act = t.get("action") or ("skipped" if t.get("error") else "thinking")
                    th = t.get("thought", "")
                    st.markdown(f"**Step {t.get('step')}** · `{act}`"
                                + (f" — {th}" if th else ""))
        _citations_block(res.get("citations", []))


def section_agent():
    st.header("Agent Reasoning")
    st.caption("How the agent investigated — it planned the focus areas itself, called tools, and "
               "rewrote its own searches when the evidence was weak or unverified (self-correction).")
    intel = _load_intel_or_stop()
    if not intel:
        return
    ag = intel.get("agent")
    if not ag:
        st.info("This intelligence run was produced by the non-agent pipeline. "
                "Regenerate with `python -m src.agent_graph` to record the agent's plan and reasoning.")
        return

    # Framework + the required workflow as a visible chain.
    fw = ag.get("framework", "agent")
    flow = ag.get("workflow") or ["Goal", "Plan", "Retrieve", "Analyze", "Decide", "Recommend", "Validate"]
    st.markdown(f'<span class="badge b-ok">⚙ {fw}</span>', unsafe_allow_html=True)
    st.markdown("**Agent workflow:** " + "  ➜  ".join(f"`{s}`" for s in flow))

    val = ag.get("validation", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plan steps", len(ag.get("plan", [])))
    c2.metric("Self-corrections", ag.get("total_retries", 0),
              help="Times the agent rewrote its own query after weak/unverified evidence")
    c3.metric("Recommendations", val.get("recommendations", len(intel.get("recommendations", []))))
    c4.metric("Validated", val.get("verified", val.get("well_supported", "—")),
              help="NLI-verified: the recommendation's claim is entailed by its evidence")

    st.markdown(f"**Objective:** {ag.get('objective','')}")

    st.subheader("Investigation plan (decided by the agent)")
    for i, p in enumerate(ag.get("plan", []), 1):
        st.markdown(f"{i}. **{p.get('focus','')}**  ·  _{p.get('lens','')}_  ·  query: `{p.get('query','')}`")

    st.subheader("Execution trace (act → observe → reflect → retry)")
    for s in ag.get("steps", []):
        rt = s.get("retries", 0)
        badge = (f'<span class="badge b-md">↻ {rt} self-correction{"s" if rt != 1 else ""}</span>'
                 if rt else '<span class="badge b-ok">✓ first try</span>')
        st.markdown(f"<div class='card'><b>{s.get('focus','')}</b> "
                    f"<span class='badge b-na'>{s.get('lens','')}</span>{badge}</div>",
                    unsafe_allow_html=True)
        for a in s.get("attempts", []):
            line = (f"&nbsp;&nbsp;**Attempt {a.get('attempt')}** — searched `{a.get('query','')[:70]}` "
                    f"→ {a.get('evidence_chunks',0)} chunks, {a.get('findings',0)} findings "
                    f"({a.get('weak_findings',0)} weak) → _{a.get('decision','')}_")
            st.markdown(line, unsafe_allow_html=True)
            if a.get("refined_query"):
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ agent rewrote query: `{a['refined_query'][:70]}`")

    st.subheader("Tools the agent can call")
    for t in ag.get("tools", []):
        st.markdown(f"- **{t.get('name')}** — {t.get('description','')}")


# ======================= shell =======================
SECTIONS = {
    # Required dashboard pages from the PDF, in the exact order.
    "Company Overview": section_overview,
    "Market Intelligence": section_stock,
    "Opportunity Monitor": section_opportunities,
    "Risk Monitor": section_risks,
    "Sentiment Analysis": section_sentiment,
    "Strategic Recommendations": section_recommendations,
    "CEO Briefing": section_briefing,

    # Extra pages after the required 7.
    "Trend Monitor": section_trends,
    "Competitive Landscape": section_competitors,
    "Source Trust": section_graph,
    "Agent Reasoning": section_agent,
    "Ask the Agent": section_ask,
}

if __name__ == "__main__":
    with st.sidebar:
        if os.path.exists(_LOGO):
            st.image(_LOGO, width=150)
        st.markdown(f"### 🧠 AI CEO\n**{cfg.COMPANY['name']}** · {cfg.COMPANY['ticker']}")
        choice = st.radio("Section", list(SECTIONS), label_visibility="collapsed")
        st.divider()
        ov, _ = load_overview()
        if ov:
            st.caption(f"{ov['total_documents']:,} documents · {ov['sources']} sources")
        st.caption(f"🔄 Last refresh: {_last_refresh()}")
        st.caption("Artifacts are pre-computed; sections load from cache.")

    # Last-refresh banner at the top of every page.
    st.caption(f"🔄 Last refresh: {_last_refresh()}  ·  {cfg.COMPANY['name']} ({cfg.COMPANY['ticker']})")

    SECTIONS[choice]()