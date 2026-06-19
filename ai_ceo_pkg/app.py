# ============================================================
#  app.py  —  AI CEO: Strategic Intelligence Dashboard
#
#  7 sections, all reading PRE-COMPUTED artifacts (fast):
#    1. Executive Overview     (DB)
#    2. Market & Stock         (DB: stock_prices + indicators)
#    3. Sentiment              (sentiment.json)
#    4. Competitive Landscape  (entities.json)
#    5. CEO Intelligence       (intelligence.json)  <- the centerpiece
#    6. Knowledge Graph        (entities.json -> NetworkX/pyvis)
#    7. Ask the Agent          (live: intelligence.brief)
#
#  Run from the project root:   streamlit run app.py
#  (Ollama only needs to be running for section 7.)
# ============================================================

import os
import json

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
</style>
""", unsafe_allow_html=True)


# ---------------- cached loaders (fast, graceful) ----------------
@st.cache_data(show_spinner=False)
def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_overview():
    try:
        return db.overview(), db.counts_by_source()
    except Exception:
        return None, {}


@st.cache_data(show_spinner=False)
def load_stock():
    try:
        return db.stock_history(800), db.stock_indicators()
    except Exception:
        return [], {}


def _f(v, default=None):
    """Indicators are stored as strings; parse a float safely."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _badge_verified(verified, confidence):
    if verified is True:
        label = "✓ verified" + (f" {confidence:.0%}" if confidence is not None else "")
        return f'<span class="badge b-ok">{label}</span>'
    if verified is False:
        return '<span class="badge b-no">✗ unverified</span>'
    return '<span class="badge b-na">— not checked</span>'


def _badge_level(level):
    cls = {"high": "b-hi", "medium": "b-md", "low": "b-lo"}.get((level or "").lower(), "b-na")
    return f'<span class="badge {cls}">{(level or "n/a").lower()} impact/risk</span>'


def _citations_block(citations):
    if not citations:
        st.caption("No supporting citations passed verification.")
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            head = f"**{c.get('source','?')}** — {c.get('title','')}"
            if c.get("section"):
                head += f"  ·  _{c['section']}_"
            st.markdown(head)
            if c.get("snippet"):
                st.markdown(f"<div class='cite'>{c['snippet']}…</div>", unsafe_allow_html=True)
            if c.get("url"):
                st.markdown(f"[source link]({c['url']})")


# ======================= sections =======================
def section_overview():
    st.header("Executive Overview")
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

    st.subheader("Corpus by source")
    if counts:
        d = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1]), columns=["source", "documents"])
        fig = go.Figure(go.Bar(x=d["documents"], y=d["source"], orientation="h",
                               marker_color=ACCENT))
        fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')

    if intel and intel.get("generated_at"):
        st.caption(f"Intelligence generated: {intel['generated_at']}  ·  "
                   f"{ov['company']} ({cfg.COMPANY['ticker']}) · {ov['industry']}")

    kw = load_json("data/clean/keywords.json")
    if kw and kw.get("overall"):
        st.subheader("Key terms across the corpus (TF-IDF)")
        st.caption("Classical, deterministic signal — most distinctive terms, no LLM involved.")
        d = pd.DataFrame(kw["overall"][:15]).sort_values("weight")
        fig = go.Figure(go.Bar(x=d["weight"], y=d["term"], orientation="h", marker_color="#6aa9ff"))
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')


def section_stock():
    st.header("Market & Stock")
    hist, ind = load_stock()
    if not hist:
        st.info("No stock data. Run `python refresh.py`.")
        return
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    df["SMA50"] = df["close"].rolling(50).mean()
    df["SMA200"] = df["close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close", line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["SMA50"], name="SMA 50", line=dict(color="#e8a838", width=1)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["SMA200"], name="SMA 200", line=dict(color="#6aa9ff", width=1)))
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


def section_intelligence():
    st.header("CEO Intelligence")
    intel = load_json(cfg.INTEL_PATH)
    if not intel:
        st.info("No analysis yet. Run `python -m src.intelligence` (Ollama must be running).")
        return
    st.caption(f"Generated {intel.get('generated_at','')}  ·  every claim is grounded in retrieved "
               "evidence and checked by an NLI entailment model.")

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
        for r in intel.get("recommendations", []):
            st.markdown(
                f"<div class='card'><b>{r.get('action','')}</b> "
                f"{_badge_level(r.get('risk_level'))}{_badge_verified(r.get('verified'), r.get('confidence'))}"
                f"<br><b>Why:</b> {r.get('rationale','')}"
                f"<br><b>Expected impact:</b> {r.get('expected_impact','')}</div>",
                unsafe_allow_html=True)


def build_graph_html(entities):
    """NVIDIA at the center, competitors + most-mentioned orgs around it. Self-contained HTML."""
    from pyvis.network import Network
    net = Network(height="600px", width="100%", bgcolor="#0e1117",
                  font_color="#e6e6e6", cdn_resources="in_line")
    center = cfg.COMPANY["name"]
    net.add_node(center, label=center, color=ACCENT, size=42, shape="dot")
    comp_names = {c["name"] for c in entities.get("competitors", [])}
    for c in entities.get("competitors", []):
        net.add_node(c["name"], label=c["name"], color="#e8a838",
                     size=16 + min(c["mentions"], 30), title=f"{c['mentions']} mentions (competitor)")
        net.add_edge(center, c["name"], color="#e8a838")
    for o in entities.get("all_orgs", [])[:18]:
        if o["name"] in comp_names:
            continue
        net.add_node(o["name"], label=o["name"], color="#5a6070",
                     size=10 + min(o["mentions"], 22), title=f"{o['mentions']} mentions")
        net.add_edge(center, o["name"], color="#33373f")
    try:
        return net.generate_html()
    except Exception:
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "kg.html")
        net.save_graph(tmp)
        with open(tmp, encoding="utf-8") as f:
            return f.read()


def section_graph():
    st.header("Knowledge Graph")
    e = load_json(ENT_PATH)
    if not e or not (e.get("competitors") or e.get("all_orgs")):
        st.info("No entities yet. Run `python -m src.entities` (or `python refresh.py`).")
        return
    st.caption("Organizations co-mentioned with the company across the corpus. "
               "Green = company, amber = known competitors, grey = other orgs (size = mentions).")
    try:
        import streamlit.components.v1 as components
        components.html(build_graph_html(e), height=620, scrolling=False)
    except Exception as ex:
        st.warning(f"Graph renderer unavailable ({ex}). Install pyvis: `pip install pyvis networkx`.")


def section_ask():
    st.header("Ask the CEO Agent")
    st.caption("A live, evidence-grounded answer: retrieves from the indexed corpus, then the local "
               "LLM answers with inline citations. (Requires Ollama running.)")
    q = st.text_input("Question", placeholder="e.g. How exposed is NVIDIA to China export controls?")
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Retrieving evidence and reasoning…"):
            try:
                from src.intelligence import brief
                res = brief(q.strip())
            except Exception as ex:
                st.error(f"Could not reach the agent: {ex}\n\nMake sure Ollama is running "
                         "(`ollama list`) and the index is built (`python -m src.repository`).")
                return
        st.markdown(res.get("answer", "_(no answer)_"))
        _citations_block(res.get("citations", []))


# ======================= shell =======================
SECTIONS = {
    "Executive Overview": section_overview,
    "Market & Stock": section_stock,
    "Sentiment": section_sentiment,
    "Competitive Landscape": section_competitors,
    "CEO Intelligence": section_intelligence,
    "Knowledge Graph": section_graph,
    "Ask the Agent": section_ask,
}

if __name__ == "__main__":
    with st.sidebar:
        st.markdown(f"### 🧠 AI CEO\n**{cfg.COMPANY['name']}** · {cfg.COMPANY['ticker']}")
        choice = st.radio("Section", list(SECTIONS), label_visibility="collapsed")
        st.divider()
        ov, _ = load_overview()
        if ov:
            st.caption(f"{ov['total_documents']:,} documents · {ov['sources']} sources")
        intel = load_json(cfg.INTEL_PATH)
        if intel:
            st.caption(f"Analysis: {intel.get('generated_at','')[:16]}")
        st.caption("Artifacts are pre-computed; sections load from cache.")

    SECTIONS[choice]()