# ============================================================
#  src/collect.py  —  Data ingestion layer
#
#  Direct publisher feeds + NVIDIA first-party + HN + arXiv +
#  Yahoo + Wikipedia + stock + SEC filings + official CDN PDFs.
#  Stores FULL content (no chunking — that's the preprocess step)
#  and loads everything into SQLite.
# ============================================================

import os
import json
import time
import urllib.parse

import requests
import feedparser
import yfinance as yf

from . import utils
from . import sec_filings
from . import pdf_ingest
from . import database
import config as cfg


# ---- Source 1: direct publisher news feeds (full-text friendly) ----
def collect_news():
    docs = []
    print("  [1/7] Direct publisher feeds (Yahoo, BBC, NDTV/Gadgets360, CNBC, MarketWatch, Register, VentureBeat, Engadget, Verge, Ars, TechCrunch, Tom's HW) ...")
    t = cfg.COMPANY["ticker"]
    feeds = {
        "yahoo_nvda":    f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US",
        "bbc_tech":      "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "ndtv_top":      "http://feeds.feedburner.com/ndtvnews-top-stories",
        "ndtv_gadgets":  "https://www.gadgets360.com/rss/news",
        "cnbc_tech":     "https://www.cnbc.com/id/19854910/device/rss/rss.html",
        "marketwatch":   "http://feeds.marketwatch.com/marketwatch/topstories/",
        "theregister":   "https://www.theregister.com/headlines.atom",
        "venturebeat":   "https://venturebeat.com/feed/",
        "engadget":      "https://www.engadget.com/rss.xml",
        "verge":         "https://www.theverge.com/rss/index.xml",
        "arstechnica":   "http://feeds.arstechnica.com/arstechnica/index",
        "techcrunch":    "https://techcrunch.com/feed/",
        "tomshardware":  "https://www.tomshardware.com/feeds/all",
        "servethehome":  "https://www.servethehome.com/feed/",          # data-center HW / silicon supply chain
        "semianalysis":  "https://semianalysis.substack.com/feed",       # semi analysis (free-post teasers only)
    }
    keywords = [k.lower() for k in cfg.COMPANY["aliases"]] + ["ai", "gpu"]
    for sid, url in feeds.items():
        try:
            for e in feedparser.parse(url).entries[:cfg.NEWS_PER_FEED]:
                title = e.get("title", "")
                summary = e.get("summary", "") or e.get("description", "")
                if any(kw in (title + summary).lower() for kw in keywords):
                    docs.append(utils.make_doc(
                        title=title, body=summary, source="news",
                        url=e.get("link", ""), published=e.get("published", "")))
        except Exception as ex:
            print(f"        ! feed '{sid}' failed ({ex})")
    print(f"        -> {len(docs)} candidate news items (filtered to NVIDIA next)")
    return docs


# ---- Source 2: NVIDIA first-party RSS ----
def collect_company():
    docs = []
    print("  [2/7] NVIDIA press + blogs ...")
    for kind, url in cfg.COMPANY_FEEDS.items():
        try:
            for e in feedparser.parse(url).entries:
                docs.append(utils.make_doc(
                    e.get("title", ""), e.get("summary", ""),
                    source="company", url=e.get("link", ""),
                    published=e.get("published", ""), section=kind))
        except Exception:
            pass
    print(f"        -> {len(docs)} company items")
    return docs


# ---- Source 3: Hacker News ----
def collect_hackernews():
    docs = []
    print("  [3/7] Hacker News ...")
    url = (f"https://hn.algolia.com/api/v1/search?query={urllib.parse.quote(cfg.HN_QUERY)}"
           f"&tags=story&hitsPerPage={cfg.HN_HITS}")
    try:
        for h in requests.get(url, timeout=10).json().get("hits", []):
            docs.append(utils.make_doc(
                h.get("title", ""), h.get("story_text", "") or "",
                source="community",
                url=h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                published=h.get("created_at", "")))
    except Exception as e:
        print(f"        ! HN skipped ({e})")
    print(f"        -> {len(docs)} HN stories")
    return docs


# ---- Source 4: arXiv ----
def collect_arxiv():
    docs = []
    print("  [4/7] arXiv research (downloading PDFs + extracting text) ...")
    import io
    from pypdf import PdfReader
    os.makedirs("data/raw/arxiv_pdfs", exist_ok=True)
    search = urllib.parse.quote(cfg.ARXIV_QUERY)
    url = (f"http://export.arxiv.org/api/query?search_query={search}"
           f"&start=0&max_results={cfg.ARXIV_MAX}&sortBy=submittedDate&sortOrder=descending")
    try:
        entries = feedparser.parse(url).entries
    except Exception:
        entries = []
    for e in entries:
        title = " ".join(e.get("title", "").split())
        abstract = e.get("summary", "")
        # Keep only papers ABOUT NVIDIA (named in title/abstract), not papers
        # that merely ran on NVIDIA GPUs. Skips the irrelevant downloads too.
        if "nvidia" not in (title + " " + abstract).lower():
            continue
        # find the PDF link
        pdf_url = ""
        for l in e.get("links", []):
            if l.get("type") == "application/pdf" or l.get("title") == "pdf":
                pdf_url = l.get("href", "")
        if not pdf_url and e.get("id"):
            pdf_url = e["id"].replace("/abs/", "/pdf/")
        aid = (e.get("id", "").rstrip("/").split("/")[-1] or "paper").replace(":", "_")

        full = abstract
        cached = os.path.join("data/raw/arxiv_pdfs", f"{aid}.pdf")
        try:
            if os.path.exists(cached):                       # reuse cached PDF (fast reruns)
                with open(cached, "rb") as fh:
                    content = fh.read()
            else:
                content = requests.get(pdf_url, timeout=cfg.FETCH_TIMEOUT,
                                       headers={"User-Agent": "Mozilla/5.0 (ai-ceo-research)"}).content
                if content[:4] == b"%PDF":
                    with open(cached, "wb") as fh:
                        fh.write(content)
            if content[:4] == b"%PDF":
                text = " ".join(utils.clean_text(p.extract_text() or "")
                                for p in PdfReader(io.BytesIO(content)).pages)
                if len(text) > len(abstract):
                    full = f"{title}. {text}"
        except Exception:
            pass
        docs.append(utils.make_doc(title, full, source="research",
                    url=e.get("link", ""), published=e.get("published", "")))
        time.sleep(0.2)                     # be polite to arXiv
    print(f"        -> {len(docs)} papers (full PDF text where parseable)")
    return docs


# ---- Source 5: Yahoo Finance news ----
def collect_market():
    docs = []
    print("  [5/7] Yahoo Finance news ...")
    try:
        for n in yf.Ticker(cfg.COMPANY["ticker"]).news:
            c = n.get("content", n) if isinstance(n, dict) else {}
            title = c.get("title", "") if isinstance(c, dict) else ""
            summ  = (c.get("summary", "") or c.get("description", "")) if isinstance(c, dict) else ""
            link  = (c.get("canonicalUrl", {}) or {}).get("url", "") or c.get("link", "") if isinstance(c, dict) else ""
            if title:
                docs.append(utils.make_doc(title, summ, source="market", url=link))
    except Exception as e:
        print(f"        ! yfinance news skipped ({e})")
    print(f"        -> {len(docs)} market items")
    return docs


# ---- Source 6: Wikipedia (full) ----
def collect_reference():
    docs = []
    print("  [6/7] Wikipedia ...")
    url = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
           "&explaintext=1&format=json&redirects=1&titles=Nvidia")
    try:
        pages = requests.get(url, timeout=15,
                             headers={"User-Agent": cfg.SEC_USER_AGENT}).json()["query"]["pages"]
        for p in pages.values():
            extract = p.get("extract", "")
            if len(extract) > 100:
                docs.append(utils.make_doc(p.get("title", "Nvidia"), extract,
                            source="reference", url="https://en.wikipedia.org/wiki/Nvidia"))
    except Exception as e:
        print(f"        ! Wikipedia skipped ({e})")
    print(f"        -> {len(docs)} reference docs")
    return docs


# ---- Source 7: stock prices + technical indicators ----
def collect_stock():
    print("  [7/7] Stock prices + technical analysis ...")
    docs = []
    try:
        hist = yf.Ticker(cfg.COMPANY["ticker"]).history(period=cfg.STOCK_PERIOD)
        if hist.empty:
            print("        ! no price history"); return docs
        series = [{"date": i.strftime("%Y-%m-%d"), "close": round(float(r["Close"]), 2),
                   "volume": int(r["Volume"])} for i, r in hist.iterrows()]
        stock = {"ticker": cfg.COMPANY["ticker"], "period": cfg.STOCK_PERIOD, "history": series}

        # Technical analysis computed from real prices (no scraping).
        try:
            c = hist["Close"]; delta = c.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rsi = (100 - 100 / (1 + gain / loss)).iloc[-1]
            latest = float(c.iloc[-1]); sma50 = c.rolling(50).mean().iloc[-1]
            sma200 = c.rolling(200).mean().iloc[-1]
            hi52 = float(c.tail(252).max()); lo52 = float(c.tail(252).min())
            stock["indicators"] = {
                "latest_close": round(latest, 2),
                "sma_50": round(float(sma50), 2) if sma50 == sma50 else None,
                "sma_200": round(float(sma200), 2) if sma200 == sma200 else None,
                "rsi_14": round(float(rsi), 1) if rsi == rsi else None,
                "high_52w": round(hi52, 2), "low_52w": round(lo52, 2),
                "pct_from_52w_high": round((latest / hi52 - 1) * 100, 1),
                "trend": ("above 200-day SMA (bullish)" if sma200 == sma200 and latest > sma200
                          else "below 200-day SMA (bearish)")}
        except Exception as e:
            print(f"        ! indicators skipped ({e})"); stock["indicators"] = {}

        os.makedirs("data/clean", exist_ok=True)
        with open("data/clean/stock.json", "w", encoding="utf-8") as f:
            json.dump(stock, f, indent=2)

        pct = hist["Close"].pct_change() * 100
        for i in pct.abs().sort_values(ascending=False).head(8).index:
            ch = pct.loc[i]; d = i.strftime("%Y-%m-%d"); dirn = "rose" if ch > 0 else "fell"
            docs.append(utils.make_doc(
                f"NVDA {dirn} {abs(ch):.1f}% on {d}",
                f"On {d}, NVIDIA (NVDA) stock {dirn} {abs(ch):.1f}% to close at ${hist['Close'].loc[i]:.2f}.",
                source="market", url=f"https://finance.yahoo.com/quote/{cfg.COMPANY['ticker']}", published=d))
        print(f"        -> {len(series)} trading days, {len(docs)} notable-move notes")
    except Exception as e:
        print(f"        ! stock skipped ({e})")
    return docs


def build_profile():
    try:
        y = yf.Ticker(cfg.COMPANY["ticker"]).info
        info = {"name": y.get("shortName", cfg.COMPANY["name"]),
                "industry": y.get("industry", cfg.COMPANY["industry"]),
                "summary": y.get("longBusinessSummary", ""),
                "financials": sec_filings.financials()}
        with open(cfg.PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
    except Exception:
        pass


# ============================================================
#  Full-text enrichment (clean extraction, no chunking)
# ============================================================
def fetch_full_text(url):
    if not url:
        return ""
    if "arxiv.org/abs/" in url:                     # full paper HTML instead of abstract
        url = url.replace("arxiv.org/abs/", "arxiv.org/html/")
    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
    try:
        r = requests.get(url, headers=headers, timeout=cfg.FETCH_TIMEOUT, allow_redirects=True)
        if r.status_code != 200:
            return ""
        import trafilatura
        text = trafilatura.extract(r.text, favor_precision=True,        # main article only
                                   include_comments=False, include_tables=False)
        return text if (text and len(text) > 400) else ""   # else keep clean summary
    except Exception:
        return ""


def enrich_full_text(docs):
    if not cfg.FETCH_FULL_TEXT:
        return docs
    from concurrent.futures import ThreadPoolExecutor
    targets = [d for d in docs if d["source"] in cfg.ENRICH_SOURCES and d.get("url")]
    print(f"  [ENRICH] fetching full text for {len(targets)} docs "
          f"({cfg.ENRICH_WORKERS} parallel) ...")

    def _one(d):
        full = fetch_full_text(d["url"])
        if len(full) > len(d["text"]):
            d["text"] = f"{d['title']}. {full}".strip(". ")
            return 1
        return 0

    with ThreadPoolExecutor(max_workers=cfg.ENRICH_WORKERS) as ex:
        ok = sum(ex.map(_one, targets))
    print(f"        -> enriched {ok}/{len(targets)} (rest kept clean summaries)")
    return docs


# ============================================================
#  Official NVIDIA investor PDFs (verified CDN URLs)
# ============================================================
def download_investor_pdfs():
    print("  [PDF SYNC] downloading official NVIDIA CDN PDFs ...")
    os.makedirs(cfg.INVESTOR_PDF_DIR, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    base = "https://s201.q4cdn.com/141608511/files/doc_financials"
    targets = [
        # ---- FY27 ----
        f"{base}/2027/Q127/NVDA-F1Q27-Quarterly-Presentation-FINAL.pdf",
        # ---- FY26 (quarterly presentations) ----
        f"{base}/2026/Q426/NVDA-F4Q26-Quarterly-Presentation.pdf",
        f"{base}/2026/q3/NVDA-F3Q26-Quarterly-Presentation.pdf",
        f"{base}/2026/q2/NVDA-F2Q26-Quarterly-Presentation-FINAL.pdf",
        f"{base}/2026/q1/NVDA-F1Q26-Quarterly-Presentation-FINAL.pdf",
        # ---- FY26 (CFO commentary + revenue trend + annual report) ----
        f"{base}/2026/Q426/Q4FY26-CFO-Commentary.pdf",
        f"{base}/2026/Q326/Q3FY26-CFO-Commentary.pdf",
        f"{base}/2026/Q226/Q2FY26-CFO-Commentary.pdf",
        f"{base}/2026/Q326/Rev_by_Mkt_Qtrly_Trend_Q326.pdf",
        f"{base}/2026/ar/2026-Annual-Report-Web.pdf",
        # ---- FY25 (quarterly presentations) ----
        f"{base}/2025/q4/NVDA-F4Q25-Quarterly-Presentation-FINAL.pdf",
        f"{base}/2025/q3/NVDA-F3Q25-Quarterly-Presentation-FINAL.pdf",
        f"{base}/2025/q2/NVDA-F2Q25-Quarterly-Presentation-FINAL.pdf",
        f"{base}/2025/q1/NVDA-F1Q25-Quarterly-Presentation-FINAL.pdf",
        # ---- FY25 (CFO commentary) ----
        f"{base}/2025/Q425/Q4FY25-CFO-Commentary.pdf",
        f"{base}/2025/Q325/Q3FY25-CFO-Commentary.pdf",
        f"{base}/2025/Q225/Q2FY25-CFO-Commentary.pdf",
        f"{base}/2025/Q1FY25-CFO-Commentary.pdf",
    ]
    got = 0
    for url in targets:
        try:
            name = "_".join(url.split("/")[-2:])
            dest = os.path.join(cfg.INVESTOR_PDF_DIR, name)
            if os.path.exists(dest):
                got += 1; continue
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code == 200 and res.content[:4] == b"%PDF":
                with open(dest, "wb") as f:
                    f.write(res.content)
                print(f"        + {name}"); got += 1
            else:
                print(f"        - skip {name} (HTTP {res.status_code})")
        except Exception as e:
            print(f"        - skip ({e})")
    print(f"        -> {got} official PDFs in {cfg.INVESTOR_PDF_DIR}")


# ============================================================
#  Orchestration
# ============================================================
def collect_reddit():
    """Consumer / retail sentiment from Reddit (PRAW, read-only OAuth, free tier).
    Searches each subreddit for company-relevant posts. Fails soft: if creds or
    praw are missing, it skips without breaking the pipeline."""
    if not (cfg.REDDIT_CLIENT_ID and cfg.REDDIT_CLIENT_SECRET):
        print("  [reddit] skipped - set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET to enable")
        return []
    try:
        import praw
    except Exception:
        print("  [reddit] skipped - praw not installed (pip install praw)")
        return []

    from datetime import datetime, timezone
    print("  [reddit] consumer/retail sentiment across subreddits ...")
    docs = []
    try:
        reddit = praw.Reddit(client_id=cfg.REDDIT_CLIENT_ID,
                             client_secret=cfg.REDDIT_CLIENT_SECRET,
                             user_agent=cfg.REDDIT_USER_AGENT)
        reddit.read_only = True
        query = " OR ".join(cfg.COMPANY["aliases"][:4])      # NVIDIA OR NVDA OR GeForce OR CUDA
        for sub in cfg.REDDIT_SUBREDDITS:
            try:
                for p in reddit.subreddit(sub).search(query, sort="new", time_filter="month",
                                                       limit=cfg.REDDIT_POSTS_PER_SUB):
                    body = f"{p.title or ''}. {p.selftext or ''}".strip()
                    pub = datetime.fromtimestamp(getattr(p, "created_utc", 0),
                                                 tz=timezone.utc).strftime("%Y-%m-%d")
                    docs.append(utils.make_doc(
                        title=p.title or "", body=body, source="social",
                        url=f"https://www.reddit.com{p.permalink}", published=pub, section=f"r/{sub}"))
            except Exception as ex:
                print(f"        ! r/{sub} failed ({ex})")
        print(f"        -> {len(docs)} reddit posts (consumer sentiment)")
    except Exception as ex:
        print(f"  [reddit] skipped ({ex})")
        return []
    return docs


def collect_reddit_rss():
    """Keyless Reddit via public RSS (no app key/secret). Uses a personal feed token
    (REDDIT_FEED_USER/REDDIT_FEED_TOKEN) to avoid the 429 rate-limiting that now hits
    unauthenticated RSS. Titles + summaries only (no comments). Fails soft."""
    import re
    auth = ""
    if cfg.REDDIT_FEED_USER and cfg.REDDIT_FEED_TOKEN:
        auth = f"&user={cfg.REDDIT_FEED_USER}&feed={cfg.REDDIT_FEED_TOKEN}"
    else:
        print("  [reddit-rss] no feed token (REDDIT_FEED_USER/REDDIT_FEED_TOKEN) - "
              "trying plain RSS, may be rate-limited (429)")
    query = "+OR+".join(cfg.COMPANY["aliases"][:4])          # NVIDIA+OR+NVDA+OR+GeForce+OR+CUDA
    headers = {"User-Agent": "Mozilla/5.0 (ai-ceo-research RSS reader)"}
    print("  [reddit-rss] consumer/retail sentiment via RSS ...")
    docs = []
    for sub in cfg.REDDIT_SUBREDDITS:
        url = (f"https://www.reddit.com/r/{sub}/search.rss?q={query}"
               f"&restrict_sr=1&sort=new&limit={cfg.REDDIT_POSTS_PER_SUB}{auth}")
        try:
            r = requests.get(url, headers=headers, timeout=cfg.FETCH_TIMEOUT)
            if r.status_code != 200:
                print(f"        ! r/{sub} -> HTTP {r.status_code}")
                time.sleep(1.5)
                continue
            for e in feedparser.parse(r.content).entries:
                title = e.get("title", "")
                summary = re.sub(r"<[^>]+>", " ", e.get("summary", ""))   # strip HTML tags
                body = f"{title}. {utils.clean_text(summary)}".strip(". ")
                docs.append(utils.make_doc(
                    title=title, body=body, source="social",
                    url=e.get("link", ""), published=e.get("published", ""), section=f"r/{sub}"))
            time.sleep(1.2)                                   # be gentle -> avoid 429
        except Exception as ex:
            print(f"        ! r/{sub} failed ({ex})")
    print(f"        -> {len(docs)} reddit posts (RSS)")
    return docs


def collect_ecosystem():
    """Snapshot the open-source GPU/AI ecosystem from the GitHub REST API.
    Tracks NVIDIA's own moat libraries and CUDA-competitive stacks as a
    structural Trends / competitive-risk signal. Structured JSON, no parsing."""
    print("  [eco] GitHub open-source ecosystem (stars, issue/PR velocity, last push) ...")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "ai-ceo-research"}
    if cfg.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {cfg.GITHUB_TOKEN}"
    docs = []
    for repo in cfg.GITHUB_REPOS:
        try:
            r = requests.get(f"https://api.github.com/repos/{repo}",
                             headers=headers, timeout=cfg.FETCH_TIMEOUT)
            if r.status_code != 200:
                print(f"        ! {repo} -> HTTP {r.status_code}")
                continue
            d = r.json()
            text = (f"GitHub repository {repo}. {d.get('description') or ''} "
                    f"Stars: {d.get('stargazers_count', 0):,}. "
                    f"Open issues and pull requests: {d.get('open_issues_count', 0):,}. "
                    f"Forks: {d.get('forks_count', 0):,}. "
                    f"Primary language: {d.get('language', 'n/a')}. "
                    f"Last code push: {d.get('pushed_at', '')}. "
                    f"Topics: {', '.join(d.get('topics', [])) or 'n/a'}.")
            docs.append(utils.make_doc(
                title=f"[ecosystem] {repo}", body=text, source="ecosystem",
                url=d.get("html_url", ""), published=d.get("pushed_at", ""), section="github"))
            time.sleep(0.4)                       # comfortably under the unauth core limit
        except Exception as ex:
            print(f"        ! {repo} failed ({ex})")
    print(f"        -> tracked {len(docs)}/{len(cfg.GITHUB_REPOS)} repos")
    return docs


def run(quick=False):
    print("=" * 60)
    print(f"  COLLECTING {'(QUICK refresh)' if quick else '(FULL)'} : {cfg.COMPANY['name']}")
    print("=" * 60)

    raw = []
    if not quick:
        download_investor_pdfs()                   # cached after first run

    # Fast, live-updating sources (always)
    raw += collect_news()
    raw += collect_company()
    raw += collect_market()
    raw += collect_stock()

    # Heavier / slow-changing sources (full runs only)
    if not quick:
        raw += collect_hackernews()
        raw += collect_reddit() or collect_reddit_rss()     # PRAW if keys set, else keyless RSS
        raw += collect_arxiv()
        raw += collect_ecosystem()
        raw += collect_reference()
        raw += sec_filings.collect_filings()
        raw += pdf_ingest.collect_pdfs()

    os.makedirs(os.path.dirname(cfg.RAW_PATH), exist_ok=True)
    with open(cfg.RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

    clean = utils.deduplicate(raw, cfg.MIN_TEXT_LEN)
    before = len(clean)
    clean = utils.filter_relevant(clean, cfg.COMPANY["aliases"], cfg.FILTER_SOURCES)
    dropped = before - len(clean)
    clean = utils.apply_caps(clean, cfg.SOURCE_CAPS)
    clean = enrich_full_text(clean)

    # Accumulate into the database, then export the FULL corpus for downstream.
    database.append_documents(clean)               # append-only (dedup by uid)
    database.export_documents_json()               # CLEAN_PATH = full accumulated corpus
    database.build_raw_table()
    database.build_stock_tables()
    build_profile()

    from collections import Counter
    print("-" * 60)
    print(f"  This run collected : {len(clean)} docs ({dropped} off-topic dropped)")
    print(f"  By source (run)    : {dict(Counter(d['source'] for d in clean))}")
    return clean


if __name__ == "__main__":
    import sys
    run(quick="--quick" in sys.argv)