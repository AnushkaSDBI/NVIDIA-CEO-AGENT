# ============================================================
#  config.py  —  Single source of truth for the whole project
#  Change the COMPANY block + CIK to retarget at another firm.
# ============================================================
import os

# Load a local .env file if present (so env vars don't have to be set by hand).
# Falls back silently to plain shell environment variables if python-dotenv isn't installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
try:
    from dotenv import load_dotenv      # auto-load a local .env if python-dotenv is installed
    load_dotenv()
except Exception:
    pass                                 # falls back to system / shell env vars

# --- 1. Target company --------------------------------------
COMPANY = {
    "name":    "NVIDIA",
    "ticker":  "NVDA",
    # High-precision NVIDIA terms. A doc from an open-web source is kept
    # ONLY if it mentions one of these (see FILTER_SOURCES below).
    "aliases": ["NVIDIA", "NVDA", "GeForce", "CUDA", "Jensen Huang", "RTX"],
    "industry": "Semiconductors / AI hardware",
    # More NVIDIA-specific topics = more on-target volume + wider timespan.
    "topics": [
        "NVIDIA",
        "NVIDIA AI chips",
        "NVIDIA earnings",
        "NVIDIA competitors",
        "NVIDIA data center revenue",
        "NVIDIA Blackwell GPU",
        "NVIDIA China export",
        "NVIDIA CUDA software",
        "Jensen Huang NVIDIA",
        "NVIDIA stock analysis",
    ],
}

# --- 2. Models ----------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SENTIMENT_MODEL = "ProsusAI/finbert"
# Sarcasm / irony handling for social (Reddit) + community text
IRONY_MODEL     = "cardiffnlp/twitter-roberta-base-irony"   # SemEval irony classifier
IRONY_THRESHOLD = 0.5            # irony probability above which a post is flagged
SARCASM_WEIGHT  = 0.3            # weight of a flagged post in the sentiment mean (down-weight, don't flip)

# Aspect-based sentiment: strategic themes + the keywords that signal them
ASPECTS = {
    "Data center / AI compute": ["data center", "datacenter", "blackwell", "hopper", "gb200",
                                  "h100", "h200", "gb300", "accelerated computing", "ai infrastructure"],
    "Gaming / GeForce":         ["geforce", "gaming", "rtx", "dlss", "gpu for gaming"],
    "China / export controls":  ["china", "export control", "export restriction", "h20", "sanction", "beijing"],
    "Valuation / stock":        ["valuation", "overvalued", "stock price", "market cap", "bubble",
                                 "sell-off", "selloff", "rally", "p/e"],
    "Competition":              ["amd", "intel", "custom silicon", "asic", "tpu", "mi300", "mi325", "competitor"],
}
# ---- LLM backend ----
# "ollama"       -> local Ollama server (your laptop)
# "transformers" -> in-process HuggingFace model on GPU (university data lab, no Ollama)
LLM_BACKEND   = os.getenv("LLM_BACKEND", "ollama")
LLM_MODEL     = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")  # HF id for transformers backend
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 1024
OLLAMA_MODEL    = "qwen2.5:7b"

# --- 3. First-party (company) RSS sources -------------------
COMPANY_FEEDS = {
    "press":    "https://nvidianews.nvidia.com/rss.xml",
    "blog":     "https://blogs.nvidia.com/feed/",
    "dev_blog": "https://developer.nvidia.com/blog/feed",
}

# --- 4. SEC EDGAR (annual report DEEP route) ----------------
CIK = "0001045810"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "AI-CEO-Project yourname@university.edu")   # SET env var or SEC returns 403
SEC_YEARS = 2                  # pull all 10-K (annual) + 10-Q (quarterly) from last N years
SEC_FORMS = ("10-K", "10-Q")
STOCK_PERIOD = "2y"            # yfinance price-history window

# Known industry rivals — used to label extracted ORG entities as competitors.
COMPETITORS = [
    "amd", "intel", "broadcom", "qualcomm", "arm", "tsmc", "micron",
    "samsung", "google", "microsoft", "amazon", "meta", "apple",
    "huawei", "cerebras", "groq", "graphcore", "marvell", "supermicro",
]

# Curated organization universe — real companies only, each tagged by its role.
# Entities NER finds are matched against this so the dashboard shows actual
# organizations (not "revenue", "gpu", "professional visualization", etc.).
ORG_TYPES = {
    "competitor": ["amd", "intel", "qualcomm", "broadcom", "arm", "huawei", "cerebras",
                   "groq", "graphcore", "tenstorrent", "sambanova", "marvell", "mediatek"],
    "customer": ["microsoft", "google", "alphabet", "amazon", "aws", "meta", "oracle",
                 "tesla", "openai", "anthropic", "tencent", "alibaba", "bytedance",
                 "dell", "supermicro", "hpe", "lenovo", "coreweave"],
    "supplier": ["tsmc", "samsung", "sk hynix", "micron", "asml", "foxconn", "amkor"],
}
# flat lookup: name -> role
ORG_ROLE = {name: role for role, names in ORG_TYPES.items() for name in names}

# Source trust weighting: how much should the CEO weight each source type when
# making decisions? "internal" = first-party / official; "external" = third-party.
# weight in [0,1] = decision trust. Drives the Source Trust view in the dashboard.
SOURCE_TRUST = {
    "filing":    ("internal", 1.00, "SEC filings — audited, legally binding"),
    "pdf":       ("internal", 0.85, "Investor decks & transcripts — official"),
    "company":   ("internal", 0.85, "NVIDIA press & blog — official but promotional"),
    "research":  ("external", 0.70, "arXiv — technical research signal"),
    "reference": ("external", 0.70, "Wikipedia — factual background"),
    "news":      ("external", 0.60, "Journalism — timely, varies by outlet"),
    "market":    ("external", 0.60, "Market/price data — factual, noisy"),
    "ecosystem": ("external", 0.50, "GitHub activity — indirect adoption signal"),
    "community": ("external", 0.40, "Hacker News — informed opinion"),
    "social":    ("external", 0.35, "Reddit — retail sentiment, noisy / sarcasm"),
}

# --- 5. Other sources ---------------------------------------
HN_QUERY = "NVIDIA"
HN_HITS  = 80
NEWS_PER_QUERY = 40
ARXIV_MAX      = 15                    # capped: arXiv PDF download is the slow step
NEWS_PER_FEED  = 60                    # more headlines per feed (wider timeframe)
ENRICH_WORKERS = 8                     # parallel full-text fetches (demo speed)
ARXIV_QUERY    = 'abs:NVIDIA'          # NVIDIA in the abstract = paper is about NVIDIA
USE_REDDIT     = False

# Bluesky returns a hard 403 to unauthenticated search (edge bot-block),
# so it is disabled. Hacker News covers the public/community signal.
USE_BLUESKY    = False

# --- Bluesky (public sentiment, no key, no auth) ------------
# searchPosts is public on this host; we avoid the `cursor` param
# (known to 403) and just run a few queries at limit<=100.
BLUESKY_QUERIES = ["NVIDIA", "NVDA", "Jensen Huang NVIDIA"]
BLUESKY_LIMIT   = 100

# --- PDF source (page-by-page) ------------------------------
PDF_DIR  = "data/pdfs"
# Verified official NVIDIA PDFs — downloaded + parsed automatically.
PDF_URLS = {
    "nvidia_q4_fy2026_results.pdf":
        "https://nvidianews.nvidia.com/_gallery/download_pdf/699f6ab43d6332ccaa689907/",
    "nvidia_q1_fy2026_results.pdf":
        "https://nvidianews.nvidia.com/_gallery/download_pdf/6837703d3d63320fddb3a9ee/",
    "nvidia_enterprise_reference_architecture.pdf":
        "https://docs.nvidia.com/enterprise-reference-architectures/white-paper.pdf",
}

# --- 6. Cleaning + chunking ---------------------------------
MIN_TEXT_LEN  = 60      # lowered from 120 so short HN/news items survive
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# --- 7. Relevance + per-source caps -------------------------
# Open-web sources are kept ONLY if they mention an alias above.
# First-party sources are NVIDIA by definition, so they are exempt.
FILTER_SOURCES = ["news", "community", "research", "market", "reference", "social"]
SOURCE_CAPS = {
    # open-web: capped for balance
    "news": 60, "community": 30, "research": 30, "market": 30, "reference": 20,
    "social": 60,                       # consumer/retail sentiment (Reddit) — a bit more headroom
    # first-party: keep (nearly) everything — these are your best evidence
    "company": 80, "filing": 300, "pdf": 300,
    # open-source ecosystem snapshot (small, structured) — keep all
    "ecosystem": 60,
}

# --- Reddit consumer / retail sentiment (PRAW, read-only OAuth; optional) ---
# Set env vars to enable:  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET  (+ optional REDDIT_USER_AGENT)
# Create a "script" app at https://www.reddit.com/prefs/apps  (no password needed for read-only).
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "ai-ceo-research/0.1")
REDDIT_SUBREDDITS    = ["nvidia", "NVDA_Stock", "wallstreetbets", "hardware", "buildapc"]
REDDIT_POSTS_PER_SUB = 25

# Keyless fallback: Reddit RSS with a personal feed token (no app needed).
# Get these from https://www.reddit.com/prefs/feeds/  (the feed= and user= values in the URLs).
# Plain unauthenticated RSS is now rate-limited (429); the token avoids that.
REDDIT_FEED_USER  = os.getenv("REDDIT_FEED_USER", "")
REDDIT_FEED_TOKEN = os.getenv("REDDIT_FEED_TOKEN", "")

# --- GitHub open-source ecosystem tracking (Trends + competitive signal) ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")     # optional; raises rate limit (10->30/min). repos endpoint is 60/hr unauth — plenty.
GITHUB_REPOS = [
    # NVIDIA's own moat / ecosystem libraries
    "NVIDIA/TensorRT", "NVIDIA/TensorRT-LLM", "NVIDIA/cutlass", "NVIDIA/cccl",
    # CUDA-competitive / lock-in-reducing stacks (early competitive signal)
    "ROCm/ROCm", "triton-lang/triton", "vllm-project/vllm", "ggml-org/llama.cpp",
]

# --- Full-text fetching (follow links, get the whole article) ---
FETCH_FULL_TEXT = True
ENRICH_SOURCES  = ["news", "market", "community", "company"]   # arrive as summaries
FETCH_TIMEOUT   = 15
MIN_DOCS = 100

# --- 8. Paths -----------------------------------------------
RAW_PATH   = "data/raw/documents.json"
CLEAN_PATH = "data/clean/documents.json"
CHUNKS_PATH = "data/clean/chunks.json"      # output of the preprocessing agent
PROFILE_PATH = "data/clean/profile.json"
INVESTOR_PDF_DIR = "data/raw/nvidia_investor_pdfs"   # official CDN PDFs land here
FAISS_PATH = "storage/faiss_index"
DB_PATH    = "storage/ai_ceo.db"      # SQLite (dashboard database endpoint)
TOP_K      = 6                        # final documents returned per search

# --- 9. Retrieval + reranking -------------------------------
RETRIEVE_K  = 20                      # hybrid over-fetch before reranking
RERANK_MODEL = "BAAI/bge-reranker-base"   # cross-encoder; v2-m3 is stronger/heavier
HYBRID_WEIGHT = 0.5                   # BM25 vs dense blend (0=all dense, 1=all BM25)
USE_MMR      = True                  # diversify retrieved chunks (MMR) to drop near-duplicates
MMR_LAMBDA   = 0.7                   # 1.0 = pure relevance, 0 = pure diversity
USE_RERANKER = True

# --- Multi-Query RAG (LLM query expansion) ------------------
MULTIQUERY_N = 3

# --- Intelligence layer (CEO agent + verification) ----------
NLI_MODEL     = "facebook/bart-large-mnli"   # entailment check for faithfulness
NLI_THRESHOLD = 0.35                         # entailment bar for "verified" (sentence-level); evidence still shown below it
CONTRADICTION_THRESHOLD = 0.5                # contradiction prob above which a finding is flagged "contested"
INTEL_PATH    = "data/clean/intelligence.json"