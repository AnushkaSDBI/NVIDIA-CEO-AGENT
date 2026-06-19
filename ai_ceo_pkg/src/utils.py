# ============================================================
#  src/utils.py  —  Shared helpers used by every collector
#  (one place for the document shape + cleaning + chunking)
# ============================================================

import re
import hashlib
from bs4 import BeautifulSoup


# --- 1. Cleaning --------------------------------------------
def clean_text(text):
    """Strip HTML tags and collapse whitespace into one clean string."""
    if not text:
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --- 2. Standard document shape -----------------------------
def make_doc(title, body, source, url, published="", section=""):
    """One record = one retrievable unit. Used by ALL sources."""
    title = clean_text(title)
    body  = clean_text(body)
    return {
        "title":     title,
        "text":      f"{title}. {body}".strip(". "),
        "source":    source,        # news / company / filing / community / research / market
        "section":   section,       # e.g. "Risk Factors (Item 1A)" for filings
        "url":       url,
        "published": published,
    }


# --- 3. Chunking (long filings -> several documents) --------
def chunk_text(text, size=1000, overlap=150):
    """Split a long string into overlapping character windows."""
    text = clean_text(text)
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap          # step back by overlap to keep context
    return [c for c in chunks if len(c) >= 200]


# --- 4. De-duplication --------------------------------------
def deduplicate(docs, min_len=120):
    """
    Drop junk/short docs and remove duplicates.
    IMPORTANT: hash the TEXT, not the title. Chunked sources (SEC
    filings, Wikipedia) share one title across many chunks, so
    hashing the title would collapse them all into a single doc.
    """
    seen, clean = set(), []
    for d in docs:
        if len(d["text"]) < min_len:
            continue
        key = hashlib.md5(d["text"].strip().lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        clean.append(d)
    return clean


# --- 5. Per-source caps (balance the index) -----------------
def apply_caps(docs, caps):
    """Keep at most caps[source] documents from each source category."""
    counts, kept = {}, []
    for d in docs:
        s = d["source"]
        if counts.get(s, 0) < caps.get(s, 9999):
            counts[s] = counts.get(s, 0) + 1
            kept.append(d)
    return kept


# --- 6. Company relevance filter ----------------------------
def mentions_company(text, aliases):
    """True if the text mentions any company alias (case-insensitive)."""
    low = text.lower()
    return any(a.lower() in low for a in aliases)


def filter_relevant(docs, aliases, filter_sources):
    """
    Drop open-web docs that never mention the company.
    First-party sources (company / filing / pdf) are NVIDIA by
    definition, so they are NOT in filter_sources and always kept.
    """
    kept = []
    for d in docs:
        if d["source"] in filter_sources and not mentions_company(d["text"], aliases):
            continue
        kept.append(d)
    return kept
