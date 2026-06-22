# ============================================================
#  src/sentiment.py  —  TASK 5 input (Sentiment Analysis)
#
#  Scores every collected document with FinBERT (finance-tuned)
#  and writes:
#    - a "sentiment" + "sentiment_score" field back onto each doc
#    - data/clean/sentiment.json  (aggregates for Dashboard Section 5)
#
#  News sentiment  = news + market sources
#  Public sentiment = social (Bluesky) + community (Hacker News)
#
#  Run (AFTER collect):  python -m src.sentiment
# ============================================================

import json
import re

import torch
from transformers import pipeline

import config as cfg

NEWS_SOURCES   = {"news", "market"}
PUBLIC_SOURCES = {"social", "community"}        # informal user text -> sarcasm-prone

# Reddit's explicit sarcasm/joke tags — the highest-precision signal there is
_TAG_RE = re.compile(r"(?:^|\s)/[sj]\b", re.IGNORECASE)


def _has_sarcasm_tag(text):
    return bool(_TAG_RE.search(text or ""))


# --- 1. Map FinBERT label -> signed compound score ----------
def _compound(label, score):
    label = label.lower()
    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0                       # neutral


# --- 2. Load the model (GPU if available) -------------------
def load_classifier():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=cfg.SENTIMENT_MODEL,
                    truncation=True, max_length=512, device=device)


# --- 3. Score every document --------------------------------
def score_docs(docs, clf, batch_size=16):
    texts = [d["text"][:2000] for d in docs]      # cap chars; FinBERT truncates tokens
    results = clf(texts, batch_size=batch_size)
    for d, r in zip(docs, results):
        d["sentiment"]       = r["label"].lower()
        d["sentiment_score"] = round(_compound(r["label"], r["score"]), 4)
    return docs


# --- 3b. Sarcasm / irony detection on social + community ----
def load_irony():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=cfg.IRONY_MODEL,
                    truncation=True, max_length=512, device=device)


def _irony_prob(scores):
    """scores = list of {label, score} for one post -> P(irony)."""
    for s in scores:
        lab = s["label"].lower()
        if lab in ("irony", "label_1", "1"):
            return s["score"]
    for s in scores:                              # only non-irony returned -> invert
        lab = s["label"].lower()
        if "non" in lab or lab in ("label_0", "0"):
            return 1.0 - s["score"]
    return 0.0


def detect_sarcasm(docs, batch_size=16):
    """Flag likely-sarcastic social/community posts. Rule layer (/s, /j tags) +
    an irony model. Fails soft: if the model can't load, the tag rule still runs."""
    for d in docs:
        d["irony_score"] = None
        d["sarcastic"] = False
    targets = [d for d in docs if d["source"] in PUBLIC_SOURCES]
    if not targets:
        return docs

    # Layer 1 — explicit Reddit tags (high precision, free)
    for d in targets:
        if _has_sarcasm_tag(d.get("text", "")):
            d["sarcastic"] = True
            d["irony_score"] = 1.0

    # Layer 2 — irony model
    try:
        iro = load_irony()
    except Exception as ex:
        print(f"  [irony] model unavailable ({ex}); using /s tag rule only")
        return docs
    preds = iro([d["text"][:512] for d in targets], batch_size=batch_size, top_k=None)
    flagged = 0
    for d, scores in zip(targets, preds):
        p = _irony_prob(scores if isinstance(scores, list) else [scores])
        d["irony_score"] = max(d["irony_score"] or 0.0, round(p, 4))
        if p >= cfg.IRONY_THRESHOLD:
            d["sarcastic"] = True
    flagged = sum(1 for d in targets if d["sarcastic"])
    print(f"  [irony] flagged {flagged}/{len(targets)} social/community posts as possibly sarcastic")
    return docs
def _agg(subset):
    if not subset:
        return {"count": 0, "positive": 0, "neutral": 0, "negative": 0, "mean": 0.0,
                "flagged_sarcastic": 0}
    c = {"count": len(subset), "positive": 0, "neutral": 0, "negative": 0, "flagged_sarcastic": 0}
    wsum = wtot = 0.0
    for d in subset:
        c[d["sentiment"]] = c.get(d["sentiment"], 0) + 1
        w = cfg.SARCASM_WEIGHT if d.get("sarcastic") else 1.0   # down-weight, don't flip
        wsum += d["sentiment_score"] * w
        wtot += w
        if d.get("sarcastic"):
            c["flagged_sarcastic"] += 1
    c["mean"] = round(wsum / wtot, 4) if wtot else 0.0
    return c


def aspect_sentiment(docs, clf, batch_size=16):
    """Aspect-based sentiment: for each strategic theme, score the SENTENCES that
    mention it (not whole docs) and aggregate. Closer to true ABSA than doc-level."""
    sent_split = re.compile(r"(?<=[.!?])\s+")
    out = {}
    for aspect, keywords in cfg.ASPECTS.items():
        kw = [k.lower() for k in keywords]
        sents = []
        for d in docs:
            for s in sent_split.split(d.get("text", "")):
                sl = s.lower()
                if 20 <= len(s) <= 400 and any(k in sl for k in kw):
                    sents.append(s)
        sents = sents[:200]                          # cap per aspect for speed
        if not sents:
            out[aspect] = {"mentions": 0, "positive": 0, "neutral": 0, "negative": 0, "mean": 0.0}
            continue
        results = clf(sents, batch_size=batch_size)
        pos = neu = neg = 0
        total = 0.0
        for r in results:
            lab = r["label"].lower()
            total += _compound(r["label"], r["score"])
            if lab == "positive":
                pos += 1
            elif lab == "negative":
                neg += 1
            else:
                neu += 1
        out[aspect] = {"mentions": len(sents), "positive": pos, "neutral": neu,
                       "negative": neg, "mean": round(total / len(sents), 4)}
    return out


def aggregate(docs):
    by_source = {}
    for d in docs:
        by_source.setdefault(d["source"], []).append(d)
    return {
        "overall":          _agg(docs),
        "news_sentiment":   _agg([d for d in docs if d["source"] in NEWS_SOURCES]),
        "public_sentiment": _agg([d for d in docs if d["source"] in PUBLIC_SOURCES]),
        "by_source":        {s: _agg(v) for s, v in by_source.items()},
    }


# --- 5. Orchestrate -----------------------------------------
def run():
    with open(cfg.CLEAN_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    print(f"Scoring {len(docs)} documents with FinBERT ...")
    clf  = load_classifier()
    docs = score_docs(docs, clf)

    print("Detecting sarcasm/irony on social + community ...")
    docs = detect_sarcasm(docs)

    print("Aspect-based sentiment across strategic themes ...")
    aspects = aspect_sentiment(docs, clf)

    with open(cfg.CLEAN_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    summary = aggregate(docs)
    summary["aspects"] = aspects
    with open("data/clean/sentiment.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    o, n, p = summary["overall"], summary["news_sentiment"], summary["public_sentiment"]
    print("-" * 52)
    print(f"  Overall : {o['positive']}+ / {o['neutral']}o / {o['negative']}-   mean={o['mean']}")
    print(f"  News    : mean={n['mean']}  (n={n['count']})")
    print(f"  Public  : mean={p['mean']}  (n={p['count']})  [Reddit + Hacker News]")
    print(f"            {p['flagged_sarcastic']} flagged sarcastic (down-weighted x{cfg.SARCASM_WEIGHT})")
    print("-" * 52)
    print("  Wrote per-doc scores -> data/clean/documents.json")
    print("  Wrote aggregates     -> data/clean/sentiment.json")


if __name__ == "__main__":
    run()