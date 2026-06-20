# ============================================================
#  src/keywords.py  —  Classical NLP track: TF-IDF key terms
#
#  A deterministic, non-LLM pass that surfaces the most distinctive
#  terms in the corpus (overall and per source) using TF-IDF. This
#  complements the neural/LLM analysis with a transparent classical
#  signal (no model, fully reproducible).
#
#  Run (after collect):  python -m src.keywords
#  Output:               data/clean/keywords.json
# ============================================================

import json

import config as cfg


def _stopwords():
    from sklearn.feature_extraction import text as _t
    extra = {a.lower() for a in cfg.COMPANY["aliases"]}
    # Reddit / RSS / web boilerplate
    extra |= {"comments", "comment", "link", "submitted", "removed", "deleted",
              "https", "http", "www", "reddit", "amp", "post", "posted", "via", "edit"}
    # generic news / finance filler that isn't a real "theme"
    extra |= {"company", "companies", "quarter", "quarterly", "year", "years", "yearly",
              "billion", "million", "trillion", "said", "say", "says", "according",
              "report", "reported", "reports", "new", "news", "also", "one", "two",
              "first", "second", "percent", "share", "shares", "week", "weeks", "day",
              "days", "time", "make", "makes", "made", "including", "include", "would",
              "could", "will", "may", "like", "just", "now", "get", "much", "many",
              "way", "back", "even", "well", "still", "going", "want", "really",
              "thing", "things", "people", "use", "used", "using", "need", "see",
              "today", "yesterday", "month", "months", "recent", "recently"}
    return list(_t.ENGLISH_STOP_WORDS.union(extra))


def _top_terms(texts, stop, n, min_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    stopset = set(stop)
    vec = TfidfVectorizer(max_features=2000, stop_words=stop,
                          ngram_range=(1, 2), min_df=min_df,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b")   # letters only, >=3 chars
    X = vec.fit_transform(texts)
    means = X.mean(axis=0).A1
    terms = vec.get_feature_names_out()
    ranked = sorted(zip(terms, means), key=lambda x: -x[1])
    out = []
    for t, w in ranked:
        words = t.split()
        if all(word in stopset for word in words):       # drop bigrams of pure stopwords
            continue
        out.append({"term": t, "weight": round(float(w), 4)})
        if len(out) >= n:
            break
    return out


def run(top_n=25):
    with open(cfg.CLEAN_PATH, encoding="utf-8") as f:
        docs = json.load(f)
    texts = [d["text"] for d in docs if d.get("text")]
    if len(texts) < 3:
        print("keywords: not enough documents.")
        return None

    stop = _stopwords()
    overall = _top_terms(texts, stop, top_n, min_df=2)

    by_source = {}
    for s in sorted({d.get("source") for d in docs}):
        s_texts = [d["text"] for d in docs if d.get("source") == s and d.get("text")]
        if len(s_texts) >= 3:
            try:
                by_source[s] = _top_terms(s_texts, stop, 12, min_df=1)
            except Exception:
                pass

    out = {"overall": overall, "by_source": by_source}
    with open("data/clean/keywords.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"keywords: top {len(overall)} terms overall, {len(by_source)} sources "
          f"-> data/clean/keywords.json")
    print("  top 10:", ", ".join(t["term"] for t in overall[:10]))
    return out


if __name__ == "__main__":
    run()