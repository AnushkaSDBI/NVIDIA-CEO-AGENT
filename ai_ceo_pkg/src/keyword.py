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
    # English stopwords + the company's own aliases (so the obvious names
    # don't dominate and genuinely distinctive terms surface).
    from sklearn.feature_extraction import text as _t
    extra = {a.lower() for a in cfg.COMPANY["aliases"]}
    extra |= {"company", "quarter", "year", "billion", "million", "said", "inc"}
    return list(_t.ENGLISH_STOP_WORDS.union(extra))


def _top_terms(texts, stop, n, min_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=2000, stop_words=stop,
                          ngram_range=(1, 2), min_df=min_df)
    X = vec.fit_transform(texts)
    means = X.mean(axis=0).A1
    terms = vec.get_feature_names_out()
    top = sorted(zip(terms, means), key=lambda x: -x[1])[:n]
    return [{"term": t, "weight": round(float(w), 4)} for t, w in top]


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