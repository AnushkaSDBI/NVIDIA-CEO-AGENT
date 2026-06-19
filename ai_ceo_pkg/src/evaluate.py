# ============================================================
#  src/evaluate.py  —  Retrieval evaluation / ablation
#
#  Compares retrieval configurations on a set of test queries:
#    BM25-only | Dense-only | Hybrid | Hybrid+Rerank | Hybrid+Rerank+MMR
#  Metrics: Hit Rate@k (a relevant chunk appears in top-k) and MRR.
#
#  Relevance is keyword-based: a chunk is "relevant" to a query if its
#  text contains any of the query's expected keywords. Lightweight but
#  legitimate, and fully reproducible.
#
#  Run (after the index is built):  python -m src.evaluate
# ============================================================

import json

import numpy as np

import config as cfg
from . import repository as repo

# (query, keywords that a genuinely relevant chunk should contain)
TESTS = [
    ("NVIDIA China export controls risk",        ["export control", "china"]),
    ("Blackwell data center revenue",            ["blackwell", "data center"]),
    ("gross margin guidance outlook",            ["gross margin"]),
    ("competition from AMD",                     ["amd"]),
    ("customer concentration risk",              ["concentrat"]),
    ("supply chain constraints",                 ["supply"]),
    ("CUDA software ecosystem moat",             ["cuda"]),
    ("automotive segment revenue",               ["automotive", "auto"]),
    ("networking InfiniBand Spectrum revenue",   ["network", "infiniband", "spectrum"]),
    ("gaming GeForce demand",                    ["gaming", "geforce"]),
    ("AI inference workloads growth",            ["inference"]),
    ("stock price 52-week trend",                ["52-week", "sma", "stock"]),
]


def _relevant(chunk, keywords):
    t = (chunk.get("text") or "").lower()
    return any(k in t for k in keywords)


def _hit_mrr(chunks, keywords):
    for rank, c in enumerate(chunks, 1):
        if _relevant(c, keywords):
            return 1, 1.0 / rank
    return 0, 0.0


def _bm25_order(state, q):
    return np.argsort(repo._minmax(state.bm25.get_scores(repo.normalize_tokens(q))))[::-1]


def _dense_order(state, q):
    qv = np.asarray(repo._get_model(state).encode([q], normalize_embeddings=True), dtype="float32")
    D, I = state.index.search(qv, len(state.chunks))
    dense = np.zeros(len(state.chunks), dtype="float32")
    for s, idx in zip(D[0], I[0]):
        dense[idx] = s
    return np.argsort(dense)[::-1]


def _hybrid_order(state, q):
    return np.argsort(repo._hybrid_scores(state, q, None))[::-1]


def run(k=None):
    k = k or cfg.TOP_K
    state = repo.get_state()
    reranker = repo._reranker()
    methods = ["BM25", "Dense", "Hybrid", "Hybrid+Rerank", "Hybrid+Rerank+MMR"]
    agg = {m: {"hits": 0, "mrr": 0.0} for m in methods}

    for q, kw in TESTS:
        bm = [state.chunks[i] for i in _bm25_order(state, q)[:k]]
        de = [state.chunks[i] for i in _dense_order(state, q)[:k]]
        hy_order = _hybrid_order(state, q)
        hy = [state.chunks[i] for i in hy_order[:k]]

        cand = [state.chunks[i] for i in hy_order[:cfg.RETRIEVE_K]]
        if reranker and cand:
            sc = reranker.predict([(q, c["text"]) for c in cand])
            cand = [c for _, c in sorted(zip(sc, cand), key=lambda p: p[0], reverse=True)]
        rer = cand[:k]

        mmr = rer
        try:
            emb = repo._get_model(state).encode([c["text"] for c in cand], normalize_embeddings=True)
            rel = np.linspace(1.0, 0.0, len(cand))
            mmr = repo._mmr(cand, rel, emb, k, cfg.MMR_LAMBDA)
        except Exception:
            pass

        for m, chunks in zip(methods, [bm, de, hy, rer, mmr]):
            h, mrr = _hit_mrr(chunks, kw)
            agg[m]["hits"] += h
            agg[m]["mrr"] += mrr

    n = len(TESTS)
    results = [{"method": m, "hit_rate": round(agg[m]["hits"] / n, 3),
                "mrr": round(agg[m]["mrr"] / n, 3)} for m in methods]

    print(f"\nRetrieval evaluation — {n} queries, k={k}")
    print(f"{'Method':24s}{'Hit@k':>8s}{'MRR':>8s}")
    print("-" * 40)
    for r in results:
        print(f"{r['method']:24s}{r['hit_rate']:>8.3f}{r['mrr']:>8.3f}")

    with open("data/clean/evaluation.json", "w", encoding="utf-8") as f:
        json.dump({"k": k, "n_queries": n, "results": results}, f, indent=2)
    print("\n  -> data/clean/evaluation.json")
    return results


if __name__ == "__main__":
    run()