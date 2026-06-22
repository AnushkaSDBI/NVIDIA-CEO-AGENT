# ============================================================
#  src/repository.py  —  Hybrid knowledge repository
#
#  Built on the course foundation (rank_bm25.BM25Okapi + FAISS),
#  improvised into a production hybrid retriever:
#
#    BM25Okapi (lemmatized tokens)  ─┐
#                                    ├─ min-max fuse ─ rerank ─ top-k
#    FAISS dense (bge embeddings)   ─┘
#
#  Vector store: FAISS (local, free) — NOT Pinecone (paid/cloud/keys).
#
#  Build:  python -m src.repository
#  Use:    from src.repository import search
# ============================================================

import os
import json
import pickle

import numpy as np

import config as cfg
from .preprocess import normalize_tokens          # same tokenizer for query + corpus

INDEX_PATH = os.path.join("storage", "faiss.index")
STORE_PATH = os.path.join("storage", "retriever.pkl")
_STATE = None


def _load_chunks():
    with open(cfg.CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(cfg.EMBEDDING_MODEL)


# --- 1. Build BM25 + FAISS and persist ----------------------
def build_index(embed_fn=None):
    import faiss
    from rank_bm25 import BM25Okapi

    chunks = _load_chunks()
    texts = [c["text"] for c in chunks]
    tokenized = [(c.get("normalized") or "").split() for c in chunks]   # course-style tokens

    bm25 = BM25Okapi(tokenized)                                          # sparse

    if embed_fn is None:
        model = _embedder()
        embed_fn = lambda t: model.encode(t, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(embed_fn(texts), dtype="float32")                   # dense (cosine via IP)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    os.makedirs("storage", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(STORE_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "bm25": bm25}, f)
    print(f"Indexed {len(chunks)} chunks  (BM25Okapi + FAISS dim {emb.shape[1]})")


# --- 2. Load persisted state (cached) -----------------------
def get_state():
    global _STATE
    if _STATE is not None:
        return _STATE
    import faiss
    with open(STORE_PATH, "rb") as f:
        store = pickle.load(f)
    state = type("State", (), {})()
    state.chunks = store["chunks"]
    state.bm25 = store["bm25"]
    state.index = faiss.read_index(INDEX_PATH)
    state.model = None                       # loaded lazily on first real query
    _STATE = state
    return state


def _get_model(state):
    if state.model is None:
        state.model = _embedder()
    return state.model


def _minmax(x):
    x = np.asarray(x, dtype="float32")
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _reranker():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(cfg.RERANK_MODEL)
    except Exception:
        return None


def _query_variants(query):
    """LLM query expansion (multi-query). Single query if Ollama is absent."""
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=cfg.OLLAMA_MODEL, temperature=0.3)
        prompt = (f"Generate {cfg.MULTIQUERY_N} alternative search queries (one per line, "
                  f"no numbering) that rephrase this for retrieval:\n{query}")
        lines = [l.strip("-•* ").strip() for l in llm.invoke(prompt).content.splitlines()]
        return [query] + [l for l in lines if l][:cfg.MULTIQUERY_N]
    except Exception:
        return [query]


def _hybrid_scores(state, query, embed_query):
    """Fused BM25 + dense score over the whole corpus for one query."""
    bm25 = _minmax(state.bm25.get_scores(normalize_tokens(query)))
    qv = embed_query([query]) if embed_query else _get_model(state).encode([query], normalize_embeddings=True)
    qv = np.asarray(qv, dtype="float32")
    D, I = state.index.search(qv, len(state.chunks))
    dense = np.zeros(len(state.chunks), dtype="float32")
    for s, idx in zip(D[0], I[0]):
        dense[idx] = s
    dense = _minmax(dense)
    return cfg.HYBRID_WEIGHT * bm25 + (1 - cfg.HYBRID_WEIGHT) * dense


# --- 3. The function the agent + dashboard call -------------
def _mmr(candidates, rel, embeds, k, lam, dedup=0.95):
    """Maximal Marginal Relevance with a hard near-duplicate guard: balances relevance
    against diversity AND skips chunks nearly identical to ones already picked."""
    selected, pool = [], list(range(len(candidates)))
    sim = embeds @ embeds.T                          # cosine (embeds are normalized)
    while pool and len(selected) < k:
        if not selected:
            best = max(pool, key=lambda i: rel[i])
        else:
            fresh = [i for i in pool if max(sim[i][j] for j in selected) < dedup]
            search_pool = fresh or pool              # if everything left is a dup, fall back
            best = max(search_pool,
                       key=lambda i: lam * rel[i] - (1 - lam) * max(sim[i][j] for j in selected))
        selected.append(best)
        pool.remove(best)
    return [candidates[i] for i in selected]


def search(query, k=None, sources=None, multi_query=False, embed_query=None):
    """Hybrid retrieve -> source filter -> cross-encoder rerank -> top-k chunks."""
    k = k or cfg.TOP_K
    state = get_state()

    queries = _query_variants(query) if multi_query else [query]
    scores = np.max([_hybrid_scores(state, q, embed_query) for q in queries], axis=0)

    candidates = []
    for idx in np.argsort(scores)[::-1]:
        c = state.chunks[idx]
        if sources and c.get("source") not in set(sources):
            continue
        candidates.append(c)
        if len(candidates) >= cfg.RETRIEVE_K:
            break

    reranker = _reranker()
    if reranker and candidates:
        rr = reranker.predict([(query, c["text"]) for c in candidates])
        candidates = [c for _, c in sorted(zip(rr, candidates), key=lambda p: p[0], reverse=True)]

    # Diversify with MMR so near-duplicate chunks don't crowd out distinct evidence
    if cfg.USE_MMR and len(candidates) > k:
        model = _get_model(state)
        emb = model.encode([c["text"] for c in candidates], normalize_embeddings=True)
        rel = np.linspace(1.0, 0.0, len(candidates))     # relevance from current rank order
        candidates = _mmr(candidates, rel, emb, k, cfg.MMR_LAMBDA)
    return candidates[:k]


if __name__ == "__main__":
    build_index()
    print("-" * 56)
    for q in ["What are NVIDIA's biggest risks?", "NVIDIA Blackwell data center revenue"]:
        print(f"\nQ: {q}")
        for c in search(q, k=3):
            print(f"   [{c['source']}] {c['title'][:55]}")