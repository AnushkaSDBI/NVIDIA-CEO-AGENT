# ============================================================
#  src/preprocess.py  —  Preprocessing agent
#
#  Pipeline:  full document  ->  chunk  ->  normalize tokens
#
#  Two tracks per chunk:
#    • text       : natural language  (for embeddings + the LLM)
#    • normalized : lowercased, stop-word-free, lemmatized tokens
#                   (for BM25 + topic modeling)
#
#  Lemmatization uses spaCy (POS-aware, more accurate than NLTK);
#  falls back to a regex tokenizer if the spaCy model is absent.
# ============================================================

import json

import config as cfg
from . import database

_NLP = None
_FALLBACK_STOP = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "her",
    "was", "our", "out", "has", "had", "his", "she", "will", "with", "that",
    "this", "from", "they", "their", "what", "which", "when", "have", "been",
    "were", "would", "there", "about", "into", "than", "them", "then", "over",
    "also", "such", "more", "most", "some", "other", "these", "those", "its",
}


def _get_nlp():
    """Load spaCy once (tagger+lemmatizer only). False if unavailable."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception:
            _NLP = False
    return _NLP


def _lemmas(spacy_doc):
    return [t.lemma_.lower() for t in spacy_doc
            if t.is_alpha and not t.is_stop and len(t) > 2]


def normalize_tokens(text):
    """Lemmatize one piece of text -> list of tokens (used for queries too)."""
    text = (text or "")[:100000]
    nlp = _get_nlp()
    if nlp:
        return _lemmas(nlp(text))
    import re
    toks = re.findall(r"[a-z][a-z\-]+", text.lower())
    return [w for w in toks if len(w) > 2 and w not in _FALLBACK_STOP]


def _chunk(text):
    """Recursive character chunking (LangChain), with a simple fallback."""
    text = text or ""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""])
        return splitter.split_text(text)
    except Exception:
        step = cfg.CHUNK_SIZE - cfg.CHUNK_OVERLAP
        return [text[i:i + cfg.CHUNK_SIZE] for i in range(0, len(text), step)]


def build():
    with open(cfg.CLEAN_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    # 1. chunk every document
    records = []                       # (piece_text, source_doc)
    for d in docs:
        for piece in _chunk(d.get("text", "")):
            if len(piece.strip()) >= 40:
                records.append((piece, d))

    # 2. normalize all chunks (batched through spaCy for speed)
    nlp = _get_nlp()
    if nlp:
        normalized = [" ".join(_lemmas(sp))
                      for sp in nlp.pipe([p[:100000] for p, _ in records], batch_size=64)]
        engine = "spaCy"
    else:
        normalized = [" ".join(normalize_tokens(p)) for p, _ in records]
        engine = "regex-fallback"

    # 3. assemble chunk records
    chunks = []
    for i, ((piece, d), norm) in enumerate(zip(records, normalized)):
        chunks.append({
            "chunk_id":  f"{d.get('source','')}-{i}",
            "source":    d.get("source", ""),
            "section":   d.get("section", ""),
            "title":     d.get("title", ""),
            "url":       d.get("url", ""),
            "published": d.get("published", ""),
            "text":       piece,        # natural -> embeddings / LLM
            "normalized": norm,         # tokens  -> BM25 / topics
        })

    with open(cfg.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    avg = sum(len(c["normalized"].split()) for c in chunks) // max(len(chunks), 1)
    print(f"Chunked {len(docs)} documents -> {len(chunks)} chunks  (lemmatizer: {engine})")
    print(f"  avg {avg} normalized tokens/chunk -> {cfg.CHUNKS_PATH}")

    database.build_chunks_table()       # store PROCESSED data in the database
    return chunks


if __name__ == "__main__":
    build()
