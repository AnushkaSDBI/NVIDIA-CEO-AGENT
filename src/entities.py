# ============================================================
#  src/entities.py  —  Competitor & entity extraction (spaCy NER)
#
#  Runs spaCy NER over the full corpus, counts the organizations
#  mentioned, drops NVIDIA itself, and flags known industry rivals
#  as competitors. Output feeds the dashboard's competitor panel
#  and the knowledge-graph view.
#
#  Setup:  pip install spacy ; python -m spacy download en_core_web_sm
#  Run:    python -m src.entities
# ============================================================

import json
import collections
import itertools

import config as cfg

ENTITIES_PATH = "data/clean/entities.json"

def _match_known(norm):
    """Map a raw NER org string to a known company in our universe, else None.
    Handles 'nvidia corp', 'amd, inc.', 'microsoft corporation' -> base name."""
    n = norm.replace(",", " ").replace(".", " ")
    n = " ".join(w for w in n.split()
                 if w not in {"inc", "corp", "corporation", "ltd", "llc", "co", "the", "plc", "ag"})
    n = n.strip()
    if n in cfg.ORG_ROLE:
        return n
    for known in cfg.ORG_ROLE:                       # token-level containment (e.g. "advanced micro devices")
        if known in n.split() or n == known:
            return known
    aliases = {"alphabet": "google", "aws": "amazon", "advanced micro devices": "amd"}
    return aliases.get(n)


def extract():
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "attribute_ruler"])

    with open(cfg.CLEAN_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    aliases = {a.lower() for a in cfg.COMPANY["aliases"]} | {"nvidia", "nvidia's"}

    org_counts = collections.Counter()
    org_samples = collections.defaultdict(set)
    pair_counts = collections.Counter()

    texts = [d["text"][:100000] for d in docs]
    titles = [d.get("title", "")[:60] for d in docs]

    for title, doc in zip(titles, nlp.pipe(texts, batch_size=16)):
        seen = set()
        for ent in doc.ents:
            if ent.label_ != "ORG":
                continue
            norm = ent.text.strip().lower().rstrip(".,")
            if norm in aliases or any(a in norm for a in aliases):
                continue
            known = _match_known(norm)               # only keep real companies in our universe
            if not known:
                continue
            if known not in seen:
                org_counts[known] += 1
                seen.add(known)
            org_samples[known].add(title)
        for a, b in itertools.combinations(sorted(seen), 2):
            pair_counts[(a, b)] += 1

    top = org_counts.most_common(40)
    top_names = {n for n, _ in top}
    edges = [{"source": a, "target": b, "weight": w}
             for (a, b), w in pair_counts.most_common()
             if a in top_names and b in top_names and w >= 2][:40]

    result = {
        "all_orgs": [{"name": n.title(), "mentions": c, "role": cfg.ORG_ROLE.get(n, "other")}
                     for n, c in top],
        "competitors": [
            {"name": n.title(), "mentions": c, "sample_docs": list(org_samples[n])[:3]}
            for n, c in top if cfg.ORG_ROLE.get(n) == "competitor"
        ],
        "edges": [{"source": e["source"].title(), "target": e["target"].title(),
                   "weight": e["weight"]} for e in edges],
    }
    with open(ENTITIES_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Tagged {len(top)} known organizations; "
          f"{len(result['competitors'])} competitors; {len(edges)} edges -> {ENTITIES_PATH}")
    for o in result["all_orgs"][:10]:
        print(f"   {o['name']:14} {o['mentions']:3}  [{o['role']}]")
    return result


if __name__ == "__main__":
    extract()