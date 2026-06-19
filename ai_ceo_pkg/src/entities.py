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

import config as cfg

ENTITIES_PATH = "data/clean/entities.json"


def extract():
    import spacy
    # Only need the NER pipe -> disable the rest for speed.
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "attribute_ruler"])

    with open(cfg.CLEAN_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    aliases = {a.lower() for a in cfg.COMPANY["aliases"]} | {"nvidia", "nvidia's"}
    competitors_known = {c.lower() for c in cfg.COMPETITORS}

    org_counts = collections.Counter()
    org_samples = collections.defaultdict(set)

    texts = [d["text"][:100000] for d in docs]            # cap per doc for speed
    titles = [d.get("title", "")[:60] for d in docs]

    for title, doc in zip(titles, nlp.pipe(texts, batch_size=16)):
        seen = set()
        for ent in doc.ents:
            if ent.label_ != "ORG":
                continue
            norm = ent.text.strip().lower().rstrip(".,").replace(",", "")
            if len(norm) < 2 or norm in aliases:
                continue
            if any(a in norm for a in aliases):           # skip "nvidia corp" etc.
                continue
            if norm not in seen:                          # count once per document
                org_counts[norm] += 1
                seen.add(norm)
            org_samples[norm].add(title)

    top = org_counts.most_common(40)
    result = {
        "all_orgs": [{"name": n, "mentions": c} for n, c in top],
        "competitors": [
            {"name": n, "mentions": c, "sample_docs": list(org_samples[n])[:3]}
            for n, c in top if n in competitors_known
        ],
    }
    with open(ENTITIES_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Extracted {len(org_counts)} organizations; "
          f"{len(result['competitors'])} known competitors -> {ENTITIES_PATH}")
    for c in result["competitors"][:10]:
        print(f"   {c['name']:14} {c['mentions']} mentions")
    return result


if __name__ == "__main__":
    extract()
