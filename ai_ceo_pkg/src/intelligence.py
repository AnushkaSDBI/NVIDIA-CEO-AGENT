# ============================================================
#  src/intelligence.py  —  CEO Intelligence Layer
#
#  Pipeline (all grounded in repository.search evidence):
#
#    RiskAnalyst        ─┐
#    OpportunityAnalyst ─┼─> findings (+citations) ─> CEOAgent ─> recommendations
#    TrendAnalyst       ─┘                                │
#                                                         ▼
#                              VerifierAgent (NLI / entailment) ─> confidence
#
#  Every finding/recommendation cites the evidence chunks it came
#  from, and is checked by an NLI model to confirm the claim is
#  actually entailed by that evidence (anti-hallucination).
#
#  Run:  python -m src.intelligence
#  Use:  from src.intelligence import run_analysis, brief
# ============================================================

import os
import re
import json
from datetime import datetime

from pydantic import BaseModel, Field

import config as cfg
from . import repository


# ---------------- structured schemas ----------------
class Finding(BaseModel):
    title: str
    detail: str
    impact: str = "medium"                       # high | medium | low
    evidence: list[int] = Field(default_factory=list)


class Recommendation(BaseModel):
    action: str
    rationale: str
    expected_impact: str = ""
    risk_level: str = "medium"                   # high | medium | low
    evidence: list[int] = Field(default_factory=list)


# ---------------- LLM (local Ollama) ----------------
def _get_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=cfg.OLLAMA_MODEL, temperature=0.2)


def _invoke(llm, prompt):
    return llm(prompt) if callable(llm) else llm.invoke(prompt).content


def _extract_json(text):
    """Robustly pull a JSON array/object out of an LLM response."""
    text = re.sub(r"```(json)?|```", "", text or "").strip()
    for op, cl in (("[", "]"), ("{", "}")):
        s, e = text.find(op), text.rfind(cl)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                pass
    return []


def _llm_json(llm, prompt):
    data = _extract_json(_invoke(llm, prompt))
    if not data:                                  # one stricter retry
        data = _extract_json(_invoke(llm, prompt + "\n\nReturn ONLY valid JSON. No prose."))
    return data if isinstance(data, list) else [data]


# ---------------- evidence + citations ----------------
def _format_evidence(chunks):
    out = []
    for i, c in enumerate(chunks, 1):
        snippet = " ".join((c.get("text") or "").split())[:400]
        out.append(f"[{i}] ({c.get('source')} — {c.get('title','')[:60]}) {snippet}")
    return "\n".join(out)


def _citations(chunks, idxs):
    cites = []
    for i in idxs:
        if 1 <= i <= len(chunks):
            c = chunks[i - 1]
            cites.append({
                "source": c.get("source"), "title": c.get("title"),
                "url": c.get("url", ""), "section": c.get("section", ""),
                "published": c.get("published", ""),
                "snippet": " ".join((c.get("text") or "").split())[:200],
            })
    return cites


# Primary / official sources are authoritative regardless of age (within our
# collection window). Only volatile secondary sources are time-decayed.
AUTHORITATIVE_SOURCES = {"filing", "pdf", "company", "reference"}


def _freshness(citation):
    """Authority-aware freshness in [0,1]: primary/official docs stay high regardless
    of age; volatile secondary sources (news, social, ...) are time-decayed."""
    if citation.get("source") in AUTHORITATIVE_SOURCES:
        return 1.0                                    # 10-K, annual report, official PDF, etc.
    return _recency_score(citation.get("published"))  # news / social / community / market


def _recency_score(published):
    """Time-decay freshness in [0,1] from a published date string (many formats)."""
    if not published:
        return 0.5                                    # unknown date -> neutral
    from datetime import datetime
    dt = None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z",
                "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
        try:
            dt = datetime.strptime(published[:31], fmt)
            break
        except (ValueError, TypeError):
            continue
    if dt is None:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(published)
        except Exception:
            return 0.5
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    age = max((datetime.now() - dt).days, 0)
    if age <= 30:
        return 1.0
    if age <= 90:
        return 0.85
    if age <= 180:
        return 0.70
    if age <= 365:
        return 0.55
    return 0.40


# ---------------- NLI verifier (faithfulness) ----------------
_NLI = None


def _get_nli():
    global _NLI
    if _NLI is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            tok = AutoTokenizer.from_pretrained(cfg.NLI_MODEL)
            mdl = AutoModelForSequenceClassification.from_pretrained(cfg.NLI_MODEL)
            _NLI = (tok, mdl, torch)
        except Exception:
            _NLI = False
    return _NLI


def verify(claim, evidence_texts):
    """Return entailment + contradiction probabilities for the claim given evidence."""
    nli = _get_nli()
    if not nli or not evidence_texts:
        return {"verified": None, "confidence": None, "contradiction": None}
    tok, mdl, torch = nli
    premise = " ".join(t for t in evidence_texts if t)[:2000]
    inp = tok(premise, claim, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = torch.softmax(mdl(**inp).logits[0], dim=-1).tolist()
    contradiction, _, entail = probs[0], probs[1], probs[-1]   # bart-mnli: [contra, neutral, entail]
    return {"verified": entail > cfg.NLI_THRESHOLD, "confidence": round(entail, 3),
            "contradiction": round(contradiction, 3)}


# ---------------- analyst agents ----------------
_ANALYST_PROMPT = """You are a {role} for {company}. Using ONLY the numbered evidence below, identify the {n} most important {kind}.

EVIDENCE:
{evidence}

Return ONLY a JSON array. Each object has keys:
  "title": short label,
  "detail": one or two sentences,
  "impact": "high" | "medium" | "low",
  "evidence": array of the evidence numbers that support it.
No markdown, no prose."""


def _score_chunks(claim, chunks, verify_fn):
    """Per-chunk NLI: {idx: (entail, contradict)} over ALL retrieved chunks."""
    scored = {}
    for i in range(1, len(chunks) + 1):
        if verify_fn:
            r = verify_fn(claim, [chunks[i - 1].get("text", "")])
            scored[i] = (r.get("confidence"), r.get("contradiction"))
        else:
            scored[i] = (None, None)
    return scored


def _corroboration(chunks, kept_idx):
    """How many INDEPENDENT source types back this finding -> strength signal."""
    sources = sorted({chunks[i - 1].get("source") for i in kept_idx if 1 <= i <= len(chunks)})
    n = len(sources)
    level = "strong" if n >= 3 else "moderate" if n == 2 else "weak" if n == 1 else "none"
    return {"score": n, "sources": sources, "level": level}


def _analyze(llm, company, role, kind, n, chunks, verify_fn):
    if not chunks:
        return []
    prompt = _ANALYST_PROMPT.format(role=role, company=company, kind=kind, n=n,
                                    evidence=_format_evidence(chunks))
    findings = []
    for item in _llm_json(llm, prompt):
        try:
            f = Finding(**item)
        except Exception:
            continue
        claim = f"{f.title}. {f.detail}"
        scored = _score_chunks(claim, chunks, verify_fn)        # {idx: (entail, contra)}

        if not verify_fn:                                       # NLI off -> keep cited as-is
            kept = [i for i in f.evidence if 1 <= i <= len(chunks)]
            confs = []
        else:
            # 1) keep cited chunks that entail the claim
            kept = [i for i in f.evidence
                    if scored.get(i, (None,))[0] is not None and scored[i][0] >= cfg.NLI_THRESHOLD]
            confs = [scored[i][0] for i in kept]
            # 2) repair: nothing cited held up -> pull best entailing chunks from the rest
            if not kept:
                cand = sorted(((i, e) for i, (e, _) in scored.items()
                               if e is not None and e >= cfg.NLI_THRESHOLD), key=lambda x: -x[1])[:3]
                kept = [i for i, _ in cand]
                confs = [e for _, e in cand]

        # 3) entailment-based verification from surviving citations
        if confs:
            entailment, verified = round(max(confs), 3), True
        elif kept:
            entailment, verified = None, None
        else:
            entailment, verified = (0.0, False) if verify_fn else (None, None)

        # 4) cross-source corroboration (how many independent source types agree)
        corroboration = _corroboration(chunks, kept)

        citations = _citations(chunks, kept)
        # authority-aware freshness: primary/official sources stay high regardless of age
        freshness = max((_freshness(c) for c in citations), default=0.5)
        # composite confidence: blend entailment + corroboration breadth + freshness
        e = entailment if entailment is not None else 0.5
        corr_norm = min(corroboration["score"] / 3.0, 1.0)
        composite = round(0.50 * e + 0.25 * corr_norm + 0.25 * freshness, 3)

        # 5) contradiction: any retrieved evidence that DISPUTES the claim -> "contested"
        contested, contra_cite = False, None
        if verify_fn:
            disputes = [(i, c) for i, (_, c) in scored.items()
                        if c is not None and c >= cfg.CONTRADICTION_THRESHOLD]
            if disputes:
                contested = True
                ci, cval = max(disputes, key=lambda x: x[1])
                cc = chunks[ci - 1]
                contra_cite = {"source": cc.get("source"), "title": cc.get("title"),
                               "url": cc.get("url", ""),
                               "snippet": " ".join((cc.get("text") or "").split())[:200],
                               "contradiction": round(cval, 3)}

        findings.append({
            "title": f.title, "detail": f.detail, "impact": f.impact,
            "citations": citations,
            "verified": verified, "confidence": composite,
            "entailment": entailment, "freshness": round(freshness, 3),
            "corroboration": corroboration,
            "score_breakdown": {"entailment": entailment, "corroboration": corr_norm,
                                "freshness": round(freshness, 3),
                                "weights": "0.50 entailment + 0.25 corroboration + "
                                           "0.25 freshness (authority-aware)"},
            "contested": contested, "contradicting_evidence": contra_cite,
        })
    return findings


# ---------------- CEO synthesis agent ----------------
_CEO_PROMPT = """You are the chief strategy advisor to the CEO of {company}.
Based on the analysis below, produce the {n} highest-priority strategic recommendations.

OPPORTUNITIES:
{opps}

RISKS:
{risks}

TRENDS:
{trends}

Return ONLY a JSON array. Each object has keys:
  "action": the recommended action,
  "rationale": why, in one or two sentences,
  "expected_impact": the expected business impact,
  "risk_level": "high" | "medium" | "low".
No markdown, no prose."""


def _bullets(items):
    return "\n".join(f"- {x['title']}: {x['detail']}" for x in items) or "- (none)"


def _recommend(llm, company, opps, risks, trends, verify_fn, n=5):
    prompt = _CEO_PROMPT.format(company=company, n=n,
                                opps=_bullets(opps), risks=_bullets(risks), trends=_bullets(trends))
    premise = [x["detail"] for x in (opps + risks + trends)]
    recs = []
    for item in _llm_json(llm, prompt):
        try:
            r = Recommendation(**item)
        except Exception:
            continue
        v = verify_fn(f"{r.action}. {r.rationale}", premise) \
            if verify_fn else {"verified": None, "confidence": None}
        recs.append({
            "action": r.action, "rationale": r.rationale,
            "expected_impact": r.expected_impact, "risk_level": r.risk_level,
            "verified": v["verified"], "confidence": v["confidence"],
        })
    return recs


# ---------------- orchestrator ----------------
def run_analysis(search_fn=None, llm=None, verify_fn=verify):
    company = cfg.COMPANY["name"]
    search_fn = search_fn or repository.search
    llm = llm or _get_llm()

    print(f"  [CEO INTEL] gathering evidence for {company} ...")
    risk_ev = search_fn("regulatory risk competition supply chain customer concentration "
                        "export controls CUDA alternatives open-source competing libraries",
                        k=8, sources=["filing", "news", "market", "ecosystem"])
    opp_ev = search_fn("growth opportunities new markets products partnerships demand expansion",
                       k=8, sources=["company", "pdf", "research", "news"])
    trend_ev = search_fn("emerging technology trends AI data center accelerated computing software "
                        "open-source ecosystem developer adoption library momentum",
                        k=8, sources=["research", "company", "news", "ecosystem"])

    print("  [CEO INTEL] running analyst agents ...")
    risks = _analyze(llm, company, "risk analyst", "strategic RISKS", 5, risk_ev, verify_fn)
    opps = _analyze(llm, company, "opportunity analyst", "growth OPPORTUNITIES", 5, opp_ev, verify_fn)
    trends = _analyze(llm, company, "technology trend analyst", "emerging TRENDS", 5, trend_ev, verify_fn)

    print("  [CEO INTEL] synthesizing recommendations ...")
    recs = _recommend(llm, company, opps, risks, trends, verify_fn)

    report = {
        "company": company,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "opportunities": opps, "risks": risks, "trends": trends,
        "recommendations": recs,
    }
    os.makedirs(os.path.dirname(cfg.INTEL_PATH), exist_ok=True)
    with open(cfg.INTEL_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  -> {len(opps)} opportunities, {len(risks)} risks, {len(trends)} trends, "
          f"{len(recs)} recommendations -> {cfg.INTEL_PATH}")
    return report


# ---------------- ad-hoc Q&A (for the dashboard) ----------------
def brief(question, k=6, llm=None):
    """Answer a strategic question, grounded + cited, via retrieve -> LLM."""
    llm = llm or _get_llm()
    chunks = repository.search(question, k=k, multi_query=True)
    prompt = (f"You are the strategic advisor to {cfg.COMPANY['name']}'s CEO. "
              f"Answer the question using ONLY the evidence. Cite evidence numbers inline like [1]. "
              f"Be concise and concrete.\n\nQUESTION: {question}\n\nEVIDENCE:\n{_format_evidence(chunks)}")
    answer = _invoke(llm, prompt)
    return {"question": question, "answer": answer, "citations": _citations(chunks, range(1, len(chunks) + 1))}


if __name__ == "__main__":
    run_analysis()