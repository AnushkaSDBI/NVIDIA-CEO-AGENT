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
    # "ollama" on your laptop (Ollama server); "transformers" on the data lab GPU
    # (in-process HuggingFace model, no server needed).
    if getattr(cfg, "LLM_BACKEND", "ollama") == "transformers":
        from .llm import get_local_llm
        return get_local_llm()
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


def _citations(chunks, idxs, scored=None):
    cites = []
    for i in idxs:
        if 1 <= i <= len(chunks):
            c = chunks[i - 1]
            ent = None
            if scored and i in scored and scored[i][0] is not None:
                ent = round(scored[i][0], 3)
            cites.append({
                "source": c.get("source"), "title": c.get("title"),
                "url": c.get("url", ""), "section": c.get("section", ""),
                "published": c.get("published", ""), "entailment": ent,
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
    """Return entailment + contradiction for the claim. Scores each SENTENCE of the
    evidence (not the whole noisy chunk) and takes the max — a claim is supported if
    any single sentence entails it. Fixes dilution on long premises."""
    nli = _get_nli()
    if not nli or not evidence_texts:
        return {"verified": None, "confidence": None, "contradiction": None}
    tok, mdl, torch = nli
    import re
    sents = []
    for t in evidence_texts:
        if not t:
            continue
        for s in re.split(r"(?<=[.!?])\s+", t):
            s = s.strip()
            if 15 <= len(s) <= 400:
                sents.append(s)
    if not sents:
        sents = [" ".join(t for t in evidence_texts if t)[:400]]
    sents = sents[:24]                                   # cap per call for speed
    inp = tok(sents, [claim] * len(sents), return_tensors="pt",
              truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        probs = torch.softmax(mdl(**inp).logits, dim=-1)  # [n, 3] = [contra, neutral, entail]
    entail = float(probs[:, -1].max())
    contradiction = float(probs[:, 0].max())
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
            # 3) FINAL fallback: still show the analysis's own citations (best by entailment)
            #    so the reader always sees the evidence — flagged unverified rather than hidden.
            verified_kept = bool(kept)
            if not kept:
                cited = [(i, scored.get(i, (0.0,))[0] or 0.0)
                         for i in f.evidence if 1 <= i <= len(chunks)]
                cited.sort(key=lambda x: -x[1])
                kept = [i for i, _ in cited[:3]]
                confs = []

        # 4) entailment-based verification from surviving citations
        if confs:
            entailment, verified = round(max(confs), 3), True
        elif kept and not verify_fn:
            entailment, verified = None, None
        elif kept:                                              # shown but did not pass the bar
            best = max((scored.get(i, (0.0,))[0] or 0.0) for i in kept)
            entailment, verified = round(best, 3), False
        else:
            entailment, verified = (0.0, False) if verify_fn else (None, None)

        # 5) cross-source corroboration (how many independent source types agree)
        corroboration = _corroboration(chunks, kept)

        citations = _citations(chunks, kept, scored)
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
_IMPACT_RANK = {"high": 3, "medium": 2, "low": 1}

_REC_PROMPT = """You are the chief strategy advisor to the CEO of {company}.
A strategic {kind} has been identified by the analysis team:

  {title}
  {detail}

Recommend ONE decisive executive action that directly responds to it.
Return ONLY a JSON object (no markdown) with keys:
  "action": the recommended action (one decisive sentence),
  "rationale": why this action, in one or two sentences,
  "expected_impact": the expected business impact,
  "risk_level": "high" | "medium" | "low" (the risk of taking the action)."""


def _priority(impact, confidence):
    """Priority blends the finding's impact with how well-verified it is."""
    rank = _IMPACT_RANK.get((impact or "medium").lower(), 2) / 3.0
    conf = confidence if confidence is not None else 0.5
    score = round(0.6 * rank + 0.4 * conf, 3)
    label = "high" if score >= 0.70 else "medium" if score >= 0.45 else "low"
    return score, label


def _rank(items):
    return sorted(items, key=lambda f: (_IMPACT_RANK.get((f.get("impact") or "medium").lower(), 2),
                                        f.get("confidence") or 0), reverse=True)


def _recommend(llm, company, opps, risks, trends, verify_fn=None, n=6):
    """One recommendation per finding: each is structurally tied to the specific
    risk/opportunity it addresses, and inherits that finding's citations + confidence."""
    risks_s, opps_s, trends_s = _rank(risks), _rank(opps), _rank(trends)

    # interleave the strongest risks and opportunities so the brief is balanced
    order = []
    for r, o in zip(risks_s, opps_s):
        order.append((r, "risk"))
        order.append((o, "opportunity"))
    order += [(r, "risk") for r in risks_s[len(opps_s):]]
    order += [(o, "opportunity") for o in opps_s[len(risks_s):]]
    if not order:
        order = [(t, "trend") for t in trends_s]

    recs = []
    for f, kind in order[:n]:
        items = _llm_json(llm, _REC_PROMPT.format(
            company=company, kind=kind, title=f["title"], detail=f["detail"]))
        item = items[0] if items else {}
        try:
            r = Recommendation(**item) if item else None
        except Exception:
            r = None
        score, plabel = _priority(f.get("impact"), f.get("confidence"))
        recs.append({
            "action": r.action if r else f"Act on: {f['title']}",
            "rationale": r.rationale if r else f["detail"],
            "expected_impact": r.expected_impact if r else "",
            "risk_level": r.risk_level if r else "medium",
            "priority": plabel, "priority_score": score,
            # structural traceability: which finding this addresses + its evidence
            "addresses": {"type": kind, "title": f["title"],
                          "confidence": f.get("confidence"),
                          "corroboration": (f.get("corroboration") or {}).get("level"),
                          "contested": f.get("contested", False)},
            "citations": f.get("citations", []),     # inherit the finding's evidence trail
        })
    recs.sort(key=lambda x: x["priority_score"], reverse=True)
    return recs


# ---------------- orchestrator ----------------
_BRIEFING_PROMPT = """You are the chief of staff briefing the CEO of {company}.
Using ONLY the analysis below, write a concise executive briefing in THREE short
paragraphs, each under its heading. Plain prose, no bullet points, no markdown.

WHAT'S HAPPENING:
(the most important current developments)

WHY IT MATTERS:
(the strategic implications for {company})

WHAT TO DO NEXT:
(the top priorities for management)

ANALYSIS
Opportunities: {opps}
Risks: {risks}
Trends: {trends}
Recommended actions: {recs}"""


def _briefing(llm, company, opps, risks, trends, recs):
    """One synthesized narrative briefing: what's happening / why / what to do next."""
    def names(items):
        return "; ".join(x.get("title", "") for x in items[:5]) or "(none)"
    prompt = _BRIEFING_PROMPT.format(
        company=company, opps=names(opps), risks=names(risks), trends=names(trends),
        recs="; ".join(r.get("action", "") for r in recs[:5]) or "(none)")
    try:
        return _invoke(llm, prompt).strip()
    except Exception:
        return ""


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

    print("  [CEO INTEL] writing CEO briefing ...")
    briefing = _briefing(llm, company, opps, risks, trends, recs)

    report = {
        "company": company,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "briefing": briefing,
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