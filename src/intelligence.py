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
    kwargs = dict(model=cfg.OLLAMA_MODEL,
                  temperature=getattr(cfg, "LLM_TEMPERATURE", 0.2))
    seed = getattr(cfg, "LLM_SEED", None)
    if seed is not None:                       # pin for reproducible plan/findings/counts
        kwargs["seed"] = seed
    return ChatOllama(**kwargs)


def _invoke(llm, prompt):
    return llm(prompt) if callable(llm) else llm.invoke(prompt).content


def _extract_json(text):
    """Robustly pull a JSON array/object out of an LLM response."""
    text = re.sub(r"```(json)?|```", "", text or "").strip()
    # 1) try to parse the whole response (handles a clean object or array directly,
    #    including an object that contains nested arrays e.g. {"action_input": {"sources": [...]}})
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) otherwise extract the OUTERMOST value, choosing object vs array by whichever
    #    delimiter appears FIRST in the text (so a nested [...] inside {...} isn't grabbed)
    first = {op: text.find(op) for op in ("{", "[") if text.find(op) != -1}
    order = sorted(first, key=first.get)                  # delimiter that starts earliest first
    for op in order:
        cl = "}" if op == "{" else "]"
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


_NLI_CACHE = {}                                   # (hash(claim), hash(text)) -> (entail, contra)


def _nli_score_many(claim, texts):
    """Score the claim against MANY chunk texts in batched forward passes (one set of
    passes for the whole list, instead of one model call per chunk). Per-chunk result is
    identical to verify([text]): split into sentences, take the max entail/contra."""
    nli = _get_nli()
    if not nli:
        return [(None, None)] * len(texts)
    tok, mdl, torch = nli
    import re
    all_sents, owner = [], []
    for ti, t in enumerate(texts):
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t or "")
                 if 15 <= len(s.strip()) <= 400][:24]
        if not sents and t:
            sents = [" ".join(t.split())[:400]]
        for s in sents:
            all_sents.append(s)
            owner.append(ti)
    ent = [0.0] * len(texts)
    con = [0.0] * len(texts)
    B = 32
    for i in range(0, len(all_sents), B):
        batch = all_sents[i:i + B]
        inp = tok(batch, [claim] * len(batch), return_tensors="pt",
                  truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            probs = torch.softmax(mdl(**inp).logits, dim=-1)  # [n,3]=[contra,neutral,entail]
        for j in range(len(batch)):
            ti = owner[i + j]
            ent[ti] = max(ent[ti], float(probs[j, -1]))
            con[ti] = max(con[ti], float(probs[j, 0]))
    return [(round(ent[k], 3), round(con[k], 3)) for k in range(len(texts))]


def _score_chunks(claim, chunks, verify_fn):
    """Per-chunk NLI: {idx: (entail, contradict)} over ALL retrieved chunks.
    With the real NLI verifier this BATCHES every chunk into one set of forward passes
    and caches by (claim, text) so retries don't re-score the same evidence."""
    if verify_fn is verify:                                    # real NLI -> batch + cache
        texts = [c.get("text", "") for c in chunks]
        out, todo, todo_i = {}, [], []
        for i, t in enumerate(texts):
            key = (hash(claim), hash(t))
            if key in _NLI_CACHE:
                out[i + 1] = _NLI_CACHE[key]
            else:
                todo.append(t)
                todo_i.append(i)
        if todo:
            for j, pair in enumerate(_nli_score_many(claim, todo)):
                i = todo_i[j]
                out[i + 1] = pair
                _NLI_CACHE[(hash(claim), hash(texts[i]))] = pair
        return out
    # custom / stub verifier (tests) -> per-chunk
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

        if not verify_fn:                                       # NLI off -> keep cited in-bounds
            kept = [i for i in f.evidence if 1 <= i <= len(chunks)]
            entailment, verified = None, None
        else:
            # Two-tier evidence bar:
            #   STRONG  (>= NLI_THRESHOLD)        -> strong enough to VERIFY the claim
            #   SUPPORT (>= CORROBORATION_THRESHOLD) -> at least weakly supportive; counts toward
            #                                          CORROBORATION so a claim verified by one strong
            #                                          source can still gather independent second sources.
            STRONG = cfg.NLI_THRESHOLD
            SUPPORT = getattr(cfg, "CORROBORATION_THRESHOLD", round(STRONG * 0.55, 3))
            strong_idx = {i for i in scored
                          if scored[i][0] is not None and scored[i][0] >= STRONG}
            supporters = sorted(((i, scored[i][0]) for i in scored
                                 if scored[i][0] is not None and scored[i][0] >= SUPPORT),
                                key=lambda x: -x[1])
            # diversify across source types FIRST: best supporter from each distinct source,
            # so a finding is backed by INDEPENDENT sources (lets corroboration reach 2-3)
            seen_src, kept = set(), []
            for i, e in supporters:
                src = chunks[i - 1].get("source")
                if src not in seen_src:
                    seen_src.add(src)
                    kept.append(i)
            # then top up with the next strongest supporters (any source) up to 5 citations
            for i, e in supporters:
                if i not in kept and len(kept) < 5:
                    kept.append(i)
            # fallback: nothing supportive -> still show the analysis's own cited evidence
            # (flagged unverified) so the reader always sees what it was based on
            if not kept:
                cited = sorted(((i, scored.get(i, (0.0,))[0] or 0.0)
                                for i in f.evidence if 1 <= i <= len(chunks)), key=lambda x: -x[1])
                kept = [i for i, _ in cited[:3]]

            # verification requires at least one STRONG (entailing) source among the kept evidence
            strong_kept = [i for i in kept if i in strong_idx]
            if strong_kept:
                entailment = round(max(scored[i][0] for i in strong_kept), 3)
                verified = True
            elif kept:                                          # shown but no source cleared the bar
                entailment = round(max((scored.get(i, (0.0,))[0] or 0.0) for i in kept), 3)
                verified = False
            else:
                entailment, verified = 0.0, False

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
    """Best findings first: verified, then breadth of independent corroboration,
    then confidence, then impact. This promotes verified + multi-source + high-confidence
    findings into the recommendations."""
    def key(f):
        verified = 1 if f.get("verified") else 0
        corr = min((f.get("corroboration") or {}).get("score", 0), 3)
        conf = f.get("confidence") or 0.0
        impact = _IMPACT_RANK.get((f.get("impact") or "medium").lower(), 2)
        return (verified, corr, round(conf, 3), impact)
    return sorted(items, key=key, reverse=True)


_REC_BATCH_PROMPT = """You are the chief strategy advisor to the CEO of {company}.
The analysis team has identified these strategic findings:

{findings}

For EACH numbered finding, recommend ONE decisive executive action that directly
responds to it. Return ONLY a JSON array (no markdown); each object has keys:
  "index": the finding number it responds to,
  "action": the recommended action (one decisive sentence),
  "rationale": why this action, in one or two sentences,
  "expected_impact": the expected business impact,
  "risk_level": "high" | "medium" | "low" (the risk of taking the action).
Return exactly {n} objects, one per finding."""


def _recommend(llm, company, opps, risks, trends, verify_fn=None, n=6):
    """One recommendation per finding: each is structurally tied to the specific
    risk/opportunity it addresses, and inherits that finding's citations + confidence.
    All recommendations are produced in ONE batched LLM call (not one call per finding)."""
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
    order = order[:n]

    # ONE batched call for all recommendations
    by_index = {}
    if order:
        listing = "\n".join(f'{i}. [{kind}] {f["title"]}: {f["detail"]}'
                            for i, (f, kind) in enumerate(order, 1))
        for it in _llm_json(llm, _REC_BATCH_PROMPT.format(
                company=company, findings=listing, n=len(order))):
            idx = it.get("index") if isinstance(it, dict) else None
            if isinstance(idx, int):
                by_index[idx] = it

    recs = []
    for i, (f, kind) in enumerate(order, 1):
        item = by_index.get(i, {})
        try:
            r = Recommendation(**item) if item.get("action") else None
        except Exception:
            r = None
        score, plabel = _priority(f.get("impact"), f.get("confidence"))
        corr = f.get("corroboration") or {}
        n_sources = corr.get("score", 0)
        recs.append({
            "action": r.action if r else f"Act on: {f['title']}",
            "rationale": r.rationale if r else f["detail"],
            "expected_impact": r.expected_impact if r else "",
            "risk_level": r.risk_level if r else "medium",
            "priority": plabel, "priority_score": score,
            # quality signals inherited from the finding this recommendation rests on
            "verified": bool(f.get("verified")),
            "confidence": f.get("confidence"),
            "evidence_sources": n_sources,                       # distinct independent source types
            # validated = the claim is NLI-verified (entailed by its evidence). The number
            # of independent sources is reported separately (evidence_sources) as a secondary
            # quality signal, but is NOT required to mark a recommendation validated.
            "well_supported": bool(f.get("verified")),
            # structural traceability: which finding this addresses + its evidence
            "addresses": {"type": kind, "title": f["title"],
                          "confidence": f.get("confidence"),
                          "corroboration": corr.get("level"),
                          "contested": f.get("contested", False)},
            "citations": f.get("citations", []),     # inherit the finding's evidence trail
        })
    # surface the best-supported recommendations first (verified + sources, then priority)
    recs.sort(key=lambda x: (x["well_supported"], x["priority_score"],
                             x.get("confidence") or 0), reverse=True)
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


def _gather(search_fn, query, sources, per_source=3):
    """Retrieve a SOURCE-BALANCED evidence pool: the top chunks from EACH source type
    separately, then combined. Without this, one query returns the global top-k, which
    long investor PDFs / filings tend to dominate — so findings end up citing only those.
    Pulling per-source guarantees the pool spans independent sources, which is what lets
    a finding be corroborated across (filing + news + market) rather than (pdf + pdf + pdf)."""
    pool, seen = [], set()
    for src in sources:
        try:
            hits = search_fn(query, k=per_source, sources=[src])
        except TypeError:
            hits = search_fn(query, k=per_source)
        for h in (hits or []):
            key = (h.get("source"), h.get("url"), (h.get("title") or "")[:60])
            if key not in seen:
                seen.add(key)
                pool.append(h)
    return pool


def run_analysis(search_fn=None, llm=None, verify_fn=verify):
    company = cfg.COMPANY["name"]
    search_fn = search_fn or repository.search
    llm = llm or _get_llm()

    print(f"  [CEO INTEL] gathering evidence for {company} ...")
    # Source-BALANCED pools (top chunks from each source type), so evidence and
    # citations span independent sources instead of being dominated by internal PDFs.
    risk_ev = _gather(search_fn,
                      "regulatory risk competition supply chain customer concentration "
                      "export controls CUDA alternatives open-source competing libraries",
                      ["filing", "news", "market", "ecosystem", "community", "social"])
    opp_ev = _gather(search_fn,
                     "growth opportunities new markets products partnerships demand expansion "
                     "data center AI software services automotive robotics",
                     ["company", "news", "research", "ecosystem", "pdf", "community", "market"])
    trend_ev = _gather(search_fn,
                       "emerging technology trends AI data center accelerated computing software "
                       "open-source ecosystem developer adoption library momentum",
                       ["research", "company", "news", "ecosystem", "community", "social"])

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