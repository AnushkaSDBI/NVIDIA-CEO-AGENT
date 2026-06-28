# ============================================================
#  src/agent.py  —  Agentic orchestration layer
#
#  Turns the FIXED pipeline (run_analysis) into AGENT BEHAVIOUR:
#
#    PLAN     the model decomposes the CEO objective into focus areas
#      |        (it decides WHAT to investigate, not a hardcoded list)
#    ACT      it calls TOOLS by name (search / stock / sentiment / ...)
#      |
#    OBSERVE  it reads the result (evidence, verification scores)
#      |
#    REFLECT  if a finding is unverified or weakly corroborated, the agent
#      |        REWRITES its own search query and tries again (self-correction)
#    LOOP     repeat until findings are verified+corroborated or max attempts
#      |
#    FINALISE recommendations + CEO briefing  (reuses intelligence.py)
#
#  Everything here REUSES the existing functions — the analyst agents,
#  the NLI verifier, the retriever — as the agent's tools. The new part
#  is the model-driven CONTROL FLOW (plan, tool choice, reflect, retry).
#
#  Run:  python -m src.agent
#  Use:  from src.agent import run_agent, agent_answer
# ============================================================

import os
import json
from datetime import datetime

import config as cfg
from . import repository, database
from .intelligence import (
    _get_llm, _invoke, _llm_json, _format_evidence, _citations,
    _gather, _analyze, _recommend, _briefing, verify,
)

# Each "lens" = (analyst role, what it looks for, which sources to draw from).
LENS = {
    "risk": ("risk analyst", "strategic RISKS",
             ["filing", "news", "market", "ecosystem", "community", "social"]),
    "opportunity": ("opportunity analyst", "growth OPPORTUNITIES",
                    ["company", "news", "research", "ecosystem", "pdf", "community", "market"]),
    "trend": ("technology trend analyst", "emerging TRENDS",
              ["research", "company", "news", "ecosystem", "community", "social"]),
}


# ============================================================
#  TOOLS — the actions the agent can choose to take.
#  Each returns (observation_text, chunks_or_None). The text is what the
#  LLM reads; the chunks (for search) are kept so answers can be cited.
# ============================================================
def tool_search_evidence(query, sources=None, per_source=3, search_fn=None):
    search_fn = search_fn or repository.search
    srcs = sources or sum((v[2] for v in LENS.values()), [])
    srcs = list(dict.fromkeys(srcs))                       # de-dup, keep order
    chunks = _gather(search_fn, query, srcs, per_source=per_source)
    return _format_evidence(chunks[:8]) or "(no evidence found)", chunks


def tool_stock_snapshot(*_a, **_k):
    try:
        ind = database.stock_indicators()
        return (f"latest close ${ind.get('latest_close','?')}, RSI(14) {ind.get('rsi_14','?')}, "
                f"trend {ind.get('trend','?')}, 52w high ${ind.get('high_52w','?')}, "
                f"{ind.get('pct_from_52w_high','?')}% from high"), None
    except Exception:
        return "(stock data unavailable)", None


def tool_sentiment_snapshot(*_a, **_k):
    try:
        with open("data/clean/sentiment.json", encoding="utf-8") as f:
            s = json.load(f)
        news = s.get("news_sentiment", {}).get("mean", "?")
        pub = s.get("public_sentiment", {}).get("mean", "?")
        return f"news sentiment mean {news}, public sentiment mean {pub}", None
    except Exception:
        return "(sentiment data unavailable)", None


def tool_competitors(*_a, **_k):
    try:
        with open("data/clean/entities.json", encoding="utf-8") as f:
            e = json.load(f)
        comp = e.get("competitors", [])[:6]
        return "competitors by mentions: " + ", ".join(
            f"{c.get('name')} ({c.get('mentions')})" for c in comp) or "(none)", None
    except Exception:
        return "(competitor data unavailable)", None


TOOLS = {
    "search_evidence": (tool_search_evidence,
                        "search the knowledge base for evidence. input: {\"query\": str, \"sources\": [optional list]}"),
    "stock_snapshot": (tool_stock_snapshot, "current stock indicators. input: {}"),
    "sentiment_snapshot": (tool_sentiment_snapshot, "news + public sentiment means. input: {}"),
    "competitors": (tool_competitors, "known competitors by mention count. input: {}"),
}


def _tool_descriptions():
    return "\n".join(f"  - {name}: {desc}" for name, (_, desc) in TOOLS.items())


# ============================================================
#  PLAN — the agent decides what to investigate.
# ============================================================
_PLAN_PROMPT = """You are the lead strategy agent for {company}. Your objective:
"{objective}"

Decompose this into 3-5 concrete investigation steps. For EACH step choose a lens
("risk", "opportunity", or "trend") and write a focused search query.

Return ONLY a JSON array; each object has keys:
  "focus": short label of what to investigate,
  "lens": "risk" | "opportunity" | "trend",
  "query": a search query to retrieve evidence for it.
No prose, no markdown."""


def _default_plan():
    return [
        {"focus": "regulatory & competitive risk", "lens": "risk",
         "query": "regulatory risk competition supply chain export controls CUDA alternatives"},
        {"focus": "growth opportunities", "lens": "opportunity",
         "query": "growth opportunities data center AI software services automotive robotics partnerships"},
        {"focus": "emerging technology trends", "lens": "trend",
         "query": "emerging technology trends AI data center accelerated computing open-source adoption"},
    ]


def plan_investigation(llm, company, objective):
    """Agent capability 1: PLANNING / decomposition (model decides the steps)."""
    try:
        steps = _llm_json(llm, _PLAN_PROMPT.format(company=company, objective=objective))
        clean = []
        for s in steps:
            lens = (s.get("lens") or "").lower()
            if lens in LENS and s.get("query"):
                clean.append({"focus": s.get("focus", lens), "lens": lens, "query": s["query"]})
        return clean or _default_plan()
    except Exception:
        return _default_plan()


# ============================================================
#  REFLECT — the agent rewrites its query when evidence is weak.
# ============================================================
_REFINE_PROMPT = """You are a {role} for {company}. Your previous search:
  "{query}"
returned findings whose evidence was weak or could not be verified:
  {weak}

Write ONE improved search query that would retrieve MORE AUTHORITATIVE, corroborating
evidence from INDEPENDENT sources (filings, news, market data) about: {focus}.
Return ONLY the query text on a single line. No quotes, no prose."""


def _refine_query(llm, company, role, focus, query, weak_titles):
    try:
        out = _invoke(llm, _REFINE_PROMPT.format(
            role=role, company=company, query=query, focus=focus,
            weak="; ".join(weak_titles) or "(unverified findings)"))
        line = next((l.strip(" -•*\"'") for l in (out or "").splitlines() if l.strip()), "")
        return line or query
    except Exception:
        return query


def _is_weak(f):
    """A finding is weak if it is not verified OR backed by < 2 independent sources."""
    corr = (f.get("corroboration") or {}).get("score", 0)
    return (not f.get("verified")) or corr < 2


def investigate(llm, company, step, verify_fn, search_fn=None, max_attempts=2):
    """Agent capabilities 2-4: ACT (search tool) -> OBSERVE (verify) ->
    REFLECT (rewrite query) -> LOOP. Returns (findings, per-attempt trace)."""
    role, kind, sources = LENS[step["lens"]]
    query = step["query"]
    attempts, findings = [], []
    for attempt in range(1, max_attempts + 1):
        obs, chunks = tool_search_evidence(query, sources=sources, search_fn=search_fn)
        findings = _analyze(llm, company, role, kind, 5, chunks, verify_fn)
        weak = [f for f in findings if _is_weak(f)]
        record = {"attempt": attempt, "tool": "search_evidence", "query": query,
                  "evidence_chunks": len(chunks), "findings": len(findings),
                  "weak_findings": len(weak)}
        if not findings or not weak or attempt == max_attempts:
            record["decision"] = "accept" if findings and not weak else (
                "give up (max attempts)" if attempt == max_attempts else "no findings")
            attempts.append(record)
            break
        # REFLECT + self-correct: rewrite the query to chase better evidence
        new_query = _refine_query(llm, company, role, step["focus"], query,
                                  [f["title"] for f in weak])
        record["decision"] = "retry with refined query"
        record["refined_query"] = new_query
        attempts.append(record)
        query = new_query
    return findings, attempts


# ============================================================
#  ReAct AGENT — tool-using Q&A (for the dashboard "Ask the Agent").
# ============================================================
_REACT_PROMPT = """You are an autonomous strategy agent for {company}. Answer the user's
question. You may use these tools, one at a time:
{tools}

Think step by step. Respond with ONLY a JSON object, either to act:
  {{"thought": "why this step", "action": "<tool name>", "action_input": {{...}}}}
or, when you have enough evidence, to finish:
  {{"thought": "why I can answer now", "final": "the answer, citing evidence inline like [1]"}}

QUESTION: {question}

OBSERVATIONS SO FAR:
{scratchpad}"""


def agent_answer(question, max_steps=4, llm=None, search_fn=None):
    """Agent capability: TOOL USE + AUTONOMY. The model decides which tool to call
    next (and with what input) until it can answer — a ReAct loop, not a fixed
    retrieve->answer. Returns answer + citations + the reasoning trace."""
    llm = llm or _get_llm()
    scratch, cite_chunks, trace = [], [], []
    answer = None
    for step in range(1, max_steps + 1):
        prompt = _REACT_PROMPT.format(
            company=cfg.COMPANY["name"], tools=_tool_descriptions(), question=question,
            scratchpad="\n".join(scratch) or "(none yet)")
        try:
            decision = next((d for d in (_llm_json(llm, prompt) or []) if isinstance(d, dict)), {})
            thought = decision.get("thought", "")
            if decision.get("final"):
                answer = decision["final"]
                trace.append({"step": step, "thought": thought, "action": "final"})
                break
            action = decision.get("action")
            ainput = decision.get("action_input") or {}
            if isinstance(ainput, str):                         # model returned a bare string
                ainput = {"query": ainput}
            elif not isinstance(ainput, dict):                  # list / number / anything odd
                ainput = {}
            fn = TOOLS.get(action, (None,))[0]
            if not action:                          # model gave neither an action nor a final:
                if not cite_chunks:                 # search on the question rather than waste a turn
                    action, ainput, fn = "search_evidence", {"query": question}, TOOLS["search_evidence"][0]
                else:
                    break                           # we already have evidence -> go synthesize
            if not fn:                                          # unknown tool -> nudge and continue
                scratch.append(f"[{step}] tried unknown tool '{action}'. Available: {list(TOOLS)}")
                trace.append({"step": step, "thought": thought, "action": action, "error": "unknown tool"})
                continue
            if action == "search_evidence":
                obs, chunks = fn(ainput.get("query", question), sources=ainput.get("sources"),
                                 search_fn=search_fn)
                if chunks:
                    cite_chunks.extend(chunks[:6])
            else:
                obs, _ = fn()
            scratch.append(f"[{step}] thought: {thought}\n      action: {action}({json.dumps(ainput)})\n      observation: {obs[:600]}")
            trace.append({"step": step, "thought": thought, "action": action, "action_input": ainput})
        except Exception as ex:                                 # one bad step never kills the agent
            scratch.append(f"[{step}] step failed: {ex}")
            trace.append({"step": step, "error": str(ex)})
            continue

    if answer is None:                                     # ran out of steps -> answer from what we have
        prompt = (f"Using ONLY these observations, answer the question with inline [n] citations.\n\n"
                  f"QUESTION: {question}\n\nOBSERVATIONS:\n" + ("\n".join(scratch) or "(none)"))
        answer = _invoke(llm, prompt)
        trace.append({"step": "final", "action": "answer_from_context"})

    # de-dup citations by (source,url,title)
    seen, uniq = set(), []
    for c in cite_chunks:
        key = (c.get("source"), c.get("url"), (c.get("title") or "")[:60])
        if key not in seen:
            seen.add(key); uniq.append(c)
    return {"question": question, "answer": answer,
            "citations": _citations(uniq, range(1, len(uniq) + 1)),
            "reasoning": trace}


# ============================================================
#  ORCHESTRATOR — the agent runs the whole investigation.
# ============================================================
def run_agent(objective=None, search_fn=None, llm=None, verify_fn=verify, max_attempts=2):
    company = cfg.COMPANY["name"]
    objective = objective or (f"If you were the CEO of {company} today, what should you do next "
                              f"and why? Identify the key risks, opportunities, and trends.")
    search_fn = search_fn or repository.search
    llm = llm or _get_llm()

    print(f"  [AGENT] planning investigation for {company} ...")
    plan = plan_investigation(llm, company, objective)        # capability 1: plan

    buckets = {"risk": [], "opportunity": [], "trend": []}
    steps_trace = []
    for step in plan:                                         # capabilities 2-4: act/observe/reflect/loop
        print(f"  [AGENT] investigating: {step['focus']}  (lens={step['lens']})")
        findings, attempts = investigate(llm, company, step, verify_fn, search_fn, max_attempts)
        buckets[step["lens"]].extend(findings)
        steps_trace.append({"focus": step["focus"], "lens": step["lens"], "attempts": attempts,
                            "final_findings": len(findings),
                            "retries": sum(1 for a in attempts if a.get("decision", "").startswith("retry"))})

    print("  [AGENT] synthesising recommendations + briefing ...")
    opps, risks, trends = buckets["opportunity"], buckets["risk"], buckets["trend"]
    recs = _recommend(llm, company, opps, risks, trends, verify_fn)   # finalise (reused)
    briefing = _briefing(llm, company, opps, risks, trends, recs)

    report = {
        "company": company,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "briefing": briefing,
        "opportunities": opps, "risks": risks, "trends": trends,
        "recommendations": recs,
        # the agent's own record — proof of plan/act/reflect/retry behaviour
        "agent": {
            "objective": objective,
            "plan": plan,
            "steps": steps_trace,
            "tools": [{"name": n, "description": d} for n, (_, d) in TOOLS.items()],
            "total_retries": sum(s["retries"] for s in steps_trace),
        },
    }
    os.makedirs(os.path.dirname(cfg.INTEL_PATH), exist_ok=True)
    with open(cfg.INTEL_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  -> {len(opps)} opportunities, {len(risks)} risks, {len(trends)} trends, "
          f"{len(recs)} recommendations, {report['agent']['total_retries']} self-corrections "
          f"-> {cfg.INTEL_PATH}")
    return report


if __name__ == "__main__":
    run_agent()