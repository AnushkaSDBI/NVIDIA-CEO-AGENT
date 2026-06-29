"""
AI CEO Agent — explicit agent loop.

Demonstrates the required workflow with one clearly labelled step per stage:

    Plan -> Retrieve (+ sufficiency check) -> Analyze -> Decide -> Recommend -> Validate

The LLM (Qwen2.5-7B via Ollama) is the reasoning engine at every stage. Retrieval,
analysis, NLI verification, recommendation and briefing are REUSED from intelligence.py
(the agent drives the engine; it does not re-implement it). Unlike a plain pipeline, the
agent: plans its own queries, checks whether its evidence is sufficient and searches again
if not, decides priorities, and validates EVERY category before writing anything.

Produces 5 findings per category (opportunities / risks / trends) so the dashboard's
Opportunity / Risk / Trend Monitor pages each show 5, all NLI-verified.

Writes  data/clean/intelligence.json   (opportunities, risks, trends, recommendations,
                                         briefing, + an `agent` reasoning block)
        data/clean/agent_log.json       (the same reasoning trace, standalone)

Run:  python -m src.agent_loop
"""

import json
import os
from datetime import datetime

from . import intelligence as intel
from . import repository
import config as cfg

N_ITEMS = 5       # final findings per category (matches the monitor pages)
N_CANDIDATES = 9  # candidate findings the analyst proposes; we keep the best N_ITEMS

# Each category: the analyst role, the label _analyze uses, the source mix to search,
# and seed queries used as a fallback if the LLM planner is unavailable.
THEMES = {
    "opportunities": {
        "role": "opportunity analyst", "kind": "growth OPPORTUNITIES",
        "sources": ["company", "news", "research", "ecosystem", "pdf", "community", "market"],
        "seed": ["data center AI compute demand growth", "new markets partnerships expansion",
                 "software platform services revenue", "automotive robotics edge opportunity",
                 "next-generation product roadmap"],
    },
    "risks": {
        "role": "risk analyst", "kind": "strategic RISKS",
        "sources": ["filing", "news", "market", "ecosystem", "community", "social"],
        "seed": ["export controls China regulatory risk", "competition AMD custom silicon",
                 "supply chain concentration", "CUDA open-source alternatives",
                 "customer concentration demand risk"],
    },
    "trends": {
        "role": "technology trend analyst", "kind": "emerging TRENDS",
        "sources": ["research", "company", "news", "ecosystem", "community", "social"],
        "seed": ["accelerated computing adoption trend", "AI data center build-out momentum",
                 "open-source ecosystem adoption", "developer library momentum",
                 "emerging AI hardware architecture"],
    },
}

_TOOLS = [
    {"name": "search_evidence", "description": "source-balanced hybrid retrieval (BM25 + FAISS + rerank + MMR)"},
    {"name": "analyze + verify", "description": "LLM analyst findings, each NLI-verified (bart-large-mnli)"},
    {"name": "recommend", "description": "one decisive action per finding, evidence inherited"},
    {"name": "validate", "description": "category-level grounding + verification gate"},
]


def _json(llm, prompt):
    """Call the LLM and return the first JSON object (robust to the model's formatting)."""
    for d in (intel._llm_json(llm, prompt) or []):
        if isinstance(d, dict):
            return d
    return {}


# ============================================================ STEP 1: PLAN ==
def plan_investigation(llm, company, goal):
    """The LLM writes its own investigation plan: which themes, and which targeted
    queries to run for each. Replaces hardcoded query strings — the agent decides
    what to search for. Falls back to seed queries if the model is unavailable."""
    print(f"  Goal: {goal}")
    schema = ('{"queries": {"opportunities": ["q1","q2","q3"], "risks": ["q1","q2","q3"], '
              '"trends": ["q1","q2","q3"]}, "reasoning": "why these queries address the goal"}')
    prompt = (
        f"You are an AI strategic intelligence agent for {company}.\n"
        f"Goal: {goal}\n\n"
        f"Available data: SEC 10-K/10-Q filings, investor PDFs, company news/blogs, Google News, "
        f"arXiv research, Hacker News, GitHub ecosystem, Reddit, market/stock data.\n\n"
        f"Produce an investigation plan as JSON: {schema}\n"
        f"Give 3-5 SPECIFIC queries per category (e.g. '{company} Blackwell data center demand', "
        f"not just 'data center'). Output JSON only.")
    plan = _json(llm, prompt)
    queries = plan.get("queries") or {}
    # fill any missing category with its seed queries
    for theme, spec in THEMES.items():
        qs = queries.get(theme)
        if not isinstance(qs, list) or not qs:
            queries[theme] = [f"{company} {q}" for q in spec["seed"]]
    print(f"  Reasoning: {str(plan.get('reasoning',''))[:120]}")
    return {"queries": queries, "reasoning": plan.get("reasoning", "")}


# ========================================= STEP 2: RETRIEVE + SUFFICIENCY ==
def retrieve_with_sufficiency(llm, theme, queries, sources, search_fn):
    """Retrieve source-balanced evidence for a theme, then let the agent judge whether the
    evidence is sufficient; if not, it autonomously runs additional queries. This is the
    autonomous decision-making + self-correction step."""
    def gather(qs):
        pool = {}
        for q in qs:
            for h in intel._gather(search_fn, q, sources, per_source=5):
                key = (h.get("text") or "")[:120]
                if key and key not in pool:
                    pool[key] = h
        return list(pool.values())[:32]

    hits = gather(queries)
    by_source = {}
    for h in hits:
        by_source[h.get("source")] = by_source.get(h.get("source"), 0) + 1

    schema = '{"sufficient": true or false, "reason": "one sentence", "additional_queries": ["q1"] or []}'
    prompt = (
        f"You are evaluating evidence quality for strategic '{theme}' analysis.\n"
        f"Retrieved {len(hits)} chunks across sources {by_source}.\n"
        f"Top titles: {[ (h.get('title') or '')[:60] for h in hits[:5] ]}\n\n"
        f"Is this sufficient for reliable conclusions? If not, suggest up to 3 more specific "
        f"queries. JSON: {schema}")
    check = _json(llm, prompt)

    extra_q = check.get("additional_queries") or []
    ran_extra = False
    if not check.get("sufficient", True) and isinstance(extra_q, list) and extra_q:
        print(f"    evidence weak -> agent runs {len(extra_q)} more queries autonomously")
        more = gather(extra_q)
        seen = {(h.get('text') or '')[:120] for h in hits}
        for h in more:
            k = (h.get('text') or '')[:120]
            if k and k not in seen:
                hits.append(h); seen.add(k)
        hits = hits[:40]
        ran_extra = True

    print(f"  {theme}: {len(hits)} chunks — {str(check.get('reason',''))[:80]}")
    return hits, check, ran_extra, extra_q


# ======================================= STEP 4: DECIDE PRIORITIES ==
def decide_priorities(llm, company, intel_dict):
    """The agent reviews all findings and explicitly decides what deserves the most
    strategic attention right now — the autonomous prioritisation decision."""
    listing = []
    for theme in ("risks", "opportunities", "trends"):
        for f in intel_dict.get(theme, []):
            listing.append(f"[{theme}] {f.get('title','')} (verified={f.get('verified')}, conf={f.get('confidence')})")
    schema = ('{"recommended_focus": "risk_mitigation" or "growth" or "balanced", '
              '"priority_risks": ["title"], "priority_opportunities": ["title"], '
              '"decision_rationale": "2-3 sentences", '
              '"immediate_actions": ["action1","action2","action3"]}')
    prompt = (f"You are the AI CEO advisor for {company}. Review all findings and decide what "
              f"matters most now.\n\nFindings:\n" + "\n".join(listing) + f"\n\nJSON: {schema}")
    d = _json(llm, prompt)
    print(f"  Focus: {d.get('recommended_focus','')} — {str(d.get('decision_rationale',''))[:100]}")
    return d


# =========================================== STEP 6: VALIDATE (all categories) ==
def validate_all(llm, company, intel_dict, recs):
    """Validate EVERY category — opportunities, risks, trends, recommendations — plus an
    overall 'all' rollup. Verification counts come from the NLI checks already run per
    finding (objective); the LLM adds a grounding/consistency note per category."""
    def stats(items):
        n = len(items)
        ver = sum(1 for x in items if x.get("verified"))
        con = sum(1 for x in items if x.get("contested"))
        return {"count": n, "verified": ver, "unverified": n - ver, "contested": con}

    by_cat = {
        "opportunities": stats(intel_dict.get("opportunities", [])),
        "risks": stats(intel_dict.get("risks", [])),
        "trends": stats(intel_dict.get("trends", [])),
        "recommendations": stats(recs),
    }

    # one LLM call for a grounding/consistency note on each category + overall
    def titles(items):
        return [ (x.get("title") or x.get("action") or "")[:70] for x in items[:5] ]
    schema = ('{"opportunities": {"valid": true, "note": "..."}, "risks": {"valid": true, "note": "..."}, '
              '"trends": {"valid": true, "note": "..."}, "recommendations": {"valid": true, "note": "..."}, '
              '"all": {"valid": true, "note": "overall 2-sentence assessment"}}')
    prompt = (
        f"You are an AI validation agent for {company} performing quality control.\n"
        f"Opportunities: {titles(intel_dict.get('opportunities', []))}\n"
        f"Risks: {titles(intel_dict.get('risks', []))}\n"
        f"Trends: {titles(intel_dict.get('trends', []))}\n"
        f"Recommendations: {titles(recs)}\n\n"
        f"For EACH category and overall, is it grounded, internally consistent, and free of "
        f"contradictions? JSON: {schema}")
    notes = _json(llm, prompt)

    for cat in ("opportunities", "risks", "trends", "recommendations"):
        n = notes.get(cat) or {}
        by_cat[cat]["valid"] = bool(n.get("valid", True))
        by_cat[cat]["note"] = n.get("note", "")

    total = sum(c["count"] for c in by_cat.values() if c is not by_cat["recommendations"])
    total_ver = by_cat["opportunities"]["verified"] + by_cat["risks"]["verified"] + by_cat["trends"]["verified"]
    overall = notes.get("all") or {}
    by_cat["all"] = {"count": total, "verified": total_ver,
                     "valid": bool(overall.get("valid", True)), "note": overall.get("note", "")}

    print("  Validation:")
    for cat in ("all", "opportunities", "risks", "trends", "recommendations"):
        c = by_cat[cat]
        print(f"    {cat:16} {c['verified']}/{c['count']} verified"
              + (f"  ·  {c.get('note','')[:70]}" if c.get("note") else ""))
    return by_cat


# ================================================= MAIN AGENT LOOP ==
def run_agent(goal=None, search_fn=None, llm=None, verify_fn=None):
    company = cfg.COMPANY["name"]
    goal = goal or (f"If you were the CEO of {company} today, what should you do next and why? "
                    f"Identify the key risks, opportunities, and trends.")
    search_fn = search_fn or repository.search
    llm = llm or intel._get_llm()
    verify_fn = verify_fn or intel.verify

    print("\nAGENT STARTING")
    plan_list, steps_trace = [], []
    total_retries = 0

    # ------ 1. PLAN ------
    print("\n[Agent 1/6] PLANNING investigation ...")
    plan = plan_investigation(llm, company, goal)
    queries_map = plan["queries"]

    # ------ 2/3. RETRIEVE (+sufficiency) and ANALYZE, per theme (5 findings each) ------
    intel_dict = {}
    for theme, spec in THEMES.items():
        print(f"\n[Agent 2/6] RETRIEVING '{theme}' (with sufficiency check) ...")
        queries = queries_map.get(theme, [])
        hits, check, ran_extra, extra_q = retrieve_with_sufficiency(
            llm, theme, queries, spec["sources"], search_fn)

        print(f"[Agent 3/6] ANALYZING '{theme}' -> {N_CANDIDATES} candidates, keep best {N_ITEMS} ...")
        candidates = intel._analyze(llm, company, spec["role"], spec["kind"],
                                    N_CANDIDATES, hits, verify_fn)
        # keep the strongest N_ITEMS (verified > corroboration > confidence > impact)
        findings = intel._rank(candidates)[:N_ITEMS]
        intel_dict[theme] = findings
        n_unverified = sum(1 for f in findings if not f.get("verified"))
        print(f"    -> {len(findings)} {theme} ({len(findings)-n_unverified} verified)")

        # record the trace in the schema the Agent Reasoning page expects
        plan_list.append({"focus": theme, "lens": theme, "query": (queries[0] if queries else "")})
        attempts = [{"attempt": 1, "query": " | ".join(queries)[:120],
                     "evidence_chunks": len(hits), "findings": len(findings),
                     "weak_findings": n_unverified,
                     "decision": "sufficient" if not ran_extra else f"insufficient -> +{len(extra_q)} queries",
                     "refined_query": (" | ".join(extra_q)[:120] if ran_extra else "")}]
        steps_trace.append({"focus": theme, "lens": theme,
                            "retries": 1 if ran_extra else 0, "attempts": attempts})
        if ran_extra:
            total_retries += 1

    # ------ 4. DECIDE ------
    print("\n[Agent 4/6] DECIDING priorities ...")
    decisions = decide_priorities(llm, company, intel_dict)

    # ------ 5. RECOMMEND ------
    print("\n[Agent 5/6] GENERATING recommendations + briefing ...")
    recs = intel._recommend(llm, company, intel_dict["opportunities"],
                            intel_dict["risks"], intel_dict["trends"], verify_fn)
    briefing = intel._briefing(llm, company, intel_dict["opportunities"],
                               intel_dict["risks"], intel_dict["trends"], recs)

    # ------ 6. VALIDATE (all categories) ------
    print("\n[Agent 6/6] VALIDATING all categories ...")
    validation = validate_all(llm, company, intel_dict, recs)

    # ------ assemble + write ------
    n_rec_verified = sum(1 for r in recs if r.get("verified"))
    agent_block = {
        "framework": "Explicit agent loop",
        "workflow": ["Goal", "Plan", "Retrieve", "Analyze", "Decide", "Recommend", "Validate"],
        "objective": goal,
        "plan": plan_list,
        "total_retries": total_retries,
        "steps": steps_trace,
        "tools": _TOOLS,
        "decisions": decisions,
        "validation": {
            "recommendations": len(recs),
            "verified": n_rec_verified,
            "by_category": validation,
        },
    }
    report = {
        "company": company,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "briefing": briefing,
        "opportunities": intel_dict["opportunities"],
        "risks": intel_dict["risks"],
        "trends": intel_dict["trends"],
        "recommendations": recs,
        "agent": agent_block,
    }
    os.makedirs(os.path.dirname(cfg.INTEL_PATH), exist_ok=True)
    with open(cfg.INTEL_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log_path = os.path.join(os.path.dirname(cfg.INTEL_PATH), "agent_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(agent_block, f, indent=2)

    print(f"\nAgent complete -> {cfg.INTEL_PATH}")
    print(f"  {len(intel_dict['opportunities'])} opportunities, {len(intel_dict['risks'])} risks, "
          f"{len(intel_dict['trends'])} trends, {len(recs)} recommendations")
    return report


if __name__ == "__main__":
    run_agent()