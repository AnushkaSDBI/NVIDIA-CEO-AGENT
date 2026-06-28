# ============================================================
#  src/agent_graph.py  —  LangGraph agent (the required workflow)
#
#  Implements the professor's agent workflow as an explicit graph:
#
#     Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate
#                         ^___________________|
#                         (reflect & retry: rewrite the query when the
#                          evidence is weak or cannot be verified)
#
#  Each arrow is a NODE in a LangGraph StateGraph; the loop back from
#  Decide -> Retrieve is a CONDITIONAL edge (the agent's autonomous
#  decision). The heavy lifting reuses the existing functions — the
#  graph supplies the agentic CONTROL FLOW (plan / decide / loop),
#  which is exactly the "agent behaviour" the assignment asks for.
#
#  Run:  python -m src.agent_graph
#  Use:  from src.agent_graph import run_agent_graph, build_graph
# ============================================================

import os
import json
from datetime import datetime
from typing import TypedDict

import config as cfg
from . import repository
from .intelligence import _analyze, _recommend, _briefing, verify, _get_llm
from .agent import (
    LENS, TOOLS, tool_search_evidence, plan_investigation, _refine_query, _is_weak,
)

WORKFLOW = ["Goal", "Plan", "Retrieve", "Analyze", "Decide", "Recommend", "Validate"]


# ---------------- graph state ----------------
class GState(TypedDict, total=False):
    objective: str
    company: str
    plan: list            # [{focus, lens, query}]
    step_index: int       # which plan step we're on
    attempt: int          # attempt number on the current step
    buckets: dict         # {risk:[], opportunity:[], trend:[]}
    steps_trace: list     # per-step record of attempts (for the dashboard)
    cur_attempts: list    # attempts log for the current step
    chunks: list          # evidence retrieved for the current step
    findings: list        # findings from the current step
    route: str            # decide -> "retrieve" | "recommend"
    recommendations: list
    briefing: str
    report: dict          # final assembled output


# ============================================================
#  NODES — one per stage of the required workflow.
# ============================================================
def build_graph(llm=None, search_fn=None, verify_fn=verify, max_attempts=2):
    """Compile the LangGraph agent. llm/search_fn/verify_fn are injectable for testing;
    they default to the real local model + retriever + NLI verifier."""
    from langgraph.graph import StateGraph, END

    llm = llm or _get_llm()
    search_fn = search_fn or repository.search
    company = cfg.COMPANY["name"]

    # --- PLAN: the agent decomposes the goal into focus areas (planning) ---
    def plan_node(state: GState) -> GState:
        plan = plan_investigation(llm, company, state["objective"])
        print(f"  [GRAPH] plan: {len(plan)} steps")
        return {"plan": plan, "step_index": 0, "attempt": 1,
                "buckets": {"risk": [], "opportunity": [], "trend": []},
                "steps_trace": [], "cur_attempts": []}

    # --- RETRIEVE: tool use — pull source-balanced evidence for this step ---
    def retrieve_node(state: GState) -> GState:
        step = state["plan"][state["step_index"]]
        _, _, sources = LENS[step["lens"]]
        _, chunks = tool_search_evidence(step["query"], sources=sources, search_fn=search_fn)
        print(f"  [GRAPH] retrieve '{step['focus']}' (attempt {state['attempt']}): {len(chunks)} chunks")
        return {"chunks": chunks}

    # --- ANALYZE: extract findings (risks/opportunities/trends) from evidence ---
    def analyze_node(state: GState) -> GState:
        step = state["plan"][state["step_index"]]
        role, kind, _ = LENS[step["lens"]]
        findings = _analyze(llm, company, role, kind, 5, state["chunks"], verify_fn)
        return {"findings": findings}

    # --- DECIDE: autonomous decision — accept, or reflect & retry, or advance ---
    def decide_node(state: GState) -> GState:
        step = state["plan"][state["step_index"]]
        findings = state["findings"]
        weak = [f for f in findings if _is_weak(f)]
        rec = {"attempt": state["attempt"], "query": step["query"],
               "evidence_chunks": len(state["chunks"]), "findings": len(findings),
               "weak_findings": len(weak)}
        cur = state.get("cur_attempts", []) + [rec]

        # reflect & retry: evidence weak AND attempts remain -> rewrite the query, loop back
        if findings and weak and state["attempt"] < max_attempts:
            role, _, _ = LENS[step["lens"]]
            refined = _refine_query(llm, company, role, step["focus"], step["query"],
                                    [f["title"] for f in weak])
            rec["decision"] = "retry with refined query"
            rec["refined_query"] = refined
            new_plan = [dict(s) for s in state["plan"]]
            new_plan[state["step_index"]]["query"] = refined
            print(f"  [GRAPH] decide -> RETRY ({len(weak)} weak findings) — rewriting query")
            return {"cur_attempts": cur, "attempt": state["attempt"] + 1,
                    "plan": new_plan, "route": "retrieve"}

        # accept: file the findings, advance to the next step (or move to recommend)
        rec["decision"] = "accept" if findings and not weak else (
            "give up (max attempts)" if findings else "no findings")
        buckets = {k: list(v) for k, v in state["buckets"].items()}
        buckets[step["lens"]].extend(findings)
        trace = state.get("steps_trace", []) + [{
            "focus": step["focus"], "lens": step["lens"], "attempts": cur,
            "final_findings": len(findings),
            "retries": sum(1 for a in cur if a.get("decision", "").startswith("retry"))}]
        nxt = state["step_index"] + 1
        route = "retrieve" if nxt < len(state["plan"]) else "recommend"
        print(f"  [GRAPH] decide -> {rec['decision'].upper()}; next: {route}")
        return {"buckets": buckets, "steps_trace": trace, "cur_attempts": [],
                "step_index": nxt, "attempt": 1, "route": route}

    # --- RECOMMEND: synthesize one action per ranked finding ---
    def recommend_node(state: GState) -> GState:
        b = state["buckets"]
        recs = _recommend(llm, company, b["opportunity"], b["risk"], b["trend"], verify_fn)
        print(f"  [GRAPH] recommend: {len(recs)} recommendations")
        return {"recommendations": recs}

    # --- VALIDATE: gate recommendations + write the briefing & output ---
    def validate_node(state: GState) -> GState:
        b = state["buckets"]
        recs = state["recommendations"]
        n_verified = sum(1 for r in recs if r.get("verified"))
        n_supported = sum(1 for r in recs if r.get("well_supported"))
        briefing = _briefing(llm, company, b["opportunity"], b["risk"], b["trend"], recs)
        report = {
            "company": company,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "briefing": briefing,
            "opportunities": b["opportunity"], "risks": b["risk"], "trends": b["trend"],
            "recommendations": recs,
            "agent": {
                "framework": "LangGraph StateGraph",
                "workflow": WORKFLOW,
                "objective": state["objective"],
                "plan": state["plan"],
                "steps": state["steps_trace"],
                "tools": [{"name": n, "description": d} for n, (_, d) in TOOLS.items()],
                "total_retries": sum(s["retries"] for s in state["steps_trace"]),
                "validation": {"recommendations": len(recs),
                               "verified": n_verified, "well_supported": n_supported},
            },
        }
        os.makedirs(os.path.dirname(cfg.INTEL_PATH), exist_ok=True)
        with open(cfg.INTEL_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  [GRAPH] validate -> {n_supported}/{len(recs)} well-supported -> {cfg.INTEL_PATH}")
        return {"briefing": briefing, "report": report}

    # --- wire the graph ---
    g = StateGraph(GState)
    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("analyze", analyze_node)
    g.add_node("decide", decide_node)
    g.add_node("recommend", recommend_node)
    g.add_node("validate", validate_node)

    g.set_entry_point("plan")                       # Goal -> Plan
    g.add_edge("plan", "retrieve")                  # Plan -> Retrieve
    g.add_edge("retrieve", "analyze")               # Retrieve -> Analyze
    g.add_edge("analyze", "decide")                 # Analyze -> Decide
    g.add_conditional_edges("decide", lambda s: s["route"],   # Decide -> (loop | go on)
                            {"retrieve": "retrieve", "recommend": "recommend"})
    g.add_edge("recommend", "validate")             # Recommend -> Validate
    g.add_edge("validate", END)
    return g.compile()


def run_agent_graph(objective=None, search_fn=None, llm=None, verify_fn=verify, max_attempts=2):
    """Build + invoke the LangGraph agent. Writes intelligence.json (same schema the
    dashboard reads) plus an 'agent' block recording the plan, the per-step retries,
    and the validation summary. Falls back to the plain-Python agent if LangGraph
    is unavailable (e.g. not installed on a lab machine)."""
    company = cfg.COMPANY["name"]
    objective = objective or (f"If you were the CEO of {company} today, what should you do next "
                              f"and why? Identify the key risks, opportunities, and trends.")
    try:
        graph = build_graph(llm=llm, search_fn=search_fn, verify_fn=verify_fn, max_attempts=max_attempts)
    except Exception as ex:
        print(f"  [GRAPH] LangGraph unavailable ({ex}); falling back to plain-Python agent.")
        from .agent import run_agent
        return run_agent(objective, search_fn, llm, verify_fn, max_attempts)

    # recursion budget: every step can loop up to max_attempts, ~3 nodes per loop
    n_steps = 5
    limit = n_steps * max_attempts * 3 + 12
    state = graph.invoke({"objective": objective, "company": company},
                         {"recursion_limit": limit})
    return state.get("report", {})


if __name__ == "__main__":
    run_agent_graph()