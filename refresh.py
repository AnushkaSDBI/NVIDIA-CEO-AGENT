#!/usr/bin/env python
"""
refresh.py  -  one-command pre-demo prep.

Runs the full offline pipeline in order so the LIVE DEMO only needs to launch
the dashboard. Everything heavy (the 12-min index build, the multi-call CEO
analysis) is pre-computed here and cached to disk:

    collect  ->  preprocess  ->  repository  ->  sentiment  ->  entities  ->  intelligence

Usage:
    python refresh.py                 # FULL rebuild  -> run this before the demo
    python refresh.py --quick         # fast data refresh only (collect --quick); no reindex/analysis
    python refresh.py --skip-collect  # keep existing data; rebuild index + analysis
    python refresh.py --skip-index    # corpus unchanged; just redo sentiment/entities/intelligence

Notes:
  * Ollama must be running for the 'intelligence' stage (qwen2.5:7b).
  * A failed optional stage (sentiment/entities) won't stop the run.
  * Stages run as isolated subprocesses, exactly like running them by hand.
"""
import sys
import time
import subprocess

PY = sys.executable


def stage(name, module_args, optional=False):
    print("\n" + "=" * 64)
    print(f"  >> {name}")
    print("=" * 64)
    t = time.time()
    rc = subprocess.run([PY, "-m"] + module_args).returncode
    dt = time.time() - t
    if rc == 0:
        print(f"  [OK]   {name}  ({dt:.0f}s)")
        return True
    print(f"  [FAIL] {name}  (exit {rc}, after {dt:.0f}s)")
    if not optional:
        print("  Stopping - fix the error above and re-run.")
        sys.exit(rc)
    print("  (optional stage - continuing)")
    return False


def main():
    args = sys.argv[1:]
    quick = "--quick" in args
    skip_collect = "--skip-collect" in args
    skip_index = "--skip-index" in args

    t0 = time.time()
    print("AI CEO  -  pipeline refresh")

    # Fast path: just accumulate fresh data, nothing heavy.
    if quick:
        stage("Collect (quick: news/company/market/stock)", ["src.collect", "--quick"])
        print("\nQuick refresh done. Data accumulated in the DB.")
        print("NOTE: retrieval index and CEO analysis were NOT rebuilt - run a full")
        print("      'python refresh.py' to fold new data into search + analysis.")
        return

    if not skip_collect:
        stage("Collect (full: all sources)", ["src.collect"])
    else:
        print("\n(skipping collect - using existing data)")

    stage("Preprocess (chunk + spaCy lemmatize)", ["src.preprocess"])

    if not skip_index:
        stage("Repository (BM25 + FAISS index)  [the slow ~12 min step]", ["src.repository"])
    else:
        print("\n(skipping index rebuild - corpus assumed unchanged)")

    stage("Sentiment (FinBERT)", ["src.sentiment"], optional=True)
    stage("Entities (competitor NER)", ["src.entities"], optional=True)
    stage("Intelligence (CEO analysis -> intelligence.json)", ["src.intelligence"])
    stage("Keywords (classical TF-IDF terms)", ["src.keywords"], optional=True)
    stage("Evaluation (retrieval ablation table)", ["src.evaluate"], optional=True)
    stage("Daily brief (what changed)", ["src.daily_brief"], optional=True)

    print("\n" + "=" * 64)
    print(f"  ALL DONE in {time.time() - t0:.0f}s  -  ready for the demo.")
    print("  Launch the dashboard; every heavy artifact is cached on disk.")
    print("=" * 64)


if __name__ == "__main__":
    main()