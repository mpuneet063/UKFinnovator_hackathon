#!/usr/bin/env python3
"""
Orchestration script — runs the full pipeline in order.

Stages 01 and 03 require ANTHROPIC_API_KEY and will be skipped (with a
warning) if it's not set — the pipeline still runs end-to-end using the
seed evidence/narratives files (or, if those are also absent, with no
qualitative evidence/narratives at all, falling back to template text).

Usage:
    cd scripts
    python3 run_all.py              # full run
    python3 run_all.py --skip-ai    # skip stages 01 and 03 even if API key is set
"""

import os
import subprocess
import sys

SKIP_AI = "--skip-ai" in sys.argv


def run(script, *args):
    cmd = [sys.executable, script, *args]
    print(f"\n{'='*70}\n>>> {' '.join(cmd)}\n{'='*70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!!! {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY")) and not SKIP_AI

    # Stage 0: financial data (real data, not AI) — best-effort, non-fatal
    run("00_fetch_financial_data.py")

    # Stage 0b: build sector-level materiality/gap/consensus data (not AI)
    run("build_sector_data.py")

    # Stage 1: evidence retrieval (AI #1) — requires API key
    if has_key:
        run("01_fetch_evidence.py")
    else:
        print("\nSkipping 01_fetch_evidence.py (ANTHROPIC_API_KEY not set, or --skip-ai). "
              "Using existing ../data/evidence.json if present (run seed_evidence.py for a demo set).")
        if not os.path.exists("../data/evidence.json"):
            run("seed_evidence.py")

    # Stage 2: explainable model (AI #2 — ML model + SHAP)
    run("02_train_model.py")

    # Stage 3: grounded narratives (AI #3) — requires API key
    if has_key:
        run("03_generate_narratives.py")
    else:
        print("\nSkipping 03_generate_narratives.py (ANTHROPIC_API_KEY not set, or --skip-ai). "
              "Using existing ../data/narratives.json if present.")

    # Stage 4: merge everything
    run("04_merge_final.py")

    print("\nPipeline complete. Final output: ../data/final_dashboard_data.json")


if __name__ == "__main__":
    main()
