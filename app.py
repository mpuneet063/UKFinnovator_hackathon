"""
Streamlit Dummy Frontend - ESG Alignment QA Viewer
====================================================

Minimal test UI to check scoring_csv.py output is correct.
NOT the production frontend - just a quick visual sanity check.

Setup:
    pip install streamlit pandas

Run:
    1. Place top3_industries.csv in this same folder
       (export top3_industries.xlsx -> CSV via Excel/Numbers/Google Sheets)
    2. python3 scoring_csv.py        # generates esg_alignment_output.json
    3. streamlit run app.py          # opens browser at localhost:8501
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
OUTPUT_FILE = BASE_DIR / "esg_alignment_output.json"
SCORING_SCRIPT = BASE_DIR / "scoring_csv.py"
CSV_FILE = BASE_DIR / "top3_industries.csv"

st.set_page_config(page_title="ESG Alignment - QA Viewer", layout="wide")

st.title("ESG Alignment Output — QA Viewer")
st.caption(
    "Dummy frontend to sanity-check `scoring_csv.py`. "
    "Not the production dashboard — just verifies the backend logic runs and "
    "the flags/z-scores look sensible before wiring into the real frontend."
)

# --- Controls -----------------------------------------------------------
col1, col2 = st.columns([1, 3])

with col1:
    if st.button("Run scoring_csv.py"):
        if not CSV_FILE.exists():
            st.error(
                f"`{CSV_FILE.name}` not found in {BASE_DIR}. "
                "Export top3_industries.xlsx to CSV and place it here."
            )
        else:
            result = subprocess.run(
                [sys.executable, str(SCORING_SCRIPT)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                st.success("Scoring ran successfully.")
                st.code(result.stdout, language="text")
            else:
                st.error("Scoring script failed:")
                st.code(result.stderr, language="text")

with col2:
    st.write("")  # spacer

if not OUTPUT_FILE.exists():
    st.warning(
        f"`{OUTPUT_FILE.name}` not found yet. Click 'Run scoring_csv.py' above "
        "(make sure top3_industries.csv is in this folder first)."
    )
    st.stop()

with open(OUTPUT_FILE) as f:
    data = json.load(f)

# --- Legend ---------------------------------------------------------------
st.markdown(
    """
**Flag legend:**
🟢 GREEN — within 1 SD of peer median (aligned) &nbsp;&nbsp;
🟠 ORANGE — 1–2 SD from peer median &nbsp;&nbsp;
🔴 RED — more than 2 SD from peer median &nbsp;&nbsp;
⬜ NO DATA — company doesn't report this metric
"""
)

FLAG_EMOJI = {"GREEN": "🟢", "ORANGE": "🟠", "RED": "🔴", "NO_DATA": "⬜"}

# --- Per-industry tabs ------------------------------------------------------
industry_names = list(data.keys())
tabs = st.tabs(industry_names)

for tab, industry in zip(tabs, industry_names):
    with tab:
        ind = data[industry]
        st.subheader(f"{industry} — peer group size: {ind['peer_group_size']}")

        benchmark = ind["materiality_benchmark"]
        metric_names = [b["metric"] for b in benchmark]

        with st.expander("Materiality benchmark (top 8 by peer coverage)"):
            bench_df = pd.DataFrame(benchmark)
            bench_df.columns = ["Metric", "Peer coverage (non-null count)"]
            st.dataframe(bench_df, use_container_width=True, hide_index=True)

        # Build flag grid
        rows = []
        for comp in ind["companies"]:
            row = {"Symbol": comp["symbol"], "Name": comp["name"]}
            for m in metric_names:
                md = comp["metrics"].get(m, {"flag": "NO_DATA"})
                row[m] = FLAG_EMOJI.get(md["flag"], "⬜")
            rows.append(row)

        st.markdown("**Alignment flags vs industry peers**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Detail view: pick a company to see raw numbers
        st.markdown("**Drill into a company**")
        symbols = [c["symbol"] for c in ind["companies"]]
        chosen = st.selectbox("Company", symbols, key=f"select_{industry}")

        comp = next(c for c in ind["companies"] if c["symbol"] == chosen)
        detail_rows = []
        for m in metric_names:
            md = comp["metrics"].get(m, {})
            detail_rows.append({
                "Metric": m,
                "Value": md.get("value"),
                "Peer median": md.get("peer_median"),
                "Peer std": md.get("peer_std"),
                "Z-score": md.get("z_score"),
                "Flag": md.get("flag", "NO_DATA"),
                "Direction": md.get("direction"),
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
