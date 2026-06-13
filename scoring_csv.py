"""
Deterministic ESG Materiality & Alignment Scoring Engine
=========================================================

NO AI involved - pure statistical/deterministic logic.
Reads from a CSV (export top3_industries.xlsx -> top3_industries.csv
via Excel/Numbers/Google Sheets: File > Download/Save As > CSV).
Uses only the Python standard library - no openpyxl/pandas required.

Pipeline
--------
1. MATERIALITY BENCHMARK (per industry / peer group)
   - For each SASB metric column, count non-empty values across all
     companies in that industry.
   - Rank columns by non-empty count (descending) = "most common indices".
   - Top N (default 8) columns = the industry's materiality benchmark set.
     (Highest data coverage => the metrics the peer group actually
      reports/discloses on => most material for that sector.)

2. ALIGNMENT SCORING (per company, vs its industry peers)
   For each benchmark metric (from step 1) that the company has a value for:
   - Compute peer-group median and standard deviation (using all
     companies in that industry that report the metric).
   - Compute company's deviation from peer median, in units of SD:
         z = (value - median) / std
   - 3-tier flag:
         GREEN  : |z| <= 1   (within 1 SD of peer median -> aligned)
         ORANGE : 1 < |z| <= 2 (moderately above/below peer norm)
         RED    : |z| > 2     (significantly above/below peer norm)
     (If std == 0, i.e. peer group is uniform, flag GREEN if value == median
      else RED.)

3. OUTPUT
   JSON structure ready to feed a frontend (per industry -> per company ->
   per metric: value, peer_median, peer_std, z_score, flag, direction).
   Frontend rendering/UI is intentionally NOT built here - this is the
   backend scoring/data layer only.
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "top3_industries.csv"
OUTPUT_FILE = Path(__file__).parent / "esg_alignment_output.json"

TOP_N_BENCHMARK_METRICS = 8

# Metadata columns that are NOT SASB metrics
META_COLS = {"Entity ID", "Name", "Symbol", "Wikipedia", "Industry"}


def load_data(path: Path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]

    # First column may be an unnamed index column (empty header) - drop it
    if header[0].strip() == "":
        header = header[1:]
        rows = [r[1:] for r in rows]

    records = []
    for r in rows[1:]:
        if not any(cell.strip() for cell in r):
            continue  # skip blank rows
        rec = {}
        for col_name, val in zip(header, r):
            val = val.strip()
            if val == "":
                rec[col_name] = None
            elif col_name in META_COLS:
                rec[col_name] = val
            else:
                try:
                    rec[col_name] = float(val)
                except ValueError:
                    rec[col_name] = None
        records.append(rec)

    metric_cols = [c for c in header if c not in META_COLS]
    return records, metric_cols


def group_by_industry(records):
    industries = defaultdict(list)
    for rec in records:
        industries[rec["Industry"]].append(rec)
    return industries


def compute_materiality_benchmark(companies, metric_cols, top_n=TOP_N_BENCHMARK_METRICS):
    """
    Rank metrics by non-null coverage across the peer group.
    Returns ordered list of (metric_name, coverage_count), top_n only.
    """
    coverage = []
    for m in metric_cols:
        count = sum(1 for c in companies if c.get(m) is not None)
        coverage.append((m, count))

    # Sort by coverage desc, then alphabetically for stable tie-break
    coverage.sort(key=lambda x: (-x[1], x[0]))

    # Drop metrics with zero coverage
    coverage = [c for c in coverage if c[1] > 0]

    return coverage[:top_n]


def flag_for_zscore(z):
    """3-tier flag based on |z| (deviation from peer median in SD units)."""
    az = abs(z)
    if az <= 1:
        return "GREEN"
    elif az <= 2:
        return "ORANGE"
    else:
        return "RED"


def direction_for_zscore(z):
    if z > 0:
        return "above_peer_norm"
    elif z < 0:
        return "below_peer_norm"
    else:
        return "at_peer_median"


def compute_alignment(companies, benchmark_metrics):
    """
    For each company, score it against the industry benchmark metrics
    using peer median/std from the same industry's companies.
    """
    metric_names = [m for m, _ in benchmark_metrics]

    # Pre-compute peer stats (median, std) per metric across companies
    # that actually report it.
    peer_stats = {}
    for m in metric_names:
        values = [c[m] for c in companies if c.get(m) is not None]
        if len(values) == 0:
            continue
        median = statistics.median(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        peer_stats[m] = {"median": median, "std": std, "n": len(values)}

    results = []
    for c in companies:
        company_result = {
            "name": c.get("Name"),
            "symbol": c.get("Symbol"),
            "entity_id": c.get("Entity ID"),
            "metrics": {},
        }
        for m in metric_names:
            value = c.get(m)
            if value is None or m not in peer_stats:
                company_result["metrics"][m] = {
                    "value": None,
                    "peer_median": peer_stats.get(m, {}).get("median"),
                    "peer_std": peer_stats.get(m, {}).get("std"),
                    "z_score": None,
                    "flag": "NO_DATA",
                    "direction": None,
                }
                continue

            median = peer_stats[m]["median"]
            std = peer_stats[m]["std"]

            if std == 0:
                z = 0.0 if value == median else (2.01 if value > median else -2.01)
            else:
                z = (value - median) / std

            company_result["metrics"][m] = {
                "value": value,
                "peer_median": median,
                "peer_std": std,
                "z_score": round(z, 3),
                "flag": flag_for_zscore(z),
                "direction": direction_for_zscore(z),
            }

        results.append(company_result)

    return results


def run():
    records, metric_cols = load_data(INPUT_FILE)
    industries = group_by_industry(records)

    output = {}

    for industry, companies in industries.items():
        benchmark = compute_materiality_benchmark(companies, metric_cols)

        alignment = compute_alignment(companies, benchmark)

        output[industry] = {
            "peer_group_size": len(companies),
            "materiality_benchmark": [
                {"metric": m, "peer_coverage_count": cnt}
                for m, cnt in benchmark
            ],
            "companies": alignment,
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {OUTPUT_FILE}")

    # Quick console summary
    for industry, data in output.items():
        print(f"\n=== {industry} (n={data['peer_group_size']}) ===")
        print("Benchmark metrics:", [m['metric'] for m in data['materiality_benchmark']])
        for comp in data["companies"]:
            flags = {m: v["flag"] for m, v in comp["metrics"].items() if v["flag"] != "NO_DATA"}
            print(f"  {comp['symbol']:<6} -> {flags}")


if __name__ == "__main__":
    run()
