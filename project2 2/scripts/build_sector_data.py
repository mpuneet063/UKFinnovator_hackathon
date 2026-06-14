"""
Maxwell Data - Sustainable Investing Silver Bullet
Rebuilt pipeline, corrected per clarification:

  - All 26 ESG factors per company, missing values filled as 0.0
    (0 = not flagged as financially material for this company's
    SASB sub-industry, per the source dataset's own convention)
  - Per-sector heatmap matrices (companies x 26 factors)
  - A "Peer Group Average" row per sector = mean of NON-ZERO values
    per factor across the sector (closest match to the
    "Maxwell Data (Target)" row provided by the frontend team;
    swap in their exact formula if/when shared)
  - NO invented performance/confidence/trend scores. The only
    numbers used are:
        - materiality_weight   (company's own value, 0-1)
        - peer_avg_weight       (sector peer-group average, 0-1)
        - target_weight         (= peer_avg_weight for now)
  - 3-lens logic (auditor / corporate / investor) now operates on
    GAP = materiality_weight - peer_avg_weight, i.e. how unusually
    material this factor is for THIS company vs its sector peers.
  - Qualitative "explanation" text is a template referencing the
    gap direction/size; real evidence text (from articles/reports)
    should be inserted manually or via a separate research step
    (kept as a clearly-marked field per factor).

Run: python3 build_sector_data.py
Outputs:
  - sector_heatmaps.json   (per sector: companies x factors matrix + peer avg row)
  - sector_dashboard_data.json (per company: 3-lens scored factors)
"""

import json
import pandas as pd
import statistics

df = pd.read_csv("../data/ftse100_materiality.csv")
esg_cols = df.columns[5:].tolist()
df[esg_cols] = df[esg_cols].fillna(0.0)

GAP_HIGH = 0.05   # gap threshold: company materiality notably higher than peer avg
GAP_LOW = -0.05   # company materiality notably lower than peer avg


def peer_avg_nonzero(series):
    nz = series[series > 0]
    if len(nz) == 0:
        return 0.0
    return round(nz.mean(), 3)


# ---------------------------------------------------------------------------
# Build per-sector heatmap matrices
# ---------------------------------------------------------------------------
sector_heatmaps = {}
for sector, group in df.groupby("Industry"):
    companies = []
    for _, row in group.iterrows():
        n_material = sum(1 for f in esg_cols if row[f] > 0)
        companies.append({
            "company": row["Name"],
            "symbol": row["Symbol"],
            "values": {f: round(float(row[f]), 3) for f in esg_cols},
            "data_status": "industry_mapping_incomplete" if n_material == 0 else "ok"
        })

    # Exclude all-zero companies from the peer-group average so they
    # don't artificially drag every factor's average toward zero.
    group_for_avg = group[group[esg_cols].sum(axis=1) > 0]
    if len(group_for_avg) == 0:
        group_for_avg = group  # fallback: whole sector is unmapped

    peer_avg = {f: peer_avg_nonzero(group_for_avg[f]) for f in esg_cols}
    sector_heatmaps[sector] = {
        "sector": sector,
        "companies": companies,
        "peer_group_average": peer_avg,  # "target" row
        "n_companies": len(companies),
        "n_companies_in_peer_average": len(group_for_avg)
    }

with open("../data/sector_heatmaps.json", "w") as f:
    json.dump(sector_heatmaps, f, indent=2)

print(f"Saved sector_heatmaps.json — {len(sector_heatmaps)} sectors")


# ---------------------------------------------------------------------------
# 3-lens scoring based on GAP = company_weight - peer_avg_weight
# ---------------------------------------------------------------------------

def auditor_lens(materiality, gap):
    """
    Auditor cares about disclosure/litigation risk relative to sector norms.
    A factor that is unusually material for THIS company (positive gap)
    and not addressed = elevated disclosure risk.
    """
    if materiality == 0:
        return "no_flag", "Not flagged as financially material for this company's sub-industry."
    if gap >= GAP_HIGH:
        return "flag", f"This factor is notably more material for this company ({materiality:.2f}) than the sector average ({materiality-gap:.2f}) — elevated disclosure/assurance scrutiny likely if not addressed in reporting."
    if gap <= GAP_LOW:
        return "watch", f"This factor is less material for this company ({materiality:.2f}) than sector peers ({materiality-gap:.2f}) — lower immediate audit priority, but worth confirming this reflects genuine business model differences."
    return "no_flag", f"Materiality ({materiality:.2f}) is broadly in line with the sector average ({materiality-gap:.2f})."


def corporate_lens(materiality, gap):
    """
    Corporate cares about whether strategic attention matches sector-relative materiality.
    """
    if materiality == 0:
        return "no_flag", "Not a SASB-flagged material factor for this company's sub-industry — limited strategic urgency from a materiality standpoint."
    if gap >= GAP_HIGH:
        return "flag", "Above-sector-average materiality suggests this factor should be a differentiated strategic priority relative to peers, not treated as a generic sector-wide issue."
    if gap <= GAP_LOW:
        return "watch", "Below-sector-average materiality — strategic resourcing here may be disproportionate to its financial relevance for this company specifically, relative to peers."
    return "no_flag", "Sector-typical materiality — likely addressed as part of standard sector-wide sustainability strategy."


def investor_lens(materiality, gap, financial_note=None):
    """
    Investor cares about whether sector-relative materiality differences
    line up with real financial signals (share price / results).
    `financial_note` is a placeholder for real data to be supplied —
    when absent, the lens reports the materiality gap only, without
    inferring a financial outcome.
    """
    if materiality == 0:
        return "no_flag", "Not flagged as financially material for this company — no investor-relevant signal from this dataset alone."
    if gap >= GAP_HIGH:
        status = "flag"
        reason = f"Materiality for this factor is {gap:+.2f} above the sector average — investors assessing this company against sector peers would reasonably expect above-average exposure here to be reflected in disclosure and, if unaddressed, in risk pricing."
    elif gap <= GAP_LOW:
        status = "watch"
        reason = f"Materiality for this factor is {gap:+.2f} below the sector average — lower relative exposure versus peers on this factor."
    else:
        status, reason = "no_flag", "Materiality is sector-typical; no notable relative signal."

    if financial_note:
        reason += f" Financial context: {financial_note}"
    else:
        reason += " (No company-specific share price / results data linked yet — add via financial data step.)"

    return status, reason


def consensus_color(a, c, i):
    """
    'flag' = materiality notably ABOVE sector peer average (positive gap)
    'watch' = materiality notably BELOW sector peer average (negative gap)
    Both directions represent a distinctive (non-average) position and
    are counted as "notable" for consensus purposes; the sign of the
    underlying gap determines the direction shown separately.
    """
    notable = sum(1 for s in (a, c, i) if s in ("flag", "watch"))
    if notable == 3:
        return "green", "All three perspectives agree this factor's materiality is distinctly different from the sector peer average."
    if notable == 2:
        return "amber", "Two of three perspectives flag this factor as distinctive vs sector peers — worth investigating."
    return "red", "Materiality is in line with the sector peer average from this data alone — no distinctive company-specific signal."


def factor_direction(materiality, gap):
    if materiality == 0:
        return "not_material"
    if gap >= GAP_HIGH:
        return "above_peer_average"
    if gap <= GAP_LOW:
        return "below_peer_average"
    return "in_line_with_peers"


def build_decision(factor, color, materiality, gap):
    if materiality == 0:
        return f"Not material: {factor} — not flagged by SASB for this company's sub-industry; no action needed on this factor specifically."
    if color == "green":
        direction = "above" if gap > 0 else "below"
        return f"Prioritize: {factor} — financial materiality is {direction} the sector peer average by {abs(gap):.3f}, a distinctive position worth addressing directly with stakeholders."
    if color == "amber":
        direction = "above" if gap > 0 else "below"
        return f"Investigate: {factor} — financial materiality is {direction} the sector peer average by {abs(gap):.3f}; partial signal, worth a closer look."
    return f"Standard sector treatment: {factor} — financial materiality ({materiality:.3f}) is in line with the sector peer average ({materiality - gap:.3f}); address as part of routine sector-wide ESG reporting, no distinctive company-specific story here."


def build_estimate(factor, materiality, gap):
    if materiality == 0:
        return {
            "metric": "N/A",
            "direction": "Not a SASB-flagged material factor for this company; no estimate generated.",
            "basis": "Based on materiality weight only (0.00)."
        }
    if gap >= GAP_HIGH:
        direction = ("This factor represents above-sector-average financial materiality for this company. "
                      "If real financial data (share price, financing terms, recent results) is linked, "
                      "compare performance/disclosure on this factor against sector peers to assess whether "
                      "this exposure is being priced in.")
    elif gap <= GAP_LOW:
        direction = ("This factor represents below-sector-average financial materiality for this company. "
                      "Limited direct estimate basis from materiality weight alone.")
    else:
        direction = "Sector-typical materiality; no distinctive estimate basis from materiality weight alone."

    return {
        "metric": "Relative financial materiality vs sector (illustrative)",
        "direction": direction,
        "basis": f"Company materiality weight: {materiality:.3f}. Sector peer-group average: {materiality-gap:.3f}. Gap: {gap:+.3f}. "
                 f"Source: FTSE100 financial materiality dataset (March 2025). No performance/confidence scores are inferred."
    }


def build_explanation(company, factor, materiality, gap, peer_avg):
    return {
        "auditor_view": (
            f"Auditor perspective: {company}'s SASB financial materiality weight for '{factor}' is {materiality:.3f}, "
            f"versus a sector peer-group average (non-zero entries) of {peer_avg:.3f} (gap {gap:+.3f}). "
            + ("This is materially above the sector norm." if gap >= GAP_HIGH else
               "This is materially below the sector norm." if gap <= GAP_LOW else
               "This is in line with the sector norm.")
            + " [Insert qualitative evidence from sustainability reports / news articles here.]"
        ),
        "corporate_view": (
            f"Corporate perspective: relative to sector peers, '{factor}' carries "
            + ("above-average" if gap >= GAP_HIGH else "below-average" if gap <= GAP_LOW else "average")
            + f" financial materiality for {company} (weight {materiality:.3f} vs peer average {peer_avg:.3f}). "
            "[Insert qualitative evidence on company's stated strategic priorities here.]"
        ),
        "investor_view": (
            f"Investor perspective: materiality gap of {gap:+.3f} vs sector peers on '{factor}'. "
            "[Link real share price / financial results data here to complete this view.]"
        ),
    }


# ---------------------------------------------------------------------------
# Build per-company scored output (all 26 factors, including zeros)
# ---------------------------------------------------------------------------
output = []
for sector, sdata in sector_heatmaps.items():
    peer_avg = sdata["peer_group_average"]
    for c in sdata["companies"]:
        company_out = {
            "company": c["company"],
            "symbol": c["symbol"],
            "sector": sector,
            "data_status": c["data_status"],
            "factors": []
        }

        if c["data_status"] == "industry_mapping_incomplete":
            company_out["note"] = (
                "No SASB financial materiality factors are mapped for this company in the source "
                "dataset (industry classification may need refinement). Excluded from this sector's "
                "peer-group average. Materiality-based scoring is not available until this is resolved; "
                "no values have been invented or borrowed from peers."
            )
            output.append(company_out)
            continue

        for factor in esg_cols:
            materiality = c["values"][factor]
            pavg = peer_avg[factor]
            gap = round(materiality - pavg, 3)

            a_status, a_reason = auditor_lens(materiality, gap)
            c_status, c_reason = corporate_lens(materiality, gap)
            i_status, i_reason = investor_lens(materiality, gap)
            color, color_reason = consensus_color(a_status, c_status, i_status)
            direction = factor_direction(materiality, gap)

            company_out["factors"].append({
                "factor": factor,
                "materiality_weight": materiality,
                "peer_group_average": pavg,
                "gap": gap,
                "direction": direction,
                "lenses": {
                    "auditor": {"status": a_status, "reason": a_reason},
                    "corporate": {"status": c_status, "reason": c_reason},
                    "investor": {"status": i_status, "reason": i_reason},
                },
                "consensus_color": color,
                "consensus_reason": color_reason,
                "decision": build_decision(factor, color, materiality, gap),
                "estimate": build_estimate(factor, materiality, gap),
                "explanation": build_explanation(c["company"], factor, materiality, gap, pavg),
            })
        output.append(company_out)

with open("../data/sector_dashboard_data.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved sector_dashboard_data.json — {len(output)} companies")

# Quick summary for Financial services
print("\n=== Financial services sample ===")
for c in output:
    if c["sector"] == "Financial services":
        print(f"\n{c['company']}")
        for f in c["factors"]:
            if f["materiality_weight"] > 0:
                print(f"   [{f['consensus_color'].upper():5}] {f['factor']:42} w={f['materiality_weight']:.3f} peer_avg={f['peer_group_average']:.3f} gap={f['gap']:+.3f}")
