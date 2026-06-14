"""
STAGE 4 — Merge all stages into final dashboard data (NOT AI; assembly only)
==============================================================================

WHAT THIS DOES
--------------
Combines outputs from all previous stages into a single
`final_dashboard_data.json` consumed by the frontend:

  - sector_dashboard_data.json   (deterministic materiality/gap/consensus)
  - model_explanations.json      (Stage 2 — SHAP attributions)
  - evidence.json                (Stage 1 — retrieved evidence, if present)
  - narratives.json              (Stage 3 — LLM narratives, if present)

For each (company, factor) where materiality > 0, the output row includes:
  - materiality_weight, peer_group_average, gap, consensus_color (as before)
  - shap_value, shap_rank, target_type (from the model)
  - evidence (if retrieved) with direction/confidence/sources
  - narrative (auditor/corporate/investor views + key_uncertainties), if
    generated; otherwise the template-based explanation from
    sector_dashboard_data.json is used as a fallback so the dashboard
    always has SOMETHING to show.

HOW TO RUN
----------
    python3 04_merge_final.py

OUTPUT
------
    ../data/final_dashboard_data.json
"""

import json
import os


def main():
    with open("../data/sector_dashboard_data.json") as f:
        companies = json.load(f)

    with open("../data/model_explanations.json") as f:
        model_out = json.load(f)
    target_type = model_out["target_type"]
    in_sample_r2 = model_out["in_sample_r2"]
    exp_by_company = {e["company"]: e for e in model_out["explanations"]}

    evidence_by_key = {}
    if os.path.exists("../data/evidence.json"):
        with open("../data/evidence.json") as f:
            for item in json.load(f):
                evidence_by_key[(item["company"], item["factor"])] = item

    narratives_by_key = {}
    if os.path.exists("../data/narratives.json"):
        with open("../data/narratives.json") as f:
            raw = json.load(f)
            items = raw.get("narratives", raw)  # handle seed file's {_readme, narratives} wrapper
            for item in items:
                narratives_by_key[(item["company"], item["factor"])] = item

    output_companies = []
    for c in companies:
        company_out = {
            "company": c["company"],
            "symbol": c["symbol"],
            "sector": c["sector"],
            "data_status": c["data_status"],
        }
        if c["data_status"] != "ok":
            company_out["note"] = c["note"]
            output_companies.append(company_out)
            continue

        exp = exp_by_company.get(c["company"])
        shap_by_factor = {}
        if exp:
            for rank, s in enumerate(exp["shap_contributions"]):
                shap_by_factor[s["factor"]] = {"shap_value": s["shap_value"], "shap_rank": rank + 1}

        out_factors = []
        for f in c["factors"]:
            row = {
                "factor": f["factor"],
                "materiality_weight": f["materiality_weight"],
                "peer_group_average": f["peer_group_average"],
                "gap": f["gap"],
                "direction": f["direction"],
                "consensus_color": f["consensus_color"],
                "decision": f["decision"],
                "estimate": f["estimate"],
            }

            shap_info = shap_by_factor.get(f["factor"])
            if shap_info:
                row["model"] = {
                    "target_type": target_type,
                    "shap_value": shap_info["shap_value"],
                    "shap_rank": shap_info["shap_rank"],
                    "in_sample_r2": in_sample_r2,
                }

            ev = evidence_by_key.get((c["company"], f["factor"]))
            if ev:
                row["evidence"] = {
                    "summary": ev["evidence_summary"],
                    "direction": ev["evidence_direction"],
                    "confidence": ev["confidence"],
                    "sources": ev.get("sources", []),
                }

            nar = narratives_by_key.get((c["company"], f["factor"]))
            if nar:
                row["explanation"] = {
                    "auditor_view": nar["auditor_view"],
                    "corporate_view": nar["corporate_view"],
                    "investor_view": nar["investor_view"],
                    "key_uncertainties": nar.get("key_uncertainties", []),
                    "source": "ai_generated",
                }
            else:
                row["explanation"] = {
                    **f["explanation"],
                    "source": "template",
                }

            out_factors.append(row)

        company_out["factors"] = out_factors
        output_companies.append(company_out)

    with open("../data/final_dashboard_data.json", "w") as f:
        json.dump({
            "model_meta": {
                "target_type": target_type,
                "in_sample_r2": in_sample_r2,
                "n_training_rows": model_out["n_training_rows"],
            },
            "companies": output_companies,
        }, f, indent=2)

    # Summary stats
    n_with_evidence = sum(1 for k in evidence_by_key)
    n_with_narrative = sum(1 for k in narratives_by_key)
    print(f"Saved ../data/final_dashboard_data.json")
    print(f"  {len(output_companies)} companies")
    print(f"  {n_with_evidence} (company, factor) pairs with retrieved evidence")
    print(f"  {n_with_narrative} (company, factor) pairs with AI-generated narratives "
          f"(others use template explanations)")
    print(f"  Model target type: {target_type} (in-sample R^2: {in_sample_r2})")


if __name__ == "__main__":
    main()
