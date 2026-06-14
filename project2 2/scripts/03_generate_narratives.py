"""
STAGE 3 — Grounded narrative generation / copilot layer (AI usage #3)
========================================================================

WHAT THIS DOES
--------------
For each company, takes:
  - The company's materiality profile + sector peer gaps
    (from sector_dashboard_data.json — Stage "0")
  - SHAP-based feature attributions for the model's prediction
    (from model_explanations.json — Stage 2)
  - Retrieved evidence summaries with source URLs
    (from evidence.json — Stage 1, if available)

...and asks an LLM (Claude) to write THREE short narrative explanations
(auditor / corporate / investor perspectives), in the style established
earlier in this project, but now grounded in REAL retrieved evidence and
REAL model attributions rather than templated gap descriptions alone.

GROUNDING / ANTI-HALLUCINATION DESIGN
----------------------------------------
The prompt explicitly:
  1. Provides ONLY the numbers from Stages 0/1/2 as context.
  2. Instructs the model to reference ONLY those numbers — it must not
     state any other materiality weight, gap, SHAP value, or financial
     figure.
  3. Instructs the model to cite evidence sources by name/URL when used,
     and to explicitly say "no qualitative evidence available" when
     evidence.json has nothing for that company/factor (rather than
     inventing context).
  4. Asks for structured JSON output (auditor_view, corporate_view,
     investor_view, key_uncertainties) so it can be slotted directly
     into the dashboard's existing explanation fields.

This is the same principle as Cameron's "explicit reasoning capture":
the narrative is downstream of, and must be traceable to, upstream
structured data — the LLM's job is articulation, not computation.

HOW TO RUN
----------
    export ANTHROPIC_API_KEY=sk-...
    python3 03_generate_narratives.py [--limit N]

INPUT
-----
    ../data/sector_dashboard_data.json
    ../data/model_explanations.json
    ../data/evidence.json   (optional — script runs without it, with a
                              note that no qualitative evidence was used)

OUTPUT
------
    ../data/narratives.json
    [{ "company": ..., "factor": ..., "auditor_view": ..., 
       "corporate_view": ..., "investor_view": ...,
       "key_uncertainties": [...] }, ...]
"""

import json
import os
import sys
import time
import requests

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"

GAP_THRESHOLD = 0.05  # match build_sector_data.py / 01_fetch_evidence.py

SYSTEM_PROMPT = """You are writing short, role-specific explanations of an ESG financial
materiality finding for a company, for three audiences: an external auditor, the company's
own corporate strategy team, and an investor.

You will be given:
  - The company name, sector, and ESG factor
  - The factor's financial materiality weight for this company (0-1)
  - The sector peer-group average for this factor
  - The gap (company weight minus peer average)
  - A SHAP-based explanation from a predictive model: how much this factor's materiality
    weight contributed to the model's prediction for this company, relative to other factors
  - (Optionally) a retrieved evidence summary with source(s), or a note that no evidence
    was retrieved

STRICT RULES — read carefully:
1. You may ONLY state numbers that are given to you in the input. Do not state any other
   materiality weight, gap, percentage, financial figure, or score.
2. If evidence is provided, you may reference it, attributing it to its source (e.g.
   "according to [source]"). Paraphrase — do not quote more than a few words.
3. If no evidence is provided, explicitly say so (e.g. "no specific evidence on this
   factor was retrieved for this company") — do not invent context, controversies, or
   commitments.
4. If the SHAP contribution for this factor is small/near-zero, say so plainly — do not
   manufacture a narrative of importance where the model attributes little weight.
5. Keep each view to 2-3 sentences.

Return ONLY a JSON object (no markdown fences, no preamble):
{
  "auditor_view": "...",
  "corporate_view": "...",
  "investor_view": "...",
  "key_uncertainties": ["...", "..."]
}

"key_uncertainties" should list 1-3 short bullet points naming what additional data
(e.g. real share price data, more evidence sources, longer time series) would most
improve confidence in this explanation."""


def build_user_message(company, sector, factor, materiality, peer_avg, gap,
                        shap_value, shap_rank, target_type, evidence_item):
    msg = (
        f"Company: {company}\n"
        f"Sector: {sector}\n"
        f"ESG Factor: {factor}\n"
        f"Materiality weight: {materiality:.3f}\n"
        f"Sector peer-group average: {peer_avg:.3f}\n"
        f"Gap (company - peer average): {gap:+.3f}\n"
        f"Model: this factor's materiality weight has a SHAP contribution of "
        f"{shap_value:+.5f} to the model's prediction of '{target_type}' for this company "
        f"(rank #{shap_rank} among this company's factors by |SHAP value|).\n"
    )
    if evidence_item:
        msg += (
            f"\nRetrieved evidence (direction: {evidence_item['evidence_direction']}, "
            f"confidence: {evidence_item['confidence']}):\n"
            f"{evidence_item['evidence_summary']}\n"
            f"Sources: {', '.join(evidence_item.get('sources', [])) or 'none listed'}\n"
        )
    else:
        msg += "\nNo qualitative evidence was retrieved for this company/factor.\n"

    msg += "\nWrite the three perspectives and key_uncertainties as specified."
    return msg


def call_claude(user_msg):
    body = {
        "model": MODEL,
        "max_tokens": 700,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
    }
    resp = requests.post(
        API_URL,
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = "".join(b["text"] for b in data["content"] if b.get("type") == "text").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}") + 1
    return json.loads(text[start:end])


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    if not API_KEY:
        raise SystemExit("Set ANTHROPIC_API_KEY environment variable first.")

    with open("../data/sector_dashboard_data.json") as f:
        companies = {c["company"]: c for c in json.load(f)}

    with open("../data/model_explanations.json") as f:
        model_out = json.load(f)
    target_type = model_out["target_type"]
    explanations_by_company = {e["company"]: e for e in model_out["explanations"]}

    evidence_by_key = {}
    if os.path.exists("../data/evidence.json"):
        with open("../data/evidence.json") as f:
            for item in json.load(f):
                evidence_by_key[(item["company"], item["factor"])] = item

    # Build worklist: distinctive (company, factor) pairs, same selection as Stage 1
    worklist = []
    for company, c in companies.items():
        if c["data_status"] != "ok":
            continue
        exp = explanations_by_company.get(company)
        if not exp:
            continue
        shap_by_factor = {s["factor"]: (rank + 1, s["shap_value"])
                           for rank, s in enumerate(exp["shap_contributions"])}
        for f in c["factors"]:
            if f["materiality_weight"] > 0 and abs(f["gap"]) >= GAP_THRESHOLD:
                rank, shap_val = shap_by_factor.get(f["factor"], (None, 0.0))
                worklist.append({
                    "company": company,
                    "sector": c["sector"],
                    "factor": f["factor"],
                    "materiality_weight": f["materiality_weight"],
                    "peer_avg": f["peer_group_average"],
                    "gap": f["gap"],
                    "shap_value": shap_val,
                    "shap_rank": rank if rank else "n/a",
                })

    print(f"{len(worklist)} (company, factor) pairs to generate narratives for.")
    if limit:
        worklist = worklist[:limit]
        print(f"Limiting to {limit} for this run.")

    if not evidence_by_key:
        print("NOTE: ../data/evidence.json not found — narratives will note that no "
              "qualitative evidence was retrieved. Run 01_fetch_evidence.py first for "
              "fuller narratives.")

    narratives = []
    for i, item in enumerate(worklist):
        key = (item["company"], item["factor"])
        evidence_item = evidence_by_key.get(key)
        print(f"[{i+1}/{len(worklist)}] {item['company']} — {item['factor']} ...",
              end=" ", flush=True)

        user_msg = build_user_message(
            item["company"], item["sector"], item["factor"],
            item["materiality_weight"], item["peer_avg"], item["gap"],
            item["shap_value"], item["shap_rank"], target_type, evidence_item
        )
        try:
            result = call_claude(user_msg)
        except Exception as e:
            result = {
                "auditor_view": f"ERROR: {e}",
                "corporate_view": "", "investor_view": "", "key_uncertainties": [],
            }
        print("done")

        narratives.append({**item, **result})
        time.sleep(0.5)

    with open("../data/narratives.json", "w") as f:
        json.dump(narratives, f, indent=2)

    print(f"\nSaved ../data/narratives.json — {len(narratives)} items")


if __name__ == "__main__":
    main()
