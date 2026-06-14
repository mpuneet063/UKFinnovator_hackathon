"""
Seed/demo evidence file.

This is a small, REAL (not synthetic) evidence.json, populated using the
same research approach as 01_fetch_evidence.py (web search + structured
extraction), for a handful of (company, factor) pairs — enough to
demonstrate the full Stage 1 -> 2 -> 3 pipeline end-to-end without
requiring an ANTHROPIC_API_KEY for a first run.

Run 01_fetch_evidence.py with a real API key to extend this to all
qualifying (company, factor) pairs (it will overwrite this file —
back it up first if you want to keep these seed entries).
"""

import json

EVIDENCE = [
    {
        "company": "Glencore plc",
        "symbol": "GLEN",
        "sector": "Mining",
        "factor": "Human Rights & Community Relations",
        "materiality_weight": 0.704,
        "gap": None,
        "peer_group_average": None,
        "evidence_summary": (
            "Dutch pension fund ABP divested its roughly EUR 57m stake in Glencore in 2024, "
            "citing unresolved human rights risks including child labour concerns in DR Congo "
            "cobalt supply chains, community conflicts, and pollution. Glencore's 2024 "
            "sustainability report states it recorded no major human rights incidents, but "
            "prior bribery convictions and resulting investor lawsuits suggest continued "
            "elevated risk in this area."
        ),
        "evidence_direction": "negative",
        "sources": [
            "https://www.ipe.com/abp-exits-glencore-investment-over-human-rights-concerns/10055740.article",
            "https://www.glencore.com/.rest/api/v1/documents/static/72cdaa4f-71c1-4901-bb31-8819f38dbfcb/GLEN-2024-Sustainability-Report.pdf",
        ],
        "confidence": "high",
    },
    {
        "company": "Glencore plc",
        "symbol": "GLEN",
        "sector": "Mining",
        "factor": "Ecological Impacts",
        "materiality_weight": 0.688,
        "gap": None,
        "peer_group_average": None,
        "evidence_summary": (
            "Glencore reports annual emissions of over 433 million tonnes of CO2 equivalent "
            "and generates approximately 2.31 billion tonnes of waste, with only around 1% "
            "historically managed sustainably. Of roughly 2 million hectares of disturbed "
            "land, only about 37,000 hectares had been rehabilitated as of 2021, with over "
            "90,000 hectares still needing restoration."
        ),
        "evidence_direction": "negative",
        "sources": [
            "https://www.greendigest.co/p/evaluating-companys-impact-case-glencore",
        ],
        "confidence": "medium",
    },
    {
        "company": "Anglo American Plc",
        "symbol": "AAL",
        "sector": "Mining",
        "factor": "Business Model Resilience",
        "materiality_weight": 0.73,
        "gap": None,
        "peer_group_average": None,
        "evidence_summary": (
            "Anglo American is mid-way through a major portfolio transformation, including "
            "the sale of its steelmaking coal business to Peabody Energy, the planned "
            "demerger of Anglo American Platinum, and divestment of De Beers, explicitly "
            "framed by the company as creating a more resilient, focused business. MSCI "
            "rates the company 'AA' (Leader) on ESG."
        ),
        "evidence_direction": "positive",
        "sources": [
            "https://www.angloamerican.com/investors/annual-reporting",
            "https://knowesg.com/esg-ratings/anglo-american-plc",
        ],
        "confidence": "high",
    },
]


if __name__ == "__main__":
    # Fill in gap/peer_group_average/materiality from sector_dashboard_data.json —
    # this is the SINGLE SOURCE OF TRUTH for these numbers; hand-written values
    # above are deliberately left as None to avoid drift/inconsistency.
    with open("../data/sector_dashboard_data.json") as f:
        companies = {c["company"]: c for c in json.load(f)}

    for item in EVIDENCE:
        c = companies[item["company"]]
        factor_data = next(f for f in c["factors"] if f["factor"] == item["factor"])
        item["gap"] = factor_data["gap"]
        item["peer_group_average"] = factor_data["peer_group_average"]
        item["materiality_weight"] = factor_data["materiality_weight"]

    with open("../data/evidence.json", "w") as f:
        json.dump(EVIDENCE, f, indent=2)

    print(f"Saved ../data/evidence.json — {len(EVIDENCE)} seed items")
