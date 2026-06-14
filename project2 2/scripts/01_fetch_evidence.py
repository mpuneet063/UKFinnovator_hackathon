"""
STAGE 1 — Evidence retrieval & structured extraction (AI usage #1)
====================================================================

WHAT THIS DOES
--------------
For each (company, ESG factor) pair where the factor is financially
material for that company (materiality weight > 0) AND the materiality
gap vs the sector peer average is notable (|gap| >= GAP_THRESHOLD), this
script uses an LLM (Claude, via the Anthropic API) WITH WEB SEARCH to:

  1. Search for recent (sustainability report / news / Wikipedia) sources
     discussing that company's stance/performance on that specific ESG
     factor.
  2. Extract a SHORT, SOURCE-ATTRIBUTED summary (2-3 sentences) plus the
     source URL(s).
  3. Classify the qualitative DIRECTION of the evidence on a simple
     3-point scale: "positive" (company appears to be addressing this
     well), "negative" (evidence of controversy/gap), "mixed/unclear".

WHY THIS IS THE RIGHT USE OF AI HERE
-------------------------------------
This mirrors the "relevance extraction" stage in the Cameron paper
(Section III.B.4): the model's job is to FIND and SUMMARISE real text,
not to invent a score. The output of this stage — `evidence.json` — is
the ONLY place qualitative judgement enters the pipeline, and every
judgement is tied to a retrieved source URL for traceability/audit.

The `evidence_direction` field (positive/negative/mixed) IS new
information contributed by the LLM (it's not in the spreadsheet), but
it is:
  (a) a coarse 3-way classification, not a continuous invented score,
  (b) always accompanied by the source text/URL it was derived from,
  (c) used downstream (Stage 2) only as ONE additional categorical
      FEATURE in a transparent, auditable ML model — never as a
      standalone "answer".

HOW TO RUN
----------
    export ANTHROPIC_API_KEY=sk-...
    python3 01_fetch_evidence.py

INPUT
-----
    ../data/sector_dashboard_data.json   (from build_sector_data.py)

OUTPUT
------
    ../data/evidence.json
    Structure: list of
    {
      "company": ..., "symbol": ..., "factor": ...,
      "materiality_weight": ..., "gap": ...,
      "evidence_summary": "...",
      "evidence_direction": "positive" | "negative" | "mixed",
      "sources": ["https://..."],
      "confidence": "high" | "medium" | "low"   # how much direct evidence was found
    }
"""

import json
import os
import time
import requests

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"

GAP_THRESHOLD = 0.05  # only fetch evidence for "distinctive" factors (matches build_sector_data.py)
MAX_ITEMS = 60        # safety cap on number of (company, factor) pairs processed per run

SYSTEM_PROMPT = """You are an ESG research analyst. You will be given a company name, an
ESG factor (from the SASB taxonomy), that company's financial materiality weight for this
factor (0-1 scale), and how that weight compares to its sector peer-group average (the "gap").

Use web search to find recent (ideally 2023-2025) information about this company's stance,
targets, performance, or controversies specifically related to this ESG factor — from
sustainability reports, news coverage, NGO/regulatory commentary, or Wikipedia.

Then return ONLY a JSON object (no markdown fences, no preamble) with this exact schema:

{
  "evidence_summary": "<2-3 sentence summary of what you found, in your own words, with no quotes longer than a few words>",
  "evidence_direction": "positive" | "negative" | "mixed",
  "sources": ["<url1>", "<url2>"],
  "confidence": "high" | "medium" | "low"
}

Guidance:
- "evidence_direction":
    "positive" = the company appears to have credible targets/track record on this factor
    "negative" = evidence of controversy, incidents, or significant disclosure gaps on this factor
    "mixed"    = both positive commitments AND unresolved issues found
- "confidence":
    "high"   = specific, recent, directly-relevant evidence found
    "medium" = general company sustainability info found, partially relevant
    "low"    = little direct evidence found
- Do NOT invent sources. If you find nothing relevant, set evidence_summary to
  "No directly relevant evidence found", evidence_direction to "mixed", sources to [],
  and confidence to "low".
- Do not include any text outside the JSON object.
- Paraphrase; do not quote more than a few words from any source (copyright)."""


def fetch_evidence_for(company, factor, materiality, gap, peer_avg):
    direction_hint = "above" if gap > 0 else "below" if gap < 0 else "in line with"
    user_msg = (
        f"Company: {company}\n"
        f"ESG Factor: {factor}\n"
        f"Financial materiality weight for this company: {materiality:.3f}\n"
        f"Sector peer-group average: {peer_avg:.3f}\n"
        f"This company's weight is {direction_hint} its sector peer average "
        f"(gap = {gap:+.3f}).\n\n"
        f"Research {company}'s stance/performance on '{factor}' and return the JSON object."
    )

    body = {
        "model": MODEL,
        "max_tokens": 1000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }

    resp = requests.post(
        API_URL,
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    text_parts = [b["text"] for b in data["content"] if b.get("type") == "text"]
    full_text = "\n".join(text_parts).strip()

    if full_text.startswith("```"):
        full_text = full_text.strip("`")
        if full_text.startswith("json"):
            full_text = full_text[4:]

    start = full_text.rfind("{")
    end = full_text.rfind("}") + 1
    json_str = full_text[start:end]

    try:
        result = json.loads(json_str)
    except Exception as e:
        result = {
            "evidence_summary": f"PARSE_ERROR: {e}",
            "evidence_direction": "mixed",
            "sources": [],
            "confidence": "low",
        }

    return result


def main():
    if not API_KEY:
        raise SystemExit("Set ANTHROPIC_API_KEY environment variable first.")

    with open("../data/sector_dashboard_data.json") as f:
        companies = json.load(f)

    # Select (company, factor) pairs worth researching: material + distinctive gap
    targets = []
    for c in companies:
        if c["data_status"] != "ok":
            continue
        for f in c["factors"]:
            if f["materiality_weight"] > 0 and abs(f["gap"]) >= GAP_THRESHOLD:
                targets.append({
                    "company": c["company"],
                    "symbol": c["symbol"],
                    "sector": c["sector"],
                    "factor": f["factor"],
                    "materiality_weight": f["materiality_weight"],
                    "gap": f["gap"],
                    "peer_group_average": f["peer_group_average"],
                })

    print(f"{len(targets)} (company, factor) pairs qualify for evidence retrieval "
          f"(materiality > 0 and |gap| >= {GAP_THRESHOLD}).")

    if len(targets) > MAX_ITEMS:
        print(f"Capping to first {MAX_ITEMS} for this run (raise MAX_ITEMS to process more).")
        targets = targets[:MAX_ITEMS]

    evidence = []
    for i, t in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] {t['company']} — {t['factor']} (gap {t['gap']:+.3f}) ...",
              end=" ", flush=True)
        try:
            result = fetch_evidence_for(
                t["company"], t["factor"], t["materiality_weight"], t["gap"], t["peer_group_average"]
            )
        except Exception as e:
            result = {
                "evidence_summary": f"ERROR: {e}",
                "evidence_direction": "mixed",
                "sources": [],
                "confidence": "low",
            }
        print(f"-> {result['evidence_direction']} ({result['confidence']} confidence, "
              f"{len(result.get('sources', []))} sources)")

        evidence.append({**t, **result})
        time.sleep(1)

    with open("../data/evidence.json", "w") as f:
        json.dump(evidence, f, indent=2)

    print(f"\nSaved ../data/evidence.json — {len(evidence)} items")


if __name__ == "__main__":
    main()
