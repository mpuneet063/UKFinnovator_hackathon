# Maxwell Data — ESG Materiality Decision-Support: Full AI Pipeline

This is the full backend pipeline: data → ML model → explainability →
AI-grounded narratives → merged dashboard data. Three distinct AI stages
are used, each with a narrow, well-defined job and explicit grounding
rules so the system stays explainable and auditable end-to-end.

## Project structure

```
project2/
├── requirements.txt
├── data/
│   ├── ftse100_materiality.csv      # source data (from the challenge xlsx)
│   ├── sector_heatmaps.json         # per-sector heatmap + peer averages (Stage 0)
│   ├── sector_dashboard_data.json   # per-company materiality/gap/consensus (Stage 0)
│   ├── financial_data.json          # real share-price metrics, if available (Stage 0)
│   ├── evidence.json                # retrieved evidence (Stage 1, AI)
│   ├── model_explanations.json      # SHAP attributions (Stage 2, AI)
│   ├── narratives.json              # grounded narratives (Stage 3, AI)
│   └── final_dashboard_data.json    # merged output (Stage 4)
├── models/
│   ├── model.json                   # saved XGBoost model
│   └── plots/                       # SHAP visualizations
├── scripts/
│   ├── 00_fetch_financial_data.py   # Stage 0: real share price data (not AI)
│   ├── build_sector_data.py         # Stage 0: materiality/gap/consensus (not AI)
│   ├── 01_fetch_evidence.py         # Stage 1 (AI #1): evidence retrieval
│   ├── seed_evidence.py             # demo evidence (no API key needed)
│   ├── 02_train_model.py            # Stage 2 (AI #2): explainable ML model
│   ├── 03_generate_narratives.py    # Stage 3 (AI #3): grounded narratives
│   ├── 04_merge_final.py            # Stage 4: merge everything
│   ├── 05_visualize_shap.py         # SHAP plots for pitch deck
│   └── run_all.py                   # orchestrates the whole pipeline
└── frontend_stub/                   # minimal reference component (not styled)
```

## Quick start

```bash
cd project2
pip install -r requirements.txt

# Without an Anthropic API key (uses seed evidence/narratives):
cd scripts
python3 run_all.py --skip-ai

# With an Anthropic API key (full pipeline, all 3 AI stages run live):
export ANTHROPIC_API_KEY=sk-...
python3 run_all.py
```

Output: `data/final_dashboard_data.json` — this is what the frontend
should consume. See `frontend_stub/src/FrontendStub.jsx` for the data
shape and a minimal reference implementation.

---

## The 3 AI stages — what each one does and why

### AI Stage 1 — Evidence retrieval & extraction (`01_fetch_evidence.py`)

**What it does:** for each (company, ESG factor) where the factor is
financially material AND the company's materiality weight notably differs
from its sector peer average (|gap| ≥ 0.05), an LLM with web search reads
sustainability reports, news, and Wikipedia, and extracts:
  - a 2-3 sentence summary (paraphrased, source-attributed)
  - a 3-way direction classification: positive / negative / mixed
  - source URLs
  - a confidence level (high/medium/low)

**Why this is the right AI use:** this is retrieval + summarisation of
real text. The model never invents a number — every claim is tied to a
retrievable source. This mirrors the "relevance extraction" stage in the
Cameron paper (the reference paper for this challenge): find the parts of
a document relevant to a specific ESG driver, before any scoring happens.

**Output:** `data/evidence.json`

---

### AI Stage 2 — Explainable predictive model (`02_train_model.py`)

**What it does:** builds a company-level feature matrix (26 columns = the
26 SASB materiality weights for that company) and trains an XGBoost
regression model to predict a financial target. SHAP (SHapley Additive
exPlanations) then attributes the model's prediction for each company back
to individual ESG factors — "this factor's materiality weight pushed the
prediction up by X, that one pushed it down by Y".

**The actual question being asked:** "does a company's ESG materiality
PROFILE help predict [target]?" Two modes:

- **Real-data mode** (`target_type: "volatility_6m"`): if real share-price
  data is available (`00_fetch_financial_data.py` succeeded for ≥10
  companies), the target is trailing 6-month annualised volatility — a
  genuine financial outcome. The model's R² tells you how well materiality
  PROFILE predicts volatility — a testable hypothesis, not an assumption.

- **Fallback mode** (`target_type: "distinctiveness_index"`): if real
  financial data isn't available (e.g. this sandbox's network
  restrictions), the target is a deterministic index — Σ |gap| ×
  materiality_weight per company — computed from the materiality data
  itself. This demonstrates the full pipeline mechanically, but
  **`final_dashboard_data.json` and the frontend must show a caveat in this
  mode** (the stub does this via `ModelMetaBanner`) — SHAP values describe
  what drives the INDEX, not a financial outcome.

**Why this is the right AI use:** SHAP is a well-established, peer-reviewed
explainability technique (not a black box on top of a black box) — every
attribution is mathematically derived from the trained model's actual
behaviour, reproducible, and auditable.

**Output:** `models/model.json`, `data/model_explanations.json`,
`models/plots/*.png`

---

### AI Stage 3 — Grounded narrative generation (`03_generate_narratives.py`)

**What it does:** for the same (company, factor) pairs as Stage 1, an LLM
is given ONLY:
  - the materiality weight, peer average, gap (from Stage 0)
  - the SHAP value + rank for that factor (from Stage 2)
  - the retrieved evidence, if any (from Stage 1)

...and asked to write three short (2-3 sentence) explanations — auditor,
corporate, investor perspectives — plus "key uncertainties".

**Strict grounding rules given to the model:**
1. It may only state numbers it was given — no new materiality weights,
   gaps, percentages, or financial figures.
2. If evidence was provided, reference it with attribution; paraphrase,
   don't quote.
3. If no evidence was provided, say so explicitly rather than inventing
   context.
4. If the SHAP value is near zero, say so — don't manufacture a narrative
   of importance.

**Why this is the right AI use:** this is the "last mile" — turning
structured numbers into prose a non-technical exec can read — without
giving the model room to introduce new claims. It's retrieval-augmented
generation in the strictest sense: the model's only job is articulation of
already-computed, already-cited facts.

**Output:** `data/narratives.json`

---

## What's deterministic (NOT AI)

- `build_sector_data.py`: materiality weights, peer averages, gaps,
  consensus colors, decisions, illustrative estimates — pure arithmetic on
  the source spreadsheet (`gap = company_weight − sector_peer_average`).
- `00_fetch_financial_data.py`: real share-price data via yfinance (a data
  fetch, not a model).

This separation is the pitch: **the numbers are auditable spreadsheet
arithmetic and real market data; AI is layered on top only for (1)
retrieving/summarising real evidence, (2) explaining a transparent ML
model via SHAP, and (3) narrating those pre-computed, cited facts — never
to invent the underlying numbers.**

## Known limitations

- This sandbox cannot reach Yahoo Finance (network egress restriction), so
  `00_fetch_financial_data.py` will report 0/79 companies with real data
  here. Run it in an environment with normal internet access — Stage 2
  will then automatically switch to `target_type: "volatility_6m"` and
  produce financially-meaningful SHAP explanations.
- In-sample R² for the fallback model (0.97) is expected to be high
  because XGBoost can closely fit a small, deterministic target — this is
  NOT evidence of predictive power and is explicitly labelled as such in
  the output.
- `seed_evidence.py` provides 3 hand-researched (company, factor) pairs so
  the pipeline can be demonstrated without an API key. Run
  `01_fetch_evidence.py` with `ANTHROPIC_API_KEY` set to cover all
  qualifying pairs (capped at `MAX_ITEMS = 60` per run by default).
