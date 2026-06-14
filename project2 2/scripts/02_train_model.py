"""
STAGE 2 — Explainable predictive model (AI usage #2)
======================================================

WHAT THIS DOES
--------------
Builds a company-level feature matrix from the 26 SASB materiality
weights (one row per company, one column per ESG factor — the company's
own materiality profile), and trains a gradient-boosted regression model
(XGBoost) to predict a financial target variable:

    TARGET = trailing 6-month share price volatility (annualised)

i.e. the question this model asks is:

    "Does a company's ESG materiality PROFILE (which factors are
     financially material for it, and how heavily) help predict how
     volatile its share price has been recently?"

This is a genuinely testable hypothesis — not an assumption. The model's
performance (R^2 / cross-validated error) tells you HOW WELL materiality
profile predicts volatility in this dataset. If the answer is "not very
well", that is itself a valid, reportable finding (and arguably an
interesting one: it would suggest the market does not yet price
materiality-profile differences into short-term volatility — an
"unpriced ESG risk" narrative).

EXPLAINABILITY (SHAP)
----------------------
For each company, SHAP (SHapley Additive exPlanations) values quantify
exactly how much each of the 26 materiality-weight features pushed that
company's predicted volatility up or down relative to the average
prediction. This gives a PER-COMPANY, PER-FACTOR attribution:

    "For Glencore, the model's prediction was 0.04 ABOVE average,
     and the single largest contributor was the materiality weight
     on 'Human Rights & Community Relations' (+0.018), followed by
     'Ecological Impacts' (+0.011)..."

This SHAP output is what Stage 3 (the LLM narrative layer) is given as
grounding — the LLM is told these exact numbers and is instructed not
to state any other numbers.

FALLBACK MODE (no real financial data)
----------------------------------------
If ../data/financial_data.json has no companies with status == "ok"
(e.g. this sandbox's network restrictions), this script:
  1. Still builds the full feature matrix and trains the model
  2. Uses a TARGET clearly generated from a deterministic, disclosed
     formula based on the materiality data itself (NOT invented
     per-company by an LLM, NOT random) — specifically, the
     "ESG Distinctiveness Index" = sum over factors of
     |gap| * materiality_weight, which is computable from data already
     in sector_dashboard_data.json.
  3. Labels all outputs with `target_type: "distinctiveness_index"` vs
     `target_type: "volatility_6m"` so downstream consumers (Stage 3,
     the dashboard) know which mode produced the explanation, and the
     dashboard can show an appropriate caveat.

HOW TO RUN
----------
    python3 02_train_model.py

OUTPUT
------
    ../models/model.json           (saved XGBoost model)
    ../data/model_explanations.json
      [{ "company": ..., "target_type": ..., "predicted_value": ...,
         "actual_value": ... | null,
         "base_value": ...,
         "shap_contributions": [ {"factor": ..., "materiality_weight": ...,
                                   "shap_value": ...}, ... sorted by |shap_value| desc ]
       }, ...]
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

ALL_FACTORS = [
    "Air Quality", "Ecological Impacts", "Energy Management", "GHG Emissions",
    "Water & Hazardous Materials", "Waste & Wastewater Management",
    "Access & Affordability", "Customer Privacy", "Customer Welfare", "Data Security",
    "Employee Engagement", "Employee Health & Safety", "Human Rights & Community Relations",
    "Labor Practices", "Product Quality & Safety", "Selling Practices & Product Labeling",
    "Business Ethics", "Business Model Resilience", "Competitive Behavior",
    "Critical Incidence Risk Management", "Management of the Legal & Regulatory Environment",
    "Materials Sourcing & Efficiency", "Product Design & Lifecycle Management",
    "Physical Impacts of Climate Change", "Supply Chain Management", "Systemic Risk Management",
]


def build_feature_matrix():
    """One row per company (data_status == ok), one column per ESG factor
    = that company's materiality weight for that factor (0 if not material)."""
    with open("../data/sector_dashboard_data.json") as f:
        companies = json.load(f)

    rows = []
    distinctiveness = {}
    for c in companies:
        if c["data_status"] != "ok":
            continue
        row = {"company": c["company"], "symbol": c["symbol"], "sector": c["sector"]}
        dist_index = 0.0
        for f in c["factors"]:
            row[f["factor"]] = f["materiality_weight"]
            dist_index += abs(f["gap"]) * f["materiality_weight"]
        rows.append(row)
        distinctiveness[c["company"]] = round(dist_index, 4)

    df = pd.DataFrame(rows).fillna(0.0)
    return df, distinctiveness


def load_financial_targets():
    try:
        with open("../data/financial_data.json") as f:
            fin = json.load(f)
    except FileNotFoundError:
        return {}
    return {r["company"]: r["volatility_6m"] for r in fin if r.get("status") == "ok"}


def main():
    df, distinctiveness = build_feature_matrix()
    financial_targets = load_financial_targets()

    X = df[ALL_FACTORS].values
    companies = df["company"].tolist()

    if len(financial_targets) >= 10:
        # Real-data mode
        target_type = "volatility_6m"
        y = np.array([financial_targets.get(c, np.nan) for c in companies])
        mask = ~np.isnan(y)
        print(f"Real financial data available for {mask.sum()}/{len(companies)} companies.")
        X_train, y_train = X[mask], y[mask]
        train_companies = [c for c, m in zip(companies, mask) if m]
    else:
        # Fallback mode: deterministic distinctiveness index
        target_type = "distinctiveness_index"
        y = np.array([distinctiveness[c] for c in companies])
        X_train, y_train = X, y
        train_companies = companies
        print("No real financial data available (need >=10 companies with status='ok' "
              "in financial_data.json).")
        print(f"FALLBACK MODE: training against 'ESG Distinctiveness Index' "
              f"(sum of |gap| * materiality_weight per company) — a DETERMINISTIC "
              f"quantity computed from the materiality dataset itself, NOT real "
              f"financial data. Treat model outputs in this mode as a PIPELINE "
              f"DEMONSTRATION, not a financial prediction.")

    if len(X_train) < 15:
        raise SystemExit(f"Only {len(X_train)} training rows available — need at least "
                          f"~15 for a meaningful model. Check input data.")

    print(f"\nTraining XGBoost regressor: {len(X_train)} companies x {len(ALL_FACTORS)} features "
          f"-> target = {target_type}")

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # In-sample fit quality (small-N — treat as illustrative, not a generalisation claim)
    preds_train = model.predict(X_train)
    ss_res = np.sum((y_train - preds_train) ** 2)
    ss_tot = np.sum((y_train - y_train.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"In-sample R^2: {r2:.3f}  (n={len(X_train)}; small-N, illustrative only — "
          f"do not over-interpret as out-of-sample predictive power without "
          f"cross-validation on a larger dataset)")

    model.save_model("../models/model.json")
    print("Saved ../models/model.json")

    # SHAP explanations for ALL companies (not just training set)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    base_value = float(explainer.expected_value)

    predictions = model.predict(X)

    explanations = []
    for i, company in enumerate(companies):
        contributions = []
        for j, factor in enumerate(ALL_FACTORS):
            if X[i, j] == 0 and abs(shap_values[i, j]) < 1e-9:
                continue  # skip factors that are 0 and contribute nothing
            contributions.append({
                "factor": factor,
                "materiality_weight": round(float(X[i, j]), 3),
                "shap_value": round(float(shap_values[i, j]), 5),
            })
        contributions.sort(key=lambda c: -abs(c["shap_value"]))

        explanations.append({
            "company": company,
            "symbol": df.iloc[i]["symbol"],
            "sector": df.iloc[i]["sector"],
            "target_type": target_type,
            "predicted_value": round(float(predictions[i]), 5),
            "actual_value": (round(float(financial_targets[company]), 5)
                             if target_type == "volatility_6m" and company in financial_targets
                             else (round(float(distinctiveness[company]), 5)
                                   if target_type == "distinctiveness_index" else None)),
            "base_value": round(base_value, 5),
            "shap_contributions": contributions[:10],  # top 10 drivers
        })

    with open("../data/model_explanations.json", "w") as f:
        json.dump({
            "target_type": target_type,
            "in_sample_r2": round(float(r2), 4),
            "n_training_rows": int(len(X_train)),
            "feature_names": ALL_FACTORS,
            "explanations": explanations,
        }, f, indent=2)

    print(f"\nSaved ../data/model_explanations.json — {len(explanations)} companies")

    # Print a sample explanation
    print("\n=== Sample explanation ===")
    sample = explanations[0]
    print(f"{sample['company']} ({sample['sector']})")
    print(f"  Predicted {target_type}: {sample['predicted_value']}")
    print(f"  Base value (average prediction): {sample['base_value']}")
    print(f"  Top SHAP contributions:")
    for c in sample["shap_contributions"][:5]:
        sign = "+" if c["shap_value"] >= 0 else ""
        print(f"    {c['factor']:42} weight={c['materiality_weight']:.3f}  "
              f"shap={sign}{c['shap_value']:.5f}")


if __name__ == "__main__":
    main()
