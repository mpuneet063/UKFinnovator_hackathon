"""
Visualization helper — SHAP summary plots and per-company waterfall charts.

Produces PNG charts in ../models/plots/ for inclusion in a pitch deck or
for sanity-checking the model. Not part of the frontend pipeline.

HOW TO RUN
----------
    python3 05_visualize_shap.py [company name]

If no company name given, produces a global SHAP summary plot
(feature importance across all companies). If a company name is given,
produces a waterfall chart for that company's prediction.

OUTPUT
------
    ../models/plots/shap_summary.png
    ../models/plots/waterfall_<company>.png
"""

import json
import os
import sys
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def load_data():
    with open("../data/sector_dashboard_data.json") as f:
        companies = [c for c in json.load(f) if c["data_status"] == "ok"]

    rows, names = [], []
    for c in companies:
        weights = {f["factor"]: f["materiality_weight"] for f in c["factors"]}
        rows.append([weights.get(factor, 0.0) for factor in ALL_FACTORS])
        names.append(c["company"])

    return np.array(rows), names


def main():
    os.makedirs("../models/plots", exist_ok=True)

    model = xgb.XGBRegressor()
    model.load_model("../models/model.json")

    X, names = load_data()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if len(sys.argv) > 1:
        company_name = " ".join(sys.argv[1:])
        if company_name not in names:
            print(f"Company '{company_name}' not found. Available: {names[:10]}...")
            return
        idx = names.index(company_name)

        plt.figure(figsize=(10, 8))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, shap_values[idx], feature_names=ALL_FACTORS, show=False, max_display=12
        )
        plt.title(f"SHAP waterfall — {company_name}")
        plt.tight_layout()
        safe_name = company_name.replace(" ", "_").replace("/", "_")
        out_path = f"../models/plots/waterfall_{safe_name}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        plt.figure(figsize=(10, 10))
        shap.summary_plot(shap_values, X, feature_names=ALL_FACTORS, show=False, max_display=15)
        plt.title("SHAP summary — materiality weight contributions across all companies")
        plt.tight_layout()
        out_path = "../models/plots/shap_summary.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
