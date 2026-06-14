"""
Maxwell Data — ESG Materiality Decision-Support
Streamlit app: type a company name, get the full 3-body materiality
analysis with heatmap, colour-coded perspective cards, and explanations.

Run:
    cd project2
    streamlit run app.py
"""

import json
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Materiality — Maxwell Data",
    page_icon="🌱",
    layout="wide",
)

# ─── load data ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "final_dashboard_data.json")

@st.cache_data
def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    companies = {
        c["company"]: c
        for c in raw["companies"]
        if c["data_status"] == "ok"
    }
    return companies, raw["model_meta"]

COMPANIES, MODEL_META = load_data()

# ─── colour helpers ───────────────────────────────────────────────────────────
CONSENSUS_COLORS = {
    "green": {"bg": "#1a3d2b", "border": "#4ade80", "dot": "#4ade80", "label": "Full consensus — distinctive vs peers (3/3 perspectives)"},
    "amber": {"bg": "#3b2f10", "border": "#fbbf24", "dot": "#fbbf24", "label": "Partial consensus (2/3 perspectives)"},
    "red":   {"bg": "#2d1515", "border": "#f87171", "dot": "#f87171", "label": "In line with peers (0–1/3 perspectives)"},
}

DIRECTION_LABELS = {
    "above_peer_average": ("⬆ Above peer average", "#f87171"),
    "below_peer_average": ("⬇ Below peer average", "#7aa8e0"),
    "in_line_with_peers": ("➡ In line with peers",  "#9bb0c5"),
    "not_material":       ("○ Not material",         "#4a5568"),
}

LENS_STATUS_LABELS = {
    "flag":    ("Flagged",    "#f87171"),
    "watch":   ("Watch",      "#7aa8e0"),
    "no_flag": ("No flag",    "#9bb0c5"),
}

# ─── sidebar: company selector ────────────────────────────────────────────────
st.sidebar.markdown("## 🌱 Maxwell Data\n### ESG Materiality Navigator")
st.sidebar.markdown("---")

company_names = sorted(COMPANIES.keys())
search = st.sidebar.text_input("Search company", placeholder="e.g. Glencore")
filtered = [n for n in company_names if search.lower() in n.lower()] if search else company_names
selected = st.sidebar.selectbox("Select company", filtered)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**3-body framework**\n\n"
    "Each ESG factor is evaluated against three perspectives:\n\n"
    "🔴 **Auditor** — disclosure & litigation risk\n\n"
    "🟡 **Corporate** — strategic alignment vs peers\n\n"
    "🔵 **Investor** — relative positioning signal\n\n"
    "Colour = how many of the 3 perspectives agree the factor is distinctive."
)

# ─── model mode caveat ────────────────────────────────────────────────────────
if MODEL_META["target_type"] == "distinctiveness_index":
    st.warning(
        "⚠️ **Demo mode**: the predictive model was trained against a materiality-based "
        "distinctiveness index (no real share-price data was available at pipeline build time). "
        "SHAP values show what drives the *index*, not a confirmed financial outcome. "
        "Run `00_fetch_financial_data.py` with internet access to switch to real volatility prediction.",
        icon="⚠️"
    )

# ─── main content ─────────────────────────────────────────────────────────────
company = COMPANIES[selected]
all_factors = company["factors"]
material_factors = [f for f in all_factors if f["materiality_weight"] > 0]
top8 = sorted(material_factors, key=lambda x: -x["materiality_weight"])[:8]

# Header
col_title, col_meta = st.columns([3, 1])
with col_title:
    st.markdown(f"# {company['company']}")
    st.markdown(f"`{company['symbol']}` · **{company['sector']}** · {len(material_factors)} material ESG factors")
with col_meta:
    st.metric("Material factors", len(material_factors))
    st.metric("In sector", company["sector"])

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Heatmap: top 8 material factors
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Top 8 Material Factors — Heatmap")
st.caption(
    "Each bar shows the company's SASB financial materiality weight (0–1) vs the sector "
    "peer-group average. The gap between them is what drives the 3-body analysis below."
)

factors_labels  = [f["factor"] for f in top8]
company_weights = [f["materiality_weight"] for f in top8]
peer_avgs       = [f["peer_group_average"] for f in top8]
gaps            = [f["gap"] for f in top8]
colors_bar      = [CONSENSUS_COLORS[f["consensus_color"]]["dot"] for f in top8]

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    name="Company materiality weight",
    y=factors_labels,
    x=company_weights,
    orientation="h",
    marker_color=colors_bar,
    text=[f"{w:.3f}" for w in company_weights],
    textposition="outside",
))
fig_bar.add_trace(go.Scatter(
    name="Sector peer average",
    y=factors_labels,
    x=peer_avgs,
    mode="markers",
    marker=dict(symbol="line-ns", size=16, color="#e8edf3", line=dict(width=2, color="#e8edf3")),
))
fig_bar.update_layout(
    template="plotly_dark",
    height=380,
    margin=dict(l=10, r=80, t=10, b=10),
    xaxis=dict(range=[0, 1.1], title="Financial materiality weight (0–1)"),
    yaxis=dict(autorange="reversed"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    plot_bgcolor="#0f151e",
    paper_bgcolor="#0a0f16",
)
st.plotly_chart(fig_bar, use_container_width=True)

# Compact gap table below the chart
with st.expander("Show raw numbers (weight / peer avg / gap)"):
    df_table = pd.DataFrame({
        "Factor": factors_labels,
        "Company weight": company_weights,
        "Peer average": peer_avgs,
        "Gap": [f"{g:+.3f}" for g in gaps],
        "Direction": [DIRECTION_LABELS[f["direction"]][0] for f in top8],
        "Consensus": [f["consensus_color"].upper() for f in top8],
    })
    st.dataframe(df_table, hide_index=True, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — 3-body colour-coded cards per factor
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔍 3-Body Materiality Analysis")
st.caption(
    "For each of the top 8 factors, the three stakeholder perspectives are evaluated "
    "based on the gap between this company's materiality weight and its sector peer average."
)

def lens_card_html(lens_name, status, reason, emoji):
    label, color = LENS_STATUS_LABELS.get(status, ("Unknown", "#9bb0c5"))
    return f"""
    <div style="background:#161d27;border:1px solid #232c39;border-radius:8px;padding:12px;height:100%;">
        <div style="font-size:11px;color:#536480;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">
            {emoji} {lens_name}
        </div>
        <div style="color:{color};font-weight:700;font-size:13px;margin-bottom:6px;">{label}</div>
        <div style="color:#cbd6e2;font-size:12px;line-height:1.5;">{reason}</div>
    </div>"""

def consensus_header_html(factor_name, consensus_color, direction, mat_weight, gap):
    cc = CONSENSUS_COLORS[consensus_color]
    dir_label, dir_color = DIRECTION_LABELS.get(direction, ("", "#9bb0c5"))
    return f"""
    <div style="background:{cc['bg']};border:1px solid {cc['border']};border-radius:10px;
                padding:14px 16px;margin-bottom:10px;display:flex;align-items:center;gap:12px;">
        <div style="width:14px;height:14px;border-radius:50%;background:{cc['dot']};
                    box-shadow:0 0 8px {cc['dot']}88;flex-shrink:0;"></div>
        <div style="flex:1;">
            <div style="font-size:15px;font-weight:700;color:#e8edf3;">{factor_name}</div>
            <div style="font-size:11px;color:#7c93ad;margin-top:2px;font-family:monospace;">
                WEIGHT {mat_weight:.3f} · GAP {gap:+.3f} · {cc['label']}
            </div>
        </div>
        <div style="font-size:11px;color:{dir_color};border:1px solid {dir_color}55;
                    border-radius:6px;padding:3px 8px;white-space:nowrap;">{dir_label}</div>
    </div>"""

def build_lens_texts(materiality, gap, peer_avg):
    """Mirrors build_sector_data.py logic for the 3-lens descriptions."""
    GAP_HIGH, GAP_LOW = 0.05, -0.05
    lenses = {}

    # Auditor
    if materiality == 0:
        lenses["auditor"] = ("no_flag", "Not flagged as financially material for this company's sub-industry.")
    elif gap >= GAP_HIGH:
        lenses["auditor"] = ("flag",    f"Above sector average ({peer_avg:.3f}) by {gap:+.3f} — elevated disclosure/assurance scrutiny likely if not addressed in reporting.")
    elif gap <= GAP_LOW:
        lenses["auditor"] = ("watch",   f"Below sector average ({peer_avg:.3f}) by {gap:.3f} — lower audit priority, but confirm this reflects genuine business model differences.")
    else:
        lenses["auditor"] = ("no_flag", f"Materiality ({materiality:.3f}) is broadly in line with sector average ({peer_avg:.3f}).")

    # Corporate
    if materiality == 0:
        lenses["corporate"] = ("no_flag", "Not a SASB-flagged material factor — limited strategic urgency from a materiality standpoint.")
    elif gap >= GAP_HIGH:
        lenses["corporate"] = ("flag",    "Above-sector-average materiality — this should be a differentiated strategic priority, not just a generic sector-wide issue.")
    elif gap <= GAP_LOW:
        lenses["corporate"] = ("watch",   "Below-sector-average materiality — strategic resourcing here may be disproportionate to its financial relevance relative to peers.")
    else:
        lenses["corporate"] = ("no_flag", "Sector-typical materiality — address as part of standard sector-wide sustainability strategy.")

    # Investor
    if materiality == 0:
        lenses["investor"] = ("no_flag", "Not flagged as financially material — no investor-relevant signal from this dataset alone.")
    elif gap >= GAP_HIGH:
        lenses["investor"] = ("flag",    f"Materiality is {gap:+.3f} above sector average — investors comparing this company against peers would expect above-average exposure here to be disclosed and, if unaddressed, may price it as a risk premium.")
    elif gap <= GAP_LOW:
        lenses["investor"] = ("watch",   f"Materiality is {gap:.3f} below sector average — lower relative exposure vs peers on this factor.")
    else:
        lenses["investor"] = ("no_flag", "Sector-typical materiality — no notable relative signal for investors.")

    return lenses


for f in top8:
    mat   = f["materiality_weight"]
    pavg  = f["peer_group_average"]
    gap   = f["gap"]
    color = f["consensus_color"]
    lenses = build_lens_texts(mat, gap, pavg)

    st.markdown(
        consensus_header_html(f["factor"], color, f["direction"], mat, gap),
        unsafe_allow_html=True,
    )

    col_a, col_c, col_i = st.columns(3)
    with col_a:
        st.markdown(lens_card_html("Auditor", *lenses["auditor"], "📋"), unsafe_allow_html=True)
    with col_c:
        st.markdown(lens_card_html("Corporate", *lenses["corporate"], "🏢"), unsafe_allow_html=True)
    with col_i:
        st.markdown(lens_card_html("Investor", *lenses["investor"], "📈"), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Explanation bullet points per factor
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 💡 Decisions & Explanations")
st.caption(
    "For each of the top 8 factors: the recommended decision, an illustrative estimate, "
    "and the auditor / corporate / investor perspective explanations. "
    "Items marked 🤖 were generated by the AI narrative layer (Stage 3); "
    "items marked 📐 use template text derived from the gap formula."
)

for f in top8:
    exp = f.get("explanation", {})
    source_icon = "🤖" if exp.get("source") == "ai_generated" else "📐"
    evidence = f.get("evidence")
    model_info = f.get("model")

    with st.expander(f"{source_icon} **{f['factor']}** — {f['consensus_color'].upper()} · gap {f['gap']:+.3f}", expanded=False):

        # Decision
        st.markdown("#### Decision")
        st.info(f["decision"])

        # Estimate
        est = f.get("estimate", {})
        st.markdown("#### Estimate")
        st.markdown(f"**{est.get('metric', '')}:** {est.get('direction', '')}")
        st.caption(est.get("basis", ""))

        # Model attribution (if available)
        if model_info:
            st.markdown("#### Model attribution (SHAP)")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("SHAP contribution", f"{model_info['shap_value']:+.5f}")
            mcol2.metric("Rank among factors", f"#{model_info['shap_rank']}")
            mcol3.metric("Model target", model_info["target_type"].replace("_", " "))
            st.caption(
                f"SHAP value = how much this factor's materiality weight pushed the model's "
                f"prediction for {company['company']} above/below the average prediction "
                f"({model_info['target_type'].replace('_',' ')}). "
                f"In-sample R² = {model_info['in_sample_r2']:.3f} (n={MODEL_META['n_training_rows']})."
            )

        # Evidence (if retrieved by Stage 1)
        if evidence:
            st.markdown("#### Evidence retrieved (Stage 1)")
            dir_colors = {"positive": "normal", "negative": "error", "mixed": "warning"}
            st.markdown(
                f"**Direction:** `{evidence['direction']}` · "
                f"**Confidence:** `{evidence['confidence']}`"
            )
            st.markdown(f"> {evidence['summary']}")
            if evidence.get("sources"):
                for src in evidence["sources"]:
                    st.markdown(f"- [{src}]({src})")

        # Perspectives
        st.markdown("#### Perspectives")
        if exp.get("auditor_view"):
            pcol_a, pcol_c, pcol_i = st.columns(3)
            with pcol_a:
                st.markdown("**📋 Auditor**")
                st.markdown(exp["auditor_view"])
            with pcol_c:
                st.markdown("**🏢 Corporate**")
                st.markdown(exp["corporate_view"])
            with pcol_i:
                st.markdown("**📈 Investor**")
                st.markdown(exp["investor_view"])

        # Key uncertainties (AI-generated explanations only)
        uncertainties = exp.get("key_uncertainties", [])
        if uncertainties:
            st.markdown("#### Key uncertainties")
            for u in uncertainties:
                st.markdown(f"- {u}")

st.markdown("---")
st.caption(
    "Source: FTSE100 financial materiality dataset (March 2025), SASB-based. "
    "Gaps are computed as company materiality weight − sector peer-group average (non-zero entries only). "
    "No performance/confidence scores are invented — all numbers trace to the source spreadsheet. "
    "AI is used for evidence retrieval (Stage 1), explainable ML (Stage 2), and grounded narrative "
    "generation (Stage 3). Not financial advice."
)
