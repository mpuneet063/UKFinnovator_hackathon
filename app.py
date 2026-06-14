"""
Maxwell Data — ESG Materiality Navigator
Clean, exec-briefing style. White/soft finance aesthetic.

Run:
    cd project2
    streamlit run app.py
"""

import json
import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Materiality — Maxwell Data",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #f8f9fb; }
  section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8ecf0; }
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }

  .factor-card {
    background: #ffffff;
    border: 1px solid #e4e8ee;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
  }

  .pill {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .03em;
    margin-right: 4px;
  }
  .pill-green { background:#dcfce7; color:#15803d; }
  .pill-amber { background:#fef9c3; color:#92400e; }
  .pill-slate { background:#f1f5f9; color:#64748b; }
  .pill-blue  { background:#dbeafe; color:#1d4ed8; }
  .pill-red   { background:#fee2e2; color:#b91c1c; }

  .persp-label {
    font-weight: 600; color: #6b7280; font-size: 11px;
    text-transform: uppercase; letter-spacing: .06em; margin-right: 6px;
  }

  .section-header {
    font-size: 11px; font-weight: 700; color: #9ca3af;
    text-transform: uppercase; letter-spacing: .1em;
    margin: 24px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #e4e8ee;
  }

  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ── data ──────────────────────────────────────────────────────────────────────
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

CONSENSUS = {
    "green": {"dot": "🟢", "label": "Priority",    "pill": "pill-green", "bar": "#16a34a"},
    "amber": {"dot": "🟡", "label": "Investigate",  "pill": "pill-amber", "bar": "#d97706"},
    "red":   {"dot": "⚪", "label": "Standard",     "pill": "pill-slate", "bar": "#d1d5db"},
}

DIRECTION_SHORT = {
    "above_peer_average": ("↑ Above peers", "pill-green"),
    "below_peer_average": ("↓ Below peers", "pill-red"),
    "in_line_with_peers": ("→ In line",     "pill-blue"),
    "not_material":       ("— N/A",         "pill-slate"),
}

# Bar chart colours — driven by DIRECTION only
# Green = above peer average (higher exposure than sector)
# Blue  = in line with peers (sector standard)
# Red   = below peer average (lower exposure than sector)
DIRECTION_BAR = {
    "above_peer_average": "#16a34a",
    "in_line_with_peers": "#4a90d9",
    "below_peer_average": "#e05252",
    "not_material":       "#e2e8f0",
}

DIRECTION_LABEL_TEXT = {
    "above_peer_average": "Above peer average",
    "in_line_with_peers": "In line with peers",
    "below_peer_average": "Below peer average",
    "not_material":       "Not material",
}

def lens_verdicts(mat, gap):
    H, L = 0.05, -0.05
    if mat == 0:
        return (
            "Not financially material for this sub-industry — no audit flag.",
            "Not flagged by SASB — no strategic urgency from materiality alone.",
            "No investor-relevant signal from materiality data alone.",
        )
    if gap >= H:
        a = f"Above-sector-average materiality (+{gap:.3f}) — disclosure scrutiny likely if not addressed in reporting."
        c = "Distinctive exposure vs peers — treat as a differentiated priority, not a generic sector issue."
        i = f"Gap of +{gap:.3f} above peers — if not disclosed adequately, may be priced as a risk premium by the market."
    elif gap <= L:
        a = f"Below sector average ({gap:.3f}) — lower audit priority; confirm this reflects genuine business model differences."
        c = "Below-peer materiality — strategic resourcing here may exceed its financial relevance relative to peers."
        i = f"Lower relative exposure vs peers ({gap:.3f}) — less likely to drive a sector-relative risk premium."
    else:
        a = f"Materiality ({mat:.3f}) in line with sector average — no standout disclosure risk from this factor alone."
        c = "Sector-typical — address as part of standard sustainability strategy."
        i = "No notable relative signal vs peers on this factor."
    return a, c, i

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 Maxwell Data")
    st.markdown("**ESG Materiality Navigator**")
    st.markdown("---")

    search   = st.text_input("Search company", placeholder="Type name…")
    names    = sorted(COMPANIES.keys())
    shown    = [n for n in names if search.lower() in n.lower()] if search else names
    selected = st.selectbox("Select", shown, label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px;color:#6b7280;line-height:1.8'>"
        "<b>Priority levels</b><br>"
        "🟢 <b>Priority</b> — all 3 perspectives agree<br>"
        "🟡 <b>Investigate</b> — 2 of 3 perspectives flag it<br>"
        "⚪ <b>Standard</b> — in line with sector peers<br><br>"
        "<b>Gap</b> = company weight − sector peer average<br>"
        "Source: FTSE100 SASB materiality (March 2025)"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    with st.expander("⚙️ AI pipeline"):
        st.markdown("""
**Stage 1** — Claude + web search retrieves cited evidence per factor

**Stage 2** — XGBoost + SHAP attributes each factor's contribution to a financial prediction

**Stage 3** — Claude writes perspectives grounded strictly in Stage 1 & 2 outputs

**Everything else** is deterministic arithmetic on the source spreadsheet
        """)

# ── main ──────────────────────────────────────────────────────────────────────
company      = COMPANIES[selected]
all_factors  = company["factors"]
material     = [f for f in all_factors if f["materiality_weight"] > 0]
non_material = [f for f in all_factors if f["materiality_weight"] == 0]
top8         = sorted(material, key=lambda x: -x["materiality_weight"])[:8]
others       = sorted(material, key=lambda x: -x["materiality_weight"])[8:]

n_green = sum(1 for f in material if f["consensus_color"] == "green")
n_amber = sum(1 for f in material if f["consensus_color"] == "amber")
n_red   = len(material) - n_green - n_amber
# direction counts — used for bar chart summary
n_above  = sum(1 for f in material if f["direction"] == "above_peer_average")
n_inline = sum(1 for f in material if f["direction"] == "in_line_with_peers")
n_below  = sum(1 for f in material if f["direction"] == "below_peer_average")

if "show_others" not in st.session_state:
    st.session_state.show_others = False
if "expanded" not in st.session_state:
    st.session_state.expanded = None

# header
st.markdown(f"## {company['company']}")
st.markdown(
    f"<span style='color:#6b7280;font-size:14px'>"
    f"`{company['symbol']}` &nbsp;·&nbsp; {company['sector']}"
    f"</span>",
    unsafe_allow_html=True,
)

# summary row — two rows: action signal + directional signal
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Material factors", len(material))
col2.metric("🟢 Priority",      n_green,  help="All 3 perspectives agree this factor needs attention")
col3.metric("🟡 Investigate",   n_amber,  help="2 of 3 perspectives flag this factor")
col4.metric("⚪ Standard",      n_red,    help="In line with sector — no distinctive signal")
col5.metric("Not material",     len(non_material), help="SASB weight = 0 for this sub-industry")

# direction summary (bar chart context)
dcol1, dcol2, dcol3, dcol4 = st.columns(4)
dcol1.markdown(
    f"<div style='font-size:12px;color:#6b7280'>vs sector peers</div>"
    f"<div style='font-size:13px;font-weight:600;color:#16a34a'>↑ {n_above} above average</div>",
    unsafe_allow_html=True)
dcol2.markdown(
    f"<div style='font-size:12px;color:#6b7280'>&nbsp;</div>"
    f"<div style='font-size:13px;font-weight:600;color:#4a90d9'>→ {n_inline} in line</div>",
    unsafe_allow_html=True)
dcol3.markdown(
    f"<div style='font-size:12px;color:#6b7280'>&nbsp;</div>"
    f"<div style='font-size:13px;font-weight:600;color:#e05252'>↓ {n_below} below average</div>",
    unsafe_allow_html=True)
dcol4.markdown(
    f"<div style='font-size:12px;color:#9ca3af;font-size:11px;margin-top:16px'>"
    f"Gap = company weight − sector peer avg</div>",
    unsafe_allow_html=True)

if MODEL_META["target_type"] == "distinctiveness_index":
    st.caption(
        "ℹ️ Model in demo mode — no real share-price data at build time. "
        "Run `00_fetch_financial_data.py` with internet access for live financial predictions."
    )

st.markdown("---")

# ── bar chart ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">Top 8 material factors — financial materiality weight vs sector peer average</div>',
    unsafe_allow_html=True,
)

fig = go.Figure()
fig.add_trace(go.Bar(
    y=[f["factor"] for f in top8],
    x=[f["materiality_weight"] for f in top8],
    orientation="h",
    marker_color=[DIRECTION_BAR.get(f["direction"], "#94a3b8") for f in top8],
    marker_line_width=0,
    text=[f"{f['materiality_weight']:.3f}" for f in top8],
    textposition="outside",
    textfont=dict(size=12, color="#374151"),
    name="Company weight",
))
fig.add_trace(go.Scatter(
    y=[f["factor"] for f in top8],
    x=[f["peer_group_average"] for f in top8],
    mode="markers",
    name="Sector peer average",
    marker=dict(symbol="line-ns", size=18, color="#94a3b8",
                line=dict(width=2.5, color="#94a3b8")),
))
fig.update_layout(
    template="plotly_white",
    height=320,
    margin=dict(l=0, r=60, t=10, b=10),
    xaxis=dict(range=[0, 1.1], showgrid=True, gridcolor="#f1f5f9",
               tickfont=dict(color="#9ca3af", size=11)),
    yaxis=dict(autorange="reversed", tickfont=dict(color="#374151", size=12)),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#f8f9fb",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                font=dict(size=11, color="#6b7280")),
)
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    "<div style='display:flex;gap:24px;font-size:12px;color:#6b7280;margin-top:4px'>"
    "<span><span style='color:#16a34a;font-weight:700'>■</span>"
    "  Above peer average</span>"
    "<span><span style='color:#4a90d9;font-weight:700'>■</span>"
    "  In line with peers</span>"
    "<span><span style='color:#e05252;font-weight:700'>■</span>"
    "  Below peer average</span>"
    "<span style='color:#94a3b8;margin-left:8px'>| tick mark = sector peer average</span>"
    "</div>",
    unsafe_allow_html=True,
)

# ── factor cards ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">Lens consensus — which factors are flagged by which perspectives</div>',
    unsafe_allow_html=True,
)

# ── Lens consensus visual ─────────────────────────────────────────────────────
# All 3 lenses (Auditor, Corporate, Investor) use the same gap threshold (±0.05),
# so a factor is flagged by ALL THREE or by NONE — binary split.
# We show this as a donut chart + factor list, which is more honest than a Venn
# that would always show empty "2 of 3" zones.

flagged_factors   = [f for f in material if abs(f["gap"]) >= 0.05 and f["materiality_weight"] > 0]
unflagged_factors = [f for f in material if abs(f["gap"]) < 0.05  and f["materiality_weight"] > 0]

donut = go.Figure(go.Pie(
    labels=["All 3 perspectives flag", "In line — no flag"],
    values=[len(flagged_factors), len(unflagged_factors)],
    hole=0.65,
    marker_colors=["#7c3aed", "#e5e7eb"],
    textinfo="label+value",
    textfont=dict(size=12),
    hovertemplate="%{label}: %{value} factors<extra></extra>",
))
donut.update_layout(
    showlegend=False,
    margin=dict(l=10, r=10, t=10, b=10),
    height=220,
    paper_bgcolor="#f8f9fb",
    annotations=[dict(
        text=f"<b>{len(flagged_factors)}</b><br><span style='font-size:10px'>flagged</span>",
        x=0.5, y=0.5, font_size=16, showarrow=False
    )],
)

vc1, vc2, vc3 = st.columns([1, 1, 2])
with vc1:
    st.plotly_chart(donut, use_container_width=True)

with vc2:
    st.markdown(
        "<div style='padding-top:20px;font-size:12px;line-height:2.2;color:#374151'>"
        f"<span style='color:#7c3aed;font-size:16px;font-weight:700'>●</span> "
        f"<b>{len(flagged_factors)}</b> factors — all 3 perspectives flag<br>"
        f"<span style='color:#e5e7eb;font-size:16px;font-weight:700;-webkit-text-stroke:1px #9ca3af'>●</span> "
        f"<b>{len(unflagged_factors)}</b> factors — in line with sector<br><br>"
        f"<span style='font-size:11px;color:#9ca3af'>"
        f"All three perspectives use gap ≥ ±0.05 as the threshold. "
        f"A factor is either flagged by all three or none.</span>"
        "</div>",
        unsafe_allow_html=True,
    )

with vc3:
    if flagged_factors:
        st.markdown(
            "<div style='padding-top:20px;font-size:12px;color:#374151'>"
            "<b style='color:#7c3aed'>Flagged by all 3 perspectives:</b><br>"
            + "".join(
                f"<div style='padding:3px 0;border-bottom:1px solid #f1f5f9'>"
                f"{'↑' if f['gap']>0 else '↓'} {f['factor']} "
                f"<span style='color:#9ca3af'>gap {f['gap']:+.3f}</span></div>"
                for f in sorted(flagged_factors, key=lambda x: -abs(x["gap"]))
            )
            + "</div>",
            unsafe_allow_html=True,
        )


st.markdown('<div class="section-header">Factor breakdown — decision, perspectives & explanation</div>',
    unsafe_allow_html=True,
)

for f in top8:
    cc  = f["consensus_color"]
    cfg = CONSENSUS[cc]
    dlabel, dpill = DIRECTION_SHORT.get(f["direction"], ("", "pill-slate"))

    a_v, c_v, i_v = lens_verdicts(f["materiality_weight"], f["gap"])

    exp      = f.get("explanation", {})
    evidence = f.get("evidence")
    narr_src = exp.get("source", "template")

    if narr_src == "ai_generated":
        a_text = exp.get("auditor_view", a_v)
        c_text = exp.get("corporate_view", c_v)
        i_text = exp.get("investor_view", i_v)
    else:
        a_text, c_text, i_text = a_v, c_v, i_v

    decision_raw   = f.get("decision", "")
    decision_short = decision_raw.split("—")[1].strip() if "—" in decision_raw else decision_raw

    fid     = f["factor"]
    is_open = st.session_state.expanded == fid

    # card header
    st.markdown(
        f"""<div class="factor-card">
          <div style="display:flex;align-items:center;gap:8px">
            <span>{cfg['dot']}</span>
            <span style="font-size:14px;font-weight:600;color:#111827;flex:1">{f['factor']}</span>
            <span class="pill {dpill}">{dlabel}</span>
            <span class="pill {cfg['pill']}">{cfg['label']}</span>
            <span style="font-size:12px;color:#9ca3af;font-family:monospace">
              {f['materiality_weight']:.3f} &nbsp;|&nbsp; gap {f['gap']:+.3f}
            </span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # toggle
    btn_label = "▲ Close" if is_open else "▼ View details"
    if st.button(btn_label, key=f"btn_{fid}", use_container_width=False):
        st.session_state.expanded = None if is_open else fid
        st.rerun()

    if is_open:
        # decision + estimate
        est = f.get("estimate", {})
        st.markdown(
            f"<div style='padding:12px 0 4px 0;font-size:14px;font-weight:600;"
            f"color:#111827'>→ {decision_short}</div>",
            unsafe_allow_html=True,
        )
        if est.get("direction"):
            st.markdown(
                f"<div style='font-size:12px;color:#6b7280;margin-bottom:12px'>"
                f"📊 {est['direction']}</div>",
                unsafe_allow_html=True,
            )

        # evidence block
        if evidence:
            ev_colors = {
                "positive": ("#dcfce7", "#15803d"),
                "negative": ("#fee2e2", "#b91c1c"),
                "mixed":    ("#fef9c3", "#92400e"),
            }
            bg, fg = ev_colors.get(evidence["direction"], ("#f1f5f9", "#374151"))
            src_links = " ".join(
                f"<a href='{s}' target='_blank' style='color:{fg};font-size:11px'>[source]</a>"
                for s in evidence.get("sources", [])
            )
            st.markdown(
                f"<div style='background:{bg};color:{fg};border-radius:6px;"
                f"padding:10px 14px;font-size:12px;margin-bottom:12px;line-height:1.6'>"
                f"<b>Evidence ({evidence['direction']} · {evidence['confidence']} confidence)</b><br>"
                f"{evidence['summary']} {src_links}</div>",
                unsafe_allow_html=True,
            )

        # three perspectives — single block, inline labels
        st.markdown(
            f"""<div style='background:#f8f9fb;border:1px solid #e4e8ee;border-radius:8px;
                           padding:14px 16px;line-height:1.8;margin-bottom:8px'>
              <div style='font-size:13px;color:#374151'>
                <span class='persp-label'>📋 Auditor</span>{a_text}
              </div>
              <div style='font-size:13px;color:#374151;margin-top:6px'>
                <span class='persp-label'>🏢 Corporate</span>{c_text}
              </div>
              <div style='font-size:13px;color:#374151;margin-top:6px'>
                <span class='persp-label'>📈 Investor</span>{i_text}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # key uncertainties
        uncertainties = exp.get("key_uncertainties", [])
        if uncertainties:
            unc_html = "".join(
                f"<div style='font-size:12px;color:#6b7280;padding:1px 0'>• {u}</div>"
                for u in uncertainties
            )
            st.markdown(
                f"<div style='margin-bottom:8px'>"
                f"<div style='font-size:11px;font-weight:600;color:#9ca3af;"
                f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px'>"
                f"What would sharpen this</div>{unc_html}</div>",
                unsafe_allow_html=True,
            )

        # SHAP (collapsed)
        model_info = f.get("model")
        if model_info:
            with st.expander("🔬 How was this calculated?"):
                mc1, mc2 = st.columns(2)
                mc1.metric("SHAP contribution", f"{model_info['shap_value']:+.5f}")
                mc2.metric("Factor rank (|SHAP|)", f"#{model_info['shap_rank']}")
                st.caption(
                    f"XGBoost model on 26 SASB materiality weights (n={MODEL_META['n_training_rows']}). "
                    f"SHAP shows how this factor pushed the prediction for {company['company']} "
                    f"vs the average. Target: {model_info['target_type'].replace('_',' ')} "
                    f"(in-sample R²={model_info['in_sample_r2']:.3f})."
                )

        if narr_src == "ai_generated":
            st.markdown(
                "<div style='font-size:11px;color:#9ca3af;margin-top:6px'>"
                "🤖 Perspectives written by AI, grounded in retrieved evidence and model "
                "attributions — no figures outside this data were introduced."
                "</div>",
                unsafe_allow_html=True,
            )

# ── other factors link ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if others or non_material:
    if st.button(
        f"{'▼' if st.session_state.show_others else '▶'} "
        f"View {len(others)} more material factors"
        + (f" + {len(non_material)} non-material" if non_material else ""),
        key="show_others_btn",
    ):
        st.session_state.show_others = not st.session_state.show_others
        st.rerun()

    if st.session_state.show_others:
        st.markdown(
            '<div class="section-header">Other material factors</div>',
            unsafe_allow_html=True,
        )
        if others:
            rows = []
            for f in others:
                dlabel, _ = DIRECTION_SHORT.get(f["direction"], ("", ""))
                rows.append({
                    "Factor": f["factor"],
                    "Weight": f["materiality_weight"],
                    "Peer avg": f["peer_group_average"],
                    "Gap": f"{f['gap']:+.3f}",
                    "Priority": CONSENSUS[f["consensus_color"]]["label"],
                    "Direction": dlabel,
                })
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Weight": st.column_config.ProgressColumn(
                        "Weight", min_value=0, max_value=1, format="%.3f"
                    ),
                    "Peer avg": st.column_config.ProgressColumn(
                        "Peer avg", min_value=0, max_value=1, format="%.3f"
                    ),
                },
            )

        if non_material:
            with st.expander(f"Non-material factors ({len(non_material)}) — SASB weight = 0 for this sub-industry"):
                st.caption(
                    "Not flagged as financially material for this company's SASB sub-industry. "
                    "May become material if the business model or SASB standards change."
                )
                st.markdown(", ".join(f["factor"] for f in non_material))

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:11px;color:#9ca3af;border-top:1px solid #e4e8ee;padding-top:12px'>"
    "Source: FTSE100 SASB financial materiality dataset (March 2025). "
    "Gap = company weight − sector peer-group average. Not financial advice."
    "</div>",
    unsafe_allow_html=True,
)
