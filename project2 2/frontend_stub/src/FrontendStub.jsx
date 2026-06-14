import React, { useState } from "react";
import DATA from "./data/final_dashboard_data.json";

/**
 * FRONTEND STUB — for the frontend team's reference.
 *
 * This is a minimal, intentionally bare-bones component showing how to
 * consume `final_dashboard_data.json` (the output of scripts/04_merge_final.py).
 * It is NOT styled to match the earlier dashboard mockups — it exists only
 * to document the data shape and demonstrate which fields are now available
 * after the 3-stage AI pipeline:
 *
 *   factor.materiality_weight / peer_group_average / gap / consensus_color
 *     -> unchanged from before (deterministic, from spreadsheet)
 *
 *   factor.model = { target_type, shap_value, shap_rank, in_sample_r2 }
 *     -> NEW: output of the explainable ML model (Stage 2). shap_value is
 *        signed (positive = pushes prediction up); shap_rank is this
 *        factor's rank (1 = most influential) among this company's factors.
 *
 *   factor.evidence = { summary, direction, confidence, sources[] }
 *     -> NEW, OPTIONAL: present only for (company, factor) pairs where
 *        Stage 1 (evidence retrieval) found something. `direction` is
 *        "positive" | "negative" | "mixed". Always check for presence.
 *
 *   factor.explanation = { auditor_view, corporate_view, investor_view,
 *                           key_uncertainties[], source }
 *     -> `source` is either "ai_generated" (Stage 3 ran for this factor)
 *        or "template" (deterministic fallback text). The frontend should
 *        probably show a small badge distinguishing these, since
 *        "ai_generated" explanations reference real evidence/SHAP and
 *        "template" ones are generic gap descriptions.
 *
 * `DATA.model_meta` (top level) describes what the model was trained to
 * predict — show this somewhere so users understand what "shap_value"
 * means. In particular, check `target_type`:
 *   - "volatility_6m"        -> model predicts real share-price volatility
 *   - "distinctiveness_index" -> FALLBACK MODE: model predicts a
 *                                 materiality-based index, not a financial
 *                                 outcome. Show a caveat banner if this is
 *                                 the case (see ModelMetaBanner below).
 */

function ModelMetaBanner({ meta }) {
  const isFallback = meta.target_type === "distinctiveness_index";
  return (
    <div style={{ padding: 12, border: "1px solid #ccc", borderRadius: 8, marginBottom: 16 }}>
      <strong>Model target:</strong> {meta.target_type} (in-sample R² = {meta.in_sample_r2}, n = {meta.n_training_rows})
      {isFallback && (
        <div style={{ marginTop: 8, color: "#a35", fontSize: 13 }}>
          No real financial data was available when this model was trained — it was trained
          against a materiality-based "distinctiveness index" computed from this dataset
          itself, NOT a financial outcome. SHAP values below show which factors drive THIS
          INDEX, not a confirmed financial effect. Re-run with real share-price data
          (00_fetch_financial_data.py) for a financial interpretation.
        </div>
      )}
    </div>
  );
}

function EvidenceBadge({ evidence }) {
  if (!evidence) return <span style={{ fontSize: 12, color: "#888" }}>No evidence retrieved</span>;
  const colors = { positive: "#2a7", negative: "#c33", mixed: "#a80" };
  return (
    <span style={{ fontSize: 12, color: colors[evidence.direction] || "#888" }}>
      Evidence: {evidence.direction} ({evidence.confidence} confidence, {evidence.sources.length} source(s))
    </span>
  );
}

function FactorRow({ f }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ border: "1px solid #ddd", borderRadius: 6, marginBottom: 8, padding: 10 }}>
      <div onClick={() => setOpen(!open)} style={{ cursor: "pointer", display: "flex", justifyContent: "space-between" }}>
        <strong>{f.factor}</strong>
        <span>
          gap {f.gap >= 0 ? "+" : ""}{f.gap.toFixed(3)}
          {f.model && (
            <> · SHAP {f.model.shap_value >= 0 ? "+" : ""}{f.model.shap_value.toFixed(4)} (rank #{f.model.shap_rank})</>
          )}
        </span>
      </div>
      {open && (
        <div style={{ marginTop: 8, fontSize: 13 }}>
          <div>{f.decision}</div>
          <div style={{ marginTop: 4 }}><EvidenceBadge evidence={f.evidence} /></div>
          {f.evidence && (
            <p style={{ fontStyle: "italic", color: "#555" }}>{f.evidence.summary}</p>
          )}
          <div style={{ marginTop: 8 }}>
            <span style={{
              fontSize: 11, padding: "2px 6px", borderRadius: 4,
              background: f.explanation.source === "ai_generated" ? "#dfe" : "#eee"
            }}>
              {f.explanation.source === "ai_generated" ? "AI-generated explanation" : "Template explanation"}
            </span>
            <p style={{ marginTop: 6 }}><strong>Auditor:</strong> {f.explanation.auditor_view}</p>
            <p><strong>Corporate:</strong> {f.explanation.corporate_view}</p>
            <p><strong>Investor:</strong> {f.explanation.investor_view}</p>
            {f.explanation.key_uncertainties && f.explanation.key_uncertainties.length > 0 && (
              <div>
                <strong>Key uncertainties:</strong>
                <ul>
                  {f.explanation.key_uncertainties.map((u, i) => <li key={i}>{u}</li>)}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function FrontendStub() {
  const companies = DATA.companies.filter(c => c.data_status === "ok");
  const [selected, setSelected] = useState(companies[0]?.company);
  const company = companies.find(c => c.company === selected);

  return (
    <div style={{ fontFamily: "sans-serif", padding: 20, maxWidth: 800 }}>
      <h2>Final dashboard data — frontend stub</h2>
      <ModelMetaBanner meta={DATA.model_meta} />

      <select value={selected} onChange={e => setSelected(e.target.value)}>
        {companies.map(c => <option key={c.company} value={c.company}>{c.company}</option>)}
      </select>

      {company && (
        <div style={{ marginTop: 16 }}>
          <h3>{company.company} ({company.sector})</h3>
          {[...company.factors]
            .filter(f => f.materiality_weight > 0)
            .sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap))
            .map(f => <FactorRow key={f.factor} f={f} />)}
        </div>
      )}
    </div>
  );
}
