# NeuroBridge-S4 Longitudinal Review Dashboard

## Purpose

The dashboard provides a local reviewer-facing interface for inspecting
within-subject biological adaptation graph trajectories. For a selected subject
and timepoint it shows what changed relative to personal baseline, what drove
the change, how it maps to NASA HRP hazard contexts, and whether the change
exceeds the current reference-calibrated variability envelope.

It is a **local research-review prototype**, intended to support expert review.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app starts a local web server (default `http://localhost:8501`). No
database, authentication, or cloud deployment is involved.

## Required inputs

The dashboard reads CSV tables from `results/tables/`.

**Required (Phase 6 core):**

- `longitudinal_node_deltas.csv`
- `longitudinal_graph_deltas.csv`

**Optional Phase 6:**

- `longitudinal_hazard_deltas.csv`
- `longitudinal_trajectory_summary.csv`
- `recovery_metrics.csv`

**Optional Phase 7 (attribution):**

- `trajectory_node_attribution.csv`
- `trajectory_graph_metric_attribution.csv`
- `trajectory_subgraph_attribution.csv`
- `trajectory_hazard_attribution.csv`
- `recovery_attribution.csv`
- `phase7_attribution_summary.csv`

**Optional Phase 8 (reference envelope):**

- `reference_calibrated_node_delta_scores.csv`
- `reference_calibrated_graph_delta_scores.csv`
- `reference_calibrated_hazard_delta_scores.csv`
- `phase8_reference_calibrated_summary.csv`
- `reference_trajectory_envelope.csv`

If the required tables are missing, the dashboard shows a friendly message and
the input readiness report instead of crashing. Missing optional tables simply
limit the corresponding panels.

### Derived hazard-context deltas

`longitudinal_hazard_deltas.csv` is optional. When it is missing, the dashboard
derives hazard-context trajectory deltas on the fly from
`longitudinal_node_deltas.csv` and the HRP hazard-domain mapping
(`hazard_domain_mapping.csv` when present, otherwise the built-in default
mapping) via
`neurobridge_graph.trajectory_features.ensure_longitudinal_hazard_deltas`. The
derived table is saved back to `results/tables/longitudinal_hazard_deltas.csv`
with these columns:

```text
subject_id, timepoint, mission_phase, hazard,
baseline_hazard_relevance, current_hazard_relevance, delta_hazard_relevance,
coverage_fraction, top_contributing_domain, interpretation
```

When per-domain baseline and current activations are available, baseline and
current hazard-context relevance are computed as weighted means over the mapped
domains that are present, and the delta is their difference. When only domain
deltas are available, the delta hazard-context relevance is computed directly as
the weighted domain delta and the baseline is documented as an assumed `0.0`.

If neither the explicit table nor the domain-delta/hazard-mapping inputs are
available, the hazard-context panel shows:

> Hazard-context trajectory data are unavailable because the required
> domain-delta and hazard-mapping inputs were not found. Run Phase 6 hazard
> delta generation or provide `longitudinal_hazard_deltas.csv`.

These derived values are **hazard-context relevance**, not exposure measurement,
not risk scoring, and not diagnosis.

## What the dashboard shows

- subject/timepoint overview;
- baseline-relative domain deltas;
- graph metric trajectories;
- HRP hazard-context shifts;
- explainable trajectory attribution;
- reference-calibrated envelope status;
- recovery behavior;
- data readiness and limitations.

The sidebar provides a subject selector, a timepoint selector, and optional
toggles to show raw tables, method notes, and the input readiness report.

## Guardrails

This is not a clinical system, not diagnosis, not treatment guidance, not
exposure measurement, and not health risk scoring. Hazard-context alignment is
not exposure measurement. Outside-envelope means a baseline-relative change is
larger than expected under the current calibration data — a candidate for
expert review, not a verdict.

## Troubleshooting

If the dashboard says Phase 6–8 outputs are missing, run notebooks `04`, `05`,
and `06` first:

- `notebooks/04_Within_Subject_Longitudinal_Graph_Trajectories.ipynb`
- `notebooks/05_Explainable_Trajectory_Attribution.ipynb`
- `notebooks/06_Reference_Calibrated_Trajectory_Envelope.ipynb`
