# Limitations and Guardrails

## Data limitations

- Phase 1 does not include live data processing.
- Later phases begin from processed outputs from the original NeuroBridge-S4 demonstration.
- NHANES is a health-filtered terrestrial reference approximation, not astronaut data.
- Cross-sectional datasets cannot fully represent mission-phase adaptation or recovery.
- Real longitudinal astronaut data are not included in this repository.
- Schema demonstration longitudinal tables test the pipeline only; they are not scientific evidence.

## Modeling limitations

- Supervised learning on n=4 is not appropriate.
- Within-subject trajectory deltas are not diagnosis.
- One-time population comparison is insufficient for astronaut monitoring.
- Reference cohorts provide calibration, not the primary monitoring endpoint.
- Correlation edges are not causal proof.
- Conceptual edges require literature support and expert review.
- Missingness can distort graph features and must be modeled explicitly.
- Hazard-context deltas are interpretation context, not exposure measurement.

## Interpretation guardrails

This project does not:

- diagnose medical or psychiatric states;
- prescribe treatment;
- infer brain chemistry directly;
- predict astronaut health outcomes;
- claim clinical validation;
- replace expert review.

This project supports:

- signal triage;
- reference-relative interpretation;
- graph-based explanation;
- monitoring-priority discussion;
- small-N methodology development.

## Phase 7 attribution limitations

- Attribution is **descriptive arithmetic, not causal inference**: contribution shares show
  which components account for the observed baseline-relative shift, not why it occurred.
- If example longitudinal data are used, they are **schema demonstration only** and are not
  scientific evidence.
- Hazard-context attribution is **hazard-context alignment, not exposure attribution**: it maps
  graph shifts onto HRP hazard categories and never measures actual hazard exposure or effect.
- Recovery attribution depends on **timepoint availability and baseline quality**; sparse or
  noisy timepoints reduce the reliability of recovery categories.
- Sparse domains limit **subgraph attribution**: a subgraph whose domains are absent from the
  dataset is reported as unavailable rather than inferred.

## Phase 8 reference-envelope limitations

- Envelope quality depends entirely on **reference/analog data quality**; a poor calibration
  source yields a poor envelope.
- **Example envelopes are not evidence**: the schema-demonstration envelope exists only to
  exercise the workflow and must be replaced with real analog/reference variability data.
- **Envelope exceedance is descriptive**, not a clinical or operational threshold.
- The envelope **does not validate clinical thresholds** and does not define a healthy-vs-
  unhealthy endpoint.
- **Domain coverage limitations** affect interpretation: features without reference calibration
  are scored as `insufficient_reference` rather than guessed.

## Phase 9 dashboard limitations

- The dashboard **depends on Phase 6–8 output tables**; if those tables are not present in
  `results/tables/`, it shows a readiness message instead of analysis.
- **Missing input tables reduce available panels**: optional Phase 7/8 tables only enrich the
  display and the dashboard runs without them, but the corresponding panels stay empty.
- It is a **local prototype only** — no database, no authentication, no cloud deployment.
- It is **not real-time monitoring** and reflects only the static CSV snapshots on disk.
- It is **not validated for operational decisions** and provides **no clinical interpretation**.
- It performs **no diagnosis, no health risk scoring, and no exposure measurement**;
  hazard-context panels show alignment context only.
- If example data are used, they are **schema demonstration only and are not scientific
  evidence**; the dashboard surfaces a provenance warning when example data are detected.
- **Derived hazard-context deltas are a fallback, not a substitute for measured inputs**: when
  `longitudinal_hazard_deltas.csv` is missing, the dashboard derives hazard-context relevance
  deltas from the longitudinal domain deltas and the HRP hazard-domain mapping. This derivation
  is hazard-context relevance only — **not exposure measurement, not risk scoring, and not
  diagnosis** — and inherits all coverage limitations of the underlying domain data. When only
  domain deltas (no baseline/current activations) are available, baseline hazard-context
  relevance is **assumed to be 0.0** and that assumption is recorded in the row interpretation.
