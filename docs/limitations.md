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

## Phase 10 data adapter limitations

- **Variable-to-domain mapping is approximate and extensible**: it is a structural interpretation
  scaffold, not a clinical assignment, and can be extended for new variables.
- **Templates are not evidence**: the example template rows
  (`schema_template_not_scientific_evidence`) exist only to illustrate the schema.
- **Unit conversion is limited unless explicitly implemented**: `expected_unit` documents an
  assumed unit. Phase 10 includes a unit-standardization placeholder
  (`standardize_units_if_known`), but broad biomedical unit conversion is not yet implemented.
  Units are tracked and unsupported conversions are reported rather than silently transformed.
- **Some variables are intentionally unmapped**: steps are left unmapped by default because their
  interpretation depends on protocol context, mission phase, workload, exercise prescription, and
  activity constraints. Users may map steps explicitly in project-specific configurations.
- **Domain scores depend on available variables**: a domain with no mapped variables in the input
  is reported as absent rather than inferred.
- **Missing data can reduce domain coverage**: per-timepoint availability is reported, and missing
  variables lower the available-variable count for a domain.
- **No clinical interpretation**: the adapter validates and transforms data only. It does not
  diagnose, score risk, infer exposure, or recommend treatment.

## Phase 11 operational resilience interpretation limitations

- **Resilience states are rule-based research interpretations**, produced by a transparent priority
  cascade over Phase 6-10 evidence.
- **They are not validated operational states**: the thresholds and rules are reasonable defaults,
  not empirically validated cut-points.
- **They are not clinical labels**: states describe baseline-relative graph adaptation patterns, not
  medical conditions.
- **They are not mission readiness categories**: nothing here classifies an astronaut as fit/unfit
  or ready/not ready.
- **They are not health risk levels**: no state is a risk score, danger level, or alert.
- **They are sensitive to available data streams**: coverage and timepoint density strongly affect
  which state is assigned.
- **Coverage-limited outputs should not be overinterpreted**: when data coverage is insufficient,
  the state is explicitly `coverage_limited_interpretation` with `coverage_limited` confidence.
- Operational resilience interpretation is a research-review layer. It is not diagnosis, treatment
  guidance, health risk scoring, exposure measurement, or an operational medical decision.
