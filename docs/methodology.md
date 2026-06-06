# Methodology

## Overview

NeuroBridge-S4 Graph Learning is primarily a **within-subject longitudinal graph trajectory
framework** for small-N human research. It extends the original NeuroBridge-S4 methodology by
converting participant-level biomedical measurements into graph objects and tracking how those
graphs change relative to each individual's personal baseline over time.

The method is designed for small-N human research settings (e.g. astronaut crews) where
population-level statistical power may be weak, but each participant contains rich multimodal
information that changes across mission phases.

## Method stages

1. Load processed biomedical/domain outputs (cross-sectional or longitudinal).
2. Map measurements into biological domains.
3. Construct one biological adaptation graph per subject-timepoint.
4. Compute node-level and graph-level features per timepoint.
5. Compute within-subject deltas from personal baseline (node, graph, subgraph, hazard-context).
6. Track recovery trajectories and mission-phase shifts.
7. Use reference cohorts for calibration (scale, noise, rarity) — secondary layer.
8. Generate explainable longitudinal adaptation profiles for human review.

## Why graph representation matters

Tables break a human system into columns. Graph representation helps assemble those columns back into a biological and operational system.

A participant is not just a row in a dataset. In human adaptation research, sleep, autonomic regulation, inflammation, metabolic regulation, cognitive load, and recovery may interact as a system.

Graph representation preserves this relational structure.

## Why this is appropriate for small-N data

The method does not train a supervised model on n=4.

Instead, it tracks **within-subject longitudinal graph trajectories** — how each individual's
biological adaptation graph changes from their own baseline across mission and recovery phases.
Reference cohorts from larger public proxy datasets provide secondary calibration (scale, noise,
rarity, feature geometry) but are not the primary comparison endpoint.

This supports:

- individualized self-baseline interpretation;
- within-subject mission-phase shift detection;
- recovery trajectory monitoring;
- coherent subgraph identification over time;
- reviewer-friendly longitudinal explanations.

## Modeling hierarchy

1. Transparent graph features
2. Similarity and distance analysis
3. Reference-relative novelty detection
4. Explainable subgraph profiles
5. Optional self-supervised graph learning
6. Future longitudinal graph trajectories

## Responsible interpretation

The project is not a diagnostic system. It is a research framework for signal triage and graph-based interpretation.

## Phase 4: Bridge between graph construction and graph learning

Phase 4 is the bridge between graph construction and graph learning.
It extracts interpretable structural features before applying embeddings or machine learning.

Each biological adaptation graph is converted into four feature levels:

1. **Graph-level features** — density, activation statistics, edge counts, BACI score.
   These allow participants to be compared as whole biological systems.
2. **Node-level features** — per-domain activation, centrality, active flags.
   These identify which domains are activated and how structurally central they are.
3. **Edge-level features** — edge type, weight, active-domain connection flags.
   These preserve the distinction between conceptual and co-activation edges.
4. **Subgraph-level features** — template-based cluster activation.
   These ask whether activation is concentrated in biologically meaningful clusters
   (cardiometabolic, immune-metabolic, hematologic-cardiovascular, etc.).

Phase 4 features are interpretable without machine-learning knowledge and do not require
training data. They are the foundation for Phase 5 graph embeddings and Phase 6 novelty detection.

## Hazard-context mapping

Phase 5 adds an HRP **hazard-context mapping** layer on top of the biological adaptation graphs.

> NeuroBridge-S4 connects individual biological adaptation patterns to NASA's five human
> spaceflight hazard categories without claiming actual exposure, diagnosis, or causal proof.

### Why the NASA five hazards matter

NASA's Human Research Program frames human spaceflight risk around five major hazards: Space
Radiation, Isolation and Confinement, Distance from Earth, Gravity Fields, and Hostile / Closed
Environments. Translating biological graph patterns into this shared HRP language lets reviewers
interpret which spaceflight hazard contexts may be most relevant for closer monitoring.

### How biological domains are mapped to hazards

Each biological domain is linked to one or more hazards with a conceptual relevance weight in
`[0, 1]` (`src/neurobridge_graph/hazard_mapping.py`). This is a transparent, editable mapping —
a conceptual HRP relevance mapping, **not** measured exposure. The mapping tolerates missing
domains: when the proxy dataset lacks a mapped domain (e.g. sleep/circadian, autonomic, emotional,
cognitive-load), coverage is reduced and reported, never silently assumed.

### Why this is context, not exposure measurement

A `hazard_relevance_score` is not a measured exposure score and not a health risk score. It is a
weighted summary of which *activated* biological domains map onto a hazard category. "Distance from
Earth" in particular is treated as an autonomy and delayed-support context: a graph pattern may be
more operationally important when real-time Earth support is limited, independent of any exposure.

### How hazard relevance scores enter graph-feature space

For each subject and hazard:

```text
hazard_relevance_score = sum(domain_activation * hazard_domain_weight) / sum(hazard_domain_weight)
```

computed over the mapped domains that are available. These scores become
`hazard_relevance__<hazard>` columns in the Phase 5 hazard-aware feature matrix, alongside graph,
subgraph, and BACI features. Phase 5 structural similarity (cosine, Euclidean, PCA) is a
demonstration layer — not the final monitoring endpoint.

## Self-baseline longitudinal methodology

The central methodological claim is that astronaut monitoring should prioritize **individual
change over time**. Generic healthy reference ranges are useful but insufficient.

> Healthy population ranges are insufficient for astronaut monitoring because operationally
> meaningful changes may occur within population-normal ranges, while stable individual
> baselines may differ from generic norms.

NeuroBridge-S4 therefore treats each astronaut's pre-mission baseline as the **primary
comparator** and tracks graph changes across mission and recovery phases:

```text
G_baseline → G_inflight → G_postflight → G_recovery
DeltaGraph(t) = Graph(t) - Graph(baseline)
```

### Trajectory signal components

- **Node deltas** — per-domain activation change from personal baseline.
- **Graph deltas** — graph-level metric changes (density, activation, active domains).
- **Subgraph deltas** — template-cluster activation shifts (decomposed in Phase 7 attribution).
- **Hazard-context deltas** — HRP hazard relevance shifts (interpretation context, not exposure).
- **Recovery metrics** — recovery slope, recovery fraction, time-to-baseline-like state.

### Reference cohorts as calibration only

Reference cohort distributions are used for:

- scale calibration;
- noise estimation;
- rarity contextualization;
- feature geometry stabilization.

They are **not** the main endpoint. The primary signal is within-subject change from the
individual's own baseline.

## Explainable trajectory attribution

Phase 7 explains *what drives* each baseline-relative graph trajectory. It decomposes the
within-subject graph change into transparent contribution shares so that a reviewer can see
which components moved the graph from baseline. Attribution is descriptive arithmetic, not a
model or a cause:

```text
contribution_share = absolute_delta / total_absolute_delta (per subject-timepoint)
```

- **Node-level attribution** identifies the biological **domain** drivers — which domains
  account for the largest share of the baseline-relative node activation change.
- **Graph-metric attribution** shows whether the trajectory is driven by broad activation,
  peak activation, active-domain count, connectivity, or co-activation changes.
- **Subgraph attribution** aggregates node attribution into biologically meaningful patterns
  (cardiometabolic, immune-metabolic, hematologic-cardiovascular, sleep-autonomic-recovery,
  cognitive-emotional-recovery), revealing whether change is localized or distributed.
- **Hazard-context attribution** aligns graph shifts to HRP hazard categories. This is
  hazard-context **alignment**, never exposure attribution or causal effect.
- **Recovery attribution** classifies each metric as returned-near-baseline, partial recovery,
  persistent shift, overshoot/reversal, or insufficient data — separating persistent from
  returning-to-baseline components.

Each subject-timepoint receives a plain-language explanation card and feeds a Phase 7 report.
Outputs are monitoring-relevant patterns and candidates for expert review — not diagnosis,
treatment guidance, exposure measurement, or causal proof.

## Reference-calibrated trajectory envelope

Phase 8 adds a **secondary calibration layer** on top of the within-subject trajectory. The
self-baseline comparison **remains primary**: Phase 8 never asks whether a subject is normal
compared with healthy people. It asks whether the subject's own change from baseline is larger
than expected under a reference-calibrated variability envelope.

> The reference envelope does not define whether a person is healthy or unhealthy. It
> calibrates how large a within-subject graph change is relative to expected variability in
> available proxy or analog data.

How reference/analog data calibrate expected variability:

- For each feature (biological domain, graph metric, or hazard context), reference deltas are
  summarized into a **median delta** (envelope center), a **MAD** (robust spread), and
  **5th/95th-percentile bounds** (the expected-variability envelope).
- A within-subject delta is scored with a **robust z-score**,
  `(delta − median) / (1.4826 × MAD)`, and classified as `within_expected_envelope`,
  `near_envelope_boundary`, `outside_expected_envelope`, or `insufficient_reference`.

What envelope bounds mean: the band represents the range of baseline-relative change that is
expected given reference/analog variability. A delta inside the band is unremarkable under the
current calibration; a delta outside the band is larger than expected.

Why envelope exceedance is not diagnosis or risk: exceedance is a **descriptive** comparison of
magnitude against a calibration band. It does not classify health status, does not score risk,
and (for hazard-context features) does not measure exposure.

How this supports expert review: in small crews where group inference is impossible, the
envelope helps reviewers avoid overreacting to expected variability while flagging unusually
large self-baseline shifts as candidates for closer expert review.

## Data adapter layer

Why an adapter layer matters: real HRP-like data rarely arrive as clean graph-ready domain
tables. They arrive as separate biomarker, sleep/activity, cognitive, questionnaire, and
environmental streams in wide or long formats with inconsistent column names. The Phase 10
adapter separates *data ingestion and validation* from *graph analysis* so the trajectory
pipeline always receives a consistent, validated, graph-ready input.

From raw data to graph-ready domain scores: the adapter detects table format (wide/long),
reconciles column-name variants to the standard longitudinal index, standardizes all inputs into
one long variable table, maps variables to canonical biological domains, applies a self-baseline
transformation, and aggregates baseline-relative variable deltas into per-domain scores.

Self-baseline transformation: for each subject and variable the adapter computes
`delta_from_baseline` (and percent change) against that subject's own baseline value — the
baseline mission phase when present, otherwise the earliest time index, with the assumption
recorded. This keeps the within-subject self-baseline principle at the point of ingestion.

Domain mapping: variable-to-domain assignment is a transparent, extensible interpretation
scaffold (see `docs/domain_mapping.md`). It is approximate and explicitly reported, never a
clinical assignment.

Coverage reporting: the adapter reports which domains are covered, which variables are unmapped,
and per-timepoint domain availability, so reviewers can see exactly how much each domain score
rests on.

Limitations: mapping is approximate; units are not auto-converted; domain scores depend on
available variables; missing data reduce coverage; template/example data are not evidence; and
no clinical interpretation is performed. The adapter does not diagnose, score risk, infer
exposure, or recommend treatment.
