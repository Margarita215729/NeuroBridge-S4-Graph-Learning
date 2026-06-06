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
