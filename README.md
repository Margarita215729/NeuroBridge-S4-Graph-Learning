# NeuroBridge-S4 Graph Learning

Graph learning extension of **NeuroBridge-S4** for small-N biomedical human adaptation modeling using real public proxy datasets.

> This is an independent research prototype. It is not an official NASA project and does not contain actual Artemis II astronaut data.

## Primary methodological direction: within-subject trajectories

**NeuroBridge-S4 Graph Learning is primarily a within-subject longitudinal graph trajectory framework for small-N human research.**

The primary target use case is longitudinal self-baseline tracking:

```text
subject baseline → mission phase → postflight → recovery
```

Reference cohorts remain useful for calibration, noise estimation, rarity context, and feature
scaling, but the most astronaut-relevant signal is how each individual's biological adaptation
graph changes relative to their own baseline.

> The primary signal is within-subject change from the individual's own baseline. Population
> reference data are used only to calibrate scale, estimate noise, contextualize rarity, and
> stabilize feature geometry.

## Why self-baseline matters

- Astronauts are not generic population subjects; individual baselines matter more than population-normal ranges.
- Mission-relevant changes can occur **inside** population-normal ranges.
- Stable individual differences can be misread if compared only to generic healthy cohorts.
- Reference cohorts remain useful for scale, noise, rarity, and feature geometry — but not as the main endpoint.

> Healthy population ranges are insufficient for astronaut monitoring because operationally
> meaningful changes may occur within population-normal ranges, while stable individual
> baselines may differ from generic norms.

**Conceptual hierarchy:**

```text
Primary comparison:    Current astronaut state vs that astronaut's own baseline.
Secondary calibration: Reference cohort distribution for scale, noise, rarity, and feature geometry.
Interpretation layer:  NASA HRP five hazards as operational context, not exposure measurement.
```

## Core question

How has this individual's biological adaptation graph changed relative to their own baseline
over time, across mission and recovery phases?

## What Phase 1 contains

Phase 1 is the **graph schema and repository foundation** stage. It defines:

- what a biological adaptation graph is;
- what nodes represent;
- what edges represent;
- what node and edge attributes should contain;
- how this project will later connect to NeuroBridge-S4 processed outputs;
- what guardrails are required for responsible interpretation.

No model training is performed in Phase 1.

## Planned workflow

```text
Processed NeuroBridge-S4 outputs
→ participant + timepoint biological domains
→ subject-timepoint biological adaptation graphs
→ graph features per timepoint
→ within-subject trajectory (delta from personal baseline)
→ hazard-context shift tracking
→ recovery trajectory metrics
→ reference-calibrated trajectory novelty (future)
```

## Phase 3: Single-Timepoint Biological Adaptation Graphs

Phase 3 builds a biological adaptation graph for a single subject-timepoint.

Run: `notebooks/01_Build_Biological_Adaptation_Graphs.ipynb`

Outputs:
- GraphML files;
- node and edge tables;
- graph summary table;
- static PNG graph figures;
- interactive HTML graph visualizations (polished, reviewer-facing);
- plain-language Phase 3 report.

Interactive graphs: Open `results/html/index.html` after running the notebook.

Interpretation: The graphs are research interpretation artifacts, not diagnostic tools.

---

## Phase 4: Single-Timepoint Graph Feature Extraction

Phase 4 extracts graph features per subject-timepoint.
This creates the foundation for trajectory deltas and hazard-context mapping.

Run: `notebooks/02_Graph_Features_and_Embeddings_Foundation.ipynb`

Outputs:
- graph-level features (`results/tables/graph_level_features.csv`);
- node-level features (`results/tables/node_level_features.csv`);
- edge-level features (`results/tables/edge_level_features.csv`);
- subgraph-level features (`results/tables/subgraph_features.csv`);
- feature comparison figures (`results/figures/phase4_*.png`);
- Phase 4 interpretation report (`results/reports/phase4_graph_feature_report.txt`).

Interpretation: These features are not diagnostic. They prepare the project for Phase 5.

---

## Phase 5: Hazard-Aware Graph-Feature Similarity Mapping

Phase 5 maps graph-feature profiles and HRP hazard-context features into a shared feature
space for structural comparison. This is a useful demonstration, but **not the final
astronaut-monitoring endpoint** — within-subject longitudinal trajectory analysis (Phase 6)
is the primary operationally relevant direction.

It adds NASA HRP's five human spaceflight hazards as an interpretation context:

- Space Radiation
- Isolation and Confinement
- Distance from Earth
- Gravity Fields
- Hostile / Closed Environments

Run:

`notebooks/03_Hazard_Aware_Graph_Embeddings_and_Similarity_Mapping.ipynb`

Outputs:
- hazard-domain mapping;
- hazard relevance scores;
- hazard coverage report;
- hazard-aware feature matrix;
- similarity and distance matrices;
- PCA graph-feature embedding;
- hazard-aware similarity figures;
- plain-language Phase 5 report.

> NeuroBridge-S4 connects individual biological adaptation patterns to NASA's five human
> spaceflight hazard categories without claiming actual exposure, diagnosis, or causal proof.

Interpretation: Hazard relevance is not exposure measurement, diagnosis, or causal attribution.
Phase 5 similarity is structural comparison in graph-feature space, not the final monitoring
endpoint.

---

## Phase 6: Within-Subject Longitudinal Graph Trajectories

Run:

`notebooks/04_Within_Subject_Longitudinal_Graph_Trajectories.ipynb`

Outputs:
- longitudinal biological adaptation graphs (one per subject-timepoint);
- node activation deltas from personal baseline;
- graph feature deltas;
- hazard-context deltas;
- recovery metrics;
- trajectory figures;
- plain-language trajectory report.

Interpretation: These outputs describe within-subject graph changes. They are not diagnosis,
exposure measurements, or treatment guidance.

---

## Phase 7: Explainable Within-Subject Trajectory Attribution

Phase 7 explains what drives each baseline-relative graph trajectory.

It decomposes longitudinal graph changes into:

- biological domain contributors;
- graph metric contributors;
- biological subgraph contributors;
- HRP hazard-context contributors;
- recovery attribution.

Run:

`notebooks/05_Explainable_Trajectory_Attribution.ipynb`

Attribution is transparent arithmetic — `contribution share = absolute delta / total absolute
delta` per subject-timepoint — so reviewers can re-derive every number by hand.

Interpretation: Attribution identifies monitoring-relevant graph-shift contributors for expert
review. It is not diagnosis, treatment guidance, exposure attribution, or causal proof.

---

## Phase 8: Reference-Calibrated Trajectory Envelope

Phase 8 keeps self-baseline tracking as the primary method and adds a reference-calibrated
variability envelope.

It does not ask whether a subject is normal compared with healthy people. It asks whether the
subject's own change from baseline is larger than expected under a calibration envelope.

Run:

`notebooks/06_Reference_Calibrated_Trajectory_Envelope.ipynb`

Outputs:
- reference trajectory envelope;
- node delta envelope scores;
- graph metric envelope scores;
- hazard-context delta envelope scores;
- envelope figures;
- plain-language Phase 8 report.

Interpretation: Outside-envelope means a baseline-relative change is larger than expected under
the current calibration data. It is not diagnosis, risk scoring, treatment guidance, or exposure
measurement.

---

## Phase 9: Interactive Longitudinal Review Dashboard

Phase 9 provides a local Streamlit dashboard for reviewing within-subject biological adaptation
graph trajectories.

Run:

```bash
streamlit run app.py
```

The dashboard displays:
- subject/timepoint overview;
- baseline-relative domain trajectories;
- graph metric trajectories;
- HRP hazard-context shifts;
- explainable trajectory attribution;
- reference-calibrated envelope status;
- recovery behavior;
- data readiness and limitations.

See `docs/dashboard.md` for details. The dashboard reads the Phase 6–8 tables from
`results/tables/`; if required tables are missing it shows a friendly message instead of
crashing.

When `longitudinal_hazard_deltas.csv` is missing, the dashboard derives hazard-context
trajectory deltas on the fly from the longitudinal domain deltas
(`longitudinal_node_deltas.csv`) and the HRP hazard-domain mapping
(`neurobridge_graph.trajectory_features.ensure_longitudinal_hazard_deltas`), saving the
derived table back to `results/tables/`. If the required domain-delta and hazard-mapping
inputs are also unavailable, the hazard-context panel shows a clear message instead of an
empty chart. These derived values are hazard-context relevance, not exposure measurement,
not risk scoring, and not diagnosis.

Interpretation: The dashboard is a research-review prototype. It is not clinical monitoring,
diagnosis, treatment guidance, exposure measurement, or health risk scoring.

---

## Interpretation guardrails

This project does **not**:

- use actual Artemis II astronaut data;
- diagnose medical or psychiatric conditions;
- prescribe treatment;
- infer brain chemistry directly;
- claim NHANES is astronaut-equivalent;
- train supervised models on n=4.

This project does:

- use real public proxy data in later phases;
- create a transparent graph schema;
- track within-subject longitudinal graph trajectories from personal baseline;
- use reference cohorts for calibration, not as the primary comparison endpoint;
- support human-review and research interpretation;
- generate explainable longitudinal adaptation profiles.

## Repository structure

```text
NeuroBridge-S4-Graph-Learning/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── notebooks/
├── src/neurobridge_graph/
├── docs/
├── data/
├── results/
└── tests/
```

## Current status

- **Phase 1** — complete: repository foundation and graph schema.
- **Phase 2** — complete: data import from NeuroBridge-S4 outputs.
- **Phase 3** — complete: biological adaptation graphs + improved interactive HTML.
- **Phase 4** — complete: interpretable graph feature extraction.
- **Phase 5** — implemented: hazard-aware graph-feature similarity mapping (structural demonstration).
- **Phase 6** — complete: within-subject longitudinal graph trajectories (primary direction).
- **Phase 7** — complete: explainable within-subject trajectory attribution.
- **Phase 8** — implemented: reference-calibrated trajectory envelope (calibration layer).
- **Phase 9** — implemented: interactive longitudinal review dashboard (Streamlit).
- **Phase 10** — next: HRP-like data adapter layer.
