# NeuroBridge-S4 Graph Learning — Roadmap

## Status summary

```text
Phase 1 — Repository foundation: complete
Phase 2 — Processed proxy data import: complete
Phase 3 — Single-timepoint biological adaptation graphs: complete
Phase 4 — Single-timepoint graph feature extraction: complete
Phase 5 — Hazard-aware graph-feature similarity mapping: complete
Phase 6 — Within-subject longitudinal graph trajectories: complete
Phase 7 — Explainable trajectory attribution: complete
Phase 8 — Reference-calibrated trajectory envelope: implemented
Phase 9 — Interactive longitudinal dashboard: next
Phase 10 — HRP-like data adapter layer: future
Phase 11 — Monitoring priority translation layer: future
Phase 12 — Experimental graph sequence models: future
Phase 13 — Portfolio release: future
```

> **Primary methodological direction:** within-subject longitudinal graph trajectories.
> Reference cohorts are a secondary calibration layer, not the main comparison endpoint.

## Phase 1 — Graph schema design and repository foundation

Status: complete in this package.

Goals:

- create a clean repository structure;
- define the biological adaptation graph schema;
- define interpretation guardrails;
- prepare for data import from original NeuroBridge-S4 outputs.

Deliverables:

- `README.md`
- `docs/graph_schema.md`
- `docs/methodology.md`
- `docs/limitations.md`
- `docs/hrp_relevance.md`
- project package structure

## Phase 2 — Data import from original NeuroBridge-S4

Goal: load processed outputs from the original NeuroBridge-S4 challenge repository.

Expected inputs:

- `pseudo_crew.csv`
- `deviation_scores.csv`
- `domain_scores.csv`
- `baci_scores.csv`
- `baci_sensitivity.csv`
- `crew_level_summary.csv`

Deliverables:

- `src/neurobridge_graph/data_loader.py`
- `notebooks/00_Project_Overview.ipynb`
- validation utilities

## Phase 3 — Single-timepoint biological adaptation graphs

Status: **complete; interactive HTML improved**.

Goal: build a biological adaptation graph for a single subject-timepoint.

Deliverables:

- `src/neurobridge_graph/graph_builder.py` — full graph construction pipeline
- `src/neurobridge_graph/visualization.py` — static PNG figures
- `src/neurobridge_graph/interactive.py` — polished interactive HTML graph export (pyvis)
- `notebooks/01_Build_Biological_Adaptation_Graphs.ipynb` — reviewer-friendly notebook
- `results/graphs/*.graphml` — one GraphML file per participant
- `results/tables/subject_nodes.csv` — all node attributes
- `results/tables/subject_edges.csv` — all edge attributes
- `results/tables/phase3_graph_summary.csv` — graph-level summary
- `results/figures/subject_graph_*.png` — static PNG graph visualizations
- `results/html/subject_graph_*.html` — interactive HTML graph visualizations (polished)
- `results/html/index.html` — index page with summary table and legend
- `results/reports/phase3_subject_graph_report.txt` — plain-language report
- `tests/test_graph_builder.py` — unit tests
- `tests/test_interactive_export.py` — HTML export tests

HTML improvements: single clean title, plain-text tooltips (no raw HTML tags),
activation-aware colours and node sizes, legend panel, guardrail note, improved
index page with participant summary table.

## Phase 4 — Single-timepoint graph feature extraction

Status: **implemented**.

Goal: extract graph features per subject-timepoint.

Deliverables:

- `src/neurobridge_graph/graph_features.py` — graph-level, node-level, edge-level features
- `src/neurobridge_graph/subgraph_features.py` — subgraph template feature extraction
- `notebooks/02_Graph_Features_and_Embeddings_Foundation.ipynb` — reviewer-friendly notebook
- `results/tables/graph_level_features.csv`
- `results/tables/node_level_features.csv`
- `results/tables/edge_level_features.csv`
- `results/tables/subgraph_features.csv`
- `results/figures/phase4_graph_level_feature_comparison.png`
- `results/figures/phase4_node_activation_heatmap.png`
- `results/figures/phase4_node_centrality_heatmap.png`
- `results/figures/phase4_subgraph_activation_heatmap.png`
- `results/reports/phase4_graph_feature_report.txt`
- `tests/test_graph_features.py` — unit tests

## Phase 5 — Hazard-aware graph-feature similarity mapping

Status: **implemented**.

Goal: map graph-feature profiles and HRP hazard-context features for structural comparison.
Phase 5 similarity is a useful demonstration, but **not the final astronaut-monitoring
endpoint**. Within-subject longitudinal trajectory analysis (Phase 6) is the primary
operationally relevant direction.

Methods (no GNN, no PyTorch):

- HRP five-hazard domain mapping and hazard relevance scores;
- hazard-aware graph-feature matrix (graph + subgraph + hazard + BACI features);
- feature scaling;
- PCA embedding (transparent visualization, not a classifier);
- cosine similarity and Euclidean distance.

Deliverables:

- `src/neurobridge_graph/hazard_mapping.py` — HRP five-hazard context mapping
- `src/neurobridge_graph/embeddings.py` — hazard-aware feature matrix + PCA
- `src/neurobridge_graph/similarity.py` — cosine similarity + Euclidean distance
- `notebooks/03_Hazard_Aware_Graph_Embeddings_and_Similarity_Mapping.ipynb`
- `results/tables/hazard_domain_mapping.csv`
- `results/tables/hazard_relevance_scores.csv`
- `results/tables/hazard_coverage_report.csv`
- `results/tables/phase5_hazard_aware_feature_matrix.csv`
- `results/tables/phase5_hazard_aware_scaled_feature_matrix.csv`
- `results/tables/graph_similarity_matrix.csv`
- `results/tables/graph_distance_matrix.csv`
- `results/tables/graph_embeddings.csv`
- `results/tables/phase5_similarity_summary.csv`
- `results/figures/phase5_*.png`
- `results/reports/phase5_hazard_aware_similarity_report.txt`
- `tests/test_hazard_mapping.py`, `tests/test_embeddings_similarity.py`

## Phase 6 — Within-subject longitudinal graph trajectories

Status: **implemented — primary operational direction**.

Goal: track how each subject's biological adaptation graph changes relative to their own
baseline across mission and recovery phases.

Deliverables:

- `src/neurobridge_graph/longitudinal.py` — longitudinal graph building and delta computation
- `src/neurobridge_graph/trajectory_features.py` — recovery metrics, hazard-context deltas
- `src/neurobridge_graph/trajectory_visualization.py` — trajectory figures
- `notebooks/04_Within_Subject_Longitudinal_Graph_Trajectories.ipynb`
- `docs/longitudinal_data_schema.md`
- `results/tables/longitudinal_*.csv`, `recovery_metrics.csv`
- `results/figures/phase6_*.png`
- `results/reports/phase6_longitudinal_trajectory_report.txt`
- `tests/test_longitudinal.py`, `tests/test_trajectory_features.py`

## Phase 7 — Explainable within-subject trajectory attribution

Status: **implemented**.

Goal: explain *what drives* each baseline-relative graph trajectory by decomposing the
within-subject graph change into transparent contribution shares across biological domains,
graph metrics, biological subgraphs, HRP hazard contexts, and recovery behavior.

Attribution is transparent arithmetic (`absolute delta / total absolute delta`), not a
black-box model. Outputs identify monitoring-relevant graph-shift contributors for expert
review. They are **not** diagnosis, treatment guidance, exposure attribution, or causal proof.

Deliverables:

- `src/neurobridge_graph/trajectory_attribution.py` — node / graph-metric / subgraph /
  hazard-context / recovery attribution + summary
- `src/neurobridge_graph/explanation_generator.py` — plain-language explanations and report
- `src/neurobridge_graph/attribution_visualization.py` — attribution figures
- `notebooks/05_Explainable_Trajectory_Attribution.ipynb`
- `results/tables/phase7_input_readiness_check.csv`
- `results/tables/trajectory_node_attribution.csv`
- `results/tables/trajectory_graph_metric_attribution.csv`
- `results/tables/trajectory_subgraph_attribution.csv`
- `results/tables/trajectory_hazard_attribution.csv`
- `results/tables/recovery_attribution.csv`
- `results/tables/phase7_attribution_summary.csv`
- `results/figures/phase7_*.png`
- `results/reports/phase7_explainable_trajectory_attribution_report.txt`
- `results/reports/explanation_cards/*.txt`
- `tests/test_trajectory_attribution.py`, `tests/test_explanation_generator.py`

## Phase 8 — Reference-calibrated trajectory envelope

Status: **implemented**.

Goal: estimate whether each within-subject baseline-relative graph change is small, moderate,
or unusually large relative to a reference-calibrated **expected-variability envelope**. The
self-baseline comparison remains primary; reference/analog data are used **only** to calibrate
expected variability (envelope bounds, noise scale, rarity context), never as a healthy-vs-
unhealthy endpoint.

> The reference envelope does not define whether a person is healthy or unhealthy. It
> calibrates how large a within-subject graph change is relative to expected variability in
> available proxy or analog data.

Outside-envelope means a baseline-relative change is larger than expected under the current
calibration data — a candidate for expert review. It is **not** diagnosis, a health risk score,
treatment guidance, or exposure measurement.

Deliverables:

- `src/neurobridge_graph/reference_envelope.py` — envelope construction, robust-z scoring,
  classification, and per-layer delta scoring + summary
- `src/neurobridge_graph/envelope_visualization.py` — envelope figures
- `src/neurobridge_graph/envelope_reporting.py` — plain-language interpretation and report
- `notebooks/06_Reference_Calibrated_Trajectory_Envelope.ipynb`
- `data/examples/example_reference_variability_envelope.csv` (schema demonstration only)
- `results/tables/phase8_input_readiness_check.csv`
- `results/tables/reference_trajectory_envelope.csv`
- `results/tables/reference_calibrated_node_delta_scores.csv`
- `results/tables/reference_calibrated_graph_delta_scores.csv`
- `results/tables/reference_calibrated_hazard_delta_scores.csv`
- `results/tables/phase8_reference_calibrated_summary.csv`
- `results/figures/phase8_*.png`
- `results/reports/phase8_reference_calibrated_trajectory_report.txt`
- `tests/test_reference_envelope.py`, `tests/test_envelope_reporting.py`

## Phase 9 — Interactive longitudinal dashboard ← next

Goal: reviewer-facing interactive exploration of trajectories, attribution, and envelope
exceedance together.

## Phase 10 — HRP-like data adapter layer

Goal: adapter to ingest HRP-style longitudinal/analog data into the trajectory pipeline.

## Phase 11 — Monitoring priority translation layer

Goal: translate envelope and attribution outputs into reviewer-facing monitoring priorities.

## Phase 12 — Experimental graph sequence models / graph autoencoders

Only after transparent trajectory baselines work. No supervised training on n=4.

## Phase 13 — Portfolio and reviewer polish

Goal: make the repository useful for scientific outreach and job applications.

Tasks:

- add Colab links;
- add architecture diagram;
- add sample figures;
- add release notes;
- add reviewer-facing summary.
