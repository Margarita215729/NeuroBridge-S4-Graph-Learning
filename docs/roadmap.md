# NeuroBridge-S4 Graph Learning — Roadmap

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

## Phase 3 — Subject-level biological adaptation graphs

Status: **implemented**.

Goal: convert each participant into a computable biological adaptation graph.

Deliverables:

- `src/neurobridge_graph/graph_builder.py` — full graph construction pipeline
- `src/neurobridge_graph/visualization.py` — static PNG figures
- `src/neurobridge_graph/interactive.py` — interactive HTML graph export (pyvis)
- `notebooks/01_Build_Biological_Adaptation_Graphs.ipynb` — reviewer-friendly notebook
- `results/graphs/*.graphml` — one GraphML file per participant
- `results/tables/subject_nodes.csv` — all node attributes
- `results/tables/subject_edges.csv` — all edge attributes
- `results/tables/phase3_graph_summary.csv` — graph-level summary
- `results/figures/subject_graph_*.png` — static PNG graph visualizations
- `results/html/subject_graph_*.html` — interactive HTML graph visualizations
- `results/html/index.html` — index page linking all subject graphs
- `results/reports/phase3_subject_graph_report.txt` — plain-language report
- `tests/test_graph_builder.py` — unit tests
- `tests/test_interactive_export.py` — HTML export tests

## Phase 4 — Graph feature extraction ← next

Goal: compute interpretable graph-level and node-level features.

Deliverables:

- centrality features;
- active domain counts;
- active subgraph sizes;
- coherence-weighted activation;
- cluster scores.

## Phase 5 — Graph embeddings and similarity maps

Goal: compare participants in graph space using transparent baseline methods.

Initial methods:

- graph feature vectors;
- PCA;
- spectral embeddings;
- cosine similarity;
- distance-to-reference centroid.

## Phase 6 — Graph novelty detection

Goal: identify reference-relative graph novelty.

Initial methods:

- Isolation Forest;
- Local Outlier Factor;
- robust z-scores;
- centroid distance.

## Phase 7 — Explainable subgraph profiles

Goal: generate human-readable graph-based adaptation summaries.

Examples:

- sleep-autonomic-recovery pattern;
- stress-immune-metabolic pattern;
- low-coherence stable pattern;
- multi-domain high-coherence pattern.

## Phase 8 — Optional GNN prototype

Only after transparent baselines work.

Safe tasks:

- self-supervised graph embeddings;
- graph reconstruction;
- representation learning;
- anomaly detection.

Avoid:

- supervised training on n=4;
- predicting astronaut health outcomes;
- diagnostic claims.

## Phase 9 — Longitudinal graph trajectories

Goal: extend static graphs into temporal graphs.

Future structure:

```text
baseline graph → stress/exposure graph → recovery graph
```

Metrics:

- graph delta from baseline;
- temporal BACI;
- recovery slope;
- time-to-baseline;
- persistent active subgraphs.

## Phase 10 — Portfolio and reviewer polish

Goal: make the repository useful for scientific outreach and job applications.

Tasks:

- add Colab links;
- add architecture diagram;
- add sample figures;
- add release notes;
- add reviewer-facing summary.
