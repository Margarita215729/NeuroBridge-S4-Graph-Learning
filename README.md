# NeuroBridge-S4 Graph Learning

Graph learning extension of **NeuroBridge-S4** for small-N biomedical human adaptation modeling using real public proxy datasets.

> This is an independent research prototype. It is not an official NASA project and does not contain actual Artemis II astronaut data.

## Purpose

The original NeuroBridge-S4 project demonstrated a methodology for small-N human research data: real proxy datasets, biological domain mapping, BACI, graph representation, and reviewer-friendly adaptation profiles.

This repository begins the next stage: making those graphs computational.

The goal is to represent each participant as a connected biological adaptation graph rather than as isolated rows and columns in a biomedical dataset.

## Core question

How can we represent each participant as a connected biological system and determine whether their adaptation graph is unusual, coherent, or research-relevant relative to a real reference population?

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
→ participant-level biological domains
→ subject-level biological adaptation graphs
→ graph features
→ graph embeddings
→ reference-relative novelty detection
→ explainable subgraph profiles
→ longitudinal graph trajectories
```

## Phase 3: Biological Adaptation Graph Construction

Phase 3 converts each pseudo-crew participant into a biological adaptation graph.

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

## Phase 4: Interpretable Graph Feature Extraction

Phase 4 turns biological adaptation graphs into measurable features.
This creates the foundation for graph embeddings, similarity mapping,
and novelty detection in Phase 5.

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
- compare small-N participants to larger reference graph spaces;
- support human-review and research interpretation;
- generate explainable graph-based adaptation profiles.

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
- **Phase 5** — next: graph embeddings and similarity mapping.
