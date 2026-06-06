# Notebook Index

Run the notebooks in `notebooks/` in the order below. Each notebook is reviewer-friendly and writes
its outputs to `results/`. All notebooks operate on within-subject, baseline-relative data and carry
the project guardrails.

| Order | Notebook | Purpose | Key expected outputs |
|------:|----------|---------|----------------------|
| 0 | `00_Project_Overview.ipynb` | Orientation and project framing. | — |
| 1 | `01_Build_Biological_Adaptation_Graphs.ipynb` | Build single-timepoint biological adaptation graphs from domain scores. | graphs in `results/graphs/`, interactive HTML in `results/html/` |
| 2 | `02_Graph_Features_and_Embeddings_Foundation.ipynb` | Extract node/graph features and build the feature foundation. | feature tables in `results/tables/` |
| 3 | `03_Hazard_Aware_Graph_Embeddings_and_Similarity_Mapping.ipynb` | Align graph features with the HRP five-hazard framework; similarity mapping. | hazard mapping + similarity tables/figures |
| 4 | `04_Within_Subject_Longitudinal_Graph_Trajectories.ipynb` | Track within-subject change from personal baseline across mission phases. | `longitudinal_node_deltas.csv`, `longitudinal_graph_deltas.csv`, `longitudinal_hazard_deltas.csv`, `recovery_metrics.csv` |
| 5 | `05_Explainable_Trajectory_Attribution.ipynb` | Attribute trajectory change to domains, subgraphs, hazards, recovery. | `trajectory_*_attribution.csv`, `recovery_attribution.csv` |
| 6 | `06_Reference_Calibrated_Trajectory_Envelope.ipynb` | Calibrate change magnitude against reference variability. | `reference_calibrated_*_scores.csv`, `phase8_reference_calibrated_summary.csv` |
| 7 | `07_HRP_Like_Data_Adapter_Layer.ipynb` | Transform raw HRP-like inputs into graph-ready domain scores. | adapter reports + graph-ready domain scores |
| 8 | `08_Operational_Resilience_Interpretation_Layer.ipynb` | Rule-based operational resilience interpretation cards. | `resilience_state_table.csv`, `mission_relevance_translation.csv`, resilience cards/figures |
| 9 | `09_PyTorch_Temporal_Graph_Autoencoder_Showcase.ipynb` | Self-supervised PyTorch autoencoder over trajectory features; builds the HTML showcase. | `phase12_*` tables/figures, `results/models/phase12_trajectory_autoencoder.pt`, `results/reports/phase12_model_card.md`, `results/html/phase12_pytorch_showcase.html` |

Notes:
- There are two legacy `01_*` notebooks; `01_Build_Biological_Adaptation_Graphs.ipynb` is the current
  graph-building entry point, while `01_Load_NeuroBridge_Outputs.ipynb` documents the original output
  import.
- Notebook 9 is the main portfolio artifact generator; its HTML output is published to the GitHub
  Pages site as `docs/phase12_pytorch_showcase.html` by `scripts/build_pages_site.py`.

Guardrails: every notebook is a research engineering prototype. None of them perform diagnosis,
treatment guidance, health risk scoring, exposure measurement, or mission readiness classification.
