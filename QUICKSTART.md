# Quickstart

NeuroBridge-S4 can be reviewed with zero installation, or run locally for a full technical review.

## Zero-install review

Open the public GitHub Pages site — no installation, no commands, no notebooks required:

- Landing page: `https://Margarita215729.github.io/NeuroBridge-S4-Graph-Learning/`
- PyTorch showcase: `https://Margarita215729.github.io/NeuroBridge-S4-Graph-Learning/phase12_pytorch_showcase.html`

(If the repository name differs, check the GitHub Pages settings — see `docs/github_pages.md`.)

## Local technical review

```bash
pip install -r requirements.txt
pytest
streamlit run app.py
```

- `pip install -r requirements.txt` installs all dependencies, including PyTorch (CPU is sufficient).
- `pytest` runs the full test suite.
- `streamlit run app.py` launches the interactive longitudinal review dashboard.

## Open the PyTorch showcase locally

After running the Phase 12 notebook, open the self-contained showcase page:

```bash
open results/html/phase12_pytorch_showcase.html
```

## Rebuild the public Pages site

```bash
python scripts/build_pages_site.py
python scripts/validate_pages_site.py
```

`build_pages_site.py` publishes the Phase 12 showcase into `docs/`, copies figures into
`docs/assets/figures/`, and prints the GitHub Pages setup instructions. `validate_pages_site.py`
runs offline structural checks on the site.

## Notebook order

Run the notebooks in `notebooks/` in this order (see `docs/notebook_index.md` for details):

1. `00_Project_Overview.ipynb`
2. `01_Build_Biological_Adaptation_Graphs.ipynb`
3. `02_Graph_Features_and_Embeddings_Foundation.ipynb`
4. `03_Hazard_Aware_Graph_Embeddings_and_Similarity_Mapping.ipynb`
5. `04_Within_Subject_Longitudinal_Graph_Trajectories.ipynb`
6. `05_Explainable_Trajectory_Attribution.ipynb`
7. `06_Reference_Calibrated_Trajectory_Envelope.ipynb`
8. `07_HRP_Like_Data_Adapter_Layer.ipynb`
9. `08_Operational_Resilience_Interpretation_Layer.ipynb`
10. `09_PyTorch_Temporal_Graph_Autoencoder_Showcase.ipynb`

## Guardrails

This is a research engineering prototype. It is not diagnosis, treatment guidance, health risk
scoring, exposure measurement, mission readiness classification, or an operational medical decision
system. Schema-demonstration data are not scientific evidence.
