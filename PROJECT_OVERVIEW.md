# NeuroBridge-S4 Project Overview

## One-line summary

NeuroBridge-S4 is an end-to-end ML research engineering prototype for within-subject biological
graph trajectory analysis in small-N human spaceflight contexts:
HRP-like longitudinal data → self-baseline biological graph trajectories → operational resilience
interpretation → PyTorch temporal graph autoencoder showcase.

## Problem

Small-N human spaceflight research cannot rely only on population-scale inference. With only a
handful of crew members, comparing individuals against a large population is statistically
inappropriate and scientifically misleading. The interesting signal is how a single organism changes
relative to its own baseline over time — across baseline, pre-flight, inflight, post-flight, and
recovery phases.

## Approach

NeuroBridge-S4 models each subject as its own reference. It converts heterogeneous HRP-like
longitudinal inputs into baseline-relative biological adaptation graphs, extracts graph features,
tracks within-subject trajectories, attributes change to interpretable drivers, calibrates change
magnitude against reference variability, and produces a transparent operational resilience
interpretation. A PyTorch self-supervised autoencoder then learns compact latent representations of
those graph-derived trajectory states.

## Architecture

```
HRP-like inputs
  → data adapter layer
  → self-baseline domain scores
  → biological adaptation graphs
  → graph features
  → longitudinal trajectories
  → attribution
  → reference-calibrated envelope
  → operational resilience interpretation
  → review dashboard
  → PyTorch temporal graph autoencoder
  → public showcase
```

See `docs/architecture.md` for the full diagram and component descriptions.

## What I built

- A reusable Python package (`src/neurobridge_graph/`) covering data adaptation, domain mapping,
  graph construction, feature extraction, hazard-aware similarity, longitudinal trajectories,
  attribution, reference-calibrated envelopes, operational resilience interpretation, a Streamlit
  dashboard data/UI layer, and a PyTorch self-supervised modeling layer.
- Reviewer-friendly notebooks for each phase.
- A self-contained, portfolio-ready PyTorch HTML showcase.
- A zero-install public GitHub Pages site.
- A comprehensive test suite and documentation.

## Why small-N is handled differently

The unit of analysis is a within-subject, baseline-relative graph state — not a person treated as a
population sample. Repeated graph-derived observations within each subject carry learnable structure
without requiring a large population. The PyTorch layer is self-supervised (reconstruction), not a
supervised population-level predictor, and it does not treat features, domains, or timepoints as
independent people.

## PyTorch showcase

The Phase 12 showcase trains a small fully connected autoencoder on graph-derived trajectory feature
vectors using reconstruction and masked-reconstruction objectives. It produces latent trajectory
embeddings, reconstruction-mismatch analysis, a trajectory similarity matrix, and a consistency view
against the Phase 11 operational resilience annotations. It is an experimental
representation-learning prototype, not a clinical predictor.

## Dashboard

A Streamlit dashboard (`app.py`) provides interactive longitudinal review of trajectories,
attribution, reference-calibrated envelope status, and operational resilience interpretation cards.
The dashboard is for local technical review and is not hosted on GitHub Pages.

## Technical stack

Python, pandas, numpy, networkx, scikit-learn, matplotlib, PyTorch, Streamlit, pytest, Jupyter, and
a static HTML/CSS GitHub Pages site.

## Outputs

- adapter-generated graph-ready domain scores;
- biological adaptation graphs and graph features;
- within-subject longitudinal trajectory tables;
- explainable attribution tables;
- reference-calibrated envelope tables;
- operational resilience interpretation cards;
- PyTorch model card, latent embeddings, reconstruction-mismatch tables, similarity matrix;
- a static public showcase site.

## Guardrails

This is a research engineering prototype. It is not diagnosis, treatment guidance, health risk
scoring, exposure measurement, mission readiness classification, or an operational medical decision
system. Schema-demonstration data are illustrative only and are not scientific evidence.

## How to review the project quickly

- **Zero-install:** open the GitHub Pages site and the PyTorch showcase. No setup required.
- **Technical:** `pip install -r requirements.txt`, then `pytest`, then `streamlit run app.py`.
- **Deep dive:** read `docs/architecture.md`, `docs/methodology.md`, and the notebooks listed in
  `docs/notebook_index.md`.

This project demonstrates end-to-end research engineering: data ingestion, validation, graph
construction, longitudinal analysis, interpretation, dashboarding, PyTorch modeling, tests,
documentation, and public static deployment.
