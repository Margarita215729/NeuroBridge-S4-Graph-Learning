# Phase 12 Model Card — Trajectory Autoencoder

## Model name
NeuroBridge-S4 Trajectory Autoencoder

## Model type
Fully connected self-supervised autoencoder (PyTorch). Reconstruction and masked-reconstruction objective.

## Purpose
Phase 12 uses PyTorch for self-supervised representation learning on structured within-subject biological graph trajectories. The model learns compact latent representations of baseline-relative trajectory patterns. It does not predict health outcomes, diagnose conditions, classify mission readiness, measure hazard exposure, or produce health risk scores.

## Intended use
- Research-review exploration of within-subject graph trajectory structure.
- Learning compact latent representations of baseline-relative trajectory states.
- Comparing trajectory shape similarity in latent space.
- Providing a consistency view against Phase 11 operational resilience annotations.

## Not intended use
- Not diagnosis, disease detection, or clinical decision-making.
- Not treatment guidance.
- Not health risk scoring or risk-level assignment.
- Not exposure measurement or causal attribution.
- Not mission readiness classification.
- Not a validated prediction model.

## Input data
Graph-derived, baseline-relative within-subject trajectory tables from Phases 6-11 (delta, attribution, reference-envelope, recovery, and resilience-annotation outputs).

## Feature families
- attribution_hazard
- attribution_node
- attribution_subgraph
- domain_delta
- envelope
- graph_metric_delta
- hazard_context_delta
- recovery

## Training objective
Self-supervised mean-squared-error reconstruction of standardized trajectory feature vectors, including a masked-reconstruction variant (reconstruct full vector from a partially masked input).

## Training data structure
- trajectory rows (subject/timepoint states): 10
- input feature dimension: 45
- latent dimension: 8
- hidden dimensions: [45, 22]
- trainable parameters: 6569
- final reconstruction loss: 0.088143

## Independent subject caveat
The model does not treat domains, features, or timepoints as independent people. It learns from structured repeated graph-derived observations within subjects. Independent subject count remains small; this is a representation-learning prototype, not a population-level predictor.

## Outputs
- latent trajectory embeddings;
- row-level and feature-level reconstruction mismatch;
- latent cosine-similarity matrix;
- reconstruction error summary: mean 0.066690, median 0.060702, min 0.022827, max 0.153449.

## Connection to Phase 11 resilience interpretation
Phase 11 operational resilience states are used purely as **annotation metadata** to visualize and group learned representations. They are not training labels and are not treated as ground truth. The resulting view is a consistency check, not a validation of either layer.

## Limitations
- Experimental; not validated for operational use.
- No clinical outcome labels; reconstruction mismatch is not risk.
- Latent clusters are not clinical categories.
- Small independent subject count limits generalization claims.
- Example/schema data, if used, are not scientific evidence.

## Guardrails
This model is an experimental self-supervised representation-learning prototype. It is not diagnosis, treatment guidance, health risk scoring, exposure measurement, mission readiness classification, or an operational medical decision system.

The model does not treat domains, features, or timepoints as independent people. It learns from structured repeated graph-derived observations within subjects.

## Data provenance
This showcase used schema-demonstration data only. It is not scientific evidence.
