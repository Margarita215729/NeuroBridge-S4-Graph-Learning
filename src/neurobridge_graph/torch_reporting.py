"""Phase 12 — Model card, plain-language report, and resilience consistency view.

All artifacts produced here carry the Phase 12 guardrail: this is an
experimental self-supervised representation-learning prototype, not diagnosis,
treatment guidance, health risk scoring, exposure measurement, mission readiness
classification, or an operational medical decision system.
"""

from __future__ import annotations

import pandas as pd

GUARDRAIL = (
    "This model is an experimental self-supervised representation-learning "
    "prototype. It is not diagnosis, treatment guidance, health risk scoring, "
    "exposure measurement, mission readiness classification, or an operational "
    "medical decision system."
)

INDEPENDENCE_NOTE = (
    "The model does not treat domains, features, or timepoints as independent "
    "people. It learns from structured repeated graph-derived observations "
    "within subjects."
)

CORE_FRAMING = (
    "Phase 12 uses PyTorch for self-supervised representation learning on "
    "structured within-subject biological graph trajectories. The model learns "
    "compact latent representations of baseline-relative trajectory patterns. It "
    "does not predict health outcomes, diagnose conditions, classify mission "
    "readiness, measure hazard exposure, or produce health risk scores."
)


def _final_loss(training_history: pd.DataFrame) -> str:
    if training_history is None or training_history.empty \
            or "reconstruction_loss" not in training_history.columns:
        return "n/a (training skipped)"
    return f"{float(training_history['reconstruction_loss'].iloc[-1]):.6f}"


def _recon_summary(reconstruction_df: pd.DataFrame) -> str:
    if reconstruction_df is None or reconstruction_df.empty \
            or "reconstruction_mse" not in reconstruction_df.columns:
        return "n/a"
    s = reconstruction_df["reconstruction_mse"]
    return (f"mean {s.mean():.6f}, median {s.median():.6f}, "
            f"min {s.min():.6f}, max {s.max():.6f}")


def generate_phase12_model_card(
    model_metadata: dict,
    feature_catalog: pd.DataFrame,
    training_history: pd.DataFrame,
    reconstruction_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    data_provenance_note: str | None = None,
) -> str:
    """Generate a Markdown model card for the Phase 12 autoencoder."""
    md = model_metadata or {}
    families = []
    if feature_catalog is not None and not feature_catalog.empty \
            and "feature_family" in feature_catalog.columns:
        families = sorted(feature_catalog["feature_family"].astype(str).unique())
    n_latent = 0
    if embeddings_df is not None and not embeddings_df.empty:
        n_latent = len([c for c in embeddings_df.columns if c.startswith("latent_")])

    lines = [
        "# Phase 12 Model Card — Trajectory Autoencoder",
        "",
        "## Model name",
        f"{md.get('model_name', 'NeuroBridge-S4 Trajectory Autoencoder')}",
        "",
        "## Model type",
        "Fully connected self-supervised autoencoder (PyTorch). Reconstruction "
        "and masked-reconstruction objective.",
        "",
        "## Purpose",
        CORE_FRAMING,
        "",
        "## Intended use",
        "- Research-review exploration of within-subject graph trajectory structure.",
        "- Learning compact latent representations of baseline-relative trajectory states.",
        "- Comparing trajectory shape similarity in latent space.",
        "- Providing a consistency view against Phase 11 operational resilience annotations.",
        "",
        "## Not intended use",
        "- Not diagnosis, disease detection, or clinical decision-making.",
        "- Not treatment guidance.",
        "- Not health risk scoring or risk-level assignment.",
        "- Not exposure measurement or causal attribution.",
        "- Not mission readiness classification.",
        "- Not a validated prediction model.",
        "",
        "## Input data",
        "Graph-derived, baseline-relative within-subject trajectory tables from "
        "Phases 6-11 (delta, attribution, reference-envelope, recovery, and "
        "resilience-annotation outputs).",
        "",
        "## Feature families",
        ("- " + "\n- ".join(families)) if families else "- (none available)",
        "",
        "## Training objective",
        "Self-supervised mean-squared-error reconstruction of standardized "
        "trajectory feature vectors, including a masked-reconstruction variant "
        "(reconstruct full vector from a partially masked input).",
        "",
        "## Training data structure",
        f"- trajectory rows (subject/timepoint states): {md.get('n_rows', 'n/a')}",
        f"- input feature dimension: {md.get('input_dim', 'n/a')}",
        f"- latent dimension: {md.get('latent_dim', n_latent or 'n/a')}",
        f"- hidden dimensions: {md.get('hidden_dims', 'n/a')}",
        f"- trainable parameters: {md.get('parameter_count', 'n/a')}",
        f"- final reconstruction loss: {_final_loss(training_history)}",
        "",
        "## Independent subject caveat",
        INDEPENDENCE_NOTE + " Independent subject count remains small; this is a "
        "representation-learning prototype, not a population-level predictor.",
        "",
        "## Outputs",
        "- latent trajectory embeddings;",
        "- row-level and feature-level reconstruction mismatch;",
        "- latent cosine-similarity matrix;",
        f"- reconstruction error summary: {_recon_summary(reconstruction_df)}.",
        "",
        "## Connection to Phase 11 resilience interpretation",
        "Phase 11 operational resilience states are used purely as **annotation "
        "metadata** to visualize and group learned representations. They are not "
        "training labels and are not treated as ground truth. The resulting view "
        "is a consistency check, not a validation of either layer.",
        "",
        "## Limitations",
        "- Experimental; not validated for operational use.",
        "- No clinical outcome labels; reconstruction mismatch is not risk.",
        "- Latent clusters are not clinical categories.",
        "- Small independent subject count limits generalization claims.",
        "- Example/schema data, if used, are not scientific evidence.",
        "",
        "## Guardrails",
        GUARDRAIL,
        "",
        INDEPENDENCE_NOTE,
    ]
    if data_provenance_note:
        lines += ["", "## Data provenance", data_provenance_note]
    return "\n".join(lines) + "\n"


def build_resilience_consistency_view(
    embeddings_df: pd.DataFrame,
    reconstruction_df: pd.DataFrame,
    resilience_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join learned embedding/reconstruction outputs with Phase 11 resilience states.

    This is a consistency view, not validation. Missing Phase 11 annotations do
    not raise; the resilience columns are filled with ``not_available``.
    """
    columns = [
        "trajectory_id", "subject_id", "timepoint", "mission_phase",
        "resilience_state_label", "dominant_adaptation_mode",
        "reconstruction_mse", "reconstruction_rmse",
    ]
    if reconstruction_df is None or reconstruction_df.empty:
        out = pd.DataFrame(columns=columns)
        out.attrs["note"] = "No reconstruction output available; consistency view is empty."
        return out

    base_cols = [c for c in ("trajectory_id", "subject_id", "timepoint", "mission_phase",
                             "reconstruction_mse", "reconstruction_rmse")
                 if c in reconstruction_df.columns]
    out = reconstruction_df[base_cols].copy()

    # Prefer resilience columns already present on the embeddings/reconstruction
    # metadata; otherwise merge from the resilience table.
    have_labels = "resilience_state_label" in reconstruction_df.columns
    if have_labels:
        for col in ("resilience_state_label", "dominant_adaptation_mode"):
            if col in reconstruction_df.columns:
                out[col] = reconstruction_df[col].values
    elif (resilience_df is not None and not resilience_df.empty
          and {"subject_id", "timepoint"}.issubset(resilience_df.columns)):
        keep = [c for c in ("subject_id", "timepoint", "resilience_state_label",
                            "dominant_adaptation_mode") if c in resilience_df.columns]
        out = out.merge(resilience_df[keep].drop_duplicates(subset=["subject_id", "timepoint"]),
                        on=["subject_id", "timepoint"], how="left")

    for col in ("resilience_state_label", "dominant_adaptation_mode"):
        if col not in out.columns:
            out[col] = "not_available"
        out[col] = out[col].fillna("not_available")

    out = out[[c for c in columns if c in out.columns]]
    out.attrs["note"] = (
        "Consistency view only: Phase 11 resilience states are annotations, not "
        "labels. This does not validate either layer.")
    return out


def generate_phase12_showcase_report(
    readiness_report: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    training_history: pd.DataFrame,
    reconstruction_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    data_provenance_note: str | None = None,
) -> str:
    """Generate a plain-language Phase 12 report."""
    n_rows = 0 if feature_matrix is None else int(len(feature_matrix))
    n_latent = 0
    if embeddings_df is not None and not embeddings_df.empty:
        n_latent = len([c for c in embeddings_df.columns if c.startswith("latent_")])

    lines = [
        "NeuroBridge-S4 Graph Learning",
        "Phase 12 — PyTorch Temporal Graph Autoencoder Showcase Report",
        "=" * 70,
        "",
        "Overview",
        "-" * 70,
        CORE_FRAMING,
        "",
        INDEPENDENCE_NOTE,
        "",
    ]
    if data_provenance_note:
        lines += ["Data provenance", "-" * 70, data_provenance_note, ""]

    lines += ["Input readiness", "-" * 70]
    if readiness_report is not None and not readiness_report.empty:
        for _, r in readiness_report.iterrows():
            lines.append(f"- {r['table_name']} [{r['required_or_optional']}]: "
                         f"{r['status']} ({int(r['rows'])} rows)")
    else:
        lines.append("- readiness report unavailable")
    lines.append("")

    lines += ["Model and training", "-" * 70,
              f"- trajectory rows: {n_rows}",
              f"- latent dimension: {n_latent}",
              f"- final reconstruction loss: {_final_loss(training_history)}",
              f"- reconstruction error (MSE): {_recon_summary(reconstruction_df)}",
              ""]

    lines += ["How to read the outputs", "-" * 70,
              "- Training loss: reconstruction optimization only.",
              "- Latent space: a visualization of learned representation structure, "
              "not validation.",
              "- Reconstruction mismatch: how hard a trajectory state is to "
              "reconstruct; it is not a risk score.",
              "- Similarity matrix: latent representation similarity, not shared "
              "health state.",
              "- Resilience annotation view: a consistency view against Phase 11 "
              "interpretation, not validation.",
              ""]

    lines += ["Limitations", "-" * 70,
              "- Experimental prototype; not validated for operational use.",
              "- No clinical outcome labels; latent clusters are not clinical categories.",
              "- Small independent subject count remains a limitation.",
              "- Phase 11 resilience states are annotations, not ground-truth labels.",
              "- Example/schema data, if used, are not scientific evidence.",
              ""]

    lines += ["Guardrails", "-" * 70, GUARDRAIL, "",
              "Next phase", "-" * 70,
              "Phase 13 — Portfolio release: package the showcase, figures, and "
              "model card for reviewer-facing presentation.", ""]
    return "\n".join(lines)
