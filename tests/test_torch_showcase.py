"""Tests for Phase 12 reporting, consistency view, and HTML showcase."""

import numpy as np
import pandas as pd

from neurobridge_graph.torch_reporting import (
    generate_phase12_model_card,
    generate_phase12_showcase_report,
    build_resilience_consistency_view,
    GUARDRAIL,
)
from neurobridge_graph.torch_showcase import (
    create_phase12_showcase_html,
    generate_phase12_figures,
)

_GUARDRAIL_FRAGMENT = ("not diagnosis, treatment guidance, health risk scoring, "
                       "exposure measurement, mission readiness classification")


def _embeddings(with_resilience=True, n=4):
    df = pd.DataFrame({
        "trajectory_id": [f"S__T{i}" for i in range(n)],
        "subject_id": ["S"] * n,
        "timepoint": [f"T{i}" for i in range(n)],
        "mission_phase": ["x"] * n,
        "latent_0": np.linspace(-1, 1, n),
        "latent_1": np.linspace(1, -1, n),
    })
    if with_resilience:
        df["resilience_state_label"] = ["Stable compensated trajectory",
                                        "Systemic strain pattern",
                                        "Localized adaptive shift",
                                        "Systemic strain pattern"][:n]
        df["dominant_adaptation_mode"] = "Multi-subgraph distributed"
    return df


def _reconstruction(with_resilience=True, n=4):
    df = _embeddings(with_resilience, n).drop(columns=["latent_0", "latent_1"])
    df["reconstruction_mse"] = np.linspace(0.1, 0.4, n)
    df["reconstruction_rmse"] = np.sqrt(df["reconstruction_mse"])
    return df


def _feature_errors():
    return pd.DataFrame({
        "feature_name": ["domain_delta__a", "graph_delta__b"],
        "mean_squared_error": [0.3, 0.1],
        "root_mean_squared_error": [0.55, 0.32],
    })


def _history():
    return pd.DataFrame({"epoch": [1, 2, 3], "reconstruction_loss": [0.9, 0.5, 0.2]})


def _catalog():
    return pd.DataFrame({
        "feature_name": ["domain_delta__a", "graph_delta__b"],
        "source_table": ["longitudinal_node_deltas", "longitudinal_graph_deltas"],
        "feature_family": ["domain_delta", "graph_metric_delta"],
        "description": ["d", "g"],
    })


def test_model_card_includes_guardrail_and_sections():
    card = generate_phase12_model_card(
        {"model_name": "Test AE", "input_dim": 5, "latent_dim": 2, "parameter_count": 123},
        _catalog(), _history(), _reconstruction(), _embeddings())
    assert "Model Card" in card
    assert GUARDRAIL in card
    assert "Connection to Phase 11 resilience interpretation" in card
    assert "independent" in card.lower()


def test_showcase_report_includes_guardrail():
    rep = generate_phase12_showcase_report(
        pd.DataFrame({"table_name": ["t"], "required_or_optional": ["required"],
                      "status": ["available"], "rows": [3]}),
        _embeddings(), _history(), _reconstruction(), _embeddings())
    assert _GUARDRAIL_FRAGMENT in rep
    assert "not a risk score" in rep.lower()


def test_consistency_view_with_resilience():
    cv = build_resilience_consistency_view(_embeddings(), _reconstruction(), None)
    assert "resilience_state_label" in cv.columns
    assert (cv["resilience_state_label"] != "not_available").any()
    assert "reconstruction_mse" in cv.columns


def test_consistency_view_without_resilience_is_safe():
    cv = build_resilience_consistency_view(
        _embeddings(with_resilience=False), _reconstruction(with_resilience=False), None)
    assert (cv["resilience_state_label"] == "not_available").all()


def test_consistency_view_empty_reconstruction():
    cv = build_resilience_consistency_view(pd.DataFrame(), pd.DataFrame(), None)
    assert cv.empty


def test_html_showcase_generated_with_key_sections(tmp_path):
    figs = generate_phase12_figures(
        _history(), _embeddings(), _reconstruction(), _feature_errors(),
        pd.DataFrame(np.eye(4), index=[f"S__T{i}" for i in range(4)],
                     columns=[f"S__T{i}" for i in range(4)]),
        build_resilience_consistency_view(_embeddings(), _reconstruction(), None),
        figures_dir=tmp_path / "figs")
    card = generate_phase12_model_card(
        {"input_dim": 5, "latent_dim": 2, "parameter_count": 123},
        _catalog(), _history(), _reconstruction(), _embeddings())
    out = create_phase12_showcase_html(
        tmp_path / "showcase.html",
        readiness_report=pd.DataFrame({"table_name": ["t"], "required_or_optional": ["required"],
                                       "status": ["available"], "rows": [3], "columns": [2],
                                       "notes": ["ok"]}),
        feature_matrix=pd.DataFrame({"trajectory_id": ["a"], "domain_delta__x": [0.1]}),
        training_history=_history(),
        reconstruction_df=_reconstruction(),
        embeddings_df=_embeddings(),
        similarity_matrix=pd.DataFrame(np.eye(4)),
        model_card_text=card,
        resilience_consistency_df=build_resilience_consistency_view(_embeddings(), _reconstruction(), None),
        figure_paths=figs,
        data_provenance_note="This showcase used schema-demonstration data only. It is not scientific evidence.",
        model_metadata={"input_dim": 5, "latent_dim": 2, "parameter_count": 123,
                        "hidden_dims": [8]},
    )
    assert out.exists()
    text = out.read_text()
    assert "NeuroBridge-S4 PyTorch Temporal Graph Learning Showcase" in text
    assert "Experimental ML showcase only" in text
    for section in ("Executive summary", "Architecture overview", "Latent trajectory embedding",
                    "Trajectory similarity map", "Consistency with Phase 11",
                    "Model card summary", "Guardrails and limitations"):
        assert section in text
    assert "schema-demonstration data only" in text


def test_html_showcase_without_resilience_does_not_crash(tmp_path):
    out = create_phase12_showcase_html(
        tmp_path / "s2.html",
        readiness_report=pd.DataFrame(),
        feature_matrix=pd.DataFrame({"trajectory_id": ["a"], "domain_delta__x": [0.1]}),
        training_history=_history(),
        reconstruction_df=_reconstruction(with_resilience=False),
        embeddings_df=_embeddings(with_resilience=False),
        similarity_matrix=pd.DataFrame(),
        model_card_text="# card\n\nguardrail",
        resilience_consistency_df=None,
        figure_paths=None,
    )
    assert out.exists()
    assert "not-available" in out.read_text() or "not available" in out.read_text().lower()
