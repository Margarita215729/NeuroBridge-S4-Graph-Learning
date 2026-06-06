"""Phase 12 — Figures and the standalone reviewer-facing HTML showcase.

The HTML page (``results/html/phase12_pytorch_showcase.html``) is the main
portfolio artifact for Phase 12. It is a self-supervised representation-learning
showcase only: not diagnosis, not treatment guidance, not health risk scoring,
not exposure measurement, and not mission readiness classification.
"""

from __future__ import annotations

import base64
import html as _html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GUARDRAIL_BANNER = (
    "Experimental ML showcase only. Not diagnosis, not treatment guidance, not "
    "health risk scoring, not exposure measurement, not mission readiness "
    "classification."
)

_CAPTION = ("Self-supervised representation learning on graph-derived "
            "within-subject trajectories — research prototype, not a clinical "
            "or risk model.")

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
            "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _placeholder(output_path: "str | Path", message: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.55, message, ha="center", va="center", wrap=True, fontsize=11)
    ax.text(0.5, 0.18, _CAPTION, ha="center", va="center", fontsize=8, color="dimgray")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _category_colors(labels: list[str]) -> dict[str, str]:
    cats = sorted({str(x) for x in labels})
    return {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(cats)}


def plot_training_loss_curve(training_history: pd.DataFrame, output_path: "str | Path") -> Path:
    """Plot the reconstruction training loss curve."""
    if (training_history is None or training_history.empty
            or "reconstruction_loss" not in training_history.columns):
        return _placeholder(output_path, "Training was skipped (insufficient data).")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(training_history["epoch"], training_history["reconstruction_loss"],
            color="#4C72B0", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss (MSE)")
    ax.set_title("Self-supervised reconstruction loss (optimization only)")
    ax.grid(alpha=0.3)
    fig.text(0.5, -0.02, _CAPTION, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_latent_space(embeddings_df: pd.DataFrame, output_path: "str | Path") -> Path:
    """Scatter of the latent trajectory embedding (representation view only)."""
    latent_cols = [c for c in (embeddings_df.columns if embeddings_df is not None else [])
                   if c.startswith("latent_")]
    if embeddings_df is None or embeddings_df.empty or not latent_cols:
        return _placeholder(output_path, "No latent embeddings available.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = embeddings_df[latent_cols[0]].to_numpy(dtype=float)
    if len(latent_cols) >= 2:
        y = embeddings_df[latent_cols[1]].to_numpy(dtype=float)
        ylabel = latent_cols[1]
    else:
        y = np.arange(len(x), dtype=float)
        ylabel = "row index"

    fig, ax = plt.subplots(figsize=(8, 6))
    if "resilience_state_label" in embeddings_df.columns:
        labels = embeddings_df["resilience_state_label"].astype(str).tolist()
        colors = _category_colors(labels)
        for cat, col in colors.items():
            m = [i for i, lab in enumerate(labels) if lab == cat]
            ax.scatter(x[m], y[m], s=90, color=col, label=cat, edgecolor="white", linewidth=0.6)
        ax.legend(title="Phase 11 resilience state (annotation)", fontsize=8,
                  loc="best", framealpha=0.9)
    else:
        ax.scatter(x, y, s=90, color="#4C72B0", edgecolor="white", linewidth=0.6)

    if "trajectory_id" in embeddings_df.columns:
        for xi, yi, tid in zip(x, y, embeddings_df["trajectory_id"].astype(str)):
            ax.annotate(tid, (xi, yi), fontsize=6, alpha=0.6,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(latent_cols[0])
    ax.set_ylabel(ylabel)
    ax.set_title("Latent trajectory embedding (learned representation, not validation)")
    ax.grid(alpha=0.3)
    fig.text(0.5, -0.02, _CAPTION, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_reconstruction_error(reconstruction_df: pd.DataFrame, output_path: "str | Path") -> Path:
    """Bar chart of per-trajectory reconstruction mismatch (not risk)."""
    if (reconstruction_df is None or reconstruction_df.empty
            or "reconstruction_mse" not in reconstruction_df.columns):
        return _placeholder(output_path, "No reconstruction errors available.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = reconstruction_df.copy()
    ids = (df["trajectory_id"].astype(str).tolist() if "trajectory_id" in df.columns
           else [str(i) for i in range(len(df))])
    vals = df["reconstruction_mse"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(ids) + 3), 4.8))
    colors = "#C44E52"
    if "resilience_state_label" in df.columns:
        cmap = _category_colors(df["resilience_state_label"].astype(str).tolist())
        colors = [cmap[str(l)] for l in df["resilience_state_label"].astype(str)]
    ax.bar(range(len(ids)), vals, color=colors)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Per-trajectory reconstruction mismatch (representation quality, not risk)")
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, -0.04, _CAPTION, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_reconstruction_error(
    feature_reconstruction_df: pd.DataFrame, output_path: "str | Path") -> Path:
    """Top features by reconstruction mismatch."""
    if (feature_reconstruction_df is None or feature_reconstruction_df.empty
            or "mean_squared_error" not in feature_reconstruction_df.columns):
        return _placeholder(output_path, "No feature-level reconstruction errors available.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top = feature_reconstruction_df.sort_values(
        "mean_squared_error", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(top) + 1.5)))
    ax.barh(top["feature_name"].astype(str), top["mean_squared_error"].astype(float),
            color="#55A868")
    ax.set_xlabel("Mean squared reconstruction error")
    ax.set_title("Feature-level reconstruction mismatch (top features)")
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.5, -0.03, _CAPTION, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_similarity_heatmap(similarity_matrix: pd.DataFrame, output_path: "str | Path") -> Path:
    """Heatmap of latent cosine similarity (representation similarity only)."""
    if similarity_matrix is None or similarity_matrix.empty:
        return _placeholder(output_path, "No similarity matrix available.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids = [str(i) for i in similarity_matrix.index]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(ids) + 2), max(5, 0.5 * len(ids) + 2)))
    im = ax.imshow(similarity_matrix.to_numpy(dtype=float), cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=7)
    ax.set_title("Latent trajectory similarity (cosine) — representation, not shared health state")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.text(0.5, -0.04, _CAPTION, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_resilience_annotation_view(
    consistency_df: pd.DataFrame, output_path: "str | Path") -> Path:
    """Mean reconstruction mismatch grouped by Phase 11 resilience state (consistency view)."""
    if (consistency_df is None or consistency_df.empty
            or "reconstruction_mse" not in consistency_df.columns
            or "resilience_state_label" not in consistency_df.columns):
        return _placeholder(output_path, "No Phase 11 resilience annotations available.")
    df = consistency_df[consistency_df["resilience_state_label"] != "not_available"]
    if df.empty:
        return _placeholder(output_path, "No Phase 11 resilience annotations available.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("resilience_state_label")["reconstruction_mse"]
    means = grouped.mean().sort_values()
    counts = grouped.count()
    colors = _category_colors(list(means.index))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * len(means) + 2)))
    ax.barh(list(means.index), means.values,
            color=[colors[c] for c in means.index])
    for i, (name, val) in enumerate(means.items()):
        ax.text(val, i, f"  n={int(counts[name])}", va="center", fontsize=8)
    ax.set_xlabel("Mean reconstruction MSE")
    ax.set_title("Reconstruction mismatch by resilience state (consistency view, not validation)")
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.5, -0.03, _CAPTION, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_phase12_figures(
    training_history: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    reconstruction_df: pd.DataFrame,
    feature_reconstruction_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    consistency_df: pd.DataFrame | None,
    figures_dir: "str | Path" = "results/figures",
) -> dict[str, str]:
    """Generate all six Phase 12 figures and return a name->path mapping."""
    figures_dir = Path(figures_dir)
    paths = {
        "training_loss": plot_training_loss_curve(
            training_history, figures_dir / "phase12_training_loss_curve.png"),
        "latent_space": plot_latent_space(
            embeddings_df, figures_dir / "phase12_latent_space.png"),
        "reconstruction_error": plot_reconstruction_error(
            reconstruction_df, figures_dir / "phase12_reconstruction_error.png"),
        "feature_reconstruction_error": plot_feature_reconstruction_error(
            feature_reconstruction_df, figures_dir / "phase12_feature_reconstruction_error.png"),
        "similarity_heatmap": plot_similarity_heatmap(
            similarity_matrix, figures_dir / "phase12_similarity_heatmap.png"),
        "resilience_annotation_view": plot_resilience_annotation_view(
            consistency_df, figures_dir / "phase12_resilience_annotation_view.png"),
    }
    return {k: str(v) for k, v in paths.items()}


# ---------------------------------------------------------------------------
# HTML showcase
# ---------------------------------------------------------------------------

def _img_data_uri(path: "str | Path") -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _figure_block(figure_paths: dict | None, key: str, alt: str) -> str:
    if not figure_paths or key not in figure_paths:
        return f'<p class="muted">Figure unavailable: {_html.escape(alt)}.</p>'
    uri = _img_data_uri(figure_paths[key])
    if uri is None:
        return f'<p class="muted">Figure unavailable: {_html.escape(alt)}.</p>'
    return f'<img class="fig" src="{uri}" alt="{_html.escape(alt)}" />'


def _table_preview(df: pd.DataFrame | None, max_rows: int = 8, max_cols: int = 12) -> str:
    if df is None or df.empty:
        return '<p class="muted">Table not available.</p>'
    view = df.copy()
    if view.shape[1] > max_cols:
        view = view.iloc[:, :max_cols]
        truncated_cols = True
    else:
        truncated_cols = False
    view = view.head(max_rows)
    table_html = view.to_html(index=False, classes="tbl", border=0, escape=True)
    note = ""
    if truncated_cols or len(df) > max_rows:
        note = (f'<p class="muted">Showing {min(max_rows, len(df))} of {len(df)} rows'
                + (f", {max_cols} of {df.shape[1]} columns" if truncated_cols else "")
                + ".</p>")
    return table_html + note


def _md_to_html(md_text: str) -> str:
    """Minimal Markdown -> HTML for the model card (headers, lists, paragraphs)."""
    out: list[str] = []
    in_list = False
    for raw in md_text.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<h3>{_html.escape(line[3:])}</h3>")
        elif line.startswith("# "):
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<h2>{_html.escape(line[2:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{_html.escape(line[2:])}</li>")
        elif not line.strip():
            if in_list:
                out.append("</ul>"); in_list = False
        else:
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<p>{_html.escape(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


_CSS = """
:root{--bg:#0b1020;--card:#151b2e;--ink:#eef2ff;--muted:#9aa6c4;--accent:#5b8cff;
--accent2:#2dd4bf;--warn:#f59e0b;--line:#26304d;}
*{box-sizing:border-box;}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
background:var(--bg);color:var(--ink);line-height:1.55;}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px 80px;}
.hero{background:linear-gradient(135deg,#1e2a55 0%,#0b1020 60%);padding:64px 20px 40px;
border-bottom:1px solid var(--line);}
.hero .wrap{padding-bottom:0;}
.eyebrow{letter-spacing:.18em;text-transform:uppercase;color:var(--accent2);font-size:12px;font-weight:700;}
h1{font-size:40px;line-height:1.1;margin:10px 0 8px;}
.subtitle{color:var(--muted);font-size:18px;max-width:760px;}
.banner{background:rgba(245,158,11,.12);border:1px solid var(--warn);color:#fde68a;
padding:12px 16px;border-radius:10px;margin:22px 0 0;font-size:14px;font-weight:600;}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:28px 0 0;}
.metric{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;}
.metric .v{font-size:26px;font-weight:800;color:var(--accent);}
.metric .k{color:var(--muted);font-size:13px;margin-top:4px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:26px;margin:22px 0;}
.card h2{margin:0 0 10px;font-size:23px;}
.card h3{margin:18px 0 6px;font-size:16px;color:var(--accent2);}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px;}
@media(max-width:820px){.grid2{grid-template-columns:1fr;}h1{font-size:30px;}}
.muted{color:var(--muted);font-size:13px;}
.fig{width:100%;border-radius:10px;border:1px solid var(--line);background:#fff;}
.tbl{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:8px;display:block;overflow-x:auto;}
.tbl th,.tbl td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;white-space:nowrap;}
.tbl th{color:var(--accent2);font-weight:700;}
.pill{display:inline-block;background:rgba(91,140,255,.14);border:1px solid var(--accent);
color:#cfe0ff;border-radius:999px;padding:3px 10px;font-size:12px;margin:3px 4px 0 0;}
.guard{background:rgba(245,158,11,.1);border-left:4px solid var(--warn);padding:14px 18px;
border-radius:8px;color:#fde68a;}
footer{color:var(--muted);font-size:12px;text-align:center;margin-top:30px;}
a{color:var(--accent);}
"""


def create_phase12_showcase_html(
    output_path: "str | Path",
    readiness_report: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    training_history: pd.DataFrame,
    reconstruction_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    model_card_text: str,
    resilience_consistency_df: pd.DataFrame | None = None,
    figure_paths: dict | None = None,
    data_provenance_note: str | None = None,
    model_metadata: dict | None = None,
) -> Path:
    """Generate a standalone, portfolio-ready HTML showcase page."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md = model_metadata or {}

    n_rows = 0 if feature_matrix is None else int(len(feature_matrix))
    latent_cols = ([c for c in embeddings_df.columns if c.startswith("latent_")]
                   if embeddings_df is not None and not embeddings_df.empty else [])
    latent_dim = md.get("latent_dim", len(latent_cols))
    input_dim = md.get("input_dim", "n/a")
    param_count = md.get("parameter_count", "n/a")
    final_loss = ("n/a" if training_history is None or training_history.empty
                  else f"{float(training_history['reconstruction_loss'].iloc[-1]):.5f}")
    recon_mean = ("n/a" if reconstruction_df is None or reconstruction_df.empty
                  or "reconstruction_mse" not in reconstruction_df.columns
                  else f"{reconstruction_df['reconstruction_mse'].mean():.5f}")

    resilience_used = (resilience_consistency_df is not None
                       and not resilience_consistency_df.empty
                       and "resilience_state_label" in resilience_consistency_df.columns
                       and (resilience_consistency_df["resilience_state_label"]
                            != "not_available").any())

    provenance_html = ""
    if data_provenance_note:
        provenance_html = f'<div class="banner">{_html.escape(data_provenance_note)}</div>'

    families = []
    # Derive family pills from the feature matrix prefixes.
    for c in (feature_matrix.columns if feature_matrix is not None else []):
        if "__" in c:
            families.append(c.split("__")[0])
    family_pills = "".join(
        f'<span class="pill">{_html.escape(f)}</span>' for f in sorted(set(families))) \
        or '<span class="muted">no feature families available</span>'

    consistency_section = (
        f"""
        <p>Phase 11 operational resilience states are joined to the learned
        representation purely as <strong>annotation metadata</strong>. The view
        below groups reconstruction mismatch by resilience state. This is a
        <strong>consistency view, not validation</strong> of either layer.</p>
        {_figure_block(figure_paths, "resilience_annotation_view", "Reconstruction mismatch by resilience state")}
        {_table_preview(resilience_consistency_df)}
        """ if resilience_used else
        """
        <p class="muted">Phase 11 resilience annotations were not available, so
        the consistency view is shown as not-available. The learning pipeline runs
        independently of Phase 11.</p>
        """)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>NeuroBridge-S4 PyTorch Temporal Graph Learning Showcase</title>
<style>{_CSS}</style>
</head>
<body>
<header class="hero">
  <div class="wrap">
    <div class="eyebrow">NeuroBridge-S4 · Phase 12 · PyTorch</div>
    <h1>NeuroBridge-S4 PyTorch Temporal Graph Learning Showcase</h1>
    <p class="subtitle">Self-supervised representation learning for within-subject
    biological adaptation trajectories.</p>
    <div class="banner">⚠️ {GUARDRAIL_BANNER}</div>
    {provenance_html}
    <div class="metrics">
      <div class="metric"><div class="v">{n_rows}</div><div class="k">trajectory states</div></div>
      <div class="metric"><div class="v">{input_dim}</div><div class="k">input features</div></div>
      <div class="metric"><div class="v">{latent_dim}</div><div class="k">latent dimensions</div></div>
      <div class="metric"><div class="v">{param_count}</div><div class="k">trainable params</div></div>
      <div class="metric"><div class="v">{final_loss}</div><div class="k">final recon loss</div></div>
      <div class="metric"><div class="v">{recon_mean}</div><div class="k">mean recon MSE</div></div>
    </div>
  </div>
</header>
<div class="wrap">

  <div class="card">
    <h2>1 · Executive summary</h2>
    <p>{_html.escape(md.get("framing", "Phase 12 uses PyTorch for self-supervised representation learning on structured within-subject biological graph trajectories. The model learns compact latent representations of baseline-relative trajectory patterns."))}</p>
    <p>It does not predict health outcomes, diagnose conditions, classify mission
    readiness, measure hazard exposure, or produce health risk scores. It is a
    representation-learning prototype that complements — and does not replace —
    the transparent, rule-based NeuroBridge-S4 analysis pipeline.</p>
  </div>

  <div class="card">
    <h2>2 · What the model learns</h2>
    <p>The autoencoder maps a graph-derived trajectory feature vector to a compact
    latent representation and back to a reconstructed vector:</p>
    <p><code>trajectory feature vector → latent representation → reconstructed trajectory feature vector</code></p>
    <p>It also learns from a masked variant — reconstructing the full vector from a
    partially masked input — which encourages it to capture internal structure
    across feature families:</p>
    <div>{family_pills}</div>
  </div>

  <div class="card">
    <h2>3 · Why this is not supervised learning on n=4</h2>
    <div class="guard">The independent subject count remains small. Phase 12 does
    not treat features, domains, or timepoints as independent people. Instead, it
    learns from structured repeated graph-derived observations within subjects.
    The objective is representation learning, not population-level prediction.</div>
  </div>

  <div class="card">
    <h2>4 · Architecture overview</h2>
    <div class="grid2">
      <div>
        <h3>Encoder</h3>
        <p>Input → hidden (GELU + dropout) → latent bottleneck.</p>
        <h3>Decoder</h3>
        <p>Latent → symmetric hidden (GELU + dropout) → reconstructed features.</p>
      </div>
      <div>
        <h3>Configuration</h3>
        <ul>
          <li>input dimension: {input_dim}</li>
          <li>hidden dimensions: {_html.escape(str(md.get("hidden_dims", "n/a")))}</li>
          <li>latent dimension: {latent_dim}</li>
          <li>trainable parameters: {param_count}</li>
          <li>objective: MSE reconstruction (+ masked reconstruction)</li>
          <li>device: CPU (deterministic seed)</li>
        </ul>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>5 · Input trajectory feature matrix</h2>
    <p class="muted">One row per graph-derived trajectory state (subject × timepoint).</p>
    {_table_preview(feature_matrix)}
  </div>

  <div class="card">
    <h2>6 · Training loss curve</h2>
    <p>Reconstruction optimization only — lower loss means the model reconstructs
    standardized trajectory features more faithfully.</p>
    {_figure_block(figure_paths, "training_loss", "Training loss curve")}
  </div>

  <div class="card">
    <h2>7 · Latent trajectory embedding</h2>
    <p>A visualization of learned representation structure, not validation. Points
    are annotated with Phase 11 resilience states where available.</p>
    {_figure_block(figure_paths, "latent_space", "Latent trajectory embedding")}
  </div>

  <div class="card">
    <h2>8 · Reconstruction mismatch analysis</h2>
    <p>Reconstruction mismatch indicates how hard a trajectory state is to
    reconstruct. It is a representation-quality signal, <strong>not a risk
    score</strong>.</p>
    <div class="grid2">
      <div>{_figure_block(figure_paths, "reconstruction_error", "Per-trajectory reconstruction error")}</div>
      <div>{_figure_block(figure_paths, "feature_reconstruction_error", "Feature-level reconstruction error")}</div>
    </div>
  </div>

  <div class="card">
    <h2>9 · Trajectory similarity map</h2>
    <p>Cosine similarity of latent embeddings — latent representation similarity,
    <strong>not shared health state</strong>.</p>
    {_figure_block(figure_paths, "similarity_heatmap", "Latent similarity heatmap")}
  </div>

  <div class="card">
    <h2>10 · Operational resilience annotation layer</h2>
    <p>Phase 11 operational resilience interpretation provides categorical
    annotations (e.g. stable compensated, localized adaptive shift, systemic strain
    pattern). Phase 12 uses these only to <strong>annotate</strong> the learned
    representation — never as training labels.</p>
  </div>

  <div class="card">
    <h2>11 · Consistency with Phase 11 resilience interpretation</h2>
    {consistency_section}
  </div>

  <div class="card">
    <h2>12 · Model card summary</h2>
    {_md_to_html(model_card_text or "")}
  </div>

  <div class="card">
    <h2>13 · Guardrails and limitations</h2>
    <div class="guard">{GUARDRAIL_BANNER}</div>
    <ul>
      <li>Experimental prototype; not validated for operational use.</li>
      <li>No clinical outcome labels; latent clusters are not clinical categories.</li>
      <li>Reconstruction mismatch is not risk.</li>
      <li>Small independent subject count remains a limitation.</li>
      <li>Phase 11 resilience states are annotations, not ground-truth labels.</li>
      <li>Example/schema data, if used, are not scientific evidence.</li>
    </ul>
  </div>

  <footer>
    NeuroBridge-S4 Graph Learning · Phase 12 · Self-supervised representation
    learning prototype · Not diagnosis, treatment guidance, health risk scoring,
    exposure measurement, or mission readiness classification.
  </footer>
</div>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")
    return output_path
