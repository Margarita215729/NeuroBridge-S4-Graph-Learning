#!/usr/bin/env python3
"""Phase 13 — Build the GitHub Pages static showcase site under ``docs/``.

This script is idempotent and never fails on missing optional figures. It:

1. creates the ``docs/assets/*`` directories;
2. publishes the self-contained Phase 12 PyTorch showcase into ``docs/`` with a
   small back-navigation bar;
3. copies selected figures from ``results/figures/`` into ``docs/assets/figures/``;
4. copies the Phase 12 model card / report into ``docs/assets/reports/`` if present;
5. validates that ``docs/index.html`` exists;
6. prints GitHub Pages setup instructions.

No new scientific modeling happens here; this is a packaging/release step.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
RESULTS = REPO / "results"

ASSET_DIRS = [
    DOCS / "assets",
    DOCS / "assets" / "css",
    DOCS / "assets" / "img",
    DOCS / "assets" / "figures",
    DOCS / "assets" / "screenshots",
    DOCS / "assets" / "reports",
    DOCS / "assets" / "data",
]

FIGURES = [
    "phase12_training_loss_curve.png",
    "phase12_latent_space.png",
    "phase12_reconstruction_error.png",
    "phase12_similarity_heatmap.png",
    "phase12_resilience_annotation_view.png",
    "phase12_feature_reconstruction_error.png",
    "phase11_resilience_state_timeline.png",
    "phase11_resilience_state_summary.png",
]

REPORTS = [
    "phase12_model_card.md",
    "phase12_pytorch_showcase_report.txt",
]

PAGES_URL = "https://Margarita215729.github.io/NeuroBridge-S4-Graph-Learning/"

_BACKBAR = (
    '<div class="nb-backbar"><a href="index.html">&larr; Back to NeuroBridge-S4 '
    'Showcase</a></div>\n'
)
_BACKBAR_STYLE = (
    "<style>.nb-backbar{position:sticky;top:0;z-index:100;"
    "background:rgba(11,16,32,.92);border-bottom:1px solid #26304d;"
    "padding:10px 20px;font:600 14px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}"
    ".nb-backbar a{color:#5b8cff;text-decoration:none;}"
    ".nb-backbar a:hover{text-decoration:underline;}</style>\n"
)


def ensure_dirs() -> None:
    for d in ASSET_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[ok] asset directories ready under {DOCS / 'assets'}")


def publish_showcase() -> bool:
    src = RESULTS / "html" / "phase12_pytorch_showcase.html"
    dst = DOCS / "phase12_pytorch_showcase.html"
    if not src.exists():
        print(f"[skip] {src} not found; run the Phase 12 notebook first. "
              "docs/phase12_pytorch_showcase.html was not (re)generated.")
        return dst.exists()
    html = src.read_text(encoding="utf-8")
    if "nb-backbar" not in html:
        if "</head>" in html:
            html = html.replace("</head>", _BACKBAR_STYLE + "</head>", 1)
        if "<body>" in html:
            html = html.replace("<body>", "<body>\n" + _BACKBAR, 1)
        else:
            html = _BACKBAR + html
    dst.write_text(html, encoding="utf-8")
    print(f"[ok] published showcase -> {dst} (with back-navigation bar)")
    return True


def copy_figures() -> int:
    src_dir = RESULTS / "figures"
    dst_dir = DOCS / "assets" / "figures"
    copied = 0
    for name in FIGURES:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied += 1
        else:
            print(f"[skip] figure not found (graceful): {name}")
    print(f"[ok] copied {copied}/{len(FIGURES)} figures -> {dst_dir}")
    return copied


def copy_reports() -> int:
    src_dir = RESULTS / "reports"
    dst_dir = DOCS / "assets" / "reports"
    copied = 0
    for name in REPORTS:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied += 1
    print(f"[ok] copied {copied}/{len(REPORTS)} reports -> {dst_dir}")
    return copied


def ensure_nojekyll() -> None:
    # Disable Jekyll so asset folders are served verbatim.
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print("[ok] docs/.nojekyll present (Jekyll disabled)")


def validate_index() -> bool:
    index = DOCS / "index.html"
    if index.exists():
        print(f"[ok] {index} exists")
        return True
    print(f"[error] {index} is missing. Phase 13 landing page must be created.")
    return False


def print_pages_instructions() -> None:
    print("\n" + "=" * 70)
    print("GitHub Pages setup")
    print("=" * 70)
    print("Repository Settings -> Pages -> Build and deployment ->")
    print("  Source: Deploy from a branch")
    print("  Branch: main")
    print("  Folder: /docs")
    print("  Save")
    print(f"\nExpected URL: {PAGES_URL}")
    print("(If the repository name differs, check the GitHub Pages settings.)")


def main() -> int:
    print("Building NeuroBridge-S4 GitHub Pages site...\n")
    ensure_dirs()
    ensure_nojekyll()
    showcase_ok = publish_showcase()
    copy_figures()
    copy_reports()
    index_ok = validate_index()
    print_pages_instructions()
    if not index_ok:
        return 1
    if not showcase_ok:
        print("\n[warning] PyTorch showcase page not present yet; "
              "run the Phase 12 notebook then re-run this script.")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
