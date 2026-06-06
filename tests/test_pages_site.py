"""Phase 13 — Offline tests for the public GitHub Pages showcase site.

These tests do not require internet access. They check that the published static
site and the portfolio documents exist, contain the expected content and
guardrails, and do not leak absolute local paths.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

ABS_PATH_RE = re.compile(r"(/Users/|/home/|[A-Za-z]:\\|file:///)")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_index_exists():
    assert (DOCS / "index.html").exists()


def test_index_contains_project_title():
    text = _norm((DOCS / "index.html").read_text(encoding="utf-8"))
    assert "NeuroBridge-S4" in text
    assert ("Operational Resilience Graph Learning for Small-N Human Spaceflight Research"
            in text)


def test_index_contains_pytorch_showcase_link():
    text = (DOCS / "index.html").read_text(encoding="utf-8")
    assert "phase12_pytorch_showcase.html" in text


def test_index_contains_guardrail_text():
    text = _norm((DOCS / "index.html").read_text(encoding="utf-8"))
    assert "Not diagnosis" in text
    assert "not mission readiness classification" in text
    assert "not exposure measurement" in text


def test_index_contains_hero_buttons():
    text = _norm((DOCS / "index.html").read_text(encoding="utf-8"))
    for label in ("View PyTorch Showcase", "View Architecture", "View Methodology",
                  "View GitHub Repository"):
        assert label in text


def test_showcase_published_or_buildable():
    showcase = DOCS / "phase12_pytorch_showcase.html"
    results_html = REPO / "results" / "html" / "phase12_pytorch_showcase.html"
    if showcase.exists():
        text = _norm(showcase.read_text(encoding="utf-8"))
        assert "NeuroBridge-S4 PyTorch Temporal Graph Learning Showcase" in text
        assert "Experimental ML showcase only" in text
    else:
        # If not yet published, the build script must be able to create it from results.
        assert (REPO / "scripts" / "build_pages_site.py").exists()
        assert results_html.exists(), (
            "Neither docs/phase12_pytorch_showcase.html nor the results showcase exists.")


def test_project_overview_exists_and_has_one_line_summary():
    p = REPO / "PROJECT_OVERVIEW.md"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "## One-line summary" in text
    assert "within-subject biological" in text


def test_quickstart_has_pages_and_local_instructions():
    p = REPO / "QUICKSTART.md"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "github.io" in text
    assert "pip install -r requirements.txt" in text
    assert "streamlit run app.py" in text


def test_release_notes_exist():
    p = REPO / "RELEASE_NOTES_v1.md"
    assert p.exists()
    assert "Portfolio Release" in p.read_text(encoding="utf-8")


def test_architecture_doc_exists():
    p = DOCS / "architecture.md"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "PyTorch temporal graph autoencoder" in text


def test_site_css_and_assets_exist():
    assert (DOCS / "assets" / "css" / "site.css").exists()
    for sub in ("figures", "img", "screenshots", "reports", "data"):
        assert (DOCS / "assets" / sub).is_dir()


def test_no_absolute_local_paths_in_public_html():
    for name in ("index.html", "phase12_pytorch_showcase.html"):
        f = DOCS / name
        if f.exists():
            assert ABS_PATH_RE.search(f.read_text(encoding="utf-8")) is None, (
                f"absolute local path leaked into {name}")


def test_nojekyll_present():
    assert (DOCS / ".nojekyll").exists()
