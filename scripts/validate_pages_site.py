#!/usr/bin/env python3
"""Phase 13 — Validate the GitHub Pages static showcase site.

Runs offline structural checks on the published ``docs/`` site and returns a
non-zero exit code if any required check fails.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

INDEX_PHRASES = [
    "NeuroBridge-S4",
    "Operational Resilience Graph Learning for Small-N Human Spaceflight Research",
    "phase12_pytorch_showcase.html",
    "Not diagnosis",
    "not mission readiness classification",
]

SHOWCASE_PHRASES = [
    "NeuroBridge-S4 PyTorch Temporal Graph Learning Showcase",
    "Experimental ML showcase only",
]

ASSET_DIRS = [
    DOCS / "assets" / "css",
    DOCS / "assets" / "figures",
    DOCS / "assets" / "reports",
    DOCS / "assets" / "img",
    DOCS / "assets" / "screenshots",
    DOCS / "assets" / "data",
]

# Absolute local paths that must never leak into public files.
ABS_PATH_RE = re.compile(r"(/Users/|/home/|[A-Za-z]:\\\\|file:///)")


def _check(name: str, ok: bool, results: list) -> None:
    results.append((name, ok))
    print(f"[{'ok' if ok else 'FAIL'}] {name}")


def main() -> int:
    results: list[tuple[str, bool]] = []

    index = DOCS / "index.html"
    showcase = DOCS / "phase12_pytorch_showcase.html"

    _check("docs/index.html exists", index.exists(), results)
    _check("docs/phase12_pytorch_showcase.html exists", showcase.exists(), results)

    # Collapse whitespace so phrases split across wrapped HTML lines still match.
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text)

    index_text = _norm(index.read_text(encoding="utf-8")) if index.exists() else ""
    for phrase in INDEX_PHRASES:
        _check(f"index contains: {phrase!r}", phrase in index_text, results)

    showcase_text = _norm(showcase.read_text(encoding="utf-8")) if showcase.exists() else ""
    for phrase in SHOWCASE_PHRASES:
        _check(f"showcase contains: {phrase!r}", phrase in showcase_text, results)

    for d in ASSET_DIRS:
        _check(f"asset dir exists: {d.relative_to(REPO)}", d.is_dir(), results)

    # No absolute local paths in the public HTML files.
    for f in (index, showcase):
        if f.exists():
            leaked = ABS_PATH_RE.search(f.read_text(encoding="utf-8"))
            _check(f"no absolute local paths in {f.name}", leaked is None, results)

    css = DOCS / "assets" / "css" / "site.css"
    _check("docs/assets/css/site.css exists", css.exists(), results)

    nojekyll = DOCS / ".nojekyll"
    _check("docs/.nojekyll exists", nojekyll.exists(), results)

    failed = [name for name, ok in results if not ok]
    print("\n" + "=" * 60)
    if failed:
        print(f"VALIDATION FAILED: {len(failed)} check(s) failed.")
        for name in failed:
            print(f"  - {name}")
        return 1
    print(f"VALIDATION PASSED: {len(results)} checks ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
