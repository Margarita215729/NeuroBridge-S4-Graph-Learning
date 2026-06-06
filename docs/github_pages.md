# Publishing the GitHub Pages showcase

The public showcase is served by GitHub Pages directly from the `/docs` folder. No build system or
backend is required — the site is plain static HTML/CSS with self-contained assets.

## Enable GitHub Pages

In the repository on GitHub:

```
Repository Settings → Pages → Build and deployment →
  Source: Deploy from a branch
  Branch: main
  Folder: /docs
  Save
```

## Expected URL

GitHub Pages publishes at:

```
https://<username>.github.io/<repository-name>/
```

For this repository the expected URL is:

```
https://Margarita215729.github.io/NeuroBridge-S4-Graph-Learning/
```

The PyTorch showcase is then available at:

```
https://Margarita215729.github.io/NeuroBridge-S4-Graph-Learning/phase12_pytorch_showcase.html
```

If your repository name differs, check the actual URL shown in the GitHub Pages settings panel after
saving.

## Rebuilding the site

The published pages are generated/refreshed by the build script:

```bash
python scripts/build_pages_site.py
python scripts/validate_pages_site.py
```

`build_pages_site.py`:
- creates the `docs/assets/*` directories;
- publishes `results/html/phase12_pytorch_showcase.html` into `docs/phase12_pytorch_showcase.html`
  with a back-navigation bar;
- copies selected figures into `docs/assets/figures/`;
- ensures `docs/.nojekyll` exists so asset folders are served verbatim;
- prints these setup instructions.

`validate_pages_site.py` runs offline checks: required files exist, key phrases and guardrails are
present, asset directories exist, and no absolute local paths leak into the public HTML.

## Notes

- `docs/.nojekyll` disables Jekyll processing so the static assets are served exactly as written.
- The Streamlit dashboard is **not** hosted on GitHub Pages; it must be run locally
  (`streamlit run app.py`). The Pages site is a static review artifact.
- The PyTorch showcase page is self-contained (figures embedded as base64), so it works when opened
  directly from GitHub Pages or locally.
