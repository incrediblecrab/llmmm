# Results

Recorded measurements, verifications and release receipts that the documentation cites.

- `huggingface_*.json` are Hugging Face release, update and deletion receipts. Each pins the repository revisions, tags and file hashes it produced or checked. Receipts are historical records: they keep the repository names and URLs they recorded after a release is superseded or a repository is deleted. `huggingface_recipe_link_release.json` records the current public dataset and demo.
- The other top-level `*.json` and `*.csv` files are measurement, training, evaluation, export and verification records, plus `ingredient_catalog_publication_scope.json`, the record of the owner's publication decisions for the ingredient dataset.
- `runs/` holds a `metrics.json` and a `manifest.json` for each training or evaluation run, plus a few sweep journals and logs.

The root README's [source-of-truth table](../../README.md#one-source-of-truth) names the file that answers each question. `make -C model docs` runs `scripts/project_status.py --check` and `scripts/check_docs.py`; the latter compares metric-shaped numbers in `README.md`, `model/README.md` and `model/ARCHITECTURE.md` with the artefacts it registers here.
