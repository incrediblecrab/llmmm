# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and GitHub Copilot CLI when working in this repository.

llmmm trains ingredient-completion and recipe-ranking models on a 4.6M-recipe corpus and publishes aggregate artifacts to Hugging Face: the model `incrediblecrab/llmmm-recipes`, the browser demo Space `llmmm-recipes-demo` and the dataset `llmmm-recipe-ingredients`. The repository is public, but the training corpus is not: source bytes stay outside git and come from a private recovery bundle. `README.md` is the current state; `model/README.md` and `model/ARCHITECTURE.md` keep historical design and investigations, and their example scores are not the current leaderboard.

## Commands

All work happens in `model/`, whose `.venv` links to the shared `~/.venvs/main`:

```bash
make -C model check-code     # tests that need no private data (pytest -m "not data") and the README results check
make -C model check          # gate, full tests, sanity, docs; needs the restored data bundle
make -C model status         # print the generated results section
model/.venv/bin/python -m pytest model/tests/<file>.py -k <name> -m "not data"
make -C model restore BUNDLE=<private bundle path> && make -C model verify
```

`model/.venv/bin/python -m ingredient_model.cli` is the `im` CLI (`gate`, `list`, `train`, `sweep`, `report`); `im gate` must pass before any result counts. Training, Hugging Face export and the cloud jobs in `model/cloud/` publish or spend money; run them only when the owner asks.

## Rules

The block between `<!-- CURRENT-RESULTS:START -->` and its end marker in `README.md` is generated: refresh it with `model/scripts/project_status.py --write`, and `make -C model docs` rejects hand edits or stale values. Results come only from recorded artifacts in `model/results/`, and releases are pinned by receipts there (for example `huggingface_recipe_link_release.json`). Evaluation splits are enforced in `model/ingredient_model/data/splits.py`: `recipe-holdout` is the honest split for every model, `edge-holdout` is for graph models only, and `check_leakage()` raises rather than warns; never pass `--allow-leakage` to produce a reported number. A model family is a folder under `model/models/` with a `@register` decorator, discovered by `ingredient_model/registry.py`, so there is no central list to edit. Publish only aggregate, authorized data: never commit or upload recipe text, instructions or images beyond what the release receipts already cover. The one exception is the encrypted off-machine backup of the corpus in the private dataset `incrediblecrab/llmmm-private-backup`, written only by `model/scripts/private_backup.py` (procedure in `model/README.md`, "Private backup"); it holds ciphertext only and must never be made public. Keep `raw-data/` to provenance (`raw-data/README.md`, `MANIFEST.md`). `prior-study/` is the original Epicure replication and `paper/` the paper copy.
