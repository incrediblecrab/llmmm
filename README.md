# llmmm

An ingredient-prediction research workspace developing toward a publishable
culinary model: compare models, inspect their evidence, and reproduce the
results behind an audit of
[Epicure](https://arxiv.org/abs/2605.22391).
The repository is the project. There is no separate website.

**Code and result records are public. Datasets and trained weights are not
redistributed here.** Some upstream data has noncommercial or unresolved terms;
see the [licensing audit](prior-study/docs/LICENSE_AUDIT.md). A local recovery
bundle is not permission to publish its contents or use them commercially.

## Current results

The section below is generated from named artifacts, not copied from terminal
output. Run `make -C model status` to inspect it and
`model/.venv/bin/python model/scripts/project_status.py --write` to refresh it.
`make -C model docs` rejects edits or stale values in this section.

<!-- CURRENT-RESULTS:START -->
**Corpus v2: 4,653,430 recipes, 36,707,624 ingredient slots, 1,790 ingredients, 29 source groups.**

Published benchmark: `all-v2`, **16 recorded runs**, `recipe-holdout`, **20,000 completion instances**. Popularity baseline: **0.3650**.

All score columns are recall@10. Native is the full predictor where available; a dash means no separate native scorer. Rows retain the published ordering: native where available, otherwise raw vectors.

| Model | Raw vectors | Centered vectors | Native predictor |
|---|---:|---:|---:|
| masked-set | 0.1229 | 0.2132 | 0.6178 |
| ease | 0.1422 | 0.1645 | 0.5872 |
| svd-ppmi | 0.4505 | 0.4493 | - |
| concat | 0.4446 | 0.4446 | - |
| residual | 0.4235 | 0.4235 | - |
| sgns-core | 0.4065 | 0.3970 | - |
| sgns-cooc | 0.3885 | 0.3871 | - |
| text-aligned | 0.3216 | 0.3038 | - |
| glove | 0.2211 | 0.4928 | - |
| lightgcn | 0.1837 | 0.2060 | - |
| text-embed | 0.1077 | 0.1123 | - |
| sgc | 0.0777 | 0.0939 | - |
| ials | 0.0426 | 0.0558 | - |
| chem-svd | 0.0152 | 0.0088 | - |
| sgns-chem | 0.0149 | 0.0201 | - |
| item2vec | 0.0123 | 0.1483 | - |

**Current training: `train-v2-20260909` (2/2 runs scored).**

| Model | Seed | Native recall@10 | Lift over popularity | Training time |
|---|---:|---:|---:|---:|
| ease | 42 | 0.5872 | +0.2222 | 1.3s |
| masked-set | 42 | 0.6150 | +0.2500 | 332.7s |

`masked-set` settings: **3 epochs**, sampling cap **600,000 recipes**, maximum training recipe length **32**.

Sources: [corpus marker](model/data/GENERATION.json), [corpus accounting](model/results/corpus_stats.json), [benchmark runs](model/results/runs/all-v2/), [current experiment](model/experiments/train-v2-20260909.yaml).
<!-- CURRENT-RESULTS:END -->

These scores measure whether a hidden ingredient is recovered in the top ten
guesses. They do not establish taste, safe substitutions, useful quantities,
or correct cooking instructions. Training-seed comparisons also do not measure
generalization to a new cuisine or a different recipe source.

The [full-partition run](model/experiments/train-v2-full-20260909.yaml) removed
the recipe-sampling cap while keeping the architecture, seed and epoch count
fixed. Its [evaluation](model/results/native_full_release.json) records the
paired comparison and export verification. The manifest counts recipes that
passed the length filter and examples processed during training.

The predictor is available as a
[private Hugging Face preview](https://huggingface.co/incrediblecrab/llmmm-recipes),
tagged `v0.2.0-preview`. The [upload record](model/results/huggingface_release.json)
contains the commit and file hashes. Public-release permissions remain unresolved.

An [all-record production fit](model/experiments/production-v2-all-20260909.yaml)
is separate from that evaluated preview. `make -C model train-all-recipes` uses
all **4,653,430** canonical recipe records, with no sampling or length filter.
Each epoch checks that every row and every ingredient slot was processed.
The old holdout is part of this training data, so the production checkpoint has
no held-out score from this corpus and does not enter the scored leaderboard.
After training finishes, `make -C model verify-all-recipes` checks the saved
per-epoch counts against the checksum-verified corpus and restores the complete
predictor. An incomplete run fails this check; having the data is not enough.

## Toward a Hugging Face release

The target includes ingredient reasoning and recipe generation. **The current
checkpoint only predicts ingredients; it is not yet a recipe-writing model.**
For technical users, a useful release needs complete predictors, matched
evaluations, identifiable training data, and clear usage rights.

The first external comparison is a local
[completion diagnostic](model/results/hf_completion_diagnostic.csv), with
[paired intervals, subgroup results and provenance](model/results/hf_completion_diagnostic.json).
It restores the current native predictors and scores the exact pinned Epicure
Cooc/Core weights plus RecipeBERT ingredient-name representations. Both raw and
centered vector results remain visible. A native predictor's score must not be
attributed to its exported embeddings.

This is **not a clean public superiority benchmark**. The pretrained baselines
have known or possible source overlap; our split excludes recipe rows, not
duplicate recipe families. The historical popularity control also uses
full-corpus frequencies. RecipeBERT's name-vector adapter is not its original
masked-language task or a contextual recipe encoder. T5 is pinned but has not
been compared: there is no llmmm generator yet.

The [original text audit](model/results/generation_data_audit.json) exposed a
Food.com quantity column being used in place of ingredient names. The
[rebuilt text-v2 audit](model/results/generation_data_audit_text_v2.json) confirms
that names are recovered. Separate quantities are retained without inventing
missing units; lists with different lengths are explicitly flagged and must
not be paired. These are coverage and extraction measurements, not certificates
of training readiness.

The new index is `recipe_text_v2.parquet`; the original is retained for comparison.
Every rebuilt row was checked against the canonical ingredient corpus. Failed
builds do not publish partial indexes, and existing indexes are not overwritten.
The canonical ingredient sets used by the predictors have not changed.

Before release, freeze a duplicate-aware, source-aware test set before further
model selection. Compare generation with T5 under equal output/token budgets,
separately reporting ingredient faithfulness, quantity/instruction consistency,
and blinded quality review. Completion recall alone cannot establish these.
Resolve applicable upstream terms before selecting a release license or
distributing weights publicly.

```bash
cd model
make setup-hf
make hf-compare
make generation-audit  # requires the optional full-text index
make train-full-native
make export-native HF_EXPORT=/private/unused/output-directory
```

Baseline downloads contain public model assets only; inference stays local.
`hf-compare` does not download T5 or call a hosted model. The audit reports
problems but is not an automatic training-readiness gate. None of these
commands publishes a model or runs GitHub Actions.

A separate `.venv-generation` environment contains MLX support so experiments
with the [pinned Qwen base](model/generation_base.lock.json) do not alter the
native PyTorch environment. Downloading a base model is not a completed culinary
fine-tune; no generator-training result or T5 generation win is asserted here.

## One source of truth

| Question | Authority |
|---|---|
| Which benchmark, training experiment and default embedding? | [`model/workspace.json`](model/workspace.json) |
| Which corpus and normalizer? | [`model/data/GENERATION.json`](model/data/GENERATION.json), verified against the corpus SHA-256 before current training |
| What actually ran and how did it score? | Each run's `manifest.json` and `metrics.json` under [`model/results/runs/`](model/results/runs/) |
| Which private artifact bytes restore this workspace? | `model/artifacts.lock.json`, generated by `make snapshot` |
| Which normalization fixes were used? | The tracked code and [`model/data/aliases/`](model/data/aliases/) |
| Where did source corpora come from? | [`raw-data/README.md`](raw-data/README.md) and [`raw-data/MANIFEST.md`](raw-data/MANIFEST.md) |
| Which external model revisions and exact assets were compared? | [`model/hf_baselines.lock.json`](model/hf_baselines.lock.json) and the diagnostic's code/weight fingerprints |
| Which model version is on Hugging Face? | [`model/results/huggingface_release.json`](model/results/huggingface_release.json) |

The published benchmark is not overwritten by new training. Older investigations
remain in [the model notes](model/README.md), [architecture notes](model/ARCHITECTURE.md)
and [prior study](prior-study/). Their historical measurements are not the current
leaderboard; the generated section above is.

## Restore and train

From a fresh checkout:

```bash
cd model
make setup-train
make check-code
make restore BUNDLE=/private/path/workspace-v2-20260909.tar.gz
make verify
make check-core
make train
make status
```

`make setup-train` installs the package, tests, and local PyTorch support.
`make check-code` runs locally and needs no private data.
`make train` reads the experiment named by `workspace.json`, verifies its corpus
generation and checksum, and resumes only that experiment's unfinished runs.
EASE is the inexpensive control; masked-set is the conditional model being trained.
Its sampling cap, length filter, epochs and other resolved parameters are recorded
in its run manifest. The corpus headline is not a claim that every model trains
on every recipe.

**A git-only clone can inspect recorded results, but cannot train without the
private data bundle or an equivalent locally rebuilt corpus.** There is no public
dataset download hidden in `make restore`: supply an authorized local bundle.
Restoration verifies archive and file checksums and refuses conflicting files.

Snapshot packing and restoration currently require macOS or Linux (POSIX
directory operations and hard links). The recovery command itself needs only
Python's standard library: from `model/`, it can also be run as
`python3 -m ingredient_model.recovery restore --bundle /private/path/archive.tar.gz`
before installing the scientific dependencies.

The core bundle deliberately excludes raw recipe downloads, the optional full-text
index, credentials, caches and virtual environments. `make check-core` checks
prepared training artifacts without claiming a raw-source audit.
`make check` additionally replays source records and requires the original
`raw-data/` files. Browsing full recipe instructions requires rebuilding the text
index with `make text` from those sources.

Existing artifact owners can create a recovery bundle:

```bash
cd model
make snapshot BUNDLE=../.artifacts/workspace-v2-20260909.tar.gz
```

Keep that archive in private storage. Commit its checksum lock, not the archive.
The input allowlist is in `workspace.json`; the command does not archive the whole
working directory.

## Inspect the models

```bash
cd model
.venv/bin/im list
.venv/bin/im report
.venv/bin/im explain tomato basil
.venv/bin/im neighbors all-v2/svd-ppmi-recipe-holdout-s0 tomato -k 10
.venv/bin/im eval train-v2-20260909/masked-set-recipe-holdout-s42
```

Reports can read tracked metadata without model weights. Prediction and
re-evaluation require the corresponding weights. Re-evaluation restores the full
predictor for models that have one instead of silently dropping its scores.
The default explainer uses the embedding explicitly named in `workspace.json`,
not whichever run sorts first.

## Repository layout

```text
model/         training, evaluation, recovery tooling and recorded results
prior-study/   replication, source readers and normalization dependencies
raw-data/      tracked provenance; source bytes stay outside git
paper/         the original paper
```

`prior-study/tools/` remains a runtime dependency of corpus normalization.
The public repository does not claim that every historical analysis is fully
reproducible: the legacy documentation scanner still records unsupported
one-off measurements. Current generated results are checked against their
specific source artifacts rather than against unrelated matching numbers.
