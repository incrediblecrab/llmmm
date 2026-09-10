# llmmm

llmmm trains and evaluates models that predict missing ingredients.
Our released model, [llmmm-recipes](https://huggingface.co/incrediblecrab/llmmm-recipes),
has its own weights and biases trained from scratch on canonical recipe
ingredient sets. No pretrained model was used to initialize it.

The code, training records and evaluations are here; the weights are on Hugging
Face. Raw recipe data is not redistributed. See the [source inventory](raw-data/README.md)
and [source-term notes](prior-study/docs/LICENSE_AUDIT.md) for provenance and
downstream obligations.

## Evaluated results

This section is generated from recorded artifacts. Run `make -C model status`
to inspect it and
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

**Evaluated training cohort: `train-v2-20260909` (2/2 runs scored).**

| Model | Seed | Native recall@10 | Lift over popularity | Training time |
|---|---:|---:|---:|---:|
| ease | 42 | 0.5872 | +0.2222 | 1.3s |
| masked-set | 42 | 0.6150 | +0.2500 | 332.7s |

`masked-set` settings: **3 epochs**, sampling cap **600,000 recipes**, maximum training recipe length **32**.

Sources: [corpus marker](model/data/GENERATION.json), [corpus accounting](model/results/corpus_stats.json), [benchmark runs](model/results/runs/all-v2/), [current experiment](model/experiments/train-v2-20260909.yaml).
<!-- CURRENT-RESULTS:END -->

Recall@10 measures whether a hidden ingredient appears in the top ten predictions.
It does not measure cooking quality, food safety or generalization to new sources.

The [full-partition run](model/experiments/train-v2-full-20260909.yaml) removed
the recipe-sampling cap while keeping the architecture, seed and epoch count
fixed. Its [evaluation](model/results/native_full_release.json) records the
paired comparison and export verification. The manifest counts recipes that
passed the length filter: **3,145,078 records**. Its weights remain available as
[v0.2.0-preview](https://huggingface.co/incrediblecrab/llmmm-recipes/tree/v0.2.0-preview),
with the original [upload record](model/results/huggingface_release.json).

## llmmm-recipes: all-record checkpoint

**Training completed on all 4,653,430 canonical recipe records.** The
[all-record declaration](model/experiments/production-v2-all-20260909.yaml)
used no sampling cap or length exclusion. Each of three epochs processed every
record exactly once: **13,960,290 recipe presentations** and **110,122,872
ingredient-slot presentations** in total. Records with one to 98 ingredients
were included intact. These are record counts, not a claim of unique recipe
content.

The [completion verification](model/results/all_record_training_validation.json)
checks the saved per-epoch counters against the checksum-verified corpus and
restores the complete predictor. The [export evidence](model/results/public_model_export.json)
records identical state tensors and exact logits on 192 synthetic reload
comparisons. This verifies the package, not its prediction quality.

The public model is
[llmmm-recipes, v0.3.1-public](https://huggingface.co/incrediblecrab/llmmm-recipes/tree/v0.3.1-public).
It is **public and ungated**: downloads and inference do not require a Hugging
Face account. The [public-release receipt](model/results/huggingface_public_release.json)
records remote file hashes, preserved version history, and fresh anonymous
download and inference without the private corpus. These are the same weights
as `v0.3.0-all-recipes`; this update changes access and documentation, not training.
No source recipe text was uploaded, and no permissive weights license is granted.

The old holdout is part of this training data, so the production checkpoint has
no held-out score from this corpus and does not enter the scored leaderboard.
The earlier evaluated checkpoint improved recall@10 from **61.5% to 65.5%** on
20,000 completion cases, as recorded in its [comparison](model/results/native_full_release.json).
Those scores do not apply to these weights. More training records alone do not
establish better predictions.

## Generation and evaluation work

The current checkpoint predicts ingredient names. Recipe generation is separate,
unfinished work.

The external [completion diagnostic](model/results/hf_completion_diagnostic.csv)
compares saved native predictors with pinned Epicure Cooc/Core weights and
RecipeBERT ingredient-name representations. It reports raw vectors, centered
vectors and native predictions separately, with
[paired intervals, subgroup results and provenance](model/results/hf_completion_diagnostic.json).

The comparison has limits: source overlap is known or possible, the row split
does not separate duplicate recipe families, and the historical popularity
control uses full-corpus frequencies. The RecipeBERT adapter embeds ingredient
names; it does not evaluate RecipeBERT's original masked-language task.
T5 is pinned but has not been compared.

The [original text audit](model/results/generation_data_audit.json) exposed a
Food.com quantity column being used in place of ingredient names. The
[rebuilt text-v2 audit](model/results/generation_data_audit_text_v2.json) confirms
that names are recovered. Separate quantities are retained without inventing
missing units; lists with different lengths are explicitly flagged and must
not be paired.

The new index is `recipe_text_v2.parquet`; the original is retained for comparison.
Every rebuilt row was checked against the canonical ingredient corpus. Failed
builds do not publish partial indexes, and existing indexes are not overwritten.
The canonical ingredient sets used by the predictors have not changed.

The next quality comparison needs new test recipes, with duplicate families and
source overlap accounted for. A future generator should be compared with T5
under equal output budgets, checking ingredient use, quantities, instructions
and blinded quality judgments.

```bash
cd model
make setup-hf
make hf-compare
make generation-audit  # requires the optional full-text index
make train-all-recipes
make verify-all-recipes
make export-all-recipes PRODUCTION_EXPORT=/private/unused/output-directory
```

Baseline downloads contain public model assets only; inference stays local.
`hf-compare` does not download T5 or call a hosted model. None of these commands
publishes a model or runs GitHub Actions.

For a later public release, export with a new `--tag` using
`scripts/export_native_model.py --production --public`, then explicitly run
`python scripts/publish_native_model.py --public --folder /private/export-directory --out results/new-release.json`.
The publisher admits only the six model-package files, removes stale evaluation
metadata, verifies remote bytes and fresh anonymous, corpus-free inference, and
preserves older tags. Publishing publicly requires the explicit `--public`
flag; without it, the publisher only accepts private repositories. Existing
version tags and publication receipts are not overwritten.

A separate `.venv-generation` environment contains MLX support so experiments
with the [pinned Qwen base](model/generation_base.lock.json) do not alter the
native PyTorch environment. The base is downloaded; culinary fine-tuning and
generation evaluation have not run.

## One source of truth

| Question | Authority |
|---|---|
| Which benchmark, evaluated training cohort and default embedding? | [`model/workspace.json`](model/workspace.json) |
| Which all-record production training and coverage? | [Declaration](model/experiments/production-v2-all-20260909.yaml) and [completion verification](model/results/all_record_training_validation.json) |
| Which corpus and normalizer? | [`model/data/GENERATION.json`](model/data/GENERATION.json), verified against the corpus SHA-256 before current training |
| What actually ran and how did it score? | Each run's `manifest.json` and `metrics.json` under [`model/results/runs/`](model/results/runs/) |
| Which private artifact bytes restore this workspace? | `model/artifacts.lock.json`, generated by `make snapshot` |
| Which normalization fixes were used? | The tracked code and [`model/data/aliases/`](model/data/aliases/) |
| Where did source corpora come from? | [`raw-data/README.md`](raw-data/README.md) and [`raw-data/MANIFEST.md`](raw-data/MANIFEST.md) |
| Which external model revisions and exact assets were compared? | [`model/hf_baselines.lock.json`](model/hf_baselines.lock.json) and the diagnostic's code/weight fingerprints |
| Which model version is on Hugging Face? | [`model/results/huggingface_public_release.json`](model/results/huggingface_public_release.json) |
| Which model-card revision is live? | [Documentation receipt](model/results/huggingface_model_card.json); text is rendered by [`export_native_model.py`](model/scripts/export_native_model.py) |

New training leaves the published benchmark unchanged. Older investigations
remain in [the model notes](model/README.md), [architecture notes](model/ARCHITECTURE.md)
and [prior study](prior-study/). Use the generated section above for the evaluated
cohorts selected in `workspace.json`.

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
Each run manifest records its sampling cap, length filter, epochs and other
resolved parameters.

**A git-only clone can inspect recorded results, but cannot train without the
private data bundle or an equivalent locally rebuilt corpus.** Supply an
authorized local bundle to `make restore`. Restoration verifies archive and file
checksums and refuses conflicting files.

Snapshot packing and restoration currently require macOS or Linux (POSIX
directory operations and hard links). The recovery command itself needs only
Python's standard library: from `model/`, it can also be run as
`python3 -m ingredient_model.recovery restore --bundle /private/path/archive.tar.gz`
before installing the scientific dependencies.

The core bundle deliberately excludes raw recipe downloads, the optional full-text
index, credentials, caches and virtual environments. `make check-core` checks
prepared training artifacts.
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
predictor for models that have one. The default explainer uses the embedding
named in `workspace.json`.

## Repository layout

```text
model/         training, evaluation, recovery tooling and recorded results
prior-study/   replication, source readers and normalization dependencies
raw-data/      tracked provenance; source bytes stay outside git
paper/         the original paper
```

`prior-study/tools/` remains a runtime dependency of corpus normalization.
Some one-off measurements in the legacy notes remain unsupported. Generated
results are checked against their declared source artifacts.

## Acknowledgements

This project started with a replication and audit of
[Epicure](https://arxiv.org/abs/2605.22391) by Jakub Radzikowski and Josef Chen.
Their ingredient-embedding work and published source inventory informed that
research. llmmm-recipes is a separately trained model with its own learned
weights and biases, not a fine-tune of Epicure or another pretrained model.

The method draws on [Transformer attention](https://arxiv.org/abs/1706.03762)
(Vaswani and coauthors), [masked prediction in BERT](https://aclanthology.org/N19-1423/)
(Devlin and coauthors), and work on
[attention over unordered sets](https://proceedings.mlr.press/v97/lee19d.html)
(Lee and coauthors). Our implementation uses a standard Transformer encoder
without positional encodings, rather than the Set Transformer reference architecture.

We credit the authors and curators of
[RecipeNLG](https://aclanthology.org/2020.inlg-1.4/) and the other datasets in our
[source inventory](raw-data/README.md).
[Epicure Cooc](https://huggingface.co/Kaikaku/epicure-cooc),
[Epicure Core](https://huggingface.co/Kaikaku/epicure-core) and
[RecipeBERT](https://huggingface.co/alexdseo/RecipeBERT) were comparison models;
their weights are not part of this checkpoint. The implementation uses
[PyTorch](https://pytorch.org/), and [Hugging Face Hub](https://huggingface.co/docs/hub)
hosts the release.

## Ideas for using this model

- Add ingredient autocomplete to a recipe editor. After a user enters at least
  two known ingredients, show ranked suggestions for them to accept or reject.
- Expand an ingredient query against a recipe catalog. Use the suggested names
  to find related entries; the catalog supplies the recipes and instructions.
- Use it in a learning experiment. Change the input ingredients and inspect how
  the rankings move, or compare it with popularity and co-occurrence baselines
  on genuinely new recipes.
