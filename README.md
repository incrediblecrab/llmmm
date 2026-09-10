# llmmm

llmmm trains ingredient predictors and recipe-ranking models.
Our released model, [llmmm-recipes](https://huggingface.co/incrediblecrab/llmmm-recipes),
has its own weights and biases trained from scratch on canonical recipe
ingredient sets. No pretrained model was used to initialize it.

The code, training records and evaluations are here; the weights are on Hugging
Face. The original training corpus is not redistributed. See the [source inventory](raw-data/README.md)
and [source-term notes](prior-study/docs/LICENSE_AUDIT.md) for provenance and
downstream obligations.

## Public recipe demo

**[Try the demo](https://huggingface.co/spaces/incrediblecrab/llmmm-recipes-demo)**
| **[Download the public sample](https://huggingface.co/datasets/incrediblecrab/llmmm-recipe-sample)**

The demo runs the published supervised ranker in your browser on a
[small, openly licensed catalog](model/demo_data/README.md). It needs no private
recipe database, login or API key. Choose pantry ingredients,
source-time and missing-item limits, required/excluded ingredients and reported
servings; switch to the simple baseline to compare the ordering.
Hosting uses a free Static Space, not paid inference hardware. Pantry inputs
are not sent to a server or stored between visits.

The sample contains **12 Wikibooks recipes under CC BY-SA 4.0**. Nine have a
reported overall time and six have reported servings. Four contain ingredient
lines that are not fully mapped; those gaps and source inconsistencies remain
visible. Original measurements, instructions, revision links and attribution are
retained. No source times or missing units are invented.

This is a separately sourced demonstration catalog, not the full training
dataset or a held-out quality benchmark. Its ingredient frequencies describe
only the sample, so the full-catalog recovery scores below do not apply to it.
The [build and publication instructions](model/demo/README.md) keep the code and
data provenance in this repository. The [release receipt](model/results/huggingface_recipe_demo_release.json)
pins the public dataset, Space, model and source revisions, and records the
documentation-only model-card update.

### Full ingredient-only search: local preview

A separate browser index now covers **all 4,653,430 canonical records**, not
just the public sample. It contains normalized ingredient names, source-reported
times and servings, source identifiers and original links. It contains no copied
titles, quantities, instructions, descriptions or images.

The [verification record](model/results/ingredient_catalog_verification.json)
compares every record and all **36,707,624 ingredient slots** with the canonical
corpus. Twenty Python/browser searches agree on feasibility, shortlist size and
ordered results. This is implementation evidence, not a recipe-quality score.
The initial compressed arrays total **36.0 MiB**; original-link files load only
when needed. The same published weights do the ranking locally.

**2,292,411 records have recorded source links; 2,361,019 do not.** The preview
defaults to records with links, so a result can lead to quantities and directions
on its original site. Uncheck that filter to search unlinked ingredient sets too.
Every record is checked against constraints; at most 2,000 feasible records,
selected by the baseline, receive learned scores. This is not a guaranteed
global learned top-k or a collection of unique, complete cooking recipes.

The owner has confirmed permission to publish this full ingredient-only dataset;
the [scope record](model/results/ingredient_catalog_publication_scope.json)
records that confirmation. The public release is being prepared. Until its
receipt is recorded here, the public Space above still serves the twelve
licensed sample recipes.
[Local build instructions](model/demo/README.md#full-ingredient-only-index)
reproduce the complete index without running a server-side model or uploading
the private corpus.

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

The current public release is
[llmmm-recipes, v0.4.0-recipe-search](https://huggingface.co/incrediblecrab/llmmm-recipes/tree/v0.4.0-recipe-search).
It is **public and ungated**. The ingredient predictor's weights remain unchanged
from `v0.3.0-all-recipes`; this release adds separate recipe-ranking policies and
the finder described below. The [release receipt](model/results/huggingface_recipe_search_release.json)
records preserved native artifact hashes and older tags, anonymous installation
of the pinned SDK source, and isolated inference before and after upload.
The [earlier access release](model/results/huggingface_public_release.json)
records when the ingredient checkpoint became public. No training-corpus recipe
text was uploaded to the model repository, and no permissive weights license is granted.

The old holdout is part of this training data, so the production checkpoint has
no held-out score from this corpus and does not enter the scored leaderboard.
The earlier evaluated checkpoint improved recall@10 from **61.5% to 65.5%** on
20,000 completion cases, as recorded in its [comparison](model/results/native_full_release.json).
Those scores do not apply to these weights. More training records alone do not
establish better predictions.

## Finding recipes

Give the finder ingredients and a time limit. It retrieves matching records from
a local recipe catalog, with source links, source-reported times, matched and
missing canonical ingredients, and the original instructions. Required and
excluded ingredients, missing-item limits, reported servings and source language
are enforced before ranking. A learned score cannot relax those constraints.

The private catalog contains **4,653,430 records**. Only **681,275** have known
source total times; unknown times do not pass a time limit. Prep-plus-cook sums
are kept separately and never substituted. The [catalog report](model/results/recipe_catalog_build.json)
records coverage and the distinction between stored text and non-whitespace
instructions. Nonempty fields are not a cooking-quality certificate.

Search requires the authorized local catalog, its derived metadata index, and
the canonical `recipe_ids.npz` ingredient corpus. The core recovery bundle
supplies the ingredient corpus; the catalog and metadata index must be rebuilt
from authorized sources. None of these data files is included in the public
model. The finder checks their hashes and filters numeric metadata and ingredient
sets before fetching recipe text. Retrieval remains bounded and reports when a
shortlist or scan limit is reached.

With the canonical corpus and authorized raw sources restored, `make -C model text`
builds the immutable text-v2 index and `make -C model recipe-catalog` builds
the catalog and metadata cache. For a catalog already restored without its cache,
use `make -C model recipe-metadata` instead. These commands refuse to overwrite
existing outputs. Reported search measurements are bound to the evaluated catalog;
changed source metadata requires a new evaluation.

```bash
cd model
.venv/bin/im find \
  --ingredients chicken rice broccoli \
  --must-use chicken --max-total-minutes 30 --max-missing 2 \
  --catalog data/recipes/recipe_search.sqlite \
  --corpus data/recipes/recipe_ids.npz \
  --revision v0.4.0-recipe-search
```

Ingredient matching uses the canonical vocabulary, not every component of a
source ingredient. Review the raw ingredient list and warnings before cooking.
Exclusions are not an allergen-safety assessment. Missing quantity units are not
invented, inconsistent name/quantity arrays are not paired, and serving counts do
not scale quantities or cooking time. Ranking scores are not probabilities.

### Ranking training

The ranker has **705 parameters**, initialized from scratch. Supervised listwise
training was followed by sampled-action REINFORCE with an entropy term and a KL
penalty to the supervised policy. Each stage processed every canonical record
once: **9,306,860 training queries**, **72,710 optimizer steps**, and
**18,613,720 sampled reinforcement actions**. The [verification record](model/results/recipe_ranker_training.json)
checks the saved per-row counts against the checksum-verified corpus.

The reward is recovery of a source recipe's canonical ingredient set, not human
preference. Pantry hashes separate training, validation and test queries; recipes
and duplicate families are not held out. Sampled candidate lists deliberately
include the feasible source recipe.

| Ranker | Source-set recovery at rank 1 |
|---|---:|
| Heuristic | 95.62% |
| Supervised | 97.23% |
| REINFORCE | 97.34% |

These are **2,852 nontrivial sampled test shortlists**, not full-catalog retrieval
results. Validation selected the supervised policy. REINFORCE did not establish
an additional validation gain, so its slightly higher test point estimate is not
a reason to switch policies.

### Live search evaluation

The [live evaluation](model/results/recipe_search_live.json) used 200 validation
and 200 test pantry queries, with no source recipe inserted into retrieval.
Half included a source-total-time limit. Supervised ranking was selected on
validation before any test scoring.

| Ranker | Test source-set recovery in the top five |
|---|---:|
| Heuristic | 75.5% |
| Supervised | 83.5% |
| REINFORCE | 84.0% |

The supervised improvement over the heuristic was **8.0 percentage points**,
with a paired query-bootstrap 95% interval of **4.5 to 12.0 points**.
Its test requests took **0.43 seconds median** and **2.05 seconds at p95**,
excluding initialization. It had no timeouts, empty results or constraint
violations in either partition. REINFORCE did not establish an extra validation
gain and had one validation timeout; the cause was not isolated.

The source set reached the test shortlist on **99%** of queries, but **162 of 200**
test queries reached a retrieval budget. These are synthetic queries over known
recipes, not evidence of taste, food safety or unseen-recipe generalization.

The supervised policy is the public default. Both trained policies are included
in the [Hugging Face release](https://huggingface.co/incrediblecrab/llmmm-recipes/tree/v0.4.0-recipe-search),
with [checkpoint-bound export evidence](model/results/recipe_search_export.json).
The existing ingredient checkpoint is unchanged. To reproduce ranking training
and verify its coverage:

```bash
make setup-hf
make train-recipe-policy RECIPE_RUN=training/new-recipe-ranking-run
make verify-recipe-policy RECIPE_RUN=training/new-recipe-ranking-run \
  RECIPE_TRAINING_REPORT=results/new-recipe-ranking-run.json
```

Training outputs and verification records are immutable. Choose fresh output
paths for another run; no GitHub Actions or hosted training services are used.

## Generation and evaluation work

The ingredient checkpoint predicts names; the finder retrieves existing recipes.
Writing new recipes is separate, unfinished work.

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

The native-only exporter and publisher admit six model-package files and refuse
unexpected remote additions. Recipe-search releases use
`scripts/export_recipe_search.py` and `scripts/publish_recipe_search.py` instead;
they preserve the existing native artifacts while adding only ranking weights,
configuration and aggregate evidence.

For a recipe-search release, complete `make evaluate-recipe-search`, commit and
push the tested source, then run `make export-recipe-search`. Publication is a
separate, explicit command:
`python scripts/publish_recipe_search.py --public --folder /private/export-directory`.
The publisher checks anonymously installable pinned source, isolated inference,
remote hashes and preserved tags. Catalogs, raw text, private query cases and
coverage arrays are excluded. Existing outputs, version tags and publication
receipts are not overwritten.

A separate `.venv-generation` environment contains MLX support so experiments
with the [pinned Qwen base](model/generation_base.lock.json) do not alter the
native PyTorch environment. The base is downloaded; culinary fine-tuning and
generation evaluation have not run.

## One source of truth

| Question | Authority |
|---|---|
| Which benchmark, evaluated training cohort and default embedding? | [`model/workspace.json`](model/workspace.json) |
| Which all-record production training and coverage? | [Declaration](model/experiments/production-v2-all-20260909.yaml) and [completion verification](model/results/all_record_training_validation.json) |
| Which recipe-ranking training, coverage and sampled evaluation? | [Verified training record](model/results/recipe_ranker_training.json) |
| Which source metadata supports recipe-search constraints? | [Catalog coverage and provenance](model/results/recipe_catalog_build.json) |
| Which public demo code, source recipes and redistribution terms? | [Browser implementation](model/demo/), [public sample card](model/demo_data/README.md) and [source revisions/mappings](model/demo_data/sources.json) |
| Which corpus and normalizer? | [`model/data/GENERATION.json`](model/data/GENERATION.json), verified against the corpus SHA-256 before current training |
| What actually ran and how did it score? | Each run's `manifest.json` and `metrics.json` under [`model/results/runs/`](model/results/runs/) |
| Which private artifact bytes restore this workspace? | `model/artifacts.lock.json`, generated by `make snapshot` |
| Which normalization fixes were used? | The tracked code and [`model/data/aliases/`](model/data/aliases/) |
| Where did source corpora come from? | [`raw-data/README.md`](raw-data/README.md) and [`raw-data/MANIFEST.md`](raw-data/MANIFEST.md) |
| Which external model revisions and exact assets were compared? | [`model/hf_baselines.lock.json`](model/hf_baselines.lock.json) and the diagnostic's code/weight fingerprints |
| Which model version and model card are on Hugging Face? | [Model release receipt](model/results/huggingface_recipe_search_release.json) and [latest demo/card update](model/results/huggingface_recipe_demo_release.json); the card is rendered by [`export_recipe_search.py`](model/scripts/export_recipe_search.py) |
| Which public demo and sample versions are deployed? | [Demo release receipt](model/results/huggingface_recipe_demo_release.json), including pinned revisions, asset hashes and anonymous browser results |
| Which bytes, data identities and model-selection results were exported? | [Recipe-search export evidence](model/results/recipe_search_export.json) |
| Which ingredient-only access and documentation updates preceded this release? | [Public-access receipt](model/results/huggingface_public_release.json) and [earlier card receipt](model/results/huggingface_model_card.json) |

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

The ranking experiment uses [REINFORCE](https://doi.org/10.1007/BF00992696)
(Ronald J. Williams), with a supervised warm-start and KL regularization.
[SQLite FTS5](https://sqlite.org/fts5.html) supplies ingredient-token retrieval.
These are method and implementation credits, not imported learned weights.

We credit the authors and curators of
[RecipeNLG](https://aclanthology.org/2020.inlg-1.4/) and the other datasets in our
[source inventory](raw-data/README.md).
[Epicure Cooc](https://huggingface.co/Kaikaku/epicure-cooc),
[Epicure Core](https://huggingface.co/Kaikaku/epicure-core) and
[RecipeBERT](https://huggingface.co/alexdseo/RecipeBERT) were comparison models;
their weights are not part of this checkpoint. The implementation uses
[PyTorch](https://pytorch.org/), and [Hugging Face Hub](https://huggingface.co/docs/hub)
hosts the release.

The public demo sample credits [Wikibooks Cookbook contributors](https://en.wikibooks.org/wiki/Cookbook:Table_of_Contents).
Its [attribution and license record](model/demo_data/README.md) includes the
additional Wikipedia credit for Oat Porridge. That recipe-text license does not
relicense the model weights or project code.

## Ideas for using this model

- Add ingredient autocomplete to a recipe editor. After a user enters at least
  two known ingredients, show ranked suggestions for them to accept or reject.
- Search an authorized recipe collection using pantry ingredients and a time
  limit. Show source links, missing canonical ingredients and data-quality warnings.
- Compare heuristic, supervised and reinforcement-trained ranking on held-out
  pantry queries. Keep the simpler policy when an improvement is not established.
- Use it in a learning experiment. Change the input ingredients and inspect how
  the rankings move, or compare it with popularity and co-occurrence baselines
  on genuinely new recipes.
