# ingredients

Everything behind *Epicure* and what came after it: the paper, the recipe corpora it was built from, the prior study that audited it, and a workspace for retraining ingredient embeddings under an evaluation protocol that does not flatter them.

Private repository, for local and educational use.

**10 GB on disk.** `raw-data/` is 7.4 GB, `model/` is 2.6 GB, `prior-study/` is 221 MB. Most of it is regenerable rather than authored.

```
paper/         epicure.pdf — the published paper
raw-data/      29 source corpora + _duplicates/ + _superseded/
model/         the ingredient-model workspace, and the `im` CLI
prior-study/   the replication and audit that preceded model/
use-cases/     empty
```

## The three parts, and how they connect

**`paper/`** holds *Epicure: Navigating the Emergent Geometry of Food Ingredient Embeddings* — Jakub Radzikowski and Josef Chen, KAIKAKU.AI, [arXiv:2605.22391v1](https://arxiv.org/abs/2605.22391) [cs.AI], May 21, 2026. It aggregates 4.14M recipes from 11 sources across seven languages, normalises them to 1,790 canonical ingredients, and trains three sibling Metapath2Vec embeddings that differ only in random-walk schema — co-occurrence only, chemistry only, and a controlled blend. It is the baseline this repository measures against, not a description of what the repository currently does.

**`raw-data/`** is the corpus, one numbered folder per source. `01-` through `09-` are the frozen 1:1 reproduction of the paper's Table A1: **4,137,626 recipes against the paper's 4,135,189 — six of nine sources reproduce exactly and the total lands within 0.059%.** Re-check it at any time with `prior-study/tools/verify_corpus_table_a1.py`, which reads this tree directly and takes about six minutes. The remaining deltas are structural (a duplicated multimodal SFT file, an undocumented dedup step), not access problems. `10-` through `29-` are corpora beyond the paper, adding eight languages. `_duplicates/` holds six corpora the expansion audit rejected, kept because five of them still carry recipe and ingredient lists; `_superseded/` holds the 2,147,248-row RecipeNLG mirror that `recipe_cooc.npz` was derived from. `MANIFEST.md` records the source-name mapping and exists so row counts can be corrected without renaming a directory.

**`model/`** turns that into a scored leaderboard. The built corpus is **4,647,847 recipes, 35,783,309 ingredient slots, a 1,790-concept vocabulary, 29 sources, 15 languages** — verified directly from `data/recipes/recipe_ids.npz`, not quoted from a log. Sixteen models across seven families train against it, all CPU-viable, because the available Azure quota is six dedicated vCPUs and no usable GPU.

**`prior-study/`** is the work that came before `model/` — 56 scripts, the pre-registration, and an independent replication of the paper. It is not archived reading material: `model/` imports `corpus.py` and `normalize.py` from it at runtime, because those two modules built the corpus every model here trains on and a reimplementation that disagreed with them would be indistinguishable from a corpus bug. `ingredient_model/data/normalizer.py` patches the base normaliser in memory rather than forking it.

Two documents in `prior-study/docs/` carry more weight than the code. **`PREREGISTRATION.md`** defines M1–M5 and hypotheses H1–H6 *before results existed*, which is what licenses the claim that results here remain comparable to the prior study, and it is where the falsified H5 that zeroes the chemistry weight is recorded. **`FINDINGS.md`** and **`REPLICATION.md`** hold the audit: the paper's own live service does not reproduce the paper (0% exact at θ=30 or θ=60), and the headline mode-coherence margin is a 6–8× overstatement against a like-for-like control.

## The one thing worth knowing before reading any number here

**Graph models and recipe models cannot share an evaluation protocol.** Hiding 10% of an ingredient-ingredient graph's edges hides nothing from a model that reads recipes, because the recipes that produced those pairs are still in the training set.

The gap is not academic. EASE scores M4 **0.6898** on the edge split and **0.5644** on the honest one — 18× the confidence interval, and the flattering number is the default if nobody intervenes. So `ingredient_model/data/splits.py` intervenes: `check_leakage()` raises rather than warns, on the reasoning that an optimistic number which is merely flagged still gets quoted six weeks later.

The same principle runs through the rest of the workspace. `control_gate()` scores pure noise and blocks everything if the metrics rate random vectors above chance. Every M6 result reports a popularity baseline beside it, because recommending onion, salt and butter to everyone scores **0.3723** and a model that has learned nothing but frequency looks competent until you put it next to that.

## Where things stand

Baselines on `recipe-holdout`, seed 0, read from `model/results/runs/baselines/`:

| model | M4 link AUC | M6 recall@10 |
|---|---|---|
| svd-ppmi | 0.6151 | **0.4430** |
| sgns-cooc | 0.7170 | 0.4068 |
| *popularity baseline* | — | *0.3723* |
| glove | 0.6527 | 0.2495 |
| ease | 0.5644 | 0.1656 — **0.5927 through its native scorer** |
| sgc | 0.6668 | 0.0144 |
| item2vec | **0.7179** | 0.0066 |

Two rows in that table are the interesting ones. EASE loses to popularity through its exported embedding and beats it by +0.22 through its actual prediction rule — squeezing an item-item autoencoder into a vector table discards more than half of it. And `item2vec` holds the best M4 on the board while sitting at chance on M6, because it was broken: an alphabetically sorted vocabulary meeting `np.triu_indices` meant each ingredient was trained in proportion to how early its name sorts, at corr(centre-share, vocabulary index) = −0.980. The fix is in and pinned by a test; the repaired run scores M4 0.7053, M6 0.0195. **A metric that can be topped by an effectively untrained embedding is not measuring what its name suggests** — which is the argument for M6 existing at all.

`model/README.md` carries the full account, including a large reproducible centring effect with no established cause and three rejected explanations for it, and `model/ARCHITECTURE.md` explains why the workspace is shaped the way it is.

## Running it

```bash
cd model
make setup          # .venv + editable install, Python >=3.10
make data           # import artefacts, build the honest split
make check          # control gate, then tests, then adversarial sanity checks
make baselines      # train every family on recipe-holdout
make report
```

`im list` enumerates models, datasets and splits; a model type is a folder under `models/`, discovered by `pkgutil` at import, so there is no registry to edit and none to forget to edit.

**Verified on this machine, August 25, 2026:** `make check` → control gate PASS, 94 tests passed / 1 skipped, **28/28 sanity checks passed**. `im list` resolves all 16 models and all 6 datasets. `prior-study/tools/verify_corpus_table_a1.py` reproduces 4,137,626 recipes against `raw-data/` in about six minutes.

The load-bearing sanity check is the last one: *stored recipes reproduce exactly from source files, 2,975/2,975*. It replays the original raw sources through the prior study's readers rather than comparing one derived artefact against another, so it is the only check that can catch a reader silently dropping a column.

## Known gaps in this repository, as of August 25, 2026

These are stated because finding them yourself costs an hour each.

- **`use-cases/` is empty.**
- **The stored baseline runs predate the centring metric.** `M6_centred_*` is implemented in `eval/harness.py` and pinned by `tests/test_completion_centring.py`, but no `metrics.json` in `results/runs/` contains it. The centring table in `model/README.md` cannot be reproduced from stored artefacts without retraining.
- **`recipe_ids_v2.npz` is built but unused.** The corpus rebuild recovered 924,315 ingredient occurrences (+2.58% slots) with the vocabulary unchanged at 1,790 — but it has 5,583 more rows than the old corpus, so row indices no longer align. Splits and the text index must be rebuilt before any family retrains on it, and until that happens the question the rebuild exists to answer — *does better coverage change the ranking?* — is open.
- **`raw-data/README.md` still describes the pre-merge layout.** It refers to an `expansion/` directory that was flattened into the numbered folders — `prior-study/_layout_migration.json` records that move and can reverse it. Its per-source content, reproduction table and counting rule are accurate.
- **Fifteen per-source READMEs under `raw-data/` have had their `license:` frontmatter stripped**, including two NonCommercial declarations (`25-halal`, `13-turkish`). Not a concern for private local use, and the full per-source licensing analysis survives in `prior-study/docs/LICENSE_AUDIT.md`.
- **`prior-study/data/derived/` duplicates about 48 MB** already present under `model/data/`. `import_data.py` copies one to the other by design; the prior-study copy is what keeps its own scripts runnable.
- **This directory is not a git repository.** `model/.gitignore` excludes `data/` and `results/runs/`, but nothing here is under version control and there is no root `.gitignore` yet — `raw-data/` alone is 7.4 GB and must not be committed.
