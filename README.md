# ingredients

Everything behind *Epicure* and what came after it: the paper, the recipe corpora it was built from, the prior study that audited it, and a workspace for retraining ingredient embeddings under an evaluation protocol that does not flatter them.

Private repository, for local and educational use.

**10 GB on disk.** `raw-data/` is 7.4 GB, `model/` is 2.6 GB, `prior-study/` is 221 MB. Most of it is regenerable rather than authored.

```
paper/         epicure.pdf — the published paper
raw-data/      29 source corpora + _duplicates/ + _superseded/
model/         the ingredient-model workspace, and the `im` CLI
prior-study/   the replication and audit that preceded model/
```

## The three parts, and how they connect

**`paper/`** holds *Epicure: Navigating the Emergent Geometry of Food Ingredient Embeddings* — Jakub Radzikowski and Josef Chen, KAIKAKU.AI, [arXiv:2605.22391v1](https://arxiv.org/abs/2605.22391) [cs.AI], May 21, 2026. It aggregates 4.14M recipes from 11 sources across seven languages, normalises them to 1,790 canonical ingredients, and trains three sibling Metapath2Vec embeddings that differ only in random-walk schema — co-occurrence only, chemistry only, and a controlled blend. It is the baseline this repository measures against, not a description of what the repository currently does.

**`raw-data/`** is the corpus, one numbered folder per source. `01-` through `09-` are the frozen 1:1 reproduction of the paper's Table A1: **4,137,626 recipes against the paper's 4,135,189 — six of nine sources reproduce exactly and the total lands within 0.059%.** Re-check it at any time with `prior-study/tools/verify_corpus_table_a1.py`, which reads this tree directly and takes about six minutes. The remaining deltas are structural (a duplicated multimodal SFT file, an undocumented dedup step), not access problems. `10-` through `29-` are corpora beyond the paper, adding eight languages. `_duplicates/` holds six corpora the expansion audit rejected, kept because five of them still carry recipe and ingredient lists; `_superseded/` holds the 2,147,248-row RecipeNLG mirror that `recipe_cooc.npz` was derived from. `MANIFEST.md` records the source-name mapping and exists so row counts can be corrected without renaming a directory.

**`model/`** turns that into a scored leaderboard. The built corpus is **4,647,847 recipes, 35,783,309 ingredient slots, a 1,790-concept vocabulary, 29 sources, 15 languages** — verified directly from `data/recipes/recipe_ids.npz`, not quoted from a log. Sixteen models across seven families train against it, all CPU-viable, because the available Azure quota is six dedicated vCPUs and no usable GPU.

**`prior-study/`** is the work that came before `model/` — 56 scripts, the pre-registration, and an independent replication of the paper. It is not archived reading material: `model/` imports `corpus.py` and `normalize.py` from it at runtime, because those two modules built the corpus every model here trains on and a reimplementation that disagreed with them would be indistinguishable from a corpus bug. `ingredient_model/data/normalizer.py` patches the base normaliser in memory rather than forking it.

Two documents in `prior-study/docs/` carry more weight than the code. **`PREREGISTRATION.md`** defines M1–M5 and hypotheses H1–H6 *before results existed*, which is what licenses the claim that results here remain comparable to the prior study, and it is where the falsified H5 that zeroes the chemistry weight is recorded. **`FINDINGS.md`** and **`REPLICATION.md`** hold the audit: the paper's own live service does not reproduce the paper (0% exact at θ=30 or θ=60), and the headline mode-coherence margin is a 6–8× overstatement against a like-for-like control.

## The one thing worth knowing before reading any number here

**Graph models and recipe models cannot share an evaluation protocol.** Hiding 10% of an ingredient-ingredient graph's edges hides nothing from a model that reads recipes, because the recipes that produced those pairs are still in the training set.

The gap is not academic. EASE scores M4 **0.6844** on the edge split and **0.5598** on the honest one — 18.9× the confidence interval, and the flattering number is the default if nobody intervenes. So `ingredient_model/data/splits.py` intervenes: `check_leakage()` raises rather than warns, on the reasoning that an optimistic number which is merely flagged still gets quoted six weeks later.

The same principle runs through the rest of the workspace. `control_gate()` scores pure noise and blocks everything if the metrics rate random vectors above chance. Every M6 result reports a popularity baseline beside it, because recommending onion, salt and butter to everyone scores **0.3650** on this split and a model that has learned nothing but frequency looks competent until you put it next to that.

## Where things stand

All sixteen registered models on `recipe-holdout`, seed 0, trained against corpus generation **v2**, read from `model/results/runs/all-v2/`. `M6@10` is the exported embedding scoring by cosine; `M6 best` is the model's own prediction rule where it has one. The popularity baseline scores **0.3650**, and is the column that decides whether a row is worth anything.

| model | M4 link AUC | M6@10 (embedding) | M6 best | vs popularity |
|---|---|---|---|---|
| masked-set | 0.5726 | 0.1229 | **0.6178** | **+0.2528** |
| ease | 0.5598 | 0.1422 | 0.5872 | +0.2222 |
| svd-ppmi | 0.6227 | **0.4505** | 0.4505 | +0.0855 |
| concat | 0.7138 | 0.4446 | 0.4446 | +0.0796 |
| residual | 0.5870 | 0.4235 | 0.4235 | +0.0585 |
| sgns-core | 0.7295 | 0.4065 | 0.4065 | +0.0415 |
| sgns-cooc | 0.7356 | 0.3885 | 0.3885 | +0.0235 |
| *popularity baseline* | — | *0.3650* | *0.3650* | — |
| text-aligned | 0.6510 | 0.3216 | 0.3216 | −0.0433 |
| glove | 0.6611 | 0.2211 | 0.2211 | −0.1439 |
| lightgcn | **0.7565** | 0.1837 | 0.1837 | −0.1813 |
| text-embed | 0.5799 | 0.1077 | 0.1077 | −0.2573 |
| sgc | 0.6806 | 0.0777 | 0.0777 | −0.2873 |
| ials | 0.5371 | 0.0426 | 0.0426 | −0.3224 |
| chem-svd | 0.5019 | 0.0152 | 0.0152 | −0.3498 |
| sgns-chem | 0.5044 | 0.0149 | 0.0149 | −0.3502 |
| item2vec | 0.7238 | 0.0123 | 0.0123 | −0.3528 |

Three things in that table are worth the reader's time.

**Most of these models lose to recommending onion, salt and butter.** Nine of sixteen sit below the popularity baseline, and the two that beat it decisively do so only through a scorer that is not an embedding at all. The best embedding on the board, `svd-ppmi`, is a truncated SVD of a PPMI matrix — the oldest and cheapest method here — and it beats every neural model that exports vectors.

The nine is not a knife-edge count. Bootstrapping the 20,000 completion instances puts it at nine with a 95% interval of nine to nine, and no model's own interval straddles the baseline: every one of the sixteen is significantly on one side or the other. The count is also replicated. `scripts/m6_intervals.py` is a second, independently written implementation of the same estimand, and it agrees with the canonical `scripts/bootstrap_m6.py` to 2.4e-08 on all sixteen point estimates, on all sixteen verdicts, and on the headline count — from a different bootstrap seed and a different number of resamples. It writes `m6_replication.json`, never the canonical artefact, and exits non-zero if the two ever part company.

That interval covers the evaluation draw. It says nothing about the training draw, so every model was retrained at five seeds on the same split — eighty trials, written to `results/runs/seeds-v2/` and summarised by `scripts/ranking_stability.py`. The popularity baseline is a fixed quantity of the corpus and comes back at exactly 0.3650 every time. The count below it, taken over exported embeddings, is eleven at every seed; the headline of nine is smaller because `ease` and `masked-set` are counted at their own scorers, which the sweep does not re-run. Four of the sixteen change rank, in two adjacent pairs — fourth with fifth, fifteenth with sixteenth — and **none of the sixteen changes which side of the baseline it is on at any seed**. Seed-pair Spearman averages 0.9971 and never falls below 0.9941. The ordering is therefore a single draw and neighbouring positions should not be read as settled; the claim made from it is not. The sweep's own seed-0 trials also reproduce the already-published numbers exactly, so the spread being measured is the seed rather than drift in the pipeline between the two runs.

**The vector table is the lossy part, not the model.** EASE scores 0.1422 through its exported embedding and 0.5872 through its own item-item rule; `masked-set` scores 0.1229 and 0.6178. Squeezing either into a fixed-width vector table discards roughly three quarters of what it knew. This is a claim about the artefact, not about the architecture.

**M4 and M6 disagree almost completely.** `lightgcn` holds the best link AUC on the board at 0.7565 while scoring 0.1837 on completion, and `item2vec` is third on M4 while last on M6. A metric that ranks an unusable model first is not measuring usefulness, which is the argument for M6 existing.

The `item2vec` history is worth keeping: its original run held the best M4 on the board while sitting at chance on M6, because an alphabetically sorted vocabulary meeting `np.triu_indices` trained each ingredient in proportion to how early its name sorts, at corr(centre-share, vocabulary index) = −0.980. The fix is in and pinned by a test.

### Does a better corpus change the ranking? No.

Generation v2 recovers 924,315 ingredient occurrences (+2.58% slots) and 5,583 recipes over v1, and rebuilding the graph on it adds 17,367 edges (203,504 → 220,871, +8.5%). Running the six models that exist in both generations through `scripts/compare_generations.py`:

| metric | Spearman rank correlation, v1 vs v2 |
|---|---|
| M6 lift over popularity | **+1.000** |
| M6 recall@10 | **+1.000** |
| M6 MRR | **+1.000** |
| M4 link AUC | +0.943 |
| M2 triplet (strict) | +0.829 |

The completion ranking is perfectly preserved. The honest caveat, which the script prints every time it runs: the two generations are scored against *different held-out sets*, because v2's split is a fresh draw rather than v1's split with rows added. Rankings are therefore comparable and individual deltas are confounded, so the deltas are printed but never used as evidence. The conclusion this supports is the weaker and more durable one — the leaderboard is not sensitive to a corpus improvement of this size.

`model/README.md` carries the full account, including a large reproducible centring effect with no established cause and three rejected explanations for it, and `model/ARCHITECTURE.md` explains why the workspace is shaped the way it is.

## Running it

```bash
cd model
make setup          # .venv + editable install, Python >=3.10
make data           # import artefacts, build the honest split
make check          # control gate, then tests, then adversarial sanity checks
im sweep experiments/all-v2.yaml    # train all sixteen models on the canonical corpus
make report
```

`im list` enumerates models, datasets and splits; a model type is a folder under `models/`, discovered by `pkgutil` at import, so there is no registry to edit and none to forget to edit.

A sweep is resumable. Completed trials are skipped on re-entry, every trial appends to `results/runs/<experiment>/journal.jsonl` as it finishes, and models that consume a sibling run are ordered after it automatically — the dependency is derived by scanning parameter values for run ids rather than declared separately, so it cannot drift out of step with the parameters it describes. `text-embed` and `text-aligned` additionally need `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`; the other fourteen run offline.

**Verified on this machine, August 30, 2026:** `make check` → control gate PASS, 94 tests passed / 1 skipped, **28/28 sanity checks passed** against generation v2. `im sweep experiments/all-v2.yaml` completes all **16/16 trials with zero failures in 67.9 minutes** on an M3 Pro. `prior-study/tools/verify_corpus_table_a1.py` reproduces 4,137,626 recipes against `raw-data/` in about six minutes.

The load-bearing sanity check is the last one: *stored recipes reproduce exactly from source files, 2,975/2,975*. It replays the original raw sources through the prior study's readers rather than comparing one derived artefact against another, so it is the only check that can catch a reader silently dropping a column.

## Known gaps in this repository, as of August 30, 2026

These are stated because finding them yourself costs an hour each.

- **Fifteen per-source READMEs under `raw-data/` have had their `license:` frontmatter stripped**, including two NonCommercial declarations (`25-halal`, `13-turkish`). Not a concern for private local use, and the full per-source licensing analysis survives in `prior-study/docs/LICENSE_AUDIT.md`.
- **`prior-study/data/derived/` duplicates about 48 MB** already present under `model/data/`. `import_data.py` copies one to the other by design; the prior-study copy is what keeps its own scripts runnable. Which copy is authoritative depends on the file, so the graphs are worth stating explicitly, and `im list` now prints each artefact's real counts rather than a description written once and left alone:

  | artefact | edges | generation | what it backs |
  |---|---|---|---|
  | `ii_graph.npz` | 220,871 | v2 | the canonical graph; differs from the prior study's copy, which is the one file `import_data.py` does not simply mirror |
  | `ii_graph_rh_train.npz` | 200,690 | v2 | **every published run.** The recipe-holdout split, which is the honest protocol |
  | `ii_graph_rh_heldout.npz` | 21,726 | v2 | its held-out edges |
  | `ii_graph_train.npz` | 183,154 | v1 | byte-identical to the prior study's copy, never rebuilt |
  | `ii_graph_heldout.npz` | 20,350 | v1 | likewise. 183,154 + 20,350 = 203,504, the v1 edge count, which is why that number appears nowhere in the v2 graph |

  The two v1 files back only the `edge-holdout` contrast, whose entire purpose is to show that an optimistic split flatters a model. Nothing on the leaderboard depends on them. They are kept rather than rebuilt because rebuilding them would silently change what that contrast is contrasting against.
- **The v1 baseline runs are not stamped with a corpus generation.** They predate the marker, so `compare_generations.py` reports their generation as `unknown` rather than inferring v1 from their age. Everything trained since carries `corpus_generation` in its manifest.
- **Fifty-six numbers in these documents cannot be traced to an artefact.** `scripts/check_docs.py` resolves every metric-shaped figure in this file, `model/README.md` and `model/ARCHITECTURE.md` against the stored `metrics.json`, `manifest.json`, `corpus_stats.json` and `GENERATION.json` files; 56 of 196 match nothing. Almost all are outputs of one-off analyses — the centring ablation, the frequency-correlation checks, the per-language alias counts — that were printed to a terminal and never written to disk. They are not known to be wrong; they are unverifiable, which is a smaller problem but not nothing. They are recorded in `scripts/docs_baseline.json` so that new drift fails `make check` while the existing gap stays visible and countable. The list only ever shrinks, and has: it started at 60 and fell to 56 as the corpus waterfall, the graph totals and the completion intervals became artefacts, without anyone editing the list by hand. `docs_baseline.json` is the authoritative count; the figure quoted here is as of the date above. The check has already earned itself twice — once catching a figure on the page arguing for honest evaluation that no run in the repository produced, and once catching a number added while writing this very paragraph.

  **What it deliberately does not check: percentages.** The pattern matches four-decimal values and comma-grouped integers, so `88.4%` is invisible to it. Extending it was tried and rejected on measurement rather than taste. A permissive pattern matched the tails of longer numbers and "resolved" round prose figures like `10%` and `96%` by coincidence against the full set of artefact values — 44 of 67 tokens, nearly all of them meaningless. A stricter pattern requiring a decimal point fixed both faults but still resolved 30 of 52, and a one-decimal percentage only pins an underlying value to about a part in a thousand, which is loose enough that some of those are luck. A gate that can pass by accident is worse than an acknowledged gap, because it converts an unchecked number into an apparently checked one. The honest route to covering these is to persist the analyses that print them, not to widen the pattern until it hits something.

  **The same failure arrives from the other direction, too.** Widening the pattern is not the only way to make the gate pass by accident; leaving the pattern alone and growing the artefact pool does it as well. The multi-seed sweep added several hundred stored values, and five baseline entries promptly started resolving. None of them was a real verification. A documented AUC was matching a model's training loss at epoch eight; two before-and-after deltas were matching an unrelated model's recall at a seed that did not exist when the sentence was written. Had the suggestion been acted on, five unverifiable numbers would have been recorded as verified. So the gate now prints the artefact and JSON path behind every entry it proposes dropping, and says plainly that a match only means some artefact holds that value, not that it holds it for that reason. Resolution is evidence to be read, not a verdict to be applied — and the baseline list is meant to shrink only when a number's own artefact appears.
- **Eight of twenty-nine sources have no text reader**, so their rows in `recipe_text.parquet` are aligned but blank. `scripts/check_text_alignment.py` lists them; they are mostly the non-English corpora.

### Closed since August 25

- ~~`raw-data/README.md` still describes the pre-merge layout.~~ It described a tree rooted at `recipe/` containing an `expansion/` subdirectory; both were flattened and renamed, and `prior-study/_layout_migration.json` records the move but was written before the rename, so its entries still use the old root. Three further path errors came out of the same cause, because the file was authored while it lived inside `prior-study/` and its relative paths silently stopped resolving when the repository was merged: the documented recount command named both an interpreter and a script that do not exist at those paths, and the stated output location was wrong. The command now reads `model/.venv/bin/python prior-study/tools/verify_corpus_table_a1.py`, which is safe to run from anywhere because the script anchors on `__file__` rather than the working directory. Every path in the file has been checked against the filesystem. The per-source content, reproduction table and counting rule were accurate throughout and are unchanged.
- ~~`use-cases/` is empty.~~ Removed. Nothing in any script, config, build step or document referred to it; it existed only so this list could record that it was empty.

- ~~`scripts/analyse_coverage.py` uses the prior study's base normalizer.~~ It was not deliberate. The script normalised recipe text with the upstream normaliser and compared the result against the v2 corpus, which the corrected one had built, so every line the correction rescued was counted as a lexicon gap. It now reads `data/GENERATION.json` and measures whichever normaliser built the canonical corpus, with `--normalizer base` retained for the historical figure. Correcting it moves kept-line share from 81.6% to 88.4% on `02-xiachufang`, 48.9% to 59.9% on `taiwan-1.8k` and 61.7% to 74.5% on `bhuvii-17k` — the same direction and rough magnitude as the independent slot-level measurement in `model/README.md`, which is the cross-check that it is now measuring the right thing.
- ~~`recipe_ids_v2.npz` is built but unused.~~ Promoted to canonical by `scripts/promote_corpus.py`, which refuses the swap unless the vocabulary is byte-identical, since embedding rows are vocabulary positions and a silent vocabulary change would invalidate every stored run rather than produce a comparable one. Splits, the ingredient-ingredient graph and the text index were all rebuilt against it, and `data/GENERATION.json` now records which corpus is canonical and which normalizer built it.
- ~~The stored baseline runs predate the centring metric.~~ Every run under `results/runs/all-v2/` carries `M6_centred_*`.
- ~~This directory is not a git repository.~~ It is, with a root `.gitignore` that excludes `raw-data/`, `prior-study/data/` and every weight blob while deliberately keeping `manifest.json` and `metrics.json` tracked, so the leaderboard is reproducible from the repository alone.

### Found and fixed while running the full sweep

- **`ials` could never have completed.** Its item solve materialised one outer product per ingredient slot — `(slots, d, d)` floats, over 100 GB at this corpus size — and was killed by the OS before finishing a single iteration. It had never appeared in a leaderboard, so nothing had exposed it. Since the vocabulary is only 1,790, each ingredient's normal equations are the Gram matrix of the recipe factors containing it, which is one BLAS call over a gathered block; the rewrite matches the original to 4.4e-07 relative on a case small enough to run both, and peaks at 3.6 GB.
- **Sibling-run resolution broke inside grouped sweeps.** `PATHS.run_dir(name)` resolves to `results/runs/<id>`, but a sweep writes to `results/runs/<experiment>/<id>`, so hybrid and text-aligned models would raise `FileNotFoundError` only after every model they depend on had already been trained. `artifacts.resolve_run()` now searches most-specific-first and raises on ambiguity rather than guessing.
- **Two call sites hardcoded the prior study's normalizer.** `sanity_check.py` and `data/text.py` both instantiated the base `Normalizer()` while v2 was built with the corrected one, which showed up as sanity check #9 falling to 97.51% and as a hard alignment failure at corpus row 89 of `01-recipenlg`. Both now read the normalizer from `GENERATION.json`, so the provenance is recorded rather than inferred.
