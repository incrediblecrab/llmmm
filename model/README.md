# ingredient-model

A workspace for learning what ingredients mean to each other, from 4.6M recipes.

Each model type lives in its own folder under `models/`. Drop a new folder in
and `im list` finds it — there is no central registry to edit, and therefore
none to forget to edit.

```bash
python -m venv .venv && ./.venv/bin/pip install -e '.[dev]'
./.venv/bin/python scripts/import_data.py     # ~50MB of prepared artefacts
./.venv/bin/python scripts/build_splits.py    # the honest evaluation split

im gate                                        # must pass before any result counts
im list                                        # models, datasets, splits
im train ease --split recipe-holdout
im sweep experiments/baselines.yaml
im report
```

## The one thing to understand first

**Graph models and recipe models cannot share an evaluation protocol.**

The corpus reaches models through two doors. Graph models read an
ingredient-ingredient co-occurrence graph; recipe models read the recipes
themselves. Hiding 10% of the *graph's edges* hides nothing from a model that
reads recipes — the recipes that produced those pairs are still right there.

This is not theoretical. EASE scores **M4 0.6898** on the edge split and
**0.5644** on the honest one. The difference is 18× the confidence interval, and
the flattering number is the one you get by default if nobody stops you.

So `ingredient_model/data/splits.py` stops you:

| split | who may use it | what it holds out |
|---|---|---|
| `edge-holdout` | graph models only | 10% of graph edges |
| `recipe-holdout` | **everyone** | 30% of recipes, graph rebuilt from the rest |
| `full` | nobody — production artefacts only | nothing |

`check_leakage()` **raises**. It does not warn, because an optimistic number
that is merely flagged still ends up quoted in a summary six weeks later.
`--allow-leakage` exists for when you want to measure the leakage itself.

## Metrics

M1–M5 come from the prior study's pre-registration and are unchanged, so results
here remain comparable to it.

| | what it measures |
|---|---|
| M1 | participation ratio — is the space actually using its dimensions |
| M2 | substitution triplet accuracy (broad / strict tiers) |
| M3 | substitution recall@10 |
| M4 | held-out link AUC |
| M5 | how much of the geometry is just word frequency |
| **M6** | **recipe completion — hide an ingredient, rank all 1,790 candidates** |

M6 is new and is the most trustworthy number here: it is leak-free for every
family at once, and it is the product's actual task. It always reports a
**popularity baseline** alongside, because recommending onion, salt and butter
to everyone scores 0.37 and a model that has learned nothing but frequency looks
competent until you put it next to that.

### The embedding is not always the model

M6 is reported twice when a model has a native scorer. This matters more than it
sounds:

| EASE | recall@10 |
|---|---|
| through its exported embedding | 0.166 — **loses** to popularity |
| through its actual prediction rule | **0.593** — beats it by +0.22 |

Squeezing an item-item autoencoder into a vector table discards more than half
its spectrum. Judged on the embedding alone, EASE looks worse than a frequency
table. Any family whose model is not literally a lookup table should return a
`scorer` — see `models/README.md`.

### Ties are broken by midrank, on purpose

M6 originally ranked the target by counting only strictly-better candidates.
That is the *optimistic* rank under ties, and it is exploitable: a degenerate
model produces a large tied block and every member of it is credited with the
best position in that block.

This was not hypothetical. A rank-1 collapsed embedding normalises to two
distinct rows, so ~900 candidates tie at the recall@10 cut, and it scored
**0.211 — thirty-eight times chance, and better than `sgc` and `residual`** —
having learned nothing whatsoever. Representation collapse is the signature
failure of exactly the two families here most at risk of it: an over-smoothed
graph network converges to the dominant eigenvector, and a transformer trained
without warmup collapses to one direction.

Ranks are now midranks. For a model with no ties this is arithmetically
identical, so **no previously recorded result changed** — verified: EASE
embedding 0.1658, native 0.5984 and popularity 0.3659 are unmoved, while the
collapsed control fell to 0.0008. Locked in `tests/test_ranking_ties.py`.

## Checking the results

`make sanity` re-derives the headline numbers by an independent path and runs
the controls that would expose a metric paying for something other than skill.
28 checks, ~12s. It is wired into `make check`.

It exists because almost every check in a project like this compares one derived
artefact against another, which cannot catch a reader that drops a column — that
failure is perfectly self-consistent downstream. So the suite also replays the
original source files and confirms the stored corpus is reproducible from them:
**2,975/2,975 recipes match exactly.**

What it verifies, briefly: vocabularies are element-wise identical across every
artefact (not merely the same length); train and test partition the corpus
exactly and share no held-out edge; the popularity baseline reproduces from raw
counts through a separate hand-written ranking loop (0.3635 vs 0.3659); random
vectors and collapsed spaces score at or below chance; row-shuffling an
embedding destroys M2 and M6; and the leakage claim reproduces end-to-end
(M4 0.6898 leaky vs 0.5644 honest, 18.2x the CI).

### What 28 passing checks did not catch

All of it passed while `item2vec` — holder of the **best M4 on the
leaderboard, 0.7179** — was broken. The suite compares artefacts to artefacts,
so a model that is internally consistent and externally meaningless goes
straight through.

What exposed it was printing nearest neighbours and reading them:

```
item2vec   butter -> buttermilk, cantal_cheese, cookie_butter, candy_coating, camembert
item2vec   yeast  -> worcestershire_sauce, yam, whole_wheat_flour, tzatziki, whipped_topping
svd-ppmi   butter -> flour, egg, cream, milk, sugar, vanilla
```

Those are alphabetical, not semantic. Quantified: mean vocabulary-index distance
to the top-10 neighbours is **213 against 597 expected** by chance, and cosine
similarity correlates −0.212 with index proximity. No other family shows it.

The cause is three innocuous decisions meeting. Recipes are stored in ascending
id order; the vocabulary is sorted alphabetically; and `np.triu_indices` emits
only `i<j`, so column 0 of every training pair is always the lower id. Skip-gram
is asymmetric — column 0 trains the input matrix, and the input matrix is what
gets exported as the embedding. So each ingredient was trained in proportion to
how early its name sorts. Measured: **corr(centre-share, vocabulary index) =
−0.980**, the first decile serving as centre 96% of the time and the last decile
4%, against 50% for a correct stream. Late-alphabet ingredients were close to
their random initialisation, which is why they look like each other.

Fixed in `models/recipe_basket/item2vec.py` by randomising pair direction, and
pinned by `tests/test_item2vec_symmetry.py` — including a test asserting the
*old* ordering trips the guard, so it cannot pass vacuously.

Three things worth keeping from this:

- **M4 did not merely miss the defect, it rewarded it.** Pairwise AUC ranked a
  model with no usable neighbourhood structure first, while M6 put it at chance
  (recall@10 0.0066, median rank 569/1790). Any metric that can be topped by an
  untrained embedding is not measuring what its name suggests.
- **The M4/M6 inversion was not a finding.** It was on its way into this README
  as the leading scientific result. A like-for-like test kills it: comparing
  every family under the same scorer gives Spearman ρ=−0.429, p=0.397, and
  removing `item2vec` alone takes it to exactly **0.00**. The ρ=−0.829, p=0.042
  version that looked publishable came from scoring EASE with its native
  ranker and everyone else with cosine — mixing two procedures along one axis.
  One broken model and one inconsistent comparison, in an n=6 sample.
- Two plausible explanations were tested and are **wrong**: frequency-dominated
  geometry (M5 — `sgc` has the lowest frequency-PC correlation, 0.098, and still
  collapses; `glove` has the highest, 0.927, and does not) and hubness (the two
  *best* M6 families have the highest neighbour-concentration gini). They are
  recorded because ruling them out is what forced the real cause into view.

`sgc`'s M6 collapse (0.0144) is a **separate, still-unexplained** failure. It is
not index-encoding (ratio 0.90) and not frequency-driven; its embedding norms
vary far more than any other family's (CV 1.99 vs 0.41–0.89), which is a lead,
not an answer.

### A second leaderboard-reordering result, with no explanation attached

Chasing `sgc` turned up something bigger. Re-scoring M6 with the embedding mean
subtracted — a one-line, cost-free post-processing — gives:

| family | raw recall@10 | centred | change |
|---|---|---|---|
| glove | 0.2509 | **0.4919** | **+0.2410** |
| item2vec (fixed) | 0.0195 | 0.1756 | +0.1562 |
| item2vec (broken) | 0.0070 | 0.0645 | +0.0575 |
| sgc | 0.0135 | 0.0539 | +0.0404 |
| ease | 0.1653 | 0.1880 | +0.0226 |
| sgns-cooc | 0.4020 | 0.3967 | −0.0054 |
| svd-ppmi | 0.4462 | 0.4443 | −0.0019 |

Centring costs the two families it hurts 0.005 and 0.002 — inside sampling
noise — and moves **glove from fourth place to first**, past `svd-ppmi`. It is
now reported as `M6_centred_*` alongside the raw number, never replacing it,
because silently redefining a metric invalidates every result recorded against
it.

**The mechanism is not known, and three explanations were tested and rejected.**
This is written down at the same prominence as the result, because the result is
the easy part:

- *Translation gauge* — the obvious story, that cosine against a summed query is
  not translation-invariant. Testable, and **false**: translating a synthetic
  embedding by a large constant moves raw M6 from 0.8395 to 0.8415. This one had
  already been written into the docstring as the explanation before it was
  tested; the test that killed it is now permanent.
- *Popularity in the shared direction* — correlation between alignment with the
  mean direction and log frequency is **+0.646 for `sgns-cooc`, which gains
  nothing**, and **−0.009 for `item2vec-fixed`, which gains +0.156**. Backwards.
- *Centroid size alone* — `glove` (0.433) and `sgns-cooc` (0.414) have nearly
  identical mean pairwise cosine and gain +0.241 and −0.005.

So the honest state is: a large, reproducible, ranking-changing effect with no
established cause. That is a reason to keep both columns and keep measuring, not
a reason to pick the flattering one.

## Reading the corpus

The corpus is `uint16` token ids, which are not inspectable — a normalisation
bug, a misaligned vocabulary and a truncated source all look identical once
everything is an integer.

```bash
im recipes                      # a uniform random sample
im recipes 4000000              # one recipe by index
im recipes --lang zh -n 5       # filter by language
im recipes --contains saffron   # every recipe using an ingredient
im recipes --verify             # all 4.6M reachable + the source mix
im recipes --raw-source 01-recipenlg   # the original lines, for comparison
```

`--verify` checks all 4,647,847 rather than a sample: offsets contiguous and
covering every token, all ids in vocabulary, and every recipe stored sorted and
de-duplicated.

Two things are deliberately *not* recoverable, and it is better to state that
than to imply otherwise. **Ingredient order and duplicates are gone** — the
builder stores `sorted(set(ids))`, which is right for set-based models but means
the corpus cannot support anything order-sensitive without a rebuild. **Titles,
quantities and instructions were never in it** — the readers only ever yielded
ingredient lists. `make text` rejoins them.

## Recipe text, and what normalisation discards

`make text` rebuilds titles, URLs, quantities and instructions and rejoins them
to the corpus, verifying **every one of the 4,647,847 rows** re-normalises to
the ingredient set already stored. Not a sample: a join that is right for 2M
rows and wrong afterwards attaches confident, plausible, wrong titles to exactly
the rows nobody spot-checks. Coverage is 96.8% titles, 2.29M URLs, 4.22M
instruction sets; eight sources have no text reader and are stored blank but
**aligned**, because a missing title is a gap while a shifted one is a corrupted
dataset.

```bash
im recipes 0 --text             # title, url, quantities, steps
make coverage                   # what normalisation drops, and why
```

The text made a previously invisible defect measurable. `make coverage`:

| source | lang | unrecognised | duplicate | kept |
|---|---|---|---|---|
| 01-recipenlg | en | 4.9% | 5.2% | 89.9% |
| 03-povarenok | ru | 5.0% | 1.1% | 93.8% |
| 02-xiachufang | zh | **13.9%** | 4.4% | 81.7% |
| taiwan-1.8k | zh | **44.9%** | 5.9% | 49.1% |

The cause is not the languages and not the quantity parser — `适量盐` and
`少许糖` normalise correctly, so prefixes are handled. It is the alias tables:
**English has 5,710 aliases, Chinese has 346**, against a shared vocabulary of
1,790 concepts. Chinese can therefore reach at most 19% of the vocabulary while
English reaches all of it. `鸡` (chicken) does not map.

Every dropped concept checked — 培根 bacon, 百香果 passion fruit, 紫薯 purple
sweet potato, `bay leaves`, `mayo` — is **already in the vocabulary**. So this
is a mapping gap, not a coverage gap: adding aliases recovers them, enlarging
the vocabulary does not.

Why it matters beyond tidiness: XiaChuFang is 31% of the corpus. Chinese recipes
are silently reduced toward the generic ingredients the lexicon can see, which
truncates the global co-occurrence statistics that `sgns-cooc`, `svd-ppmi`,
`glove` and `ease` all learn from, and **confounds any comparison of model
quality across languages**. The English rate is also not what it looks like —
most of its 4.9% is section headers (`Filling`, `Topping`, `FOR THE CAKE:`)
correctly discarded, so the true English lexicon gap is smaller still and the
real asymmetry is wider than the table suggests.

One trap, recorded because the first attempt fell into it: tokens-per-recipe
over lines-per-recipe says English 90% and Chinese 89% and concludes there is no
gap. Chinese recipes list more ingredients, so a larger proportional loss lands
on a similar token count. Recipe length is the confound; measuring per line and
splitting by cause is what exposes the effect.

## Fixing it: a quantity bug, then aliases

The alias gap was real but it was not the whole story. Mining the unmapped terms
surfaced a worse defect in the upstream Chinese quantity stripper: both of its
prefix groups are optional, so its measure-word character class matched
**anywhere in a word, including the middle**. `培根` (bacon) became `培`,
`巧克力` (chocolate) became `巧力`, `面包` (bread) became `面`. **Twelve of the
346 Chinese lexicon entries could never be matched by construction** — chocolate,
bread, noodle, bacon, cabbage, broth, chicken broth, matcha among them. They were
not missing from the lexicon; they were being destroyed before lookup.

`ingredient_model/data/normalizer.py` anchors the pattern to both ends of the
string. Two constraints in it are not obvious, and each is pinned by a test that
fails without it:

- a Chinese **numeral** must be followed by a measure word, or `八角` (star
  anise) reads as "8 corners" and `三文鱼` (salmon) as "3 …";
- a **bare leading measure character must not be stripped**, or `包菜`
  (cabbage) becomes `菜`.

The fix damages **0** lexicon entries and also repairs cases upstream missed
(`1瓶啤酒`, `两瓣蒜`, `100g低粉`). It patches the base normaliser in memory and
never mutates the upstream tools.

Aliases were then proposed by `gpt-oss-120b` on Azure, with one design choice
that makes hallucination survivable: **the model is never shown the vocabulary
and never asked to choose from it.** It only translates a term to a plain English
ingredient name, and that name is bound through llmmm's own English alias table.
An invented concept fails to bind and is discarded automatically. Wrong-but-
bindable answers are still possible; this bounds the failure mode, it does not
eliminate it.

Measured on 25,000 recipes per language, same recipes across all three variants:

| lang | upstream | + regex fix | + aliases | tokens/recipe |
|---|---|---|---|---|
| en | 94.9% | 94.9% | **95.7%** | +1.0% |
| ru | 95.0% | 95.0% | **98.8%** | +4.0% |
| zh | 86.1% | 87.1% | **93.1%** | +8.6% |
| **all** | 92.3% | 92.7% | **96.1%** | **+4.2%** |

Chinese unrecognised lines **halved, 13.9% → 6.9%**.

The A/B is not decoration. Russian first measured **+0.0% despite 342 aliases**:
the foreign matcher lowercases text before matching and the generated keys were
capitalised, so they could never fire. Nothing else in the pipeline would have
caught that — the aliases loaded, validated and bound correctly, and reported a
gain of nothing. Without the measurement it would have shipped as a win.

What is still **unanswered**: whether any of this changes the leaderboard. Every
family reads the same corpus, so the relative order is probably stable — but
"probably" is a hypothesis, and it stays open until the corpus is rebuilt and
the families retrained on it.

### The rebuilt corpus

`make rebuild` writes `recipe_ids_v2.npz`. Before trusting it, the same script
was run in `--upstream` mode to replay the *original* normalisation, and the
output was diffed element-wise against the shipped corpus:

```
flat       identical  (35783309,) uint16      offsets  identical  (4647848,) int64
itos       identical  (1790,)     <U29        lang     identical  (4647847,) <U3
source     identical  (4647847,)  <U20
```

Byte-for-byte across 35.8M ingredient slots. Counts matching would not have been
enough — a reader that drops one column and adds another keeps the count — so
the check is element-wise, and it is what justifies the real rebuild.

The rebuild itself:

| | before | after |
|---|---|---|
| recipes | 4,647,847 | 4,653,430 (+0.12%) |
| ingredient slots | 35,783,309 | **36,707,624 (+2.58%)** |
| per recipe | 7.70 | 7.89 |
| matched nothing | 22,182 | 16,599 |

**924,315 ingredient occurrences recovered**, and 5,583 recipes that previously
matched nothing at all. The vocabulary is unchanged at 1,790 concepts — nothing
was invented, the same concepts are simply reachable now. Per language:

| lang | recipes | before | after | |
|---|---|---|---|---|
| zh | 1,440,314 | 6.63 | 7.10 | **+7.0%** |
| ru | 162,815 | 8.53 | 8.87 | +4.0% |
| en | 2,799,449 | 8.11 | 8.17 | +0.7% |

The 25k-recipe A/B predicted +8.6% / +4.0% / +1.0%. It called Russian exactly
and was within 1.6pp on Chinese, which is the useful result here: the cheap
measurement forecast the expensive one, so it can be trusted for the next round
of alias work without rebuilding the corpus each time. The corpus-wide figure
(+2.58%) is lower than the A/B's +4.2% because the A/B sampled languages
equally rather than by their true share.

**Not yet done, and it is the point of the whole exercise:** `recipe_ids_v2.npz`
has 5,583 more rows than the old corpus, so row indices no longer align. The
splits and the text index must be rebuilt against it before any family is
retrained, and until that happens the question this fix exists to answer —
*does better coverage change the ranking?* — is still open.

## Layout

```
ingredient_model/       core library — no model implementations live here
    config.py           paths and the global seed
    spec.py             TrainContext / TrainResult — the contract
    registry.py         @register + folder auto-discovery
    data/               corpus, graphs, labels, and splits.py (the leakage rule)
    eval/               metrics, harness, M6, reporting
    reasoning/          evidence blending, explanations, near-duplicate filtering
    experiments.py      declarative sweep runner
models/                 one folder per model type — see models/README.md
    sgns_walk/          SGNS over random walks       (3 models)
    factorization/      SVD / GloVe                  (3)
    recipe_basket/      EASE / iALS / item2vec       (3)
    graph_neural/       LightGCN / SGC               (2)
    set_transformer/    masked ingredient modelling  (1)
    text_embedding/     hosted text vectors, cold-start (2)
    hybrid/             blends of the above          (2)
cloud/                  Azure ML submission — see cloud/README.md
experiments/            declarative sweeps, versioned with the code
tests/                  the leakage rule and the product rules are asserted
```

## Reasoning

An embedding answers "how similar" with a number. That is not an explanation,
and on its own it is not even the best available answer — the prior study found
raw co-occurrence statistics rank held-out pairs at AUC 0.984 against the best
embedding's 0.873.

So the Reasoner treats the model as one witness among several, and surfaces
disagreement rather than averaging it away:

```bash
im explain tomato basil
im neighbors ease-rh tomato -k 10
```

```
  tomato  <->  basil
    blended score        +0.323
    co-occurrence NPMI   +0.348   (53,000 recipes together)
    embedding cosine     +0.286
    chemistry overlap    +0.486   (0 shared compounds)
    connected through:
      oregano  olive_oil  garlic  pasta
```

Chemistry is carried at **weight 0**. The prior study falsified the hypothesis
that it adds anything (H5), so it is shown for transparency and given no
influence — a deleted signal would not record that it was tested and rejected.

## Development

```bash
./.venv/bin/python -m pytest tests/ -q
```

The tests assert the two things that fail *silently* if broken: the leakage rule
(`test_splits.py`), and that M6's test recipes really are unseen and really are
sampled across all corpus sources rather than sliced by index
(`test_completion_corpus.py`). Both failures would make every model look better.
