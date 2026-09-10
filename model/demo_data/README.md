---
pretty_name: llmmm public recipe demonstration catalog
language:
  - en
license: cc-by-sa-4.0
task_categories:
  - text-retrieval
size_categories:
  - n<1K
tags:
  - recipes
  - wikibooks
  - demonstration
configs:
  - config_name: default
    data_files:
      - split: demo
        path: recipes.jsonl
---

# Public recipe demonstration catalog

[Try the browser demo](https://huggingface.co/spaces/incrediblecrab/llmmm-recipes-demo)
or read the [model and its evidence](https://huggingface.co/incrediblecrab/llmmm-recipes).

This is a small public demonstration catalog for the llmmm browser demo.
It contains 12 separately sourced English Wikibooks Cookbook recipes.
It is **not the original 4,653,430-row (4.65m) training dataset** and **not a
held-out quality benchmark**. No private corpus was used to create it.
The 83.5% full-catalog evaluation result must not be attributed to this sample.
Demo-catalog statistics describe only these records.

## Source and license

The adapted recipe text and this catalog are distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The [Wikibooks copyright policy](https://en.wikibooks.org/w/index.php?title=Wikibooks%3ACopyrights&oldid=4622060),
[Wikimedia Terms of Use, section 7](https://foundation.wikimedia.org/w/index.php?title=Policy%3ATerms_of_Use&oldid=554823#7._Licensing_of_Content),
and [license legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)
describe attribution, share-alike, change notices and license notices.
Each selected revision's HTML explicitly links to CC BY-SA 4.0. Recipe text,
talk-page snapshots and earliest revisions were reviewed for external-source
notices. Images, captions and other media are not included.

Credit belongs to the English Wikibooks contributors to each linked page.
Every record contains its revision URL, contributor-history URL, attribution,
license and change notice. Oat Porridge's earliest edit credits the
[English Wikipedia Porridge contributors](https://en.wikipedia.org/w/index.php?title=Porridge&action=history);
that additional credit is retained. `sources.json` records the evidence URLs,
actual fetched-byte SHA256s, retrieval timestamps and line-by-line mappings.

When redistributing, retain the attribution, source, license and change notices,
including the additional Wikipedia credit. Keep a link to the license, indicate
further changes, and license adaptations under CC BY-SA 4.0 (or a permitted
later/compatible license). Do not impose additional restrictions. The recipe
content is not relicensed under the application code's license.

## Extraction and limitations

Every source Ingredients bullet and Procedure step is retained in order as
plain text. Wiki links, formatting and images are removed; entities and
whitespace are normalized. Explicit temperature-template arguments are
displayed, not converted. Introductions, optional notes/variations, nutrition
tables, media and category boilerplate are outside the extraction.
Source measurements, ranges, alternatives and unmeasured seasonings remain
unchanged, including apparent source inconsistencies. No quantities, units,
instructions or missing ingredient lines are invented.

`canonical_ingredients` uses reviewed names from the public
[v0.4.0-recipe-search vocabulary](https://huggingface.co/incrediblecrab/llmmm-recipes/resolve/v0.4.0-recipe-search/config.json).
Some mappings are deliberately broad: lemon juice becomes `lemon`, red lentils
become `lentil`, and chocolate chips become `chocolate`. Explicit alternatives,
optional items and named mixture examples can all occur in the canonical set;
this does not mean that every alternative is required. Unspecified mixture
contents and unavailable names are flagged in `unmapped_ingredients`, which
contains the original line even when that line is partly mapped. Ingredient
sets do not cover every later serving suggestion or all constituents of a
compound ingredient. Raw ingredient lines remain authoritative; there are no
guessed quantity arrays.

**Ingredient exclusions are not allergy-safety certification.** Incomplete
ingredient coverage, substitutions, product composition and cross-contact are
not resolved here. Read the full source, product labels and relevant safety
guidance. These recipes have not been kitchen-tested or independently validated.

## Missing metadata

9 recipes have an explicit source-reported overall summary time;
8 of those report at most 30 minutes.
Hours may be converted to minutes, but preparation, cooking, resting and
per-item times are never added or multiplied. An explicitly labeled total is
used when supplied. Missing or ambiguous overall times stay `null`; the
available exact summary text is kept in `time_evidence`. Thus Waffles has
component-time evidence but no total; Red Lentil Soup and Apple Crisp have
step-level cooking times but no overall time.

Only unambiguous source serving counts become numbers. Ranges remain `null`
(for example, Potato Curry's `4-6`); no endpoint or midpoint is chosen.
Cookie/waffle counts and other item yields are not treated as servings.
`servings_evidence` retains an available exact serving field; otherwise it is
`null`. Per-recipe explanations and original summary fields are in `sources.json`.

## Reproduce and verify

The [source repository](https://github.com/incrediblecrab/llmmm) contains the
[curation script](https://github.com/incrediblecrab/llmmm/blob/main/model/scripts/curate_recipe_demo.py)
and its declared recipe revisions. Run from that repository's root using the
model environment:

```sh
model/.venv/bin/python model/scripts/curate_recipe_demo.py
model/.venv/bin/python model/scripts/curate_recipe_demo.py --verify
model/.venv/bin/python model/scripts/curate_recipe_demo.py --verify --offline
model/.venv/bin/python -m pytest model/tests/test_recipe_demo_data.py -q
```

The first command fetches only the declared public sources, verifies pinned
recipe/config hashes and page licenses, cross-checks raw text against rendered
lists, and generates `recipes.jsonl`, this card and `sources.json`. Requests are
bounded and HTTP failures stop without automatic retries. Source responses are
cached only under the Git-ignored `.artifacts/demo-sources/`.

`--verify` checks the distributed files without network access or a source
cache. `--verify --offline` additionally regenerates in memory from the cached
responses and requires byte-identical outputs. `--offline` alone regenerates
files from that cache. Fresh retrievals retain pinned recipe content but have
new retrieval timestamps and potentially different rendered-HTML hashes;
retaining the original cache permits byte-for-byte reproduction.
