# Browser recipe demo

The live [demo](https://huggingface.co/spaces/incrediblecrab/llmmm-recipes-demo)
searches the full public ingredient-only index, then runs the released supervised
ranker in JavaScript. It needs no private corpus, Python backend or API key.
The [release receipt](../results/huggingface_ingredient_demo_release.json) pins
the actual dataset and application revisions.

The original twelve-recipe sample is also retained. Its complete recipe text,
license notices and revision links live in [`../demo_data/`](../demo_data/).
The sample and full-index modes share the ranking implementation.

The weights are downloaded from the exact public model commit recorded in
[`huggingface_recipe_search_release.json`](../results/huggingface_recipe_search_release.json).
The builder checks the original safetensors and configuration hashes before
exporting their numbers to JSON. The full published vocabulary is retained:
changing its size would change the feature normalizer. The sample mode uses
sample frequencies; the full-index mode uses full-corpus frequencies.

## Build the licensed sample locally

From the repository root, with the model environment prepared:

```bash
model/.venv/bin/pip install -e './model[demo,dev]'
model/.venv/bin/python model/scripts/build_recipe_demo.py --out .artifacts/demo-preview
python3 -m http.server 7860 --bind 127.0.0.1 --directory .artifacts/demo-preview
```

Open `http://127.0.0.1:7860`. The build needs Node as well as Python because it
compares browser feature values, scores and constrained search results against
the existing Python implementation. It uses only public weights and the tracked
public sample. Choose a fresh output path for another build; outputs are not
overwritten. Add `--ipv4` to the build command if the Hub's IPv6 route fails on
your network.

## Browser checks

Install the declared development tools:

```bash
npm --prefix model/demo ci
(cd model/demo && npx playwright install chromium)
make -C model demo-test
```

With the local server running, use a second terminal:

```bash
npm --prefix model/demo run test:e2e
```

The browser suite covers desktop and mobile layouts, actual ranking, original
recipe text, hard constraints, input privacy and explicit failure states. It
deliberately corrupts a policy response and aborts a catalog download. Both must
disable the app rather than substitute fallback recommendations.

Serving assets are checksum-verified. Hugging Face adds its own
`window.huggingface` variables bootstrap to static HTML; the browser checks remove
only that recognized bootstrap before comparing the served HTML to the source.
Other asset bytes must match exactly.
The Hub renders `README.md` as HTML rather than serving its original Markdown.
Its source hash is recorded under `documentation_files` and checked through the
exact Hub revision, separately from browser assets.

## Publish the licensed sample

Commit and push the source, including the curation inputs, before publication:

```bash
model/.venv/bin/python model/scripts/publish_recipe_demo.py \
  --source-revision "$(git rev-parse HEAD)" \
  --out .artifacts/hf/recipe-demo-release \
  --report model/results/huggingface_recipe_demo_release.json \
  --public
```

The publisher creates a public dataset and a **static** Space. It never requests
paid hardware. It pins the dataset revision, builds a fresh app, runs the browser
checks locally, uploads only the serving allowlist, then repeats the checks
anonymously against the actual hosting origin returned by the Hub.
This preserves the historical sample workflow; its guards refuse to replace the
full ingredient dataset's current deployment.

Only after the live app works does it add demo links to the model card. All
non-README model files and existing model tags must remain unchanged. A failed
attempt leaves its intermediate upload receipts in the chosen artifact
directory; it does not write a successful final receipt.

Use a new artifact directory and receipt for each release. `--update` permits an
inspected update of the existing demo repositories; changed dataset bytes require
a new `--dataset-tag`. Old tags are never moved. Regeneration and hosting stay
local/Hugging Face; no GitHub Actions are used.

## Sample scope

The demo retrieves existing recipes; it is not a recipe-writing model. Ingredient
selection is a canonical-name picker, not the ingredient predictor. Filters
cannot certify allergy safety, source time estimates have not been timed
independently, and servings do not scale quantities. Unmapped source ingredients
remain visible and are not silently turned into guesses.

The sample is not a held-out benchmark. Numerical parity establishes that the
browser executes the same ranking calculation, not that users prefer its meals.
The full-catalog recovery measurements must not be applied to this demo.

## Full ingredient-only index

The full mode scans every canonical ingredient record, including singletons,
duplicates and records without readable instructions. It does not copy recipe
titles, quantities, raw ingredient sentences, instructions, images or descriptive
prose. Known overall time and servings must carry their source-reported status;
unknown values remain unknown.

The public dataset has 4,653,430 records and 36,707,624 ingredient slots.
Seven compressed arrays total 37,769,040 bytes (36.0 MiB). The browser reconstructs
offsets, verifies sorted unique sets and full-corpus document frequencies, then
searches in a worker. The user must request the initial download. Loaded arrays
and offsets occupy about 181 MiB; browser overhead is additional. The measured
Node process footprint is not a phone memory or latency guarantee.

Only 2,292,411 records have a recorded original URL. Links are retrieved from
285 small, checksum-bound shards as results need them, with bounded concurrency.
The default link-only filter avoids presenting an ingredient set as if it were
a complete cooking recipe. Disabling it includes records without links; the UI
says when a link is missing. No URLs, source times or serving counts are invented.

Every record is checked against the hard constraints. The best 2,000 feasible
records under the deterministic baseline form a shortlist for the real published
supervised model. Both ranking choices use that same shortlist, and the UI
discloses truncation. This is a different retrieval pipeline from the private
SQLite/FTS finder: neither its recovery benchmark nor the public sample's
statistics measure this mode.

### Rebuild from public data

From the repository root with the model environment prepared, fetch only the
index at the revision recorded in the release receipt:

```bash
model/.venv/bin/python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from ingredient_model.ingredient_demo import build_ingredient_demo

root = Path.cwd()
release = json.loads((root / "model/results/huggingface_ingredient_demo_release.json").read_text())
snapshot = snapshot_download(
    release["dataset"]["repository"],
    repo_type="dataset",
    revision=release["dataset"]["revision"],
    allow_patterns=["index/**"],
    token=False,
)
build_ingredient_demo(root, Path(snapshot) / "index", root / ".artifacts/public-demo-preview")
PY
python3 -m http.server 7860 --bind 127.0.0.1 \
  --directory .artifacts/public-demo-preview
```

This uses public weights and public ingredient facts, not the private SQLite
catalog or original corpus arrays. Existing preview directories are not
overwritten. The full dataset also provides a standard Parquet `train` split
with IDs, ingredient IDs/names, source/language, times, servings and source URLs.

### Rebuild from original inputs

With the authorized canonical corpus and private catalog restored, run:

```bash
make -C model ingredient-index
make -C model verify-ingredient-index
make -C model ingredient-demo
python3 -m http.server 7860 --bind 127.0.0.1 \
  --directory .artifacts/ingredient-demo-preview
```

The Make variables `INGREDIENT_INDEX`, `INGREDIENT_REPORT` and `INGREDIENT_DEMO`
select new output paths relative to `model/`. Existing outputs are never
overwritten. The two build commands require Git-ignored output directories.
Their Python scripts also accept `--ipv4` for Hub downloads where applicable.

`verify-ingredient-index` independently compares the full exported arrays against
the canonical corpus, validates every URL shard and compares ten queries under
both policies with the existing Python feature/ranking kernels. Its
[recorded result](../results/ingredient_catalog_verification.json) is aggregate
evidence only. The preview additionally binds the index to the released model's
vocabulary, corpus and catalog identities.

With the preview served on port 7860:

```bash
LLMMM_DEMO_URL=http://127.0.0.1:7860 \
LLMMM_DEMO_BUILD="$PWD/.artifacts/ingredient-demo-preview" \
LLMMM_DEMO_REPORT="$PWD/.artifacts/ingredient-browser-checks.json" \
  npm --prefix model/demo run test:ingredients
```

The browser checks cover deliberate download, full-population coverage, actual
source links, hard constraints, shortlist disclosure and corrupted-index
rejection in desktop/mobile layouts. The original public-sample suite is
separate and still uses `test:e2e`.

**Publication authorization:** the owner confirmed permission for this complete
ingredient-only release. The [scope record](../results/ingredient_catalog_publication_scope.json)
records the confirmation; it is not a license for copied source-page prose or
images, and does not relicense the model weights.

## Publish the full ingredient dataset and demo

The full publisher is separate from the twelve-recipe sample publisher. It
packages every canonical ingredient row as bounded Parquet shards and a compact
browser index. Before upload, a separate comparison checks every Parquet ID,
ingredient name/ID, source/language value, source time, serving count and URL
against the verified index. Only the fixed dataset/application inventories can
be published.

Commit and push the implementation and authorization record first, then run
from the repository root:

```bash
model/.venv/bin/python model/scripts/publish_ingredient_demo.py \
  --index .artifacts/ingredient-catalog-v2 \
  --source-revision "$(git rev-parse HEAD)" \
  --out .artifacts/hf/ingredient-demo-release \
  --report model/results/huggingface_ingredient_demo_release.json \
  --public --update
```

Use new artifact/receipt paths for another attempt; add `--ipv4` if needed.
The public dataset is `incrediblecrab/llmmm-recipe-ingredients`. Its train split
contains ingredient facts, not the separately sourced sample's instructions.
The sample dataset is preserved unchanged.

The Space downloads the index from an immutable public dataset revision rather
than storing a second full copy. Its manifest also pins the source commit,
model and application bytes. Browser checks run locally against the public
dataset, then anonymously on the actual Static Space origin. The dataset viewer
must expose the complete row count and matching fields. Only then are the model
card's demo links updated; model files and existing model tags are preserved.

The existing sample Space revision is preserved under `v0.1.0-sample`.
The ingredient-only app uses `v0.2.0-ingredient-search`. Tags are never moved.
No GitHub Actions, paid hardware or inference endpoint is requested.
