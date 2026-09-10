# Browser recipe demo

The demo filters a small public recipe catalog, then runs the released supervised
ranker in JavaScript. It does not need the private corpus, a Python backend or an
API key. The source recipe text, license notices and revision links live in
[`../demo_data/`](../demo_data/).

The weights are downloaded from the exact public model commit recorded in
[`huggingface_recipe_search_release.json`](../results/huggingface_recipe_search_release.json).
The builder checks the original safetensors and configuration hashes before
exporting their numbers to JSON. The full published vocabulary is retained:
changing its size would change the feature normalizer. Document frequencies
and recipe counts describe this sample only, not the original training corpus.

## Build and run locally

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

## Publish

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

Only after the live app works does it add demo links to the model card. All
non-README model files and existing model tags must remain unchanged. A failed
attempt leaves its intermediate upload receipts in the chosen artifact
directory; it does not write a successful final receipt.

Use a new artifact directory and receipt for each release. `--update` permits an
inspected update of the existing demo repositories; changed dataset bytes require
a new `--dataset-tag`. Old tags are never moved. Regeneration and hosting stay
local/Hugging Face; no GitHub Actions are used.

## Scope

The demo retrieves existing recipes; it is not a recipe-writing model. Ingredient
selection is a canonical-name picker, not the ingredient predictor. Filters
cannot certify allergy safety, source time estimates have not been timed
independently, and servings do not scale quantities. Unmapped source ingredients
remain visible and are not silently turned into guesses.

The sample is not a held-out benchmark. Numerical parity establishes that the
browser executes the same ranking calculation, not that users prefer its meals.
The full-catalog recovery measurements must not be applied to this demo.
