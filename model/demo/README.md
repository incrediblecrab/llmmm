# Browser recipe demo

The [demo](https://huggingface.co/spaces/incrediblecrab/llmmm-recipes-demo) searches the full public index, then runs the released supervised ranker in JavaScript. Result cards show each recipe's recorded title and original ingredient lines. It needs no private corpus, Python backend or API key. The [release receipt](../results/huggingface_recipe_card_release.json) pins the live dataset, Space, model-card and source revisions and records the browser checks. The earlier ingredient-only app remains pinned by [its receipt](../results/huggingface_ingredient_demo_release.json) and the Space tag `v0.2.0-ingredient-search`.

The original twelve-recipe Wikibooks sample is kept for local previews only. Its complete recipe text, license notices and revision links live in [`../demo_data/`](../demo_data/). Its Hugging Face dataset, `incrediblecrab/llmmm-recipe-sample`, is retired and nothing here publishes it; [`incrediblecrab/llmmm-recipe-ingredients`](https://huggingface.co/datasets/incrediblecrab/llmmm-recipe-ingredients) is the project's only public dataset. The sample and full-index modes share the ranking implementation.

The weights are downloaded from the exact public model commit recorded in [`huggingface_recipe_search_release.json`](../results/huggingface_recipe_search_release.json). The builder checks the original safetensors and configuration hashes before exporting their numbers to JSON. The full published vocabulary is retained: changing its size would change the feature normalizer. The sample mode uses sample frequencies; the full-index mode uses full-corpus frequencies.

## Build the licensed sample locally

From the repository root, with the model environment prepared:

```bash
model/.venv/bin/pip install -e './model[demo,dev]'
model/.venv/bin/python model/scripts/build_recipe_demo.py --out .artifacts/demo-preview
python3 -m http.server 7860 --bind 127.0.0.1 --directory .artifacts/demo-preview
```

Open `http://127.0.0.1:7860`. The build needs Node as well as Python because it compares browser feature values, scores and constrained search results against the existing Python implementation. It uses only public weights and the tracked public sample. Choose a fresh output path for another build; outputs are not overwritten. Add `--ipv4` to the build command if the Hub's IPv6 route fails on your network.

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

The browser suite covers desktop and mobile layouts, actual ranking, original recipe text, hard constraints, input privacy and explicit failure states. It deliberately corrupts a policy response and aborts a catalog download. Both must disable the app rather than substitute fallback recommendations.

Serving assets are checksum-verified. Hugging Face adds its own `window.huggingface` variables bootstrap to static HTML; the browser checks remove only that recognized bootstrap before comparing the served HTML to the source. Other asset bytes must match exactly. The Hub renders `README.md` as HTML rather than serving its original Markdown. Its source hash is recorded under `documentation_files` and checked through the exact Hub revision, separately from browser assets.

## Sample scope

The demo retrieves existing recipes; it is not a recipe-writing model. Ingredient selection is a canonical-name picker, not the ingredient predictor. Filters cannot certify allergy safety, source time estimates have not been timed independently, and servings do not scale quantities. Unmapped source ingredients remain visible and are not silently turned into guesses.

The sample is not a held-out benchmark. Numerical parity establishes that the browser executes the same ranking calculation, not that users prefer its meals. The full-catalog recovery measurements must not be applied to this demo.

## Full index with recipe cards

The full mode scans every canonical ingredient record, including singletons, duplicates and records without readable instructions. A result card shows the record's recorded title and original ingredient lines, with the amounts the source recorded, beside its canonical matches and recipe link. Cooking instructions, descriptions, author names and images are not copied; the recipe link leads to the steps. Known overall time and servings must carry their source-reported status; unknown values remain unknown.

Of the 4,653,430 records, 4,503,160 have a recorded title and 4,503,161 have ingredient lines, 39,045,568 lines in all. A card without a title shows its canonical ingredient names instead, and a card without lines says so. The export decodes complete HTML character references, collapses whitespace, turns HTML ingredient tables into one `name: amount` line per row and writes lines recorded as name and amount fields (povarenok-detail, taiwan-1.8k) as `name: amount`. The private catalog wrote each 03-povarenok amount recorded as null as `None` (`Майонез: None`); the export drops it, so the card shows `Майонез`. A whole list recorded on one line becomes one line per item only where it is joined with ` ; ` (or `|` in filipino-2k); commas never split a line. Wording is otherwise unchanged. Some source lines carry a unit without its amount, such as the XiaChuFang line `克盐` ("g salt"), and some RecipeNLG lines lost their fraction slash in the source dataset (`12 cup sugar` for 1/2 cup); the card shows them as recorded.

The example pantries show their cards before anything large downloads. The build ranks each example over the complete index with the browser's own search code, then stores those five results with their text and links. The cards are labeled as precomputed until the index loads, and the browser checks require the live search to return the same cards.

The public dataset has 4,653,430 records and 36,707,624 ingredient slots. Seven compressed arrays total 37,892,178 bytes (36.1 MiB). The browser reconstructs offsets, verifies sorted unique sets and full-corpus document frequencies, then searches in a worker. The user must request the initial download. Loaded arrays and offsets occupy about 181 MiB; browser overhead is additional. The measured Node process footprint is not a phone memory or latency guarantee.

Titles and ingredient lines live in 2,273 checksum-bound text shards, 312,659,863 compressed bytes in all and at most 923,037 bytes each, listed in a compressed 222,126-byte manifest. The browser fetches only the shards holding the results shown, so showing all 100 results can fetch up to 100 shards; in the release's anonymous browser check this took 6.4 seconds from click to 100 cards in each viewport, one run each. Any index, link or recipe-text download interrupted by a network error is retried twice, after 0.5 and 1.5 seconds; HTTP errors and checksum mismatches are not retried. A missing or corrupted shard fails the search rather than showing cards without their text.

Only 2,292,411 records have a recorded original URL. Each card link comes from that URL by its site's rule, stated in [`recipe_links.py`](../ingredient_model/recipe_links.py) and backed by the [per-site checks](../results/recipe_link_health.json). 1,652,643 records open the recorded page over HTTPS, on the site's current host: `https://www.cookbooks.com` resets the connection, so cookbooks.com records open `https://cookbooks.com`, where their `http://` addresses redirect, and NYT Cooking records get the title slug its pages now require. 266,182 open the Internet Archive's copy of the recorded URL, for sites whose own pages failed the September 23, 2026 check and whose archived copies passed it; www.povarenok.ru passed that check but failed a September 24 recheck made while its pages were answering errors, so its records open 2025 copies. The other 373,586, from sites that passed neither, have no link, and their cards say so. The recorded URL is kept beside the link, with `http://` added where it was recorded without a scheme. Links are retrieved from 285 small, checksum-bound link shards as results need them, with bounded concurrency. The link-only filter, on by default, shows only records with a card link, since that link is the only route to the cooking steps. Disabling it includes records without links; the UI says when a link is missing. No source times or serving counts are invented, and every link is derived from a recorded URL.

Every record is checked against the hard constraints. The best 2,000 feasible records under the deterministic baseline form a shortlist for the real published supervised model. Both ranking choices use that same shortlist, and the UI discloses truncation. This is a different retrieval pipeline from the private SQLite/FTS finder: neither its recovery benchmark nor the public sample's statistics measure this mode.

### Rebuild from public data

From the repository root with the model environment prepared, fetch only the index at the revision recorded in the release receipt:

```bash
model/.venv/bin/python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from ingredient_model.ingredient_demo import build_ingredient_demo

root = Path.cwd()
release = json.loads((root / "model/results/huggingface_recipe_card_release.json").read_text())
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

This uses public weights and the public index, including its text shards, not the private SQLite catalog or original corpus arrays. Existing preview directories are not overwritten. The full dataset also provides a standard Parquet `train` split with IDs, ingredient IDs/names, source/language, times, servings, source URLs, recipe links and link statuses; titles and ingredient lines are only in the index.

### Rebuild from original inputs

With the authorized canonical corpus and private catalog restored, run:

```bash
make -C model ingredient-index
make -C model verify-ingredient-index
make -C model ingredient-demo
python3 -m http.server 7860 --bind 127.0.0.1 \
  --directory .artifacts/ingredient-demo-preview
```

The Make variables `INGREDIENT_INDEX`, `INGREDIENT_REPORT` and `INGREDIENT_DEMO` select new output paths relative to `model/`. Existing outputs are never overwritten. The two build commands require Git-ignored output directories. Their Python scripts also accept `--ipv4` for Hub downloads where applicable.

`verify-ingredient-index` independently compares the full exported arrays against the canonical corpus. It recomputes every exported link, title and ingredient line from the private catalog in ID order, validates every URL and text shard, and compares ten queries under both policies with the existing Python feature/ranking kernels. That comparison checks alignment and completeness; unit tests check the normalization rules. Its [recorded result](../results/ingredient_catalog_verification.json) is aggregate evidence only. The preview additionally binds the index to the released model's vocabulary, corpus and catalog identities.

With the preview served on port 7860:

```bash
LLMMM_DEMO_URL=http://127.0.0.1:7860 \
LLMMM_DEMO_REPORT="$PWD/.artifacts/ingredient-browser-checks.json" \
  npm --prefix model/demo run test:ingredients
```

The browser checks run in desktop and mobile layouts. Before any download, the example cards must show the stored titles, ingredient lines and links, with no numbered cooking steps. After a deliberate download, the live learned ranking must return the same cards; the checks then cover full-population coverage, 100 results, hard constraints, the baseline comparison, shortlist disclosure and input privacy. Corrupted arrays, a failed index download and corrupted recipe text must each fail visibly rather than show live-looking cards, while a single dropped recipe-text download must be retried and the search completed. The original public-sample suite is separate and still uses `test:e2e`.

**Publication authorization:** the owner confirmed permission for the complete ingredient-only release. The September 23, 2026 request to show each recipe on its card is recorded in the [scope record](../results/ingredient_catalog_publication_scope.json) as the owner's authorization to add recorded titles and ingredient lines; that is an interpretation of the request, not a new permission grant. The record is not a license for cooking instructions, other source-page prose or images, and does not relicense the model weights.

## Publish the full ingredient dataset and demo

This is the project's only publisher. It packages every canonical ingredient row as bounded Parquet shards and a compact browser index with its text shards. Before upload, a separate comparison checks every Parquet ID, ingredient name/ID, source/language value, source time, serving count and URL against the verified index, and every text shard's bytes and hash against the verified index's manifest. Only the fixed dataset/application inventories can be published.

Commit and push the implementation and authorization record first, then run from the repository root:

```bash
model/.venv/bin/python model/scripts/publish_ingredient_demo.py \
  --index .artifacts/ingredient-catalog-v7 \
  --source-revision "$(git rev-parse HEAD)" \
  --out .artifacts/hf/recipe-link-release \
  --report model/results/huggingface_recipe_link_release.json \
  --public --update --dataset-tag v0.3.0 --space-tag v0.4.0-recipe-links
```

Use new artifact/receipt paths for another attempt; add `--ipv4` if needed. The public dataset is `incrediblecrab/llmmm-recipe-ingredients`. It is the project's only public dataset. Its train split contains ingredient facts and recipe links; its index adds the recorded titles and ingredient lines, not cooking instructions.

An update uploads only new or changed files. It uses one commit when at most 99 files change, following the Hub's advice to keep manual commits to about 50–100 files. A larger update adds new files first, then replacements, in chained 99-file commits; the card, dataset manifest and index descriptor go in the final commit. Until that commit lands, the dataset's `main` branch mixes old and new files, but the Space and tags pin only the verified final revision. A rerun compares against the current `main` again and uploads what is still missing.

The Space downloads the index from an immutable public dataset revision rather than storing a second full copy. Its manifest also pins the source commit, model and application bytes. Browser checks run locally against the public dataset, then anonymously on the actual Static Space origin. The dataset viewer must expose the complete row count and matching fields; while it still serves the previous release's rows, the publisher waits up to 15 minutes. Only then are the model card's demo links updated; model files and existing model tags are preserved.

Both release tags must be new, and every existing dataset and Space tag must still point at the same commit afterwards. The existing sample Space revision is preserved under `v0.1.0-sample` and the ingredient-only app under `v0.2.0-ingredient-search`; the dataset's ingredient-only release keeps `v0.1.0`, and the recipe-card release keeps dataset tag `v0.2.0` and Space tag `v0.3.0-recipe-cards`. This release adds dataset tag `v0.3.0` and Space tag `v0.4.0-recipe-links`. Tags are never moved. No GitHub Actions, paid hardware or inference endpoint is requested.
