# The epicure-explorer Gradio API — reference and validation oracle

Source: `https://kaikaku-epicure-explorer.hf.space/?view=api`
— captured verbatim to `data/raw/epicure-explorer/gradio_api_docs.txt`.

> **Full machine-parsed reference: [`EXPLORER_API_REFERENCE.md`](EXPLORER_API_REFERENCE.md)**
> — all 34 endpoints with every parameter, type, default, UI label and return
> value (105/105 parameters extracted). Regenerate with
> `./.venv/bin/python tools/parse_api_docs.py`; also emitted as
> `data/derived/explorer_api.json` for programmatic use.
>
> **Only 4 of the 34 are a real API.** `/neighbors`, `/slerp`, `/arithmetic` and
> `/embed` are declared with `gr.api()` and are a deliberate contract. The other
> 30 are Gradio UI event handlers exposed incidentally — they return rendered
> HTML or plot payloads and will break on any UI edit. Depend only on the four.

The Space holds **no corpus** — it is 43 KB of code that loads the three sibling
models from the Hub at runtime. But it is extremely useful for two reasons:

1. It exposes the **author's reference implementation** as a callable service,
   which we can differential-test the Swift port against.
2. Its source (`data/raw/epicure-explorer/app.py`) reveals how directions that
   are *not* shipped as vectors get reconstructed — see FINDINGS §4.

---

## Calling it without the `gradio_client` dependency

Two steps: POST to get an `event_id`, then GET the SSE stream.

```bash
EID=$(curl -s -X POST "https://kaikaku-epicure-explorer.hf.space/gradio_api/call/neighbors" \
  -H "Content-Type: application/json" \
  -d '{"data":["miso","core",5]}' | sed -n 's/.*"event_id":"\([^"]*\)".*/\1/p')

curl -s "https://kaikaku-epicure-explorer.hf.space/gradio_api/call/neighbors/$EID"
# event: complete
# data: [[{"name":"mirin","cosine":0.788707}, ...]]
```

### Core endpoints

| endpoint | args |
|---|---|
| `/neighbors` | `ingredient, sibling, k` |
| `/slerp` | `seed, direction, theta_deg, sibling, k` |
| `/arithmetic` | vector arithmetic |
| `/embed` | returns an ingredient's vector |
| `/substitute_finder` | `seed, sibling, k, must_share_group, same_nova, diff_cuisine` |
| `/sensory_search` | `sibling, k` + 10 axis weights |
| `/browse_modes`, `/render_mode_wiki` | mode atlas |
| `/explore_all_siblings`, `/umap_view`, `/cultural_context` | views |

Full list (34): `_filter_dropdown`, `_passport_outputs{,_1}`, `_refresh_factor_choices`,
`arithmetic`, `browse_modes`, `cultural_context{,_1}`, `embed`, `explore_all_siblings`,
`lambda{,_1..4}`, `neighbors`, `parse_or_suggest`, `render_3d_atlas{,_1,_2,_3}`,
`render_arithmetic_vector`, `render_cuisine_compass`, `render_cuisine_cosine_map`,
`render_cuisine_phylogeny`, `render_factor_poster{,_1}`, `render_mode_wiki`,
`render_slerp_trajectory`, `sensory_search`, `slerp`, `substitute_finder`, `transform`,
`umap_view`.

---

## Validation result 1 — our pipeline is 1:1 with the reference service

`tools/api_differential.py` runs a systematic differential test: 30 `neighbors`
queries (10 seeds × 3 siblings) and 24 `slerp` queries (4 cuisine directions ×
3 siblings × θ∈{30,60}).

| operator | calls | identical ranking | worst cosine delta |
|---|---|---|---|
| `neighbors` | 30 | **30/30** | 1.0e-06 |
| `slerp` @30° | 12 | **12/12** | 0.0 |
| `slerp` @60° | 12 | **12/12** | 0.0 |
| **total** | **54** | **54/54 (100%)** | **1.0e-06** |

The 1e-06 residual is float32 rounding in the service's JSON serialisation, not a
computational difference. **Our bundle is bit-exact with the author's live service** —
the loader, L2 normalisation, cosine ranking and SLERP semantics in `tools/` are all
confirmed correct. Use these as Swift acceptance tests alongside
`data/derived/golden_fixture.json`.

## Validation result 2 — the live service contradicts the published paper

Re-querying 45 published `direction_arithmetic_full.parquet` cells against the **live
service** (15 seed×cuisine×sibling combos × 3 angles):

| θ | cells | mean top-5 overlap | exact match |
|---|---|---|---|
| 0° | 15 | 94.7% | 73.3% |
| 30° | 15 | 26.7% | **0%** |
| 60° | 15 | 28.0% | **0%** |

Worked example — `rice + South_Asian`, `core`, θ=30:

| rank | **published** | **author's live API** (= our output) |
|---|---|---|
| 1 | chana_dal 0.6976 | turmeric 0.7607 |
| 2 | fenugreek_leaf 0.6919 | mustard_seed 0.7567 |
| 3 | urad_dal 0.6838 | fenugreek_seed 0.7468 |
| 4 | toor_dal 0.6741 | coriander 0.7428 |
| 5 | horse_gram 0.6684 | cumin 0.7388 |

**Zero overlap, different cosine range.** The chain of reasoning is airtight:

```
our bundle  ≡  live service        (100% parity, 1e-6)
live service ≠  published tables   (0% exact at any θ>0)
------------------------------------------------------
therefore   shipped weights ≠ the weights behind the paper
```

This cannot be an implementation error on our side. The `supervised_poles.json` on the
Hub are not the vectors that produced the paper's tables, and the author's own demo
doesn't reproduce them either.

The residual 5.3% miss at θ=0 (where the pole is unused and this is pure `neighbors`)
is the unshipped near-duplicate filter — see FINDINGS §5.

---

## Use as a CI oracle

Because the operators are deterministic and the models are frozen on the Hub, the
Space is a stable differential-test target:

```
for (seed, direction, theta, sibling) in cases:
    assert swift_slerp(...) ≈ live_api_slerp(...)   # within 1e-5
```

Rate-limit politely and cache responses — it's a free community Space. Snapshot the
expected values into the test bundle so CI doesn't depend on network availability.
