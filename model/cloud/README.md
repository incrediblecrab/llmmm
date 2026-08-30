# Azure notes

Verified live state of `cookbook-regression` (westus2), read through the ARM
REST API on the date this was written.

## What exists

| resource | detail |
|---|---|
| ML workspace | `cookbook-regression`, Succeeded |
| Compute | `cpu-cluster` — `Standard_F2s_v2` × 3 nodes, Succeeded |
| Storage | `cookingrecipes`, `cookbookstoragee947b0d86` |
| Foundry | `llmmm-foundry` (Cognitive Services) |
| Registry | `bb242b5cea774fec882f341b59db9f4c` |

The cluster is already sized to the full quota, so `cloud/submit.py` will find
it rather than create one.

## Foundry deployments

| deployment | model |
|---|---|
| `gpt-oss-120b` | gpt-oss-120b |
| `llama-3.3-70b` | Llama-3.3-70B-Instruct |
| `phi4-reasoning` | Phi-4-reasoning |
| `phi4-vision` | Phi-4-multimodal-instruct |

**There is no embedding deployment.** These are all chat/reasoning models, so:

- `models/text_embedding/` needs a `text-embedding-3-small` deployment created
  before it will run. It is otherwise ready.
- The models that *are* deployed suit the reasoning layer rather than the model
  layer — generating explanations over evidence the Reasoner has already
  assembled, not producing vectors.

## Constraints, and why they are not negotiable

These were established the expensive way in the prior study.

- **6 dedicated regional vCPUs, 0 low-priority.** Spot is not a cost saving that
  is merely unavailable; requesting it fails. `Standard_F2s_v2` × 3 is the
  entire budget.
- **No usable GPU.** The families with quota are retired hardware; the families
  that exist have zero quota. Every model in this workspace must be CPU-viable,
  which is why the compartments favour closed-form and sparse methods.
- **Use a prebuilt curated image.** Supplying a conda file makes Azure ML start
  a hidden 4-vCPU cluster to build it, consuming most of the 6-core budget and
  blocking the real job with no obvious explanation.
- **Outputs must use `UPLOAD`, not `RW_MOUNT`.** The FUSE write-mount fails with
  a 409 reported as an authentication error, which sends you debugging
  credentials that were never the problem.

## Budget

Not the binding constraint. Roughly **$4 of the $150** covers 40 node-hours on
this hardware. **Wall-clock is the real limit** — 6 vCPUs is not much, so prefer
many cheap experiments over one large one, and keep anything that fits on a
laptop on the laptop.

## `az ml` is unusable here

`az ml workspace show` and `az ml compute list` hang for over two minutes with
no output. The ARM REST API answers instantly:

```bash
TOKEN=$(az account get-access-token --query accessToken -o tsv)
SUB=<subscription-id>
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://management.azure.com/subscriptions/$SUB/resourceGroups/cookbook-regression/providers/Microsoft.MachineLearningServices/workspaces/cookbook-regression/computes?api-version=2023-10-01"
```

`cloud/setup.py` wraps CLI calls in timeouts for this reason — a setup script
that appears frozen gets killed and re-run, which is worse than one that reports
a timeout.

## Note on the package name

This directory is `cloud/`, not `azure/`. `azure` is a namespace package owned by
Microsoft's SDK; a local package with that name shadows it and breaks
`from azure.ai.ml import MLClient` in a way that is confusing to diagnose.
