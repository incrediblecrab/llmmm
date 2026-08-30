#!/usr/bin/env python3
"""Submit Epicure embedding training jobs to Azure ML.

Provisioning lives in azure_setup.py; this module only submits work, and imports
its configuration from there so the two cannot drift apart.

    python tools/azure_train.py --check                  # validate, submit nothing
    python tools/azure_train.py cooc --epochs 1          # one variant
    python tools/azure_train.py                          # cooc, chem, core

Each variant is a separate job rather than one job looping over three variants,
so that a failure in `chem` does not lose `cooc` and `core`, and so all three
occupy the cluster concurrently. Three 2-vCPU nodes exactly saturate the 6-vCPU
regional ceiling, so three variants is also the maximum useful fan-out.

Inputs are mounted read-only from blob and outputs are written straight back to
blob, so no artefact round-trips through the local machine.
"""
from __future__ import annotations

import argparse
import sys

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential

from azure_setup import (CLUSTER, CLUSTER_MAX, ENVIRONMENT, RG, SUB, WS,
                         cred)

VARIANTS = ("cooc", "chem", "core")
EXPERIMENT = "cookbook-regression"


def client() -> MLClient:
    return MLClient(cred, SUB, RG, WS)


def build_job(name: str, script: str, args: str, inputs_extra: dict | None = None,
              tags: dict | None = None,
              graph: str = "ii_graph_train.npz") -> command:
    """Generic single-node CPU job.

    Everything the program runs -- SGNS training, matrix factorisation, the
    cuisine analysis -- is one script plus an argument string, so the queue can
    stay declarative and every phase takes the identical, proven code path.

    `graph` defaults to the train split. Any value must be a split that already
    has the held-out edges removed, or M4 is void.
    """
    return command(
        display_name=f"epicure-{name}",
        experiment_name=EXPERIMENT,
        code="./tools",
        command=("export EPICURE_DERIVED=${{inputs.graphs}} && "
                 "export EPICURE_OUT=${{outputs.models}} && "
                 f"export EPICURE_II_GRAPH={graph} && "
                 f"python {script} {args}"),
        inputs={
            "graphs": Input(type=AssetTypes.URI_FOLDER, path=GRAPHS_URI,
                            mode=InputOutputModes.DOWNLOAD),
            **(inputs_extra or {}),
        },
        outputs={
            # UPLOAD, not RW_MOUNT: the FUSE write-mount fails with a 409 that
            # rslex reports as an auth error. We write once at the end, so a
            # mount buys nothing.
            "models": Output(type=AssetTypes.URI_FOLDER,
                             path=f"azureml://datastores/{OUT_DATASTORE}"
                                  f"/paths/epicure-models/{name}/",
                             mode=InputOutputModes.UPLOAD),
        },
        environment=ENVIRONMENT,
        compute=CLUSTER,
        tags={"name": name, "script": script, **(tags or {})},
    )


def build(variant: str, epochs: int, tag: str, graph: str,
          graphs_uri: str, ii_repeat: float | None = None) -> command:
    """SGNS training job (H1/H5). EPICURE_II_GRAPH defaults to the train split:
    training on the full graph would leak held-out edges and void M4."""
    name = f"{variant}{tag}"
    args = (f"{variant} --device cpu --epochs {epochs}"
            + ("" if ii_repeat is None else f" --ii-repeat {ii_repeat}"))
    return build_job(
        name, "train_epicure.py", args,
        tags={"variant": variant, "epochs": str(epochs), "graph": graph,
              **({} if ii_repeat is None else {"ii_repeat": str(ii_repeat)})})


# The workspace's own datastore. Mounting the separate cookingrecipes account
# fails inside the job with a 409 that rslex reports as an auth error, and the
# 1.2MB of graphs is not worth debugging FUSE over -- the bulk corpus stays in
# cookingrecipes, only the small training inputs are mirrored here.
OUT_DATASTORE = "workspaceblobstore"
GRAPHS_URI = f"azureml://datastores/{OUT_DATASTORE}/paths/epicure-graphs/"


def graphs_version(ml: MLClient) -> str:
    """Pin an explicit version. The @latest label is not resolvable inside a
    job input, and pinning also records exactly which data a result used."""
    return max((d.version for d in ml.data.list(name="epicure-graphs")), key=int)


def preflight(ml: MLClient, n_jobs: int) -> None:
    """Fail loudly here rather than leaving jobs wedged in the queue."""
    c = ml.compute.get(CLUSTER)
    print(f"compute      {c.name}  {c.size}  max={c.max_instances}")
    if n_jobs > c.max_instances:
        print(f"  note: {n_jobs} jobs > {c.max_instances} nodes; "
              f"{n_jobs - c.max_instances} will queue")
    print(f"input        {GRAPHS_URI}")
    print(f"environment  {ENVIRONMENT.rsplit('/', 3)[1]}  (prebuilt)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--tag", default="", help="suffix for the output model name")
    ap.add_argument("--graph", default="ii_graph_train.npz",
                    help="graph file inside the mount; the train split by "
                         "default so held-out edges stay unseen")
    ap.add_argument("--ii-repeat", type=float, default=None,
                    help="core only: I-I mixing weight for the H1 sweep")
    ap.add_argument("--check", action="store_true",
                    help="run preflight only, submit nothing")
    a = ap.parse_args()

    unknown = set(a.variants) - set(VARIANTS)
    if unknown:
        sys.exit(f"unknown variants: {sorted(unknown)}; expected {VARIANTS}")

    ml = client()
    preflight(ml, len(a.variants))
    uri = GRAPHS_URI
    if a.check:
        print("\npreflight only; nothing submitted")
        return

    submitted = []
    for v in a.variants:
        r = ml.jobs.create_or_update(
            build(v, a.epochs, a.tag, a.graph, uri, a.ii_repeat))
        submitted.append(r)
        print(f"\nsubmitted {v}: {r.name}\n  {r.studio_url}")
    print(f"\n{len(submitted)} job(s); poll with tools/azure_status.py")


if __name__ == "__main__":
    main()
