#!/usr/bin/env python3
"""Provision the Cookbook Regression training environment on Azure ML.

Idempotent and declarative: safe to re-run.

    python tools/azure_setup.py            # read-only status check
    python tools/azure_setup.py --apply    # converge to the desired state

Design is dictated by one hard constraint discovered the hard way:

    Total Cluster Dedicated Regional vCPUs .... 6      <- ceiling for everything
    Total Cluster LowPriority Regional vCPUs .. 0      <- spot is unavailable

Consequences baked in below:

1. Nodes are 2-vCPU (`Standard_F2s_v2`) rather than 4-vCPU, so three nodes fit
   inside the ceiling and the three model variants train concurrently instead
   of queueing. A 1,790x300 embedding is ~2 MB; 2 vCPUs is not the bottleneck.

2. The environment references a prebuilt curated image rather than supplying a
   conda file. A conda environment forces Azure ML to spin up a hidden 4-vCPU
   image-build cluster, which ate 4 of the 6 available vCPUs and left our own
   job stuck at ClusterCoreQuotaReached.

3. GPU is unreachable on this subscription, so CPU is the only target: every
   GPU family with quota (NC/NV v1) is retired hardware, and every family with
   live hardware (A10/A100/H100) has zero quota.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (AccountKeyConfiguration, AmlCompute,
                                  AzureBlobDatastore, Data, Workspace)
from azure.identity import DefaultAzureCredential

SUB = "994c9e26-012e-4a65-b6c5-c491e658d65a"
RG = "cookbook-regression"
WS = "cookbook-regression"
REGION = "westus2"
STORAGE = "cookingrecipes"
ROLE = "Storage Blob Data Contributor"

CLUSTER = "cpu-cluster"
CLUSTER_VM = "Standard_F2s_v2"      # 2 vCPU
CLUSTER_MAX = 3                     # 3 x 2 = 6 vCPU, the regional ceiling
IDLE_SECONDS = 180

# Prebuilt: referencing it means no image build and no builder cluster.
# CUDA image, but torch falls back to CPU cleanly when no GPU is present.
ENVIRONMENT = ("azureml://registries/azureml/environments/"
               "acpt-pytorch-2.2-cuda12.1/versions/55")

# blob container -> datastore name; everything the pipeline reads or writes
DATASTORES = {
    "graphs": "graphs_ds",          # training inputs (derived graphs)
    "models": "models_ds",          # trained embeddings
    "catalog": "catalog_ds",        # ingredient catalog
    "recipes": "recipes_ds",        # raw corpus, 29 sources
    "embeddings": "embeddings_ds",  # published reference embeddings
}

DATA_ASSETS = {
    "epicure-graphs": ("graphs_ds", "ii_graph.npz + flavor_graph.npz: training inputs"),
    "epicure-corpus": ("recipes_ds", "4.3M recipes across 29 flat NN-name sources"),
    "epicure-catalog": ("catalog_ds", "1,790-ingredient catalog + substitutions"),
}

cred = DefaultAzureCredential()


def account_key() -> str:
    r = subprocess.run(["az", "storage", "account", "keys", "list", "-n", STORAGE,
                        "-g", RG, "--query", "[0].value", "-o", "tsv"],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def ensure_workspace(apply: bool) -> MLClient | None:
    root = MLClient(cred, SUB, RG)
    try:
        root.workspaces.get(WS)
        print(f"[ok]      workspace {WS}")
    except Exception:
        if not apply:
            print(f"[MISSING] workspace {WS}")
            return None
        print(f"[create]  workspace {WS} ...")
        root.workspaces.begin_create(Workspace(
            name=WS, location=REGION,
            description="Epicure ingredient embeddings",
            tags={"project": "llmmm", "env": "Cookbook Regression"})).result()
    return MLClient(cred, SUB, RG, WS)


def ensure_compute(ml: MLClient, apply: bool) -> None:
    want = f"{CLUSTER_VM} min=0 max={CLUSTER_MAX}"
    existing = None
    try:
        existing = ml.compute.get(CLUSTER)
    except Exception:
        pass

    if existing is not None:
        have = f"{existing.size} min={existing.min_instances} max={existing.max_instances}"
        if (existing.size.lower() == CLUSTER_VM.lower()
                and existing.max_instances == CLUSTER_MAX):
            print(f"[ok]      compute {CLUSTER}: {have}")
            return
        print(f"[DRIFT]   compute {CLUSTER}: have {have}, want {want}")
        if not apply:
            return
        print(f"[delete]  compute {CLUSTER} (AmlCompute VM size is immutable) ...")
        ml.compute.begin_delete(CLUSTER).result()
    else:
        print(f"[MISSING] compute {CLUSTER}: want {want}")
        if not apply:
            return

    print(f"[create]  compute {CLUSTER}: {want} ...")
    ml.compute.begin_create_or_update(AmlCompute(
        name=CLUSTER, type="amlcompute", size=CLUSTER_VM,
        min_instances=0, max_instances=CLUSTER_MAX,
        idle_time_before_scale_down=IDLE_SECONDS, tier="Dedicated")).result()
    print(f"[ok]      compute {CLUSTER} created")


def ensure_datastores(ml: MLClient, apply: bool) -> None:
    have = {d.name for d in ml.datastores.list()}
    missing = {c: n for c, n in DATASTORES.items() if n not in have}
    if not missing:
        print(f"[ok]      datastores: {len(DATASTORES)} present")
        return
    if not apply:
        print(f"[MISSING] datastores: {sorted(missing.values())}")
        return
    key = account_key()
    for container, name in missing.items():
        ml.datastores.create_or_update(AzureBlobDatastore(
            name=name, account_name=STORAGE, container_name=container,
            credentials=AccountKeyConfiguration(account_key=key),
            description=f"{STORAGE}/{container}"))
        print(f"[create]  datastore {name} -> {STORAGE}/{container}")


def ensure_data_assets(ml: MLClient, apply: bool) -> None:
    have = {d.name for d in ml.data.list()}
    missing = {k: v for k, v in DATA_ASSETS.items() if k not in have}
    if not missing:
        print(f"[ok]      data assets: {len(DATA_ASSETS)} present")
        return
    if not apply:
        print(f"[MISSING] data assets: {sorted(missing)}")
        return
    for name, (ds, desc) in missing.items():
        ml.data.create_or_update(Data(
            name=name, type=AssetTypes.URI_FOLDER,
            path=f"azureml://datastores/{ds}/paths/", description=desc))
        print(f"[create]  data asset {name} -> {ds}")


def ensure_rbac(apply: bool) -> None:
    """Grant the workspace MSI blob data access on the storage account.

    Datastores registered with an account key still hand out `azureml://` URIs
    for *outputs*, and those are resolved on the compute node using the
    workspace managed identity -- not the stored key. Without this role a job
    mounts its inputs fine and then dies at the output mount with
    ScriptExecution.StreamAccess.Authentication. Subscription Owner does not
    imply data-plane access, so this must be granted explicitly.
    """
    ident = subprocess.run(
        ["az", "rest", "--method", "get", "--url",
         f"https://management.azure.com/subscriptions/{SUB}/resourceGroups/{RG}"
         f"/providers/Microsoft.MachineLearningServices/workspaces/{WS}"
         "?api-version=2023-04-01", "--query", "identity.principalId", "-o", "tsv"],
        capture_output=True, text=True).stdout.strip()
    if not ident:
        print("[warn]    could not read workspace identity; skipping RBAC")
        return
    scope = (f"/subscriptions/{SUB}/resourceGroups/{RG}/providers/"
             f"Microsoft.Storage/storageAccounts/{STORAGE}")
    have = subprocess.run(
        ["az", "role", "assignment", "list", "--assignee", ident,
         "--scope", scope, "--query", "[].roleDefinitionName", "-o", "tsv"],
        capture_output=True, text=True).stdout
    if ROLE in have:
        print(f"[ok]      rbac: workspace MSI has {ROLE}")
        return
    if not apply:
        print(f"[DRIFT]   rbac: workspace MSI missing {ROLE} on {STORAGE}")
        return
    subprocess.run(
        ["az", "role", "assignment", "create", "--assignee-object-id", ident,
         "--assignee-principal-type", "ServicePrincipal", "--role", ROLE,
         "--scope", scope, "-o", "none"], check=True)
    print(f"[create]  rbac: granted {ROLE}; allow ~70s to propagate")


def report_quota() -> None:
    r = subprocess.run(
        ["az", "rest", "--method", "get", "--url",
         f"https://management.azure.com/subscriptions/{SUB}/providers/"
         f"Microsoft.MachineLearningServices/locations/{REGION}/usages"
         "?api-version=2023-04-01",
         "--query", "value[?contains(name.value,'Total')].{n:name.localizedValue,"
                    "c:currentValue,l:limit}", "-o", "tsv"],
        capture_output=True, text=True)
    print("\nquota (westus2):")
    for line in r.stdout.strip().splitlines():
        print("   ", line.replace("\t", "  "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="make changes; default is a read-only status check")
    a = ap.parse_args()
    if not a.apply:
        print("(status check; pass --apply to converge)\n")

    ml = ensure_workspace(a.apply)
    if ml is None:
        sys.exit("workspace missing; re-run with --apply")
    ensure_compute(ml, a.apply)
    ensure_datastores(ml, a.apply)
    ensure_rbac(a.apply)
    ensure_data_assets(ml, a.apply)
    print(f"\nenvironment: {ENVIRONMENT.rsplit('/environments/', 1)[1]} "
          f"(prebuilt, no build cluster)")
    report_quota()


if __name__ == "__main__":
    main()
