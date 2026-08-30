#!/usr/bin/env python3
"""Report Azure ML job and cluster state for the Cookbook Regression workspace.

    python tools/azure_status.py             # recent jobs + cluster + quota
    python tools/azure_status.py --watch     # poll until nothing is running
    python tools/azure_status.py --log NAME  # tail one job's driver log

Surfaces cluster allocation errors explicitly: a job can sit in Queued
indefinitely because the cluster silently failed to scale, and that error is
attached to the compute rather than the job.
"""
from __future__ import annotations

import argparse
import subprocess
import time

from azure.ai.ml import MLClient

from azure_setup import CLUSTER, REGION, RG, SUB, WS, cred

LIVE = {"Queued", "Starting", "Preparing", "Running", "Finalizing",
        "NotStarted", "Provisioning"}


def client() -> MLClient:
    return MLClient(cred, SUB, RG, WS)


def cluster_errors() -> list[str]:
    r = subprocess.run(
        ["az", "rest", "--method", "get", "--url",
         f"https://management.azure.com/subscriptions/{SUB}/resourceGroups/{RG}"
         f"/providers/Microsoft.MachineLearningServices/workspaces/{WS}"
         f"/computes/{CLUSTER}?api-version=2023-04-01",
         "--query", "properties.properties.errors[].error.code", "-o", "tsv"],
        capture_output=True, text=True)
    return [x for x in r.stdout.strip().splitlines() if x]


def snapshot(ml: MLClient, n: int = 8) -> list:
    jobs = list(ml.jobs.list())[:n]
    try:
        c = ml.compute.get(CLUSTER)
        print(f"cluster  {c.size} max={c.max_instances}  ", end="")
    except Exception:
        print("cluster  <missing>  ", end="")
    errs = cluster_errors()
    print(f"errors={errs if errs else 'none'}")
    for j in jobs:
        print(f"  {j.status:<12} {j.display_name or '':<22} {j.name}")
    return jobs


def tail_log(name: str) -> None:
    subprocess.run(["az", "ml", "job", "stream", "-n", name, "-g", RG, "-w", WS],
                   check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--log", help="stream a job's logs by name")
    a = ap.parse_args()

    ml = client()
    if a.log:
        tail_log(a.log)
        return

    while True:
        print(time.strftime("%H:%M:%S"))
        jobs = snapshot(ml)
        if not a.watch or not any(j.status in LIVE for j in jobs):
            break
        time.sleep(a.interval)
        print()


if __name__ == "__main__":
    main()
