"""What is running, what it cost, and what quota is left.

Written because the Azure portal answers these questions slowly and this
subscription's limits are tight enough that the answer changes what you submit
next.
"""
from __future__ import annotations

import argparse
import sys

from .config import MAX_VCPUS, load_config
from .submit import _client


def show_quota(cfg) -> None:
    ml = _client(cfg)
    print(f"compute in {cfg.workspace_name}:")
    used = 0
    for c in ml.compute.list():
        size = getattr(c, "size", "?")
        nodes = getattr(c, "max_instances", 0) or 0
        per = {"Standard_F2s_v2": 2, "Standard_F4s_v2": 4}.get(size, 2)
        used += per * nodes
        print(f"  {c.name:<24}{size:<20}max {nodes} nodes  "
              f"{getattr(c, 'provisioning_state', '?')}")
    print(f"\n  {used}/{MAX_VCPUS} dedicated vCPUs committed")
    if used > MAX_VCPUS:
        print("  ! over quota — clusters exist that can never allocate a node")


def show_jobs(cfg, limit: int) -> None:
    ml = _client(cfg)
    print(f"\nrecent jobs in {cfg.experiment_name}:")
    for i, j in enumerate(ml.jobs.list()):
        if i >= limit:
            break
        print(f"  {j.name:<38}{getattr(j, 'status', '?'):<12}"
              f"{getattr(j, 'display_name', '')}")


def show_one(cfg, name: str) -> None:
    ml = _client(cfg)
    j = ml.jobs.get(name)
    print(f"{j.name}\n  status  {j.status}\n  {j.studio_url}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cloud.status", description=__doc__)
    ap.add_argument("job", nargs="?", help="a job name; omit for an overview")
    ap.add_argument("-n", type=int, default=10)
    a = ap.parse_args(argv)
    cfg = load_config()
    if problems := cfg.validate():
        for p in problems:
            print(f"  {p}")
        return 2
    if a.job:
        show_one(cfg, a.job)
    else:
        show_quota(cfg)
        show_jobs(cfg, a.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
