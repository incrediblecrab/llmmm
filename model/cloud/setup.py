"""One-time Azure setup: verify access, record the subscription, size the cluster.

Run this before the first submission. It checks what actually exists rather than
assuming, because the failure mode otherwise is a job that queues forever
against a cluster that cannot allocate a node.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .config import CONFIG_PATH, MAX_VCPUS, AzureConfig


def _az(*args: str, timeout: int = 60) -> dict | list | None:
    """Call the Azure CLI with a timeout.

    A timeout is not optional here: `az ml` calls against a cold workspace can
    hang for minutes with no output, and an interactive setup script that
    appears to have frozen gets killed and re-run, which is worse.
    """
    try:
        out = subprocess.run(["az", *args, "-o", "json"], capture_output=True,
                             text=True, timeout=timeout)
    except FileNotFoundError:
        raise SystemExit("the Azure CLI is not installed — see "
                         "https://aka.ms/azure-cli")
    except subprocess.TimeoutExpired:
        print(f"  ! `az {' '.join(args)}` timed out after {timeout}s")
        return None
    if out.returncode != 0:
        print(f"  ! az {' '.join(args)}: {out.stderr.strip().splitlines()[-1:]}")
        return None
    return json.loads(out.stdout) if out.stdout.strip() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cloud.setup", description=__doc__)
    ap.add_argument("--init", action="store_true",
                    help="write config.json from the active az login")
    ap.add_argument("--check", action="store_true", help="verify access only")
    a = ap.parse_args(argv)

    cfg = AzureConfig.load()
    print("account")
    acct = _az("account", "show", timeout=30)
    if not acct:
        print("  not logged in — run `az login`")
        return 2
    print(f"  {acct.get('name')}  ({acct.get('id')})")
    if a.init or not cfg.subscription_id:
        cfg.subscription_id = acct.get("id", "")

    print(f"\nworkspace {cfg.workspace_name}")
    ws = _az("ml", "workspace", "show", "-n", cfg.workspace_name,
             "-g", cfg.resource_group, timeout=120)
    if ws:
        print(f"  {ws.get('location')}  storage={str(ws.get('storage_account','')).rsplit('/',1)[-1]}")
    else:
        print("  not reachable. Either it does not exist, or `az ml` is slow "
              "on this subscription — the portal is the faster check.")

    print("\nquota")
    print(f"  {cfg.vm_size} x {cfg.max_nodes} = "
          f"{cfg.vcpus_per_node * cfg.max_nodes}/{MAX_VCPUS} dedicated vCPUs")
    print("  low-priority quota on this subscription is 0 — spot is unavailable")
    print("  no usable GPU family — every model must be CPU-viable")

    for p in cfg.validate():
        print(f"  ! {p}")

    if a.check:
        return 0
    cfg.save()
    print(f"\nwrote {CONFIG_PATH}")
    print("  next: python -m cloud.submit experiments/baselines.yaml --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
