#!/usr/bin/env python3
"""Mirror the flattened corpus layout into Azure blob storage, server-side.

The container was uploaded before the local flatten, so it still has the old
shape: `recipes/recipe/expansion/<name>/...` -- a redundant `recipe/` prefix
inside a container already called `recipes`, plus the `expansion/` nesting.

Target is `recipes/<NN>-<name>/...` matching the local tree.

Every copy is server-side (blob URL -> blob URL in the same account), so this
moves no bytes over the local network regardless of corpus size.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = "cookingrecipes"
CONTAINER = "recipes"
BASE = f"https://{ACCOUNT}.blob.core.windows.net/{CONTAINER}"
LEDGER = ROOT / "recipe" / "_layout_migration.json"


def mapping() -> dict[str, str]:
    """old blob prefix -> new blob prefix, composing both migration steps."""
    led = json.loads(LEDGER.read_text())
    # step 2 renames, applied on top of step 1 moves
    renamed = {r["from"]: r["to"] for r in led.get("renames", [])}
    m = {}
    for x in led["moves"]:
        if x["from"].startswith("expansion/") and not x["to"].startswith("_"):
            final = renamed.get(x["to"], x["to"])
            m[f"recipe/{x['from']}"] = final
    # the nine baseline sources only lose the redundant recipe/ prefix
    for d in sorted(p.name for p in (ROOT / "recipe").iterdir()
                    if p.is_dir() and p.name[:2].isdigit() and int(p.name[:2]) < 10):
        m[f"recipe/{d}"] = d
    return m


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout + r.stderr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    m = mapping()
    for old, new in sorted(m.items(), key=lambda kv: kv[1]):
        print(f"  {old}/  ->  {new}/")
    print(f"\n{len(m)} server-side copies")
    if not a.apply:
        print("dry run; pass --apply to execute")
        return

    import os
    os.environ["AZCOPY_AUTO_LOGIN_TYPE"] = "AZCLI"
    fail = 0
    for old, new in sorted(m.items(), key=lambda kv: kv[1]):
        out = run(["azcopy", "copy", f"{BASE}/{old}/*", f"{BASE}/{new}/",
                   "--recursive", "--overwrite=true", "--output-level=essential"])
        status = "OK" if "Final Job Status: Completed" in out else "FAIL"
        if status == "FAIL":
            fail += 1
            print(f"  FAIL {new}\n{out[-400:]}")
        else:
            print(f"  OK   {new}")
    print(f"\n{len(m) - fail} copied, {fail} failed")
    if fail:
        sys.exit("not deleting old prefix while copies are failing")

    print("\nremoving old recipe/ prefix ...")
    out = run(["azcopy", "remove", f"{BASE}/recipe/", "--recursive",
               "--output-level=essential"])
    print("  removed" if "Final Job Status: Completed" in out else out[-400:])


if __name__ == "__main__":
    main()
