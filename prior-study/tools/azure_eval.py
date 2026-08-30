#!/usr/bin/env python3
"""Fetch trained embeddings from Azure and score them with the eval harness.

    python tools/azure_eval.py cooc-smoke          # one output folder
    python tools/azure_eval.py --all               # everything trained so far

Results land in results/<name>.json and are printed as a comparison table, so
hypotheses are judged against the pre-registered metrics rather than by eyeball.
Downloads to results/_emb/ which is scratch; models/ stays quarantined per R4.
"""
from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as EH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCRATCH = RESULTS / "_emb"
# Jobs write to the workspace's own datastore; see azure_train.OUT_DATASTORE.
ACCOUNT = "cookbookstoragee947b0d86"
CONTAINER = "azureml-blobstore-bb242b5c-ea77-4fec-882f-341b59db9f4c"
PREFIX = "epicure-models"
RG = "cookbook-regression"


@functools.lru_cache(maxsize=1)
def key() -> str:
    r = subprocess.run(["az", "storage", "account", "keys", "list", "-n", ACCOUNT,
                        "-g", RG, "--query", "[0].value", "-o", "tsv"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def blob(*args: str) -> str:
    cmd = ["az", "storage", "blob", *args, "--account-name", ACCOUNT,
           "--account-key", key(), "-o", "tsv"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr.strip() or "az storage failed")
    return r.stdout


def list_runs() -> list[str]:
    out = blob("list", "--container-name", CONTAINER, "--prefix", f"{PREFIX}/",
               "--query", "[].name")
    names = {p.split("/")[1] for p in out.split() if p.count("/") >= 2}
    return sorted(names)


def fetch(name: str) -> Path | None:
    """Pull the embedding matrix for one run. Returns None if absent."""
    dst = SCRATCH / name
    dst.mkdir(parents=True, exist_ok=True)
    files = blob("list", "--container-name", CONTAINER,
                 "--prefix", f"{PREFIX}/{name}/", "--query", "[].name").split()
    emb = [f for f in files if f.endswith(".npy") or f.endswith(".npz")]
    if not emb:
        print(f"  {name}: no embedding written (job may have failed)")
        return None
    for f in emb:
        local = dst / Path(f).name
        if not local.exists():
            blob("download", "--container-name", CONTAINER, "--name", f,
                 "--file", str(local), "--no-progress")
    return dst


def load_matrix(d: Path) -> np.ndarray | None:
    for p in sorted(d.glob("*.npy")):
        return np.load(p)
    for p in sorted(d.glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        for k in ("emb", "W", "vectors", "embeddings"):
            if k in z:
                return z[k]
        for k in z.files:
            a = z[k]
            if a.ndim == 2 and a.shape[0] > 100:
                return a
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    names = list_runs() if a.all or not a.names else a.names
    if not names:
        sys.exit("no trained runs found in the models container")

    RESULTS.mkdir(exist_ok=True)
    ctx = EH.load_context()
    print(f"vocab {ctx['n']}  held-out {len(ctx['held'][0]):,}  runs {len(names)}\n")

    table = {}
    for n in names:
        d = fetch(n)
        W = load_matrix(d) if d else None
        if W is None:
            continue
        r = EH.evaluate(np.asarray(W, dtype=np.float64), ctx)
        table[n] = r
        (RESULTS / f"{n}.json").write_text(json.dumps(r, indent=2))
        print(EH.render(n, r))

    if len(table) > 1:
        print(f"\n{'run':<22}{'M1':>8}{'M2broad':>10}{'M4auc':>9}{'M5freq':>9}")
        for n, r in sorted(table.items(),
                           key=lambda kv: -kv[1]["M2_triplet_accuracy_broad"]):
            print(f"{n:<22}{r['M1_participation_ratio']:>8.1f}"
                  f"{r['M2_triplet_accuracy_broad']:>10.4f}"
                  f"{r['M4_link_auc']:>9.4f}{r['M5_max_pc_freq_corr']:>9.4f}")


if __name__ == "__main__":
    main()
