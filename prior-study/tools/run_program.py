#!/usr/bin/env python3
"""Run the pre-registered model program on Azure, unattended.

    python tools/run_program.py --seed-queue     # write the default queue
    python tools/run_program.py                  # run until the queue drains

Keeps at most MAX_PARALLEL jobs alive (the 6 vCPU regional ceiling / 2 vCPU per
node), submits in priority order, and as each job lands it downloads the vectors
and scores them with the pre-registered harness. State is on disk, so this can
be killed and restarted without losing or repeating work, and results/program.json
can be extended while it runs.

Design note: the queue is data, not code. Every phase -- SGNS, factorisation,
cuisine analysis -- is a script plus an argument string going through the one
job path proven by the smoke test.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import azure_setup as S          # noqa: E402
import azure_train as AT         # noqa: E402
import eval_harness as EH        # noqa: E402
from azure.ai.ml import MLClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
QUEUE = RESULTS / "program.json"
STATE = RESULTS / "program_state.json"
LOG = RESULTS / "program.log"
REPORT = RESULTS / "REPORT.md"

MAX_PARALLEL = 3
POLL_SECONDS = 60
TERMINAL = {"Completed", "Failed", "Canceled"}


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


# ------------------------------------------------------------------ queue
def default_queue() -> list[dict]:
    """The pre-registered program, in priority order.

    Priority 1 is the H1 collapse curve, which is the study's core claim and
    also supplies the H5 comparison for free. Replicates come last: they refine
    confidence in a result rather than produce a new one, so if the night runs
    short they are the right thing to lose.
    """
    q: list[dict] = []

    def sgns(name, variant, ii=None, seed=0, prio=1, hyp="H1"):
        args = f"{variant} --device cpu --epochs 20 --seed {seed}"
        if ii is not None:
            args += f" --ii-repeat {ii}"
        q.append({"name": name, "script": "train_epicure.py", "args": args,
                  "priority": prio, "hypothesis": hyp})

    # H1: one knob traces chem -> cooc. ii_repeat=0 is Chem's schema, 10 is the
    # published Core, 100 approaches Cooc.
    sgns("cooc", "cooc", hyp="H1,H5")
    sgns("chem", "chem", hyp="H1,H5")
    for ii in (0, 0.1, 1, 10, 100):
        sgns(f"core-ii{ii}", "core", ii=ii, hyp="H1,H5")

    # H2: does a closed-form factorisation match SGNS at this vocabulary size?
    for m in ("svd-ppmi", "glove", "chem-svd"):
        q.append({"name": m, "script": "train_factorization.py",
                  "args": f"--method {m}", "priority": 2, "hypothesis": "H2"})

    # H4: the headline. Cross-cultural food-pairing asymmetry, no training.
    q.append({"name": "cuisines", "script": "analyze_cuisines.py", "args": "",
              "priority": 2, "hypothesis": "H4"})

    # H7 (Amendment A2): does the participation ratio predict the width the
    # model actually needs? cooc trains at 300 but uses ~115 directions.
    for d in (32, 64, 128, 600):
        q.append({"name": f"cooc-d{d}", "script": "train_epicure.py",
                  "args": f"cooc --device cpu --epochs 20 --seed 0 --d-model {d}",
                  "priority": 2, "hypothesis": "H7"})

    # Replicates: a difference smaller than seed variance is not a difference.
    for seed in (1, 2):
        sgns(f"cooc-s{seed}", "cooc", seed=seed, prio=3, hyp="H1,var")
        sgns(f"core-ii10-s{seed}", "core", ii=10, seed=seed, prio=3, hyp="H1,var")
        sgns(f"chem-s{seed}", "chem", seed=seed, prio=3, hyp="H1,var")

    # H4 convergence check: if Delta moves between 30k and 200k recipes per
    # cuisine, the headline sample was too small to trust.
    q.append({"name": "cuisines-full", "script": "analyze_cuisines.py",
              "args": "--max-recipes 200000 --n-null 25",
              "priority": 4, "hypothesis": "H4"})
    return q


def load_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(p)


# ------------------------------------------------------------------ azure
_KEY: list[str] = []


def blob_key() -> str:
    """Cached; an az call per download was both slow and a hang risk."""
    if not _KEY:
        r = subprocess.run(["az", "storage", "account", "keys", "list",
                            "-n", AT_ACCOUNT, "-g", S.RG, "--query", "[0].value",
                            "-o", "tsv"], capture_output=True, text=True,
                           timeout=120)
        if r.returncode or not r.stdout.strip():
            raise RuntimeError(f"could not read storage key: {r.stderr[:200]}")
        _KEY.append(r.stdout.strip())
    return _KEY[0]


AT_ACCOUNT = "cookbookstoragee947b0d86"
AT_CONTAINER = "azureml-blobstore-bb242b5c-ea77-4fec-882f-341b59db9f4c"


def download_outputs(name: str, dst: Path) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["az", "storage", "blob", "download-batch", "--account-name", AT_ACCOUNT,
         "--account-key", blob_key(), "--source", AT_CONTAINER,
         "--pattern", f"epicure-models/{name}/*", "--destination", str(dst),
         "--no-progress", "-o", "none"], capture_output=True, text=True,
        timeout=600)
    if r.returncode:
        raise RuntimeError(f"download {name} failed: {r.stderr[:300]}")
    return sorted(dst.rglob("*"))


def score(name: str, ctx: dict) -> dict | None:
    """Download and evaluate one finished job."""
    with tempfile.TemporaryDirectory() as td:
        files = download_outputs(name, Path(td))
        npys = [p for p in files if p.suffix == ".npy"]
        jsons = [p for p in files if p.suffix == ".json"]
        # A non-training job (the cuisine analysis) reports its own findings.
        if not npys:
            for j in jsons:
                try:
                    payload = json.loads(j.read_text())
                except Exception:
                    continue
                if "cuisines" in payload or "delta" in payload:
                    save_json(RESULTS / f"{name}.json", payload)
                    return {"kind": "analysis", "payload": payload}
            raise RuntimeError(f"{name}: no .npy and no analysis json")
        W = np.load(npys[0])
        r = EH.evaluate(np.asarray(W, dtype=np.float64), ctx)
        r["kind"] = "embedding"
        # H3 is a post-hoc geometric correction, so it costs no cluster time:
        # remove the top principal directions and re-measure.
        r["whitened"] = EH.evaluate(EH.all_but_top(W, k=3), ctx)
        save_json(RESULTS / f"{name}.json", r)
        return r


# ------------------------------------------------------------------ report
def write_report(queue: list[dict], state: dict) -> None:
    done = {n: v for n, v in state.items() if v.get("result")}
    lines = [
        "# Program results",
        "",
        f"_updated {datetime.now(timezone.utc).astimezone():%Y-%m-%d %H:%M %Z}_",
        "",
        f"{sum(1 for v in state.values() if v.get('status') == 'Completed')} completed, "
        f"{sum(1 for v in state.values() if v.get('status') == 'Failed')} failed, "
        f"{sum(1 for v in state.values() if v.get('status') not in TERMINAL and v.get('job'))} running, "
        f"{sum(1 for j in queue if j['name'] not in state)} not started",
        "",
        "Chance is 0.50 on M2 and M4. Random-vector control scored "
        "M2 0.5017 / M4 0.4932, so anything near those is noise.",
        "",
        "| model | hypothesis | M1 PR | M2 broad | M2 strict | M3@10 | M4 AUC | M5 freq |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = {j["name"]: i for i, j in enumerate(queue)}
    for n in sorted(done, key=lambda x: order.get(x, 999)):
        r = done[n]["result"]
        if r.get("kind") != "embedding":
            continue
        hyp = next((j["hypothesis"] for j in queue if j["name"] == n), "")
        lines.append(
            f"| {n} | {hyp} | {r['M1_participation_ratio']:.1f} "
            f"| {r['M2_triplet_accuracy_broad']:.4f} "
            f"| {r['M2_triplet_accuracy_strict']:.4f} "
            f"| {r['M3_recall_at_10_broad']:.4f} "
            f"| {r['M4_link_auc']:.4f} | {r['M5_max_pc_freq_corr']:.3f} |")

    lines += ["", "## H3: after removing the top 3 principal directions", "",
              "| model | M1 PR | M2 broad | M4 AUC | M5 freq |", "|---|---|---|---|---|"]
    for n in sorted(done, key=lambda x: order.get(x, 999)):
        r = done[n]["result"]
        w = r.get("whitened")
        if not w:
            continue
        lines.append(f"| {n} | {w['M1_participation_ratio']:.1f} "
                     f"| {w['M2_triplet_accuracy_broad']:.4f} "
                     f"| {w['M4_link_auc']:.4f} | {w['M5_max_pc_freq_corr']:.3f} |")

    an = [(n, v["result"]["payload"]) for n, v in done.items()
          if v["result"].get("kind") == "analysis"]
    for n, payload in an:
        lines += ["", f"## {n}", "", "```",
                  json.dumps(payload, indent=2)[:4000], "```"]

    fails = [n for n, v in state.items() if v.get("status") == "Failed"]
    if fails:
        lines += ["", "## Failed", ""] + [f"- {n}" for n in fails]
    REPORT.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ loop
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-queue", action="store_true")
    ap.add_argument("--once", action="store_true", help="one pass, no loop")
    a = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if a.seed_queue or not QUEUE.exists():
        save_json(QUEUE, default_queue())
        log(f"seeded queue with {len(default_queue())} jobs -> {QUEUE}")
        if a.seed_queue:
            return

    ml = MLClient(S.cred, S.SUB, S.RG, S.WS)
    ctx = EH.load_context()
    log(f"harness ready: vocab {ctx['n']}, held-out {len(ctx['held'][0]):,}")

    while True:
        queue = load_json(QUEUE, [])
        state = load_json(STATE, {})

        # 1. reap finished jobs
        for name, st in list(state.items()):
            if st.get("job") and st.get("status") not in TERMINAL:
                try:
                    st["status"] = ml.jobs.get(st["job"]).status
                except Exception as e:
                    log(f"poll {name}: {e}")
                    continue
                if st["status"] in TERMINAL:
                    log(f"{name} -> {st['status']}")

            # Score any Completed job that still has no result, regardless of
            # when it finished. Previously this sat inside the branch above, so
            # a transient download failure stranded the job permanently.
            if st.get("status") == "Completed" and not st.get("result"):
                try:
                    r = score(name, ctx)
                    if r:
                        st["result"] = r
                    if r and r.get("kind") == "embedding":
                        log(f"{name} scored: PR {r['M1_participation_ratio']:.1f} "
                            f"M2 {r['M2_triplet_accuracy_broad']:.4f} "
                            f"M4 {r['M4_link_auc']:.4f}")
                    elif r:
                        log(f"{name} analysis collected")
                    else:
                        log(f"{name}: no output found, will retry")
                except Exception:
                    log(f"score {name} failed:\n{traceback.format_exc()}")
        save_json(STATE, state)

        # 2. submit while slots are free
        live = sum(1 for v in state.values()
                   if v.get("job") and v.get("status") not in TERMINAL)
        todo = [j for j in sorted(queue, key=lambda x: (x.get("priority", 9), x["name"]))
                if j["name"] not in state]
        for job in todo:
            if live >= MAX_PARALLEL:
                break
            try:
                r = ml.jobs.create_or_update(
                    AT.build_job(job["name"], job["script"], job["args"],
                                 tags={"hypothesis": job.get("hypothesis", "")},
                                 graph=job.get("graph", "ii_graph_train.npz")))
                state[job["name"]] = {"job": r.name, "status": r.status,
                                      "submitted": datetime.now().isoformat(),
                                      "hypothesis": job.get("hypothesis", "")}
                live += 1
                log(f"submitted {job['name']} ({job.get('hypothesis')}) -> {r.name}")
            except Exception:
                log(f"submit {job['name']} failed:\n{traceback.format_exc()}")
                state[job["name"]] = {"job": None, "status": "Failed",
                                      "error": "submit"}
        save_json(STATE, state)
        write_report(queue, state)

        remaining = [j for j in queue if j["name"] not in state
                     or state[j["name"]].get("status") not in TERMINAL]
        if not remaining:
            log("queue drained; all jobs terminal")
            return
        if a.once:
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
