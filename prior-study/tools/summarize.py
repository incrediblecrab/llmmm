#!/usr/bin/env python3
"""Turn raw metrics into verdicts against the pre-registered predictions.

    python tools/summarize.py            # writes results/FINDINGS.md

Reads results/*.json and judges each hypothesis by the thresholds fixed in
docs/PREREGISTRATION.md before any model ran. Verdicts are mechanical on
purpose: the point of pre-registering is that the conclusion follows from the
numbers rather than from how they look in the morning.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "FINDINGS.md"

CI = 0.0069          # 95% half-width on M2/M4 at current sample sizes
CHANCE = 0.50
RANDOM_M2, RANDOM_M4 = 0.5017, 0.4932


def load() -> dict:
    out = {}
    for p in sorted(RESULTS.glob("*.json")):
        if p.name in ("program.json", "program_state.json"):
            continue
        try:
            out[p.stem] = json.loads(p.read_text())
        except Exception:
            pass
    return out


def emb(r: dict) -> bool:
    return isinstance(r, dict) and "M1_participation_ratio" in r


def fmt(v, n=4):
    return "—" if v is None else f"{v:.{n}f}"


def verdict(ok: bool | None, yes: str, no: str, pending: str = "pending") -> str:
    if ok is None:
        return f"**PENDING** — {pending}"
    return f"**{'SUPPORTED' if ok else 'FALSIFIED'}** — {yes if ok else no}"


def h1(R: dict) -> list[str]:
    """Chem's collapse is structural: it comes from having no I-I edges."""
    curve = [("core-ii0", 0.0), ("core-ii0.1", 0.1), ("core-ii1", 1.0),
             ("core-ii10", 10.0), ("core-ii100", 100.0)]
    have = [(n, x) for n, x in curve if n in R and emb(R[n])]
    L = ["## H1 — the chem collapse is structural, not a training bug", "",
         "One knob (`ii_repeat`) interpolates from Chem's pure ingredient→compound",
         "schema (0) to Cooc's pure ingredient–ingredient schema (∞). If collapse were",
         "an optimiser or hardware artefact it would not track this knob.", "",
         "| ii_repeat | M1 PR | M2 broad | M4 AUC |", "|---|---|---|---|"]
    for n, x in have:
        r = R[n]
        L.append(f"| {x:g} | {r['M1_participation_ratio']:.1f} "
                 f"| {r['M2_triplet_accuracy_broad']:.4f} | {r['M4_link_auc']:.4f} |")
    for n in ("chem", "cooc"):
        if n in R and emb(R[n]):
            r = R[n]
            L.append(f"| _{n}_ | {r['M1_participation_ratio']:.1f} "
                     f"| {r['M2_triplet_accuracy_broad']:.4f} | {r['M4_link_auc']:.4f} |")
    ok = None
    if len(have) >= 3:
        prs = [R[n]["M1_participation_ratio"] for n, _ in have]
        aucs = [R[n]["M4_link_auc"] for n, _ in have]
        mono = all(b >= a - 5 for a, b in zip(prs, prs[1:]))
        lifts = aucs[-1] - aucs[0] > 0.05
        ok = mono and lifts
    L += ["", verdict(
        ok,
        "PR and held-out AUC both rise with I-I mixing, so the collapse is a "
        "property of the walk schema — chemistry-only walks cannot express "
        "ingredient–ingredient structure.",
        "PR does not track ii_repeat, so collapse is not explained by the schema.",
        "needs at least three points on the curve")]
    if "chem" in R and emb(R["chem"]):
        c = R["chem"]
        L += ["", f"Chem alone sits at PR {c['M1_participation_ratio']:.1f} with "
                  f"held-out AUC {c['M4_link_auc']:.4f} against a 0.50 chance and a "
                  f"{RANDOM_M4:.4f} random control — it is close to uninformative "
                  "about which ingredients actually co-occur."]
    return L


def h2(R: dict) -> list[str]:
    """Closed-form factorisation vs SGNS at n=1,790."""
    base = R.get("cooc")
    L = ["## H2 — a closed-form factorisation matches SGNS at this scale", "",
         "| model | M1 PR | M2 broad | M4 AUC | vs SGNS (M2) |", "|---|---|---|---|---|"]
    ok = None
    if base and emb(base):
        for n in ("cooc", "svd-ppmi", "glove", "chem-svd"):
            r = R.get(n)
            if not (r and emb(r)):
                continue
            d = r["M2_triplet_accuracy_broad"] - base["M2_triplet_accuracy_broad"]
            L.append(f"| {n} | {r['M1_participation_ratio']:.1f} "
                     f"| {r['M2_triplet_accuracy_broad']:.4f} | {r['M4_link_auc']:.4f} "
                     f"| {d:+.4f} |")
        cands = [n for n in ("svd-ppmi", "glove") if n in R and emb(R[n])]
        if cands:
            ok = any(abs(R[n]["M2_triplet_accuracy_broad"]
                         - base["M2_triplet_accuracy_broad"]) <= 0.02 for n in cands)
    L += ["", verdict(
        ok,
        "at least one factorisation lands within 0.02 of SGNS, so the signal is in "
        "the corpus statistics rather than in the sampling procedure — and these "
        "baselines train in seconds and cannot collapse.",
        "factorisation does not reach SGNS; the random-walk sampling contributes "
        "something the co-occurrence matrix alone does not capture.",
        "factorisation jobs not finished")]
    return L


def h3(R: dict) -> list[str]:
    """Is popularity a removable low-rank artefact?"""
    L = ["## H3 — popularity degeneration is low-rank and removable", "",
         "Pre-registered test: removing the top 3 principal directions should push",
         "M5 below 0.3 while costing M2 no more than 0.02.", "",
         "| model | M5 before | M5 after | M2 before | M2 after | M2 cost |",
         "|---|---|---|---|---|---|"]
    rows = 0
    passes = []
    for n, r in R.items():
        w = r.get("whitened") if isinstance(r, dict) else None
        if not (emb(r) and w):
            continue
        cost = r["M2_triplet_accuracy_broad"] - w["M2_triplet_accuracy_broad"]
        L.append(f"| {n} | {r['M5_max_pc_freq_corr']:.3f} | {w['M5_max_pc_freq_corr']:.3f} "
                 f"| {r['M2_triplet_accuracy_broad']:.4f} "
                 f"| {w['M2_triplet_accuracy_broad']:.4f} | {cost:+.4f} |")
        passes.append(w["M5_max_pc_freq_corr"] < 0.3 and cost <= 0.02)
        rows += 1
    ok = (all(passes) if passes else None) if rows else None
    L += ["", verdict(
        ok,
        "whitening removes the popularity axis cheaply.",
        "whitening either fails to remove popularity or costs more accuracy than "
        "the pre-registered budget. Where M5 stays high after removing three "
        "directions, popularity is spread across many dimensions rather than "
        "concentrated in a few — it is not a low-rank artefact and cannot be "
        "projected away for free.",
        "no whitened results yet")]
    return L


def h4(R: dict) -> list[str]:
    """Ahn et al. 2011 cross-cultural asymmetry."""
    L = ["## H4 — the food-pairing asymmetry across cuisines (headline)", ""]
    c = R.get("cuisines")
    payload = c.get("payload") if isinstance(c, dict) and "payload" in c else c
    if not payload or "cuisines" not in payload:
        return L + [verdict(None, "", "", "cuisine analysis not finished")]
    cu = payload["cuisines"]
    L += [f"{len(cu)} cuisines. Δ > 0 means a cuisine pairs ingredients that share "
          "flavour compounds more than its own ingredient frequencies would predict.",
          "", "| cuisine | region | recipes | Δ | relative | z |", "|---|---|---|---|---|---|"]
    for n, r in sorted(cu.items(), key=lambda kv: -kv[1]["rel_delta"]):
        L.append(f"| {n} | {r['region']} | {r['recipes_used']:,} | {r['delta']:+.3f} "
                 f"| {r['rel_delta']:+.2%} | {r.get('z', 0):+.1f} |")
    L += ["", "| region | mean relative Δ | cuisines |", "|---|---|---|"]
    for k, v in sorted(payload["regions"].items(),
                       key=lambda kv: -kv[1]["mean_rel_delta"]):
        L.append(f"| {k} | {v['mean_rel_delta']:+.2%} | {v['cuisines']} |")
    reg = payload["regions"]
    west = [reg[k]["mean_rel_delta"] for k in
            ("North America", "Western Europe", "Southern Europe") if k in reg]
    east = [reg[k]["mean_rel_delta"] for k in ("East Asia",) if k in reg]
    ok = (sum(west) / len(west) > sum(east) / len(east)) if west and east else None
    L += ["", verdict(
        ok,
        "western cuisines pair shared-compound ingredients more than East Asian "
        "cuisines do, reproducing Ahn et al. 2011 on natively-sourced regional "
        "corpora rather than on a western-dominated corpus. That is the part of "
        "their claim most open to sampling bias, and it holds.",
        "the asymmetry does not reproduce once each cuisine is measured on its own "
        "native corpus, which would suggest the original result was partly an "
        "artefact of where the recipes came from.",
        "need both western and East Asian regions")]
    return L


def h5(R: dict) -> list[str]:
    """Does chemistry add anything over co-occurrence?"""
    L = ["## H5 — does the chemistry graph add value?", "",
         "Pre-registered: a chemistry-informed model should beat pure co-occurrence "
         "on held-out link AUC by more than 0.02.", "",
         "| model | M4 AUC | vs cooc |", "|---|---|---|"]
    base = R.get("cooc")
    ok = None
    if base and emb(base):
        b = base["M4_link_auc"]
        for n in ("cooc", "chem", "chem-svd", "core-ii1", "core-ii10", "core-ii100"):
            r = R.get(n)
            if r and emb(r):
                L.append(f"| {n} | {r['M4_link_auc']:.4f} | {r['M4_link_auc'] - b:+.4f} |")
        cores = [R[n]["M4_link_auc"] for n in ("core-ii1", "core-ii10", "core-ii100")
                 if n in R and emb(R[n])]
        if cores:
            ok = max(cores) - b > 0.02
    L += ["", verdict(
        ok,
        "a chemistry-informed model beats pure co-occurrence, so FlavorDB earns "
        "its licensing risk.",
        "no chemistry-informed model beats pure co-occurrence by the pre-registered "
        "margin. FlavorDB can be dropped, which removes a licensing risk from the "
        "product without costing measured quality.",
        "core variants not finished")]
    return L


def main() -> None:
    R = load()
    state = {}
    try:
        state = json.loads((RESULTS / "program_state.json").read_text())
    except Exception:
        pass
    done = sum(1 for v in state.values() if v.get("status") == "Completed")
    fail = sum(1 for v in state.values() if v.get("status") == "Failed")

    L = ["# Findings", "",
         f"_{done} jobs completed, {fail} failed, {len(state)} submitted._", "",
         "Verdicts are mechanical against thresholds fixed in "
         "`docs/PREREGISTRATION.md` before any model ran.",
         "Random-vector control: M2 "
         f"{RANDOM_M2:.4f}, M4 {RANDOM_M4:.4f}. 95% CI on both is ±{CI:.4f}, "
         "so smaller gaps are not differences.", ""]
    for fn in (h1, h2, h3, h4, h5):
        L += fn(R) + [""]

    reps = {}
    for n, r in R.items():
        if emb(r) and "-s" in n:
            reps.setdefault(n.rsplit("-s", 1)[0], []).append(r["M2_triplet_accuracy_broad"])
    for base, vals in list(reps.items()):
        if base in R and emb(R[base]):
            vals.append(R[base]["M2_triplet_accuracy_broad"])
    if any(len(v) > 1 for v in reps.values()):
        L += ["## Seed variance", "",
              "A difference smaller than this spread is not a result.", "",
              "| model | seeds | M2 spread |", "|---|---|---|"]
        for b, v in sorted(reps.items()):
            if len(v) > 1:
                L.append(f"| {b} | {len(v)} | {max(v) - min(v):.4f} |")
        L.append("")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")
    print("\n".join(L[:4]))


if __name__ == "__main__":
    main()
