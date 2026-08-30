#!/usr/bin/env python3
"""Build iOS-ready binary artefacts from the raw Epicure repos.

Emits little-endian float32 blobs (mmap-able straight into Swift `Data` and fed
to Accelerate/vDSP) plus compact JSON sidecars.

Layout per sibling under data/derived/bundle/:
    {sib}.emb.f32      1790*300 float32, L2-NORMALISED (row-major)
    {sib}.poles.f32    P*300 float32, L2-normalised
    {sib}.meta.json    vocab, pole index, mode atlas, stats
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "derived" / "bundle"
SIBLINGS = ["cooc", "core", "chem"]


def unit(v, axis=-1, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def emit_recipe_cooc(manifest: dict) -> None:
    """Ship real co-occurrence counts alongside the embeddings.

    This is the single highest-value item in the bundle per byte. The `cooc`
    embedding is a lossy 300-d compression of this same signal: on held-out
    human-judged pairs the raw statistics score AUC 0.984 against the
    embedding's 0.873, and the embedding rates chicken+lemon a clash despite
    23,940 real recipes using both. Costs ~1.2 MB pruned, against 2.1 MB for
    one sibling's vectors.

    Layout is three parallel arrays sorted by key, so the app can binary-search
    `key = min(i,j) * n + max(i,j)` with no hash table and no parsing.
    """
    src = ROOT / "data" / "derived" / "recipe_cooc.npz"
    if not src.exists():
        print("\nrecipe_cooc.npz missing - run tools/build_recipe_cooc.py; skipping")
        return
    z = np.load(src, allow_pickle=True)
    pairs, cnt, npmi = z["pairs"], z["count"], z["npmi"]
    itos = [str(s) for s in z["itos"]]

    keep = (cnt >= 3) & np.isfinite(npmi)
    pairs, cnt, npmi = pairs[keep], cnt[keep], npmi[keep]
    n = len(itos)
    key = pairs[:, 0].astype(np.int64) * n + pairs[:, 1]
    o = np.argsort(key)

    (OUT / "recipe_cooc.keys.bin").write_bytes(key[o].astype("<i8").tobytes())
    (OUT / "recipe_cooc.count.bin").write_bytes(
        np.minimum(cnt[o], 65535).astype("<u2").tobytes())
    (OUT / "recipe_cooc.npmi.bin").write_bytes(npmi[o].astype("<f2").tobytes())
    (OUT / "recipe_cooc.meta.json").write_text(json.dumps({
        "n_ingredients": n,
        "n_pairs": int(len(o)),
        "n_recipes": int(z["n_recipes"]),
        "min_count": 3,
        "key": "min(i,j) * n_ingredients + max(i,j), ascending; binary-search it",
        "count_dtype": "uint16 little-endian, saturated at 65535",
        "npmi_dtype": "float16 little-endian",
        "note": "counts over 2.1M English-language recipes (corbt/all-recipes). "
                "Skews American home cooking: miso appears 272x, butter 87,215x.",
    }, indent=2))

    sizes = {p.name: p.stat().st_size for p in OUT.glob("recipe_cooc.*")}
    manifest["recipe_cooc"] = {"n_pairs": int(len(o)),
                               "n_recipes": int(z["n_recipes"]),
                               "bytes": sizes}
    print(f"\nrecipe co-occurrence: {len(o):,} pairs over "
          f"{int(z['n_recipes']):,} recipes, "
          f"{sum(sizes.values())/1024/1024:.2f} MB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "Kaikaku/epicure-{cooc,core,chem}",
        "paper": "arXiv:2605.22391",
        "license": "CC BY 4.0",
        "attribution": "Radzikowski, J. and Chen, J. (2026). Epicure: Navigating the "
                       "Emergent Geometry of Food Ingredient Embeddings. arXiv:2605.22391",
        "layout": {
            "emb": "float32 LE, row-major, shape [n_ingredients, 300], L2-normalised",
            "poles": "float32 LE, row-major, shape [n_poles, 300], L2-normalised",
            "sensory": "float32 LE, row-major, shape [n_sensory_axes, 300], L2-normalised; reconstructed per epicure-explorer app.py::_aggregate_pole",
        },
        "siblings": {},
    }

    for sib in SIBLINGS:
        d = RAW / f"epicure-{sib}"
        E_raw = load_file(d / "embeddings.safetensors")["embeddings"].astype(np.float32)
        E = unit(E_raw).astype("<f4")
        vocab = json.loads((d / "vocab.json").read_text())
        modes = json.loads((d / "modes.json").read_text())
        poles = json.loads((d / "supervised_poles.json").read_text())

        # index -> name, ordered by embedding row
        itos = [None] * len(vocab)
        for n, i in vocab.items():
            itos[i] = n
        assert all(x is not None for x in itos), "vocab is not a dense 0..n-1 index"

        (OUT / f"{sib}.emb.f32").write_bytes(E.tobytes(order="C"))

        # --- supervised poles: stable ordering, normalised ---
        pole_keys = sorted(poles)
        P = unit(np.stack([np.asarray(poles[k], np.float32) for k in pole_keys])).astype("<f4")
        (OUT / f"{sib}.poles.f32").write_bytes(P.tobytes(order="C"))

        # --- sensory axes: reconstructed with the author's own recipe -------
        # Source: epicure-explorer/app.py::_aggregate_pole. Sensory directions
        # are not shipped as vectors; they are the unit-mean of every supervised
        # pole under a property prefix. Valid because 98% of modes have
        # prop_z_mean > 0, i.e. modes are the HIGH end of their property.
        # floral / nutty / high-water are deliberately absent: no pole family.
        SENSORY = {
            "sweet":   ["sweet_score/", "cf_sweet/"],
            "sour":    ["sour_score/", "cf_sour/"],
            "bitter":  ["bitter_score/", "cf_bitter/"],
            "umami":   ["umami_score/", "cf_umami/", "cf_meaty/"],
            "fatty":   ["fatty_score/", "cf_fatty/"],
            "pungent": ["pungent_score/", "cf_pungent/"],
            "savory":  ["cf_savory/"],
            "citrus":  ["cf_citrus/"],
            "woody":   ["cf_woody/"],
            "earthy":  ["cf_earthy/"],
        }
        sensory_keys, sensory_vecs, sensory_support = [], [], {}
        for axis, prefixes in SENSORY.items():
            src = [k for p in prefixes for k in pole_keys if k.startswith(p)]
            if not src:
                continue
            v = unit(np.stack([unit(np.asarray(poles[k], np.float32)) for k in src]).mean(0))
            sensory_keys.append(axis)
            sensory_vecs.append(v)
            sensory_support[axis] = len(src)
        S = np.stack(sensory_vecs).astype("<f4")
        (OUT / f"{sib}.sensory.f32").write_bytes(S.tobytes(order="C"))

        # --- mode atlas: pole vectors live inline in modes.json ---
        mode_keys = [m["mode_id"] for m in modes]
        M = unit(np.stack([np.asarray(m["pole"], np.float32) for m in modes])).astype("<f4")
        (OUT / f"{sib}.modes.f32").write_bytes(M.tobytes(order="C"))

        # Member names in the atlas are SPACE separated; vocab keys are UNDERSCORE
        # separated. Normalise to vocab keys and record the miss rate.
        def to_key(name: str) -> str:
            return name.strip().replace(" ", "_")

        total = miss = 0
        mode_meta = []
        for m in modes:
            ids = []
            for nm in m["members"]:
                total += 1
                k = to_key(nm)
                if k in vocab:
                    ids.append(vocab[k])
                else:
                    miss += 1
            mode_meta.append({
                "id": m["mode_id"],
                "kind": m["kind"],
                "property": m["property"],
                "label": m["label"],
                "members": ids,
            })

        meta = {
            "sibling": sib,
            "n_ingredients": len(itos),
            "dim": 300,
            "ingredients": itos,
            "pole_keys": pole_keys,
            "cuisine_poles": [k for k in pole_keys if k.startswith("cuisine:")],
            "sensory_axes": sensory_keys,
            "sensory_axis_support": sensory_support,
            "modes": mode_meta,
            "member_name_resolution": {
                "total": total, "unresolved": miss,
                "rate": round(1 - miss / total, 5),
            },
        }
        (OUT / f"{sib}.meta.json").write_text(json.dumps(meta, separators=(",", ":")))

        sizes = {p.name: p.stat().st_size for p in OUT.glob(f"{sib}.*")}
        manifest["siblings"][sib] = {
            "n_ingredients": len(itos),
            "n_poles": len(pole_keys),
            "n_cuisine_poles": len(meta["cuisine_poles"]),
            "cuisine_poles": meta["cuisine_poles"],
            "n_sensory_axes": len(sensory_keys),
            "sensory_axes": sensory_keys,
            "n_modes": len(mode_meta),
            "member_resolution_rate": meta["member_name_resolution"]["rate"],
            "bytes": sizes,
        }
        print(f"epicure-{sib}: {len(itos)} ingredients, {len(pole_keys)} poles "
              f"({len(meta['cuisine_poles'])} cuisine), {len(mode_meta)} modes, "
              f"member-name resolution {100*meta['member_name_resolution']['rate']:.2f}% "
              f"({miss}/{total} unresolved)")

    emit_recipe_cooc(manifest)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"\nbundle total: {total/1024/1024:.2f} MB  ->  {OUT.relative_to(ROOT)}")

    # Round-trip check: read the blob back the way Swift will and re-run a query.
    print("\n=== round-trip check (reading blobs the way Swift will) ===")
    for sib in SIBLINGS:
        meta = json.loads((OUT / f"{sib}.meta.json").read_text())
        n, dim = meta["n_ingredients"], meta["dim"]
        E = np.frombuffer((OUT / f"{sib}.emb.f32").read_bytes(), dtype="<f4").reshape(n, dim)
        names = meta["ingredients"]
        i = names.index("miso")
        sims = E @ E[i]
        sims[i] = -np.inf
        top = [(names[j], float(sims[j])) for j in np.argsort(-sims)[:5]]
        norm_ok = np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-5)
        print(f"  {sib:5s} L2-normalised={norm_ok}  miso -> "
              + ", ".join(f"{a}({b:.3f})" for a, b in top))


if __name__ == "__main__":
    main()
