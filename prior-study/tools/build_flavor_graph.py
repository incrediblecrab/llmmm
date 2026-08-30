#!/usr/bin/env python3
"""Build the typed ingredient-compound graph from the FlavorDB scrape.

The paper seeds its Chem and Core walks from "an 80,019-edge typed FlavorDB
ingredient-compound graph, 2,247 typed compound nodes across 15 categories".
Two things about that are worth stating plainly, because they drive the design:

  * The edges are ingredient-compound where *ingredient* means a term in the
    1,790-entry canonical vocabulary, not a FlavorDB entity. FlavorDB exposes
    935 entities; the vocabulary is finer-grained, so a single entity ("pepper")
    legitimately backs several vocabulary terms and the edge count is a function
    of the vocab->entity mapping, not of FlavorDB's own size.
  * FlavorDB does not ship a compound type field. The "15 categories" have to be
    derived, and the only per-molecule signal it publishes that is chemical
    rather than sensory is `functional_groups`. So compounds are typed by a
    priority-ordered chemical taxonomy over those groups.

Typing by chemistry rather than by flavour is deliberate: the whole point of the
Chem variant is to sit at the opposite end of the spectrum from recipe context,
and flavour descriptors ("sweet", "green") are closer to co-occurrence than to
structure. Priority order matters because most molecules carry 2-4 groups; the
most structurally specific class wins so that esters aren't all filed as
"carbonyl".

Writes data/derived/flavor_graph.npz.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_recipe_cooc as brc

ROOT = Path(__file__).resolve().parents[1]
ENT = ROOT / "data" / "raw" / "flavordb" / "entities"

# Priority-ordered chemical taxonomy over FlavorDB's `functional_groups` labels.
# First match wins, so structurally specific classes come first: most molecules
# carry 2-4 groups, and a lactone is also tagged "carboxylic acid ester" while
# nearly every ketone/aldehyde is also tagged "carbonyl compound". Ordering by
# specificity keeps those from collapsing into one giant bucket.
#
# The class list is derived from the 86 group labels FlavorDB actually publishes,
# not from a textbook: an earlier version reserved slots for "terpenoid" and
# "hydrocarbon", which sound obvious for aroma chemistry but never appear in the
# data, leaving 2 of the 15 categories permanently empty.
CLASSES: list[tuple[str, tuple[str, ...]]] = [
    ("organosulfur",   ("thiol", "thioether", "sulfenic", "sulfide", "sulfanyl",
                        "isothiocyanate", "disulfide", "thiocarboxylic",
                        "thioacetal", "sulf")),
    ("organonitrogen", ("amine", "amide", "nitrile", "aminoacid", "imine",
                        "azo", "nitro", "cyanate")),
    ("lactone",        ("lactone", "lactam")),
    ("phenol",         ("phenol", "hydroxyhetarene", "diphenol")),
    ("acetal",         ("acetal", "ketal", "hemiacetal")),
    ("aldehyde",       ("aldehyde",)),
    ("ester",          ("carboxylic acid ester", "carboxylic acid derivative",
                        "ester")),
    ("carboxylic_acid", ("carboxylic acid", "hydroxyacid", "enol")),
    ("ketone",         ("ketone", "quinone", "carbonyl")),
    ("alcohol",        ("alcohol", "diol", "hydroxy compound", "polyol")),
    ("ether",          ("ether", "oxide", "epoxide")),
    ("heterocycle",    ("heterocyclic", "hetarene", "furan", "pyran",
                        "pyrazine", "pyridine", "pyrrole", "thiophene")),
    ("aromatic",       ("aromatic", "arene", "benzene")),
    ("alkene",         ("alkene", "alkyne", "olefin", "alkane")),
    ("other",          ()),
]
CLASS_NAMES = [c for c, _ in CLASSES]
assert len(CLASS_NAMES) == 15, len(CLASS_NAMES)

_PAREN = re.compile(r"\s*\(.*?\)\s*")
# Suffixes FlavorDB appends that the vocabulary doesn't carry. "Hops Oil" and
# "Mentha Oil" are the same food as "hops"/"mint" for graph purposes, and
# "Bakery Products" is a category label wearing an entity's clothes.
_SUFFIX = re.compile(
    r"\s+(?:oil|oils|products?|extracts?|essence|essential|juice|leaf|leaves"
    r"|seeds?|roots?|fruits?|flowers?|nuts?|beans?|powder|paste)$")


def classify(groups: str) -> int:
    g = (groups or "").lower()
    for i, (_, keys) in enumerate(CLASSES):
        if any(k in g for k in keys):
            return i
    return len(CLASSES) - 1


def entity_aliases(e: dict) -> set[str]:
    """Every surface form FlavorDB offers for one entity, normalised.

    Generated aggressively on purpose: an entity that fails to resolve costs the
    graph every one of its ~64 compound edges, whereas a spurious alias has to
    collide with a real vocabulary term to do any damage, and `lookup` only
    accepts exact hits.
    """
    out = set()
    for key in ("entity_alias_readable", "entity_alias", "natural_source_name"):
        v = e.get(key)
        if v:
            out.add(str(v))
    syn = e.get("entity_alias_synonyms") or ""
    out.update(s for s in re.split(r"[,;@]", str(syn)) if s.strip())

    clean = set()
    for s in out:
        s = _PAREN.sub(" ", str(s)).strip().lower()
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) <= 2:
            continue
        forms = {s}
        stripped = _SUFFIX.sub("", s).strip()
        if len(stripped) > 2:
            forms.add(stripped)
        for f in list(forms):
            forms.add(f.replace(" ", "_"))
            forms.add(f.replace(" ", ""))      # "lemon grass" -> "lemongrass"
            if " " in f:
                forms.add(f.rsplit(" ", 1)[-1])  # head noun: "clary sage" -> sage
        clean.update(f for f in forms if len(f) > 2)
    return clean


def main() -> None:
    vocab, itos = brc.load_vocab()
    alias = brc.build_alias(vocab)

    files = sorted(glob.glob(str(ENT / "*.json")))
    if not files:
        raise SystemExit("no FlavorDB entities; run tools/scrape_flavordb.py first")

    mol_index: dict[int, int] = {}
    mol_rows: list[dict] = []
    ent_mols: list[tuple[dict, list[int]]] = []
    for f in files:
        e = json.load(open(f))
        idxs = []
        for m in e.get("molecules", []):
            pid = m["pubchem_id"]
            j = mol_index.get(pid)
            if j is None:
                j = mol_index[pid] = len(mol_rows)
                mol_rows.append(m)
            idxs.append(j)
        ent_mols.append((e, idxs))

    ctype = np.array([classify(m.get("functional_groups")) for m in mol_rows],
                     dtype=np.int16)

    # vocab term -> FlavorDB entities. One term may hit several entities and one
    # entity may back several terms; both are correct and both are kept.
    hits: dict[int, set[int]] = {}
    unmatched = []
    for ei, (e, _) in enumerate(ent_mols):
        matched = False
        for a in entity_aliases(e):
            vid = brc.lookup(alias, a)
            if vid is not None:
                hits.setdefault(vid, set()).add(ei)
                matched = True
        if not matched:
            unmatched.append(e.get("entity_alias_readable"))

    edges = set()
    for vid, eis in hits.items():
        for ei in eis:
            for mj in ent_mols[ei][1]:
                edges.add((vid, mj))
    edges = sorted(edges)
    src = np.array([a for a, _ in edges], dtype=np.int32)
    dst = np.array([b for _, b in edges], dtype=np.int32)

    used = sorted(set(dst.tolist()))
    hubs = sorted(hits)
    tally = Counter(ctype[used].tolist())

    print(f"FlavorDB entities            {len(ent_mols):,}")
    print(f"  entities matched to vocab  {sum(len(v) for v in hits.values()):,}"
          f" (unmatched {len(unmatched):,})")
    print(f"ingredient hubs              {len(hubs):,}   (paper 523)")
    print(f"non-hubs                     {len(vocab)-len(hubs):,}   (paper 1,267)")
    print(f"compound nodes used          {len(used):,}   (paper 2,247)")
    print(f"typed I-C edges              {len(edges):,}   (paper 80,019)")
    print(f"compound categories          {len(CLASS_NAMES)}   (paper 15)")
    print("\ncompounds per category:")
    for i, n in tally.most_common():
        print(f"  {CLASS_NAMES[i]:16}{n:>6}")

    out = ROOT / "data" / "derived" / "flavor_graph.npz"
    np.savez_compressed(
        out, src=src, dst=dst, ctype=ctype,
        class_names=np.array(CLASS_NAMES),
        pubchem=np.array([m["pubchem_id"] for m in mol_rows], dtype=np.int64),
        mol_name=np.array([str(m.get("common_name") or "") for m in mol_rows]),
        hubs=np.array(hubs, dtype=np.int32), itos=np.array(itos),
    )
    print(f"\nwrote {out.relative_to(ROOT)}")
    if unmatched:
        print(f"sample unmatched entities: {unmatched[:15]}")


if __name__ == "__main__":
    main()
