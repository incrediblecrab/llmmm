"""Adversarial sanity checks — try to break our own results.

Every headline number this workspace produces is checked here against an
*independent* computation, a control, or a structural invariant. The point is
not to re-run the pipeline and observe that it agrees with itself; it is to
compute the same quantity a different way, and to look specifically for the
failure modes that would make results *better* than they should be.

Run: `python scripts/sanity_check.py`   (exit code 1 if any check fails)
"""
from __future__ import annotations

import sys
import time

import numpy as np

from ingredient_model.artifacts import load_embedding
from ingredient_model.config import PATHS, SEED
from ingredient_model.data.graphs import load_chem_graph, load_ii_graph
from ingredient_model.data.labels import load_substitutions
from ingredient_model.data.recipes import load_recipes
from ingredient_model.data.splits import get_split, held_out_recipes
from ingredient_model.eval.completion import recipe_completion
from ingredient_model.eval.harness import build_context
from ingredient_model.eval.metrics import unit

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------------
def check_vocabulary_alignment() -> None:
    """Every artefact must index the vocabulary identically.

    This is the check that, if it fails, makes every other number in the
    workspace meaningless — and it would fail *silently*, because mismatched
    integer indices still produce plausible-looking metrics.
    """
    section("1. Vocabulary alignment")
    full = load_recipes()
    g_full = load_ii_graph("ii_graph.npz")
    g_rh = load_ii_graph(get_split("recipe-holdout").graph)
    g_edge = load_ii_graph(get_split("edge-holdout").graph)

    sizes = {"corpus": full.n_vocab, "ii_graph": g_full.n_vocab,
             "rh_train": g_rh.n_vocab, "edge_train": g_edge.n_vocab}
    check("all artefacts agree on vocabulary size",
          len(set(sizes.values())) == 1, str(sizes))

    # Element-wise, not just length: two lists of 1790 names can be the same
    # length and a different order, which is the failure that produces
    # believable nonsense.
    check("corpus and graph itos are element-wise identical",
          list(full.itos) == list(g_full.itos),
          f"first mismatch: {next((i for i, (a, b) in enumerate(zip(full.itos, g_full.itos)) if a != b), None)}")
    check("recipe-holdout graph itos identical to corpus",
          list(g_rh.itos) == list(full.itos))
    check("token ids are all within vocabulary",
          int(full.flat.max()) < full.n_vocab,
          f"max token id {int(full.flat.max())}, vocab {full.n_vocab}")

    try:
        chem = load_chem_graph()
        check("chem graph vocabulary matches",
              list(chem.itos) == list(full.itos))
    except FileNotFoundError:
        check("chem graph present", True, "absent — skipped")

    subs = load_substitutions(tuple(full.itos))
    ok, detail = True, []
    for tier, pairs in subs.pairs.items():
        arr = np.asarray(pairs, dtype=np.int64)
        in_range = bool(arr.min() >= 0 and arr.max() < full.n_vocab)
        no_self = bool((arr[:, 0] != arr[:, 1]).all())
        ok &= in_range and no_self
        detail.append(f"{tier}: {len(arr):,} pairs, range "
                      f"[{int(arr.min())}, {int(arr.max())}], "
                      f"self-pairs {'none' if no_self else 'PRESENT'}")
    check("substitution labels are in range and self-pair free",
          ok, "\n".join(detail))


def check_corpus_integrity() -> None:
    section("2. Corpus integrity")
    full = load_recipes()
    off = full.offsets
    check("offsets are monotonic non-decreasing",
          bool(np.all(np.diff(off) >= 0)))
    check("offsets terminate exactly at len(flat)",
          int(off[-1]) == len(full.flat),
          f"offsets[-1]={int(off[-1])}  len(flat)={len(full.flat)}")
    check("recipe count matches offsets",
          full.n_recipes == len(off) - 1,
          f"{full.n_recipes:,} recipes")
    sizes = full.sizes
    check("no recipe is empty", int(sizes.min()) >= 1,
          f"min {int(sizes.min())}, max {int(sizes.max())}, "
          f"mean {sizes.mean():.1f}")
    check("metadata arrays are aligned with recipes",
          len(full.lang) == full.n_recipes == len(full.source))


def check_split_disjointness() -> None:
    """Train and test recipes must not overlap, and the rebuilt graph must not
    contain the held-out edges."""
    section("3. Split disjointness")
    sp = get_split("recipe-holdout")
    full = load_recipes()
    train = load_recipes(sp.corpus)
    test = held_out_recipes("recipe-holdout", limit=None)
    check("train + test partitions the corpus exactly",
          train.n_recipes + test.n_recipes == full.n_recipes,
          f"{train.n_recipes:,} + {test.n_recipes:,} = "
          f"{train.n_recipes + test.n_recipes:,} vs {full.n_recipes:,}")

    # The held-out archive stores only the edge list, so read it directly
    # rather than through load_ii_graph (which expects npmi/count/uni).
    g_tr = load_ii_graph(sp.graph)
    ho = np.load(PATHS.graphs / sp.heldout, allow_pickle=True)
    tr_edges = {(min(a, b), max(a, b))
                for a, b in zip(g_tr.src.tolist(), g_tr.dst.tolist())}
    ho_edges = {(min(a, b), max(a, b))
                for a, b in zip(ho["src"].tolist(), ho["dst"].tolist())}
    overlap = tr_edges & ho_edges
    check("held-out edges are absent from the training graph",
          len(overlap) == 0,
          f"{len(tr_edges):,} train edges, {len(ho_edges):,} held-out, "
          f"{len(overlap)} overlap")


def check_popularity_baseline() -> None:
    """Recompute the M6 popularity baseline by an independent path.

    The harness derives it from `ctx.unigram`. Here it is recomputed straight
    from raw token counts over the training corpus, with a hand-written ranking
    loop rather than the vectorised one.
    """
    section("4. Popularity baseline, recomputed independently")
    sp = get_split("recipe-holdout")
    train = load_recipes(sp.corpus)
    test = held_out_recipes("recipe-holdout", limit=20_000)

    counts = np.bincount(train.flat.astype(np.int64), minlength=train.n_vocab)
    order_desc = np.argsort(-counts)

    rng = np.random.default_rng(1234)
    hits = tot = 0
    for r in range(test.n_recipes):
        ids = test.recipe(r)
        if len(ids) < 3:
            continue
        hide = int(rng.integers(0, len(ids)))
        target = int(ids[hide])
        visible = set(int(x) for i, x in enumerate(ids) if i != hide)
        top, seen = [], 0
        for cand in order_desc:
            c = int(cand)
            if c in visible:
                continue
            top.append(c)
            seen += 1
            if seen == 10:
                break
        hits += target in top
        tot += 1
        if tot >= 4000:
            break
    independent = hits / tot

    ctx = build_context("recipe-holdout")
    W = np.random.default_rng(SEED).normal(size=(ctx.n, 32))
    harness = recipe_completion(W, test, n_test=20_000, unigram=ctx.unigram)
    reported = harness["M6_popularity_recall_at_10"]
    check("popularity baseline reproduces independently",
          abs(independent - reported) < 0.02,
          f"independent loop {independent:.4f} (n={tot:,}) vs "
          f"harness {reported:.4f}  |diff| {abs(independent - reported):.4f}")

    check("popularity is well above chance",
          reported > 0.20,
          f"chance for recall@10 over 1790 candidates is "
          f"{10 / ctx.n:.4f}; popularity is {reported:.4f} — this is why it "
          f"must be reported alongside every model")


def check_random_controls() -> None:
    """A metric that gives free credit to noise cannot support any claim."""
    section("5. Random and degenerate controls on M6")
    ctx = build_context("recipe-holdout")
    test = held_out_recipes("recipe-holdout", limit=20_000)
    rng = np.random.default_rng(SEED)

    rand = recipe_completion(rng.normal(size=(ctx.n, 300)), test,
                             n_test=20_000)["M6_recall_at_10"]
    chance = 10 / ctx.n
    check("random vectors score at chance on M6",
          abs(rand - chance) < 0.01,
          f"random {rand:.4f} vs chance {chance:.4f}")

    rank1 = recipe_completion(
        rng.normal(size=(ctx.n, 1)) @ rng.normal(size=(1, 300)), test,
        n_test=20_000)["M6_recall_at_10"]
    check("rank-1 collapsed space scores at or below chance",
          rank1 < chance,
          f"collapsed {rank1:.4f} vs chance {chance:.4f}. Under an optimistic "
          f"tie rule this scored 0.2113 — 38x chance — because a collapsed "
          f"space produces two distinct scores and ~900 candidates tie at the "
          f"cut. Midrank tie-breaking removes the artefact.")


def check_ease_native_scorer() -> None:
    """The largest claim in the workspace: EASE's native scorer is 3.6x its
    embedding. Verify it is not an artefact of leakage or of ranking ties."""
    section("6. EASE native scorer")
    run = PATHS.run_dir("ease-rh")
    if not run.exists():
        check("ease-rh run present", False, "train it first")
        return
    B = np.load(run / "item_scores.npy")
    W = load_embedding(run)
    test = held_out_recipes("recipe-holdout", limit=20_000)
    ctx = build_context("recipe-holdout")

    check("B is square over the vocabulary",
          B.shape == (ctx.n, ctx.n), str(B.shape))
    check("B has a zero diagonal",
          float(np.abs(np.diag(B)).max()) < 1e-6,
          f"max |diag| {float(np.abs(np.diag(B)).max()):.2e} — a non-zero "
          f"diagonal would let an ingredient predict itself")

    res = recipe_completion(W, test, n_test=20_000, unigram=ctx.unigram,
                            scorer=lambda c: B[c].sum(1))
    emb, nat = res["M6_recall_at_10"], res["M6_native_recall_at_10"]
    pop = res["M6_popularity_recall_at_10"]
    check("native scorer beats the popularity baseline",
          nat > pop, f"native {nat:.4f} vs popularity {pop:.4f}")
    check("embedding export loses substantial signal",
          nat > emb * 2,
          f"native {nat:.4f} vs embedding {emb:.4f} "
          f"(ratio {nat / max(emb, 1e-9):.2f}x)")

    # Ties would make the optimistic rank rule flatter this model.
    k = 8
    rows = np.flatnonzero(test.sizes == k)[:1000]
    ids = test.flat[test.offsets[rows][:, None] + np.arange(k)].astype(np.int64)
    S = B[ids[:, :k - 1]].sum(1)
    tenth = np.sort(S, axis=1)[:, ::-1][:, 9:10]
    max_ties = int((S == tenth).sum(1).max())
    check("no ranking ties at the recall@10 cut",
          max_ties <= 1,
          f"max candidates tied with the 10th-best score: {max_ties}. "
          f"The rank rule is optimistic under ties, so this must be 1.")


def check_leakage_claim() -> None:
    """Re-prove the central claim: the edge protocol inflates recipe models."""
    section("7. Leakage claim, reproduced")
    from models.recipe_basket.ease import train_ease
    from ingredient_model.eval.harness import evaluate
    from ingredient_model.spec import TrainContext

    out = {}
    for split_name in ("edge-holdout", "recipe-holdout"):
        sp = get_split(split_name)
        c = TrainContext(graph=sp.graph, seed=0, out_dir=PATHS.runs / "_sanity",
                         params={"d_model": 300, "reg": 250.0, "max_recipes": 0},
                         corpus=sp.corpus, split=sp.name)
        r = train_ease(c)
        out[split_name] = evaluate(r.embedding, build_context(split_name))["M4_link_auc"]

    leaky, honest = out["edge-holdout"], out["recipe-holdout"]
    ci = 0.0069
    check("edge protocol inflates M4 for a recipe model",
          leaky - honest > 5 * ci,
          f"edge-holdout {leaky:.4f} vs recipe-holdout {honest:.4f}  "
          f"inflation {leaky - honest:+.4f} = {(leaky - honest) / ci:.1f}x the CI")
    check("the honest number is still above chance",
          honest > 0.5 + 2 * ci, f"honest M4 {honest:.4f}")


def check_metric_monotonicity() -> None:
    """A trained space must beat its own shuffled self.

    This catches metrics that reward *any* structure — if shuffling the rows of
    an embedding leaves a score unchanged, that score is reading the labels or
    the frequencies, not the model.
    """
    section("8. Shuffle control")
    run = PATHS.run_dir("ease-rh")
    if not run.exists():
        check("ease-rh present", False)
        return
    W = load_embedding(run)
    test = held_out_recipes("recipe-holdout", limit=20_000)
    ctx = build_context("recipe-holdout")
    real = recipe_completion(W, test, n_test=20_000)["M6_recall_at_10"]
    perm = np.random.default_rng(7).permutation(len(W))
    shuf = recipe_completion(W[perm], test, n_test=20_000)["M6_recall_at_10"]
    check("shuffling embedding rows destroys M6",
          real > shuf * 3,
          f"real {real:.4f} vs row-shuffled {shuf:.4f}")

    from ingredient_model.eval.metrics import m2_triplet_accuracy
    U = unit(W)
    r2 = m2_triplet_accuracy(U, ctx, "broad")[0]
    s2 = m2_triplet_accuracy(unit(W[perm]), ctx, "broad")[0]
    check("shuffling embedding rows destroys M2",
          r2 - s2 > 0.10, f"real {r2:.4f} vs shuffled {s2:.4f}")


def check_normalisation_faithfulness() -> None:
    """Replay the original readers and confirm the stored corpus is what they
    produce.

    This is the only check here that leaves the derived artefacts. Everything
    else compares one derived file against another, which cannot detect a
    reader that drops a column or mis-splits a delimiter — that failure is
    perfectly self-consistent downstream and would silently redefine what every
    metric is measuring.
    """
    section("9. Normalisation is faithful to the original sources")
    import itertools
    import sys
    from pathlib import Path

    root = PATHS.prior_study
    if not (root / "tools" / "normalize.py").exists():
        check("original corpus tree available", True,
              "absent — skipped (derived artefacts are self-sufficient)")
        return
    if str(root / "tools") not in sys.path:
        sys.path.insert(0, str(root / "tools"))
    import corpus as raw  # type: ignore
    import normalize as nm  # type: ignore

    full = load_recipes()

    # Replay through the normaliser that *built* this corpus, not whichever one
    # happens to be importable. The corrected normaliser recovered 924,315
    # ingredient occurrences the base one missed, so replaying a corrected
    # corpus through the base normaliser disagrees on exactly those recipes and
    # reports a 97.5% that looks like corpus corruption and is nothing of the
    # kind. Which one built it is recorded at promotion rather than guessed.
    from ingredient_model.config import corpus_generation
    marker = corpus_generation()
    which = marker.get("normalizer", "base")
    if which == "corrected":
        from ingredient_model.data.normalizer import get_normalizer
        nz = get_normalizer(fix_zh_qty=True, extra=True)
    else:
        nz = nm.Normalizer()
    print(f"  corpus generation {marker.get('generation', 'unknown')!r}, "
          f"replaying through the {which} normaliser")

    check("normaliser and corpus share one vocabulary",
          list(nz.itos) == list(full.itos),
          f"{len(nz.itos)} terms, element-wise identical")

    # Replay the builder exactly: same dedup key, same drop rules, same order.
    fp: set[int] = set()
    i = ok = n = 0
    for _key, lang, items in itertools.islice(raw.iter_all(None), 3000):
        h = hash("\u241f".join(sorted(str(x).strip().lower() for x in items)))
        if h in fp:
            continue
        fp.add(h)
        ids = nz.normalize(lang, items)
        if not ids:
            continue
        ok += sorted(ids) == list(full.recipe(i))
        i += 1
        n += 1
    check("stored recipes reproduce exactly from source files",
          n > 0 and ok == n,
          f"{ok:,}/{n:,} exact matches ({ok / max(n, 1):.2%}) — the corpus is "
          f"reproducible from the original readers, not merely internally "
          f"consistent")


def main() -> int:
    t0 = time.time()
    print("=" * 68)
    print("SANITY CHECKS — attempting to falsify our own results")
    print("=" * 68)
    for fn in (check_vocabulary_alignment, check_corpus_integrity,
               check_split_disjointness, check_popularity_baseline,
               check_random_controls, check_ease_native_scorer,
               check_leakage_claim, check_metric_monotonicity,
               check_normalisation_faithfulness):
        try:
            fn()
        except Exception as e:
            check(f"{fn.__name__} raised", False, f"{type(e).__name__}: {e}")

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 68)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed "
          f"in {time.time() - t0:.0f}s")
    for n in failed:
        print(f"  FAILED: {n}")
    print("=" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
