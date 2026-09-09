"""A permutation-invariant masked-ingredient model."""
from __future__ import annotations

import time
import hashlib
from pathlib import Path

import numpy as np

from ingredient_model.data import load_recipes
from ingredient_model.data.recipes import RECIPE_IDS
from ingredient_model.registry import register
from ingredient_model.spec import TrainContext, TrainResult

DEFAULTS = dict(d_model=256, n_heads=4, n_layers=2, ff_mult=2, dropout=0.1,
                epochs=3, lr=1e-3, batch_size=512, max_recipes=600_000,
                min_len=3, max_len=32, expected_recipes=None,
                tie_output=True, warmup=500)


def _build(vocab: int, p: dict, device: str):
    import torch
    from torch import nn

    d = int(p["d_model"])
    tie = bool(p["tie_output"])

    class MaskedSetModel(nn.Module):
        """Encode the visible ingredients, then score every candidate.

        No positional encoding: a recipe is a set, and the order it was scraped
        in carries no culinary information. Omitting positions makes the encoder
        permutation-equivariant by construction rather than by hoping the model
        learns to ignore them.
        """

        def __init__(self):
            super().__init__()
            # One extra row for [MASK]. Keeping it inside the same table means
            # the mask token lives in the ingredient space and its learned
            # position is interpretable as "the average missing ingredient".
            self.tok = nn.Embedding(vocab + 1, d)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=int(p["n_heads"]),
                dim_feedforward=d * int(p["ff_mult"]),
                dropout=float(p["dropout"]), batch_first=True,
                norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, int(p["n_layers"]))
            self.norm = nn.LayerNorm(d)
            self.bias = nn.Parameter(torch.zeros(vocab))
            # Untied models need their own output table; tied ones score
            # against `tok` directly. Assigning a *slice* of `tok.weight` to an
            # nn.Linear would copy it and silently untie the two, so the tied
            # path multiplies by the live weight instead of holding a second
            # parameter.
            self.head = None if tie else nn.Parameter(
                torch.randn(vocab, d) * d ** -0.5)
            nn.init.normal_(self.tok.weight, std=d ** -0.5)

        def output_table(self) -> torch.Tensor:
            return self.tok.weight[:vocab] if self.head is None else self.head

        def encode(self, ids, pad_mask, mask_pos):
            h = self.enc(self.tok(ids), src_key_padding_mask=pad_mask)
            return self.norm(h[torch.arange(len(ids), device=ids.device), mask_pos])

        def forward(self, ids, pad_mask, mask_pos):
            return self.encode(ids, pad_mask, mask_pos) @ self.output_table().T \
                + self.bias

    return MaskedSetModel().to(device)


def _make_scorer(model, mask_id: int, device: str):
    """Wrap a built model as the conditional scorer the evaluation calls.

    Shared by training and by :func:`restore` so that a rebuilt model scores
    through exactly the code the reported number came from. A second
    implementation here would be free to drift, and the whole point of
    restoring the model is to reproduce a published figure.
    """
    import torch

    def scorer(ctx_ids: np.ndarray) -> np.ndarray:
        """Score every candidate against the encoded context.

        This is the model. Ranking by cosine between the exported token table
        and a summed context — which is what the embedding path does — measures
        a bag-of-words shadow of an attention model, so both numbers are
        reported and this is the one that reflects what would be served.
        """
        was_training = model.training
        model.eval()
        m, k = ctx_ids.shape
        padded = np.concatenate(
            [ctx_ids, np.full((m, 1), mask_id, np.int64)], axis=1)
        outs = []
        with torch.no_grad():
            for i in range(0, m, 4096):
                chunk = torch.from_numpy(padded[i:i + 4096]).to(device)
                pad = torch.zeros(chunk.shape, dtype=torch.bool, device=device)
                pos = torch.full((len(chunk),), k, dtype=torch.long, device=device)
                outs.append(model(chunk, pad, pos).cpu().numpy())
        if was_training:
            model.train()
        return np.concatenate(outs)

    return scorer


def restore(run_dir, vocab: int, device: str = "cpu"):
    """Rebuild this run's trained scorer from the state tensors it saved.

    The set transformer is the one model whose served scorer is not a matrix
    that can simply be loaded — it is a forward pass — so anything wanting to
    re-score it later (a bootstrap over the completion instances, say) would
    otherwise have to retrain. The weights were already persisted as
    `extra_arrays`; this is the inverse of that flattening.

    Hyperparameters come from the run's own manifest rather than from
    ``DEFAULTS``, so a run trained with overrides restores as itself.
    """
    import json
    from pathlib import Path

    import torch

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "manifest.json").read_text()).get("metadata", {})
    p = {**DEFAULTS, **{k: v for k, v in meta.items() if k in DEFAULTS}}

    state = {}
    for f in sorted(run_dir.glob("state__*.npy")):
        key = f.name[len("state__"):-len(".npy")].replace("__", ".")
        state[key] = torch.from_numpy(np.load(f))
    if not state:
        raise FileNotFoundError(f"no state__*.npy tensors in {run_dir}")

    model = _build(vocab, p, device)
    model.load_state_dict(state)
    model.eval()
    return _make_scorer(model, vocab, device)


@register(name="masked-set", family="set_transformer", cost_hint="heavy",
          defaults=DEFAULTS, tags=("recipes", "conditional", "transformer"),
          requires=("recipes",),
          description="Permutation-invariant transformer predicting a masked ingredient")
def train_masked_set(ctx: TrainContext) -> TrainResult:
    import torch
    from torch import nn

    p = {**DEFAULTS, **dict(ctx.params)}
    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    corpus = load_recipes(ctx.corpus or RECIPE_IDS)
    max_r = int(p["max_recipes"])
    if max_r and corpus.n_recipes > max_r:
        rng0 = np.random.default_rng(ctx.seed)
        corpus = corpus.select(
            np.sort(rng0.choice(corpus.n_recipes, max_r, replace=False)))

    min_len = int(p["min_len"])
    max_len = None if p["max_len"] is None else int(p["max_len"])
    if min_len < 1 or int(p["epochs"]) < 1:
        raise ValueError("min_len and epochs must be positive")
    recipe_sizes = corpus.sizes
    expected_rows = recipe_sizes >= min_len
    if max_len is not None:
        expected_rows &= recipe_sizes <= max_len
    eligible = int(expected_rows.sum())
    if not eligible:
        raise ValueError("no recipes satisfy the masked-set training length bounds")
    if p["expected_recipes"] is not None:
        expected = p["expected_recipes"]
        if type(expected) is not int or expected < 1:
            raise ValueError("expected_recipes must be a positive integer")
        if corpus.n_recipes != expected or eligible != expected:
            raise ValueError(
                f"expected all {expected:,} recipes, found {corpus.n_recipes:,} "
                f"sampled and {eligible:,} eligible")
    expected_slots = int(recipe_sizes[expected_rows].sum())
    torch.manual_seed(ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    vocab, device = corpus.n_vocab, ctx.device
    model = _build(vocab, p, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(p["lr"]),
                            weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    mask_id, warmup = vocab, int(p["warmup"])
    print(f"  {corpus.n_recipes:,} sampled recipes, {eligible:,} eligible, vocab {vocab}, "
          f"{sum(x.numel() for x in model.parameters()):,} parameters", flush=True)

    history, coverage, step, examples_seen, slots_seen, t0 = [], [], 0, 0, 0, time.time()
    for ep in range(int(p["epochs"])):
        tot, nb, epoch_examples, epoch_slots = 0.0, 0, 0, 0
        visits = np.zeros(corpus.n_recipes, dtype=np.uint32)
        for ids_np, keep_np, rows in corpus.indexed_batches(
                int(p["batch_size"]), min_size=min_len, seed=ctx.seed + ep,
                max_len=max_len):
            m = len(ids_np)
            lengths = keep_np.sum(1)
            hide = (rng.random(m) * lengths).astype(np.int64)
            target = ids_np[np.arange(m), hide].copy()

            ids = torch.from_numpy(ids_np).to(device)
            pad = torch.from_numpy(~keep_np).to(device)
            pos = torch.from_numpy(hide).to(device)
            ids[torch.arange(m), pos] = mask_id
            logits = model(ids, pad, pos)
            loss = lossf(logits, torch.from_numpy(target).to(device))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            step += 1
            if step <= warmup:
                # Linear warmup. Without it the first few hundred steps of an
                # untrained attention stack produce gradients large enough to
                # push the token table into a degenerate configuration it does
                # not recover from.
                for gparam in opt.param_groups:
                    gparam["lr"] = float(p["lr"]) * step / warmup
            opt.step()
            np.add.at(visits, rows, 1)
            examples_seen += m
            epoch_examples += m
            slots_seen += int(lengths.sum())
            epoch_slots += int(lengths.sum())
            tot += float(loss.detach())
            nb += 1
            if nb % 1000 == 0:
                print(f"  epoch {ep + 1}/{p['epochs']}  "
                      f"{epoch_examples:,}/{eligible:,} recipes  "
                      f"loss {tot / nb:.4f}", flush=True)
        if not np.array_equal(visits, expected_rows.astype(np.uint32)):
            raise RuntimeError("training batches did not visit each eligible recipe exactly once")
        if epoch_slots != expected_slots:
            raise RuntimeError("training batches omitted or duplicated ingredient slots")
        coverage.append({"epoch": ep + 1, "unique_recipes": int(np.count_nonzero(visits)),
                         "examples_seen": epoch_examples,
                         "ingredient_slots_seen": epoch_slots,
                         "every_eligible_recipe_once": True})
        history.append(tot / max(nb, 1))
        print(f"  epoch {ep + 1}/{p['epochs']}  loss {history[-1]:.4f}  "
              f"steps {nb:,}  {time.time() - t0:.0f}s", flush=True)

    W = model.tok.weight.detach().cpu().numpy()[:vocab]
    state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

    scorer = _make_scorer(model, mask_id, device)

    return TrainResult(
        embedding=W,
        scorer=scorer,
        metadata={"loss_history": history, "n_recipes": corpus.n_recipes,
                  "n_eligible_recipes": eligible, "n_examples_seen": examples_seen,
                  "n_ingredient_slots_seen": slots_seen,
                  "observed_min_len": int(recipe_sizes[expected_rows].min()),
                  "observed_max_len": int(recipe_sizes[expected_rows].max()),
                  "n_optimizer_steps": step, "torch_num_threads": torch.get_num_threads(),
                  "epoch_coverage": coverage, "training_code_sha256": code_sha256,
                  "perplexity": float(np.exp(history[-1])), **p},
        extra_arrays={f"state__{k.replace('.', '__')}": v
                      for k, v in state.items()})
