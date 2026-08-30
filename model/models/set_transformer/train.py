"""A permutation-invariant masked-ingredient model."""
from __future__ import annotations

import time

import numpy as np

from ingredient_model.data import load_recipes
from ingredient_model.data.recipes import RECIPE_IDS
from ingredient_model.registry import register
from ingredient_model.spec import TrainContext, TrainResult

DEFAULTS = dict(d_model=256, n_heads=4, n_layers=2, ff_mult=2, dropout=0.1,
                epochs=3, lr=1e-3, batch_size=512, max_recipes=600_000,
                max_len=32, tie_output=True, warmup=500)


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


@register(name="masked-set", family="set_transformer", cost_hint="heavy",
          defaults=DEFAULTS, tags=("recipes", "conditional", "transformer"),
          requires=("recipes",),
          description="Permutation-invariant transformer predicting a masked ingredient")
def train_masked_set(ctx: TrainContext) -> TrainResult:
    import torch
    from torch import nn

    p = {**DEFAULTS, **dict(ctx.params)}
    corpus = load_recipes(ctx.corpus or RECIPE_IDS)
    max_r = int(p["max_recipes"])
    if max_r and corpus.n_recipes > max_r:
        rng0 = np.random.default_rng(ctx.seed)
        corpus = corpus.select(
            np.sort(rng0.choice(corpus.n_recipes, max_r, replace=False)))

    torch.manual_seed(ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    vocab, device = corpus.n_vocab, ctx.device
    model = _build(vocab, p, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(p["lr"]),
                            weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    mask_id, warmup = vocab, int(p["warmup"])
    print(f"  {corpus.n_recipes:,} recipes, vocab {vocab}, "
          f"{sum(x.numel() for x in model.parameters()):,} parameters", flush=True)

    history, step, t0 = [], 0, time.time()
    for ep in range(int(p["epochs"])):
        tot, nb = 0.0, 0
        for ids_np, keep_np in corpus.batches(
                int(p["batch_size"]), min_size=3, seed=ctx.seed + ep,
                max_len=int(p["max_len"])):
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
            tot += float(loss.detach())
            nb += 1
        history.append(tot / max(nb, 1))
        print(f"  epoch {ep + 1}/{p['epochs']}  loss {history[-1]:.4f}  "
              f"steps {nb:,}  {time.time() - t0:.0f}s", flush=True)

    W = model.tok.weight.detach().cpu().numpy()[:vocab]
    state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

    def scorer(ctx_ids: np.ndarray) -> np.ndarray:
        """Score every candidate against the encoded context.

        This is the model. Ranking by cosine between the exported token table
        and a summed context — which is what the embedding path does — measures
        a bag-of-words shadow of an attention model, so both numbers are
        reported and this is the one that reflects what would be served.
        """
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
        model.train()
        return np.concatenate(outs)

    return TrainResult(
        embedding=W,
        scorer=scorer,
        metadata={"loss_history": history, "n_recipes": corpus.n_recipes,
                  "perplexity": float(np.exp(history[-1])), **p},
        extra_arrays={f"state__{k.replace('.', '__')}": v
                      for k, v in state.items()})
