"""Recover recipe text and rejoin it to the corpus.

The corpus stores `sorted(set(ingredient_ids))` and nothing else — no title, no
quantities, no instructions, no back-pointer to the row it came from. That was
the right call for co-occurrence models, but it throws away the three things the
next phase needs: **what the dish is called**, **how much of each ingredient**,
and **what you do with them**.

All of it still exists in the original source files. The problem is purely one of
alignment: the builder iterated 29 heterogeneous readers, dropped exact-duplicate
ingredient lists and dropped recipes that normalised to nothing, then wrote the
survivors in order. Nothing records which source row became which corpus row.

The join is recoverable because that iteration is deterministic, and this module
does not assume so — it *proves* it. :func:`build_index` replays the readers, and
for every single row asserts that re-normalising the source record reproduces the
stored corpus row exactly. If any source ever drifts, the build fails at the
offending index instead of silently emitting a corpus where recipe 3,000,000 is
labelled with recipe 2,999,999's title.

Two layers, deliberately separated:

* **Alignment** is generic and covers 100% of rows. It rides on
  ``corpus.iter_all``, which reports the source key of every record, so position
  can never drift even for a source whose text this module cannot read.
* **Text extraction** is per-source and best-effort. A source with no readable
  title yields an empty one; it is counted and reported, never guessed. A missing
  title is a gap. A misaligned title is a corrupted dataset, and the two must not
  be allowed to look alike.
"""
from __future__ import annotations

import ast
import csv
import glob
import hashlib
import json
import os
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import PATHS

TEXT_FILE = "recipe_text_v2.parquet"
TEXT_SCHEMA_VERSION = 2
LLMMM = PATHS.prior_study

# Per-source text columns. Only the ingredient column is load-bearing for
# alignment — these are additive, so an omission costs coverage, not correctness.
EXPANSION_TEXT: dict[str, dict[str, str | None]] = {
    "foodcom-522k": {"title": "Name", "steps": "RecipeInstructions",
                     "raw": "RecipeIngredientParts",
                     "ingredient_quantities": "RecipeIngredientQuantities"},
    "foodcom-raw-231k": {"title": "name", "steps": "steps"},
    "povarenok-detail": {"title": "title", "url": "page_url"},
    "turkish-102k": {"title": "Yemek İsmi", "url": "URL", "steps": "Yapılış"},
    "allrecipes-33k": {"title": "title", "url": "url", "steps": "directions"},
    "kaggle-food-13k": {"title": "Title", "steps": "Instructions"},
    "indian-7k": {"title": "name", "steps": "instructions"},
    "persian-6k": {"title": "name"},
    "greek-5k": {"title": "name", "steps": "Instructions"},
    "filipino-2k": {"title": "recipe_name", "steps": "instructions"},
    "taiwan-1.8k": {"title": "title", "url": "url", "steps": "steps"},
    "japanese-3k": {"title": "タイトル", "steps": "方向"},
    "romanian-881": {"title": "0"},
    "halal-2k": {"title": "title", "steps": "instructions"},
    "bhuvii-17k": {"title": "Title", "steps": "Instructions"},
    "thai-1k": {"title": "ชื่อเมนู", "steps": "วิธีทำ (Instructions)"},
    "thai-1k-seasoning": {"title": "ชื่อเมนู", "steps": "วิธีทำ (Instructions)"},
    # The whole recipe is one text blob whose first line is the title.
    "thefoodprocessor-74k": {"title": None, "title_from_value": "first_line"},
}

# Candidates tried when a source is not listed above, so a newly added reader
# picks up a title without anyone editing this file.
TITLE_FALLBACK = ("title", "Title", "name", "Name", "recipe_name", "RecipeName")


@dataclass(frozen=True)
class RecipeText:
    """Text for one recipe. Empty strings mean "not available", never "unknown"."""

    index: int
    source: str
    title: str
    url: str
    raw_ingredients: str
    steps: str
    ingredient_quantities: str = "[]"
    quantity_status: str = ""
    text_status: str = ""

    @property
    def has_text(self) -> bool:
        return bool(self.title)


# --------------------------------------------------------------------------
# metadata streams — each yields one dict per record its reader yields
# --------------------------------------------------------------------------

def _blank() -> dict:
    return {"title": "", "url": "", "raw": "", "steps": "",
            "ingredient_quantities": "[]", "quantity_status": "", "text_status": ""}


def _r_sequence(value) -> list[str | None]:
    """Decode serialized R vectors without dropping NA positions or executing code."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text in ("NA", "NULL", "NaN"):
        return []
    if text.startswith("c("):
        if not text.endswith(")"):
            raise ValueError("unterminated R vector")
        expression = "[" + text[2:-1] + "]"
    elif len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        escapes = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "'": "'", "\\": "\\"}
        return [re.sub(r"""\\([nrt"'\\])""",
                       lambda match: escapes[match.group(1)], text[1:-1])]
    elif text.startswith("["):
        expression = text
    else:
        return [text]
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        try:
            node = ast.parse(expression, mode="eval").body
        except (SyntaxError, SyntaxWarning) as error:
            raise ValueError("malformed R sequence") from error
    elements = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    values = []
    for element in elements:
        if isinstance(element, ast.Name) and element.id in {
                "NA", "NULL", "NaN", "NA_character_", "NA_real_", "NA_integer_"}:
            values.append(None)
        elif isinstance(element, ast.Constant) and (
                element.value is None or type(element.value) in (str, int, float)):
            if isinstance(element.value, str):
                literal = ast.get_source_segment(expression, element)
                if literal is None or not re.fullmatch(
                        r"""(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')""",
                        literal, flags=re.DOTALL):
                    raise ValueError("ingredient strings require explicit separators")
            values.append(None if element.value is None else str(element.value))
        else:
            raise ValueError("unsupported expression in an R sequence")
    return values


def _foodcom_fields(names_value, quantities_value, steps_value) -> dict:
    names = _r_sequence(names_value)
    quantities = _r_sequence(quantities_value)
    ingredient_names = []
    for name in names:
        if name is None or not name.strip():
            raise ValueError("Food.com contains an empty ingredient name")
        ingredient_names.append(name)
    status = "values_without_units" if quantities else "not_supplied"
    if quantities and len(quantities) != len(names):
        status = "count_mismatch"
    text_status = ""
    try:
        steps = "\x1f".join(value for value in _r_sequence(steps_value) if value)
    except ValueError:
        steps = _join(steps_value)
        text_status = "unparsed_steps"
    return {
        "raw": "\x1f".join(ingredient_names),
        "ingredient_quantities": json.dumps(quantities, ensure_ascii=False),
        "quantity_status": status,
        "steps": steps,
        "text_status": text_status,
    }


def _join(v) -> str:
    """Source instruction/ingredient fields are variously a JSON array, a Python
    list literal, a numpy array or a plain string."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, (list, tuple, np.ndarray)):
        return "\x1f".join(str(x) for x in v)
    s = str(v)
    if s[:1] in "[{":
        # literal_eval tokenises before it parses, so a fragment like
        # "[1 1/2 cups flour]" emits SyntaxWarning on the way to the
        # SyntaxError we already handle. Only the exception is informative.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            for parse in (json.loads, ast.literal_eval):
                try:
                    d = parse(s)
                except Exception:
                    continue
                if isinstance(d, dict):
                    return "\x1f".join(f"{k}: {x}" for k, x in d.items())
                if isinstance(d, (list, tuple)):
                    return "\x1f".join(str(x) for x in d)
    return s


def _meta_recipenlg(raw):
    f = raw.BASE / "01-recipenlg" / "RecipeNLG_dataset.csv"
    with f.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            ner = row.get("NER")
            if not ner:
                continue
            items = [x.strip().strip('"[]\' ').lower() for x in ner.split(",")]
            if not [x for x in items if x]:
                continue
            yield {"title": str(row.get("title") or ""),
                   "url": str(row.get("link") or ""),
                   "raw": _join(row.get("ingredients")),
                   "steps": _join(row.get("directions"))}


def _meta_xiachufang(raw):
    f = raw.BASE / "02-xiachufang" / "recipe_corpus_full.json"
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            ing = d.get("recipeIngredient") or []
            if not ing:
                continue
            items = [raw._ZH_QTY.sub("", s).strip() for s in ing]
            if not [x for x in items if x]:
                continue
            name = str(d.get("name") or "")
            dish = str(d.get("dish") or "")
            yield {"title": name or dish,
                   "url": "",
                   "raw": _join(ing),
                   "steps": _join(d.get("recipeInstructions"))}


def _meta_povarenok(raw):
    f = raw.BASE / "03-povarenok" / "povarenok.csv"
    with f.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                d = ast.literal_eval(row["ingredients"])
            except Exception:
                continue
            if not d:
                continue
            yield {"title": str(row.get("name") or ""),
                   "url": str(row.get("url") or ""),
                   "raw": _join(row.get("ingredients")),
                   "steps": ""}


BESPOKE = {"01-recipenlg": _meta_recipenlg,
           "02-xiachufang": _meta_xiachufang,
           "03-povarenok": _meta_povarenok}


def _resolve_cols(available, key):
    spec = dict(EXPANSION_TEXT.get(key) or {})
    spec.pop("title_from_value", None)
    if "title" not in spec:
        spec["title"] = next((c for c in TITLE_FALLBACK if c in available), None)
    return {k: (v if v in available else None) for k, v in spec.items()}


def _mirrorable(kind, col) -> bool:
    """Whether this module can reproduce a reader's yield sequence.

    Returning False is safe — the source then carries blank text while
    ``iter_all`` keeps its position. Returning True wrongly is not: the metadata
    stream would drift against the reader and mislabel every later row, so the
    bespoke `hebrew` and `moroccan` kinds, whose skip rules live inside the
    reader rather than in the ingredient column, are excluded deliberately.
    """
    return col is not None and kind in ("csv", "parquet", "jsonl")


def _meta_expansion(raw, key, kind, rel, col, split):
    """Mirror `_read_csv_col` / `_read_parquet_col` exactly.

    The skip rules there depend only on the ingredient column — a null value, or
    a splitter that returns nothing — so re-evaluating them on the same rows
    reproduces the yield sequence without duplicating the reader itself.
    """
    import pandas as pd

    p = raw.EXP / rel
    from_value = (EXPANSION_TEXT.get(key) or {}).get("title_from_value")
    if kind == "csv":
        frames = pd.read_csv(p, chunksize=100_000, engine="c",
                             on_bad_lines="skip", dtype=str)
    elif kind == "parquet":
        frames = (pd.read_parquet(f) for f in sorted(glob.glob(str(p))))
    elif kind == "jsonl":
        for line in p.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not split(d.get(col)):
                continue
            cols = _resolve_cols(set(d), key)
            yield {"title": str(d.get(cols.get("title")) or ""),
                   "url": str(d.get(cols.get("url")) or ""),
                   "raw": _join(d.get(col)),
                   "steps": _join(d.get(cols.get("steps")))}
        return
    else:
        return

    for chunk in frames:
        if col not in chunk.columns:
            return
        cols = _resolve_cols(set(chunk.columns), key)
        series = {k: chunk[v] for k, v in cols.items() if v}
        for i, v in enumerate(chunk[col]):
            if v is None or (isinstance(v, float) and v != v):
                continue
            if not split(v):
                continue
            out = _blank()
            for k, s in series.items():
                if k == "ingredient_quantities":
                    continue
                out["raw" if k == "raw" else k] = (
                    _join(s.iloc[i]) if k in ("steps", "raw")
                    else _join(s.iloc[i]))
            if not out["raw"]:
                out["raw"] = _join(v)
            if key == "foodcom-522k":
                quantities = series.get("ingredient_quantities")
                steps = series.get("steps")
                out.update(_foodcom_fields(
                    v, None if quantities is None else quantities.iloc[i],
                    None if steps is None else steps.iloc[i]))
            if from_value == "first_line" and not out["title"]:
                out["title"] = str(v).split("\n", 1)[0].strip()
            yield out


def _meta_stream(raw, key):
    if key in BESPOKE:
        return BESPOKE[key](raw)
    for k, lang, kind, rel, col, split in raw.EXPANSION:
        if k == key:
            if not _mirrorable(kind, col):
                return None
            return _meta_expansion(raw, key, kind, rel, col, split)
    return None


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def _load_llmmm():
    if not (LLMMM / "tools" / "corpus.py").exists():
        raise FileNotFoundError(
            f"original corpus tree not found at {LLMMM} — the text index can "
            f"only be built where the source files live")
    if str(LLMMM / "tools") not in sys.path:
        sys.path.insert(0, str(LLMMM / "tools"))
    import corpus as raw  # type: ignore
    import normalize as nm  # type: ignore
    return raw, nm


def _corpus_normalizer(nm):
    """The normaliser that built the canonical corpus.

    The index is verified row by row against the stored corpus, so it must
    replay through the same normaliser that produced it. Using the base one
    against a corrected corpus fails at the first recipe the correction touched
    — row 89 of RecipeNLG — which reads as a reader bug and is not one. Which
    one built it is recorded at promotion rather than inferred.
    """
    from ..config import corpus_generation
    if corpus_generation().get("normalizer", "base") == "corrected":
        from .normalizer import get_normalizer
        return get_normalizer(fix_zh_qty=True, extra=True)
    return nm.Normalizer()


def build_index(out: Path | None = None, *, limit: int | None = None,
                with_steps: bool = True, chunk: int = 100_000) -> Path:
    """Replay the readers, verify every row, and write the text index.

    Verification is not a sample. Every row is re-normalised and compared to the
    stored corpus, because a join that is right for the first two million rows
    and wrong afterwards is worse than no join at all — it would attach
    confident, plausible, wrong titles to exactly the recipes nobody spot-checks.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .recipes import load_recipes
    from ..config import corpus_generation

    out = out or (PATHS.recipes / TEXT_FILE)
    if out.exists():
        raise FileExistsError(f"{out}: choose a new output path; existing indexes are not overwritten")
    if chunk <= 0 or (limit is not None and limit <= 0):
        raise ValueError("chunk and limit must be positive")
    if limit is not None and out.resolve() == (PATHS.recipes / TEXT_FILE).resolve():
        raise ValueError("a partial index requires an explicit noncanonical --out path")
    raw, nm = _load_llmmm()
    corpus = load_recipes()
    nz = _corpus_normalizer(nm)
    if list(nz.itos) != list(corpus.itos):
        raise RuntimeError("normaliser and corpus disagree on the vocabulary; "
                           "the text index would be built against the wrong ids")

    schema = pa.schema([("idx", pa.int32()), ("source", pa.string()),
                        ("title", pa.string()), ("url", pa.string()),
                        ("raw_ingredients", pa.string()), ("steps", pa.string()),
                        ("ingredient_quantities", pa.string()), ("quantity_status", pa.string()),
                        ("text_status", pa.string())],
                       metadata={
                           "text_schema_version": str(TEXT_SCHEMA_VERSION),
                           "corpus_sha256": corpus_generation()["sha256"],
                           "reader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "partial": str(limit is not None).lower(),
                       })
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".parquet", delete=False) as temp:
        staged = Path(temp.name)
    writer = pq.ParquetWriter(staged, schema, compression="zstd")

    streams: dict[str, object] = {}
    missing: set[str] = set()
    buf: list[list] = [[] for _ in range(9)]
    fp: set[int] = set()
    idx = 0
    seen = 0
    n_title = 0
    n_url = 0
    n_steps = 0
    n_unparsed_steps = 0
    per_source: dict[str, list[int]] = {}

    def flush():
        if not buf[0]:
            return
        writer.write_table(pa.Table.from_arrays(
            [pa.array(values, type=field.type) for values, field in zip(buf, schema)],
            schema=schema))
        for b in buf:
            b.clear()

    try:
        for key, lang, items in raw.iter_all(None):
            seen += 1
            # The metadata stream must advance in lockstep with the reader,
            # including for records the builder is about to discard — otherwise
            # every later row in that source is shifted by one.
            if key not in streams and key not in missing:
                s = _meta_stream(raw, key)
                if s is None:
                    missing.add(key)
                else:
                    streams[key] = s
            meta = _blank()
            if key in streams:
                meta = next(streams[key], None)
                if meta is None:
                    raise RuntimeError(
                        f"metadata stream for {key} ran out at corpus row {idx}; "
                        f"its skip rules no longer match the reader")

            h = hash("\x1f".join(sorted(str(x).strip().lower() for x in items)))
            if h in fp:
                continue
            fp.add(h)
            ids = nz.normalize(lang, items)
            if not ids:
                continue

            if sorted(ids) != list(corpus.recipe(idx)):
                raise RuntimeError(
                    f"alignment lost at corpus row {idx} (source {key}): the "
                    f"replayed record does not reproduce the stored recipe")

            buf[0].append(idx)
            buf[1].append(key)
            buf[2].append(meta["title"])
            buf[3].append(meta["url"])
            buf[4].append(meta["raw"])
            buf[5].append(meta["steps"] if with_steps else "")
            buf[6].append(meta.get("ingredient_quantities", "[]"))
            buf[7].append(meta.get("quantity_status", ""))
            buf[8].append(meta.get("text_status", ""))
            st = per_source.setdefault(key, [0, 0])
            st[0] += 1
            st[1] += bool(meta["title"])
            n_title += bool(meta["title"])
            n_url += bool(meta["url"])
            n_steps += bool(meta["steps"]) and with_steps
            n_unparsed_steps += meta.get("text_status") == "unparsed_steps"
            idx += 1
            if len(buf[0]) >= chunk:
                flush()
                print(f"  {idx:,} rows  ({n_title / max(idx, 1):.1%} titled)",
                      flush=True)
            if limit and idx >= limit:
                break
        flush()
        if not limit and idx != corpus.n_recipes:
            raise RuntimeError(f"wrote {idx:,} rows but the corpus holds "
                               f"{corpus.n_recipes:,}")
        writer.close()
        os.link(staged, out)
    finally:
        writer.close()
        staged.unlink()
    print(f"\nverified {idx:,} rows against the corpus — every one reproduces\n"
          f"  titles {n_title:,} ({n_title / max(idx, 1):.1%})  "
          f"urls {n_url:,}  steps {n_steps:,}")
    if missing:
        print(f"  no text reader for: {', '.join(sorted(missing))}")
    if n_unparsed_steps:
        print(f"  WARNING: {n_unparsed_steps:,} instruction fields retain malformed "
              "source serialization; flagged as unparsed_steps, not generation-ready")
    return out


# --------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------

_CACHE = None


def load_text(columns: list[str] | None = None):
    """The text index as a pandas frame, indexed by corpus row."""
    global _CACHE
    import pandas as pd

    path = PATHS.recipes / TEXT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build it with `im text build`")
    if columns is not None:
        return pd.read_parquet(path, columns=columns)
    if _CACHE is None:
        _CACHE = pd.read_parquet(path)
    return _CACHE


def text_of(index: int) -> RecipeText:
    df = load_text()
    r = df.iloc[index]
    return RecipeText(index=int(r["idx"]), source=str(r["source"]),
                      title=str(r["title"]), url=str(r["url"]),
                      raw_ingredients=str(r["raw_ingredients"]),
                      steps=str(r["steps"]),
                      ingredient_quantities=str(r.get("ingredient_quantities", "[]")),
                      quantity_status=str(r.get("quantity_status", "")),
                      text_status=str(r.get("text_status", "")))


def has_text() -> bool:
    return (PATHS.recipes / TEXT_FILE).exists()
