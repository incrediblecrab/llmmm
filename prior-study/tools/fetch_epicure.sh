#!/usr/bin/env bash
# Fetch all Epicure artefacts (CC BY 4.0) from HuggingFace.
# Paper: Radzikowski & Chen 2026, arXiv:2605.22391
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"

MODEL_FILES=(
  LICENSE README.md config.json cuisine_pole_provenance.json
  embeddings.safetensors epicure.py factor_pole_index.json factor_poles.npy
  itos.json modes.json paper_slerp_results.csv supervised_poles.json vocab.json
)

DATA_FILES=(
  LICENSE README.md
  data/canonical_vocabulary.parquet data/cross_modal.parquet
  data/cuisine_macroregions.json data/direction_arithmetic_full.parquet
  data/direction_orthogonal.parquet
  data/epicure_chem.csv data/epicure_cooc.csv data/epicure_core.csv
  data/factor_top_alignments_chem.parquet data/factor_top_alignments_cooc.parquet
  data/factor_top_alignments_core.parquet
  data/linear_probe_categorical.parquet data/linear_probe_continuous.parquet
  data/mode_atlas_chem.parquet data/mode_atlas_cooc.parquet data/mode_atlas_core.parquet
  data/procrustes_sensory.parquet data/supplement.pdf data/vocab.csv data/weat.parquet
)

get() { # url dest
  if [[ -s "$2" ]]; then printf '  = %s\n' "$(basename "$2")"; return 0; fi
  if curl -sfL --retry 3 --retry-delay 2 -o "$2" "$1"; then
    printf '  + %-38s %s\n' "$(basename "$2")" "$(du -h "$2" | cut -f1)"
  else
    printf '  ! FAILED %s\n' "$1" >&2; rm -f "$2"; return 1
  fi
}

for sib in cooc core chem; do
  echo "== Kaikaku/epicure-$sib"
  mkdir -p "$RAW/epicure-$sib"
  for f in "${MODEL_FILES[@]}"; do
    get "https://huggingface.co/Kaikaku/epicure-$sib/resolve/main/$f" "$RAW/epicure-$sib/$f"
  done
done

echo "== Kaikaku/epicure-corpus-resources"
mkdir -p "$RAW/epicure-corpus-resources/data"
for f in "${DATA_FILES[@]}"; do
  get "https://huggingface.co/datasets/Kaikaku/epicure-corpus-resources/resolve/main/$f" \
      "$RAW/epicure-corpus-resources/$f"
done

echo
echo "Total: $(du -sh "$RAW" | cut -f1)"
