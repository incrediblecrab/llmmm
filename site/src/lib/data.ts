/* The site's only route to its own data.
 *
 * Everything here is read from `public/data/`, which is written by
 * `model/scripts/export_site.py` and by nothing else. No number is typed into
 * a template. That constraint is the whole reason the site can claim its
 * figures match its runs: if a value is not in an artefact, there is no way
 * to put it on the page.
 */

import leaderboardRaw from "../../public/data/leaderboard.json";
import comparisonRaw from "../../public/data/comparison.json";
import vocabRaw from "../../public/data/vocab.json";
import provenanceRaw from "../../public/data/provenance.json";
import splitsRaw from "../../public/data/splits.json";
import corpusRaw from "../../public/data/corpus.json";

export interface ModelRow {
  model: string;
  family: string;
  run_id: string;
  run_dir: string;
  seed: number;
  generation: string;
  duration_s: number;
  dims: number | null;
  /** True when the model factorises something richer than the vectors it
   *  exports, so it has a second, better score under its own rule. */
  has_native: boolean;
  /** The number the model would actually serve — native where one exists,
   *  otherwise the embedding. This is the column the leaderboard sorts on. */
  best_recall_at_10: number;
  best_lift: number;
  M6_recall_at_10?: number;
  M6_native_recall_at_10?: number;
  M6_centred_recall_at_10?: number;
  M6_lift_over_popularity?: number;
  M6_mrr?: number;
  M4_link_auc?: number;
  M4_link_auc_ci95?: number;
  M2_triplet_accuracy_strict?: number;
  M2_triplet_accuracy_strict_ci95?: number;
  M1_participation_ratio?: number;
  M5_max_pc_freq_corr?: number;
  M6_n?: number;
  /** 95% bootstrap interval on the served score, and on its distance from
   *  the baseline. Present only once `scripts/bootstrap_m6.py` has run. */
  ci95?: [number, number];
  lift_ci95?: [number, number];
  p_two_sided?: number;
  /** True when no bootstrap replicate landed on the other side, so the
   *  p-value is the resolution limit rather than a measurement. */
  p_is_bound?: boolean;
}

export interface Bootstrap {
  n_boot: number;
  n_instances: number;
  popularity_ci95: [number, number];
  n_below_popularity_ci95: [number, number];
  n_below_popularity_distribution: Record<string, number>;
  n_models_with_ci: number;
}

export interface Leaderboard {
  generation: string;
  split: string;
  popularity_recall_at_10: number;
  n_models: number;
  n_below_popularity: number;
  /** Rank correlation between each offline metric and the completion task,
   *  with a p-value. Computed at export time from the runs themselves. */
  agreement: Record<string, { spearman: number; p: number; n: number }>;
  /** Null until the bootstrap has been run, so the site degrades to point
   *  estimates rather than failing to build. */
  bootstrap: Bootstrap | null;
  models: ModelRow[];
}

export const leaderboard = leaderboardRaw as unknown as Leaderboard;
export const comparison = comparisonRaw as unknown as {
  caveat: string;
  before: { sweep: string; generation: string };
  after: { sweep: string; generation: string };
  metrics: string[];
  models: Array<Record<string, any>>;
};
export const vocab = vocabRaw as unknown as {
  n: number;
  total_slots: number;
  by_frequency: Array<{ id: number; name: string; count: number }>;
};
export const provenance = provenanceRaw as unknown as {
  commit: string;
  generation: Record<string, unknown>;
};

export const POPULARITY = leaderboard.popularity_recall_at_10;

/** The leakage demonstration: one model, one metric, two split protocols.
 *  Both sides are runs in this repository — the page that argues for honest
 *  evaluation is the last place a number should be typed by hand. */
export const splits = splitsRaw as unknown as {
  note: string;
  metric: string;
  model: string;
  leaky: { split: string; value: number; ci95: number | null; run_dir: string };
  honest: { split: string; value: number; ci95: number; run_dir: string };
  gap: number;
  gap_in_ci_units: number | null;
};

/** The corpus the models were trained on, as stamped into every run manifest
 *  by `data/GENERATION.json`. The site quotes these rather than any number
 *  written in a README, because the README is downstream of this file too. */
export const generation = (corpusRaw as any).generation as {
  generation: string;
  recipes: number;
  slots: number;
  vocab: number;
  ii_graph_edges: number;
  min_count: number;
  sha256: string;
  previous: { generation: string; recipes: number; slots: number };
};

/** Written by `scripts/rebuild_corpus.py --stats-only`, in the same pass that
 *  decides each recipe's fate, so the waterfall cannot disagree with the
 *  artefact it describes. The exporter refuses to publish it unless the kept
 *  count equals the corpus and the per-source columns sum to the waterfall.
 *  Optional: the site omits the figures rather than inventing them. */
export interface CorpusSource {
  source: string;
  scanned: number;
  kept: number;
  dup: number;
  empty: number;
  slots: number;
}

export interface CorpusStats {
  note: string;
  n_sources: number;
  waterfall: {
    scanned: number;
    kept: number;
    dropped_duplicate: number;
    dropped_matched_nothing: number;
    slots: number;
  };
  per_source: CorpusSource[] | Record<string, Omit<CorpusSource, "source">>;
}

export const corpusStats = ((corpusRaw as any).stats ?? null) as CorpusStats | null;

/** Per-source rows, largest surviving corpus first. Accepts either shape the
 *  scan may write, so the site does not care which the script chose. */
export const corpusSources: CorpusSource[] = (() => {
  if (!corpusStats) return [];
  const p = corpusStats.per_source;
  const rows = Array.isArray(p)
    ? p
    : Object.entries(p).map(([source, v]) => ({ source, ...v }));
  return rows.slice().sort((a, b) => b.kept - a.kept);
})();

/** Share of the modelled corpus contributed by a source. */
export const sourceShare = (kept: number): number =>
  corpusStats ? kept / corpusStats.waterfall.kept : 0;

/** Thousands separators, once, so 4,653,430 is never hand-typed. */
export const n = (v: number): string => v.toLocaleString("en-US");

/** Every internal link on the site.
 *
 *  Relative hrefs are wrong here: the site builds to directory routes, so
 *  `./native/` written on `/models/` resolves to `/models/native/` and 404s.
 *  Routing through one helper also means a change of deployment base — a
 *  project page under a repository name, say — is a single edit rather than a
 *  search across nine pages.
 */
const BASE = import.meta.env.BASE_URL.replace(/\/+$/, "");
export const href = (path: string): string =>
  BASE + (path.startsWith("/") ? path : `/${path}`);

/** Ingredient names in vocabulary-id order, for turning model output into
 *  words. Built once rather than per lookup. */
export const NAMES: string[] = (() => {
  const out = new Array<string>(vocab.n);
  for (const v of vocab.by_frequency) out[v.id] = v.name;
  return out;
})();

/** Underscores are how the normaliser stores multi-word ingredients; they are
 *  an implementation detail and should never reach a reader. */
export const pretty = (name: string): string => name.replace(/_/g, " ");

/** A model is "below the line" when the best score it can produce still loses
 *  to ranking ingredients by frequency. Defined once, here, because it is the
 *  site's central claim and must mean the same thing on every page. */
export const isBelow = (m: ModelRow): boolean => m.best_lift < 0;

export const belowLine = leaderboard.models.filter(isBelow);
export const aboveLine = leaderboard.models.filter((m) => !isBelow(m));

/** The uncertainty attached to that claim, or null before the bootstrap has
 *  been run. Every page that quotes "nine of sixteen" reads the interval from
 *  here, so the qualifier can never drift away from the number. */
export const bootstrap = leaderboard.bootstrap;

/** How often the resampled leaderboard produced the published count. The
 *  bootstrap resamples evaluation instances, so this speaks to how the models
 *  were scored, not to how they were trained — the seed sweep is the separate
 *  question and is answered on the robustness page. */
export const countIsExact = (): boolean => {
  const b = bootstrap;
  if (!b) return false;
  const [lo, hi] = b.n_below_popularity_ci95;
  return lo === hi && lo === leaderboard.n_below_popularity;
};

/** A 95% interval rendered the way a table should render one. */
export const ci = (r?: [number, number], digits = 4): string =>
  r ? `[${r[0].toFixed(digits)}, ${r[1].toFixed(digits)}]` : "";

/** Sorted for display: best first, and always by the score the model would
 *  actually serve rather than by whichever column flatters it. */
export const ranked = [...leaderboard.models].sort(
  (a, b) => b.best_recall_at_10 - a.best_recall_at_10,
);

export const byName = (n: string): ModelRow | undefined =>
  leaderboard.models.find((m) => m.model === n);

/** Multi-seed stability, written by `scripts/ranking_stability.py` and
 *  exported only once every model has run at every seed.
 *
 *  Loaded through `import.meta.glob` rather than a static import because the
 *  exporter deliberately withholds the file until the sweep is complete: a
 *  static import of a missing file fails the build, and a half-finished sweep
 *  should degrade the page, not break it. `stability` is null until then, and
 *  every consumer has to say what it shows in that case.
 */
export interface StabilityModel {
  model: string;
  mean: number;
  sd: number;
  min: number;
  max: number;
  rank_best: number;
  rank_worst: number;
  rank_mean: number;
  by_seed: Record<string, number>;
  below_popularity: boolean[];
}

export interface Stability {
  note: string;
  metric: string;
  seeds: number[];
  n_models_complete: number;
  models_partial: string[];
  baseline_by_seed: Record<string, number>;
  n_below_popularity_by_seed: Record<string, number>;
  n_models_changing_rank: number;
  /** How many models are on different sides of the popularity baseline at
   *  different seeds. This, not the rank count, is what decides whether the
   *  site's claim survives reseeding: two models trading places matters only
   *  if the swap carries one of them across the line. */
  n_models_crossing_baseline: number;
  /** Why this file's count of models below the baseline is larger than the
   *  leaderboard's. The count here is over exported embeddings; the headline
   *  scores each model at its best available scorer. The sweep never re-runs
   *  a native scorer, so it holds no seed evidence about those models. */
  below_popularity_scope: {
    basis: string;
    note: string;
    native_not_reseeded: {
      model: string;
      embedding: number;
      native: number;
      embedding_below_baseline: boolean;
      native_below_baseline: boolean;
    }[];
  };
  /** Models whose score is bit-identical at every seed. Their variance is
   *  absent rather than small — a truncated SVD returns the same
   *  factorisation every time — so averaging them into a mean sd would
   *  describe nothing. */
  seed_invariant: string[];
  max_seed_spread: number;
  min_adjacent_gap: number;
  spearman_between_seeds: {
    pairs: number[][];
    values: number[];
    mean: number;
    min: number;
    ci95: [number, number];
    n_boot: number;
  };
  models: StabilityModel[];
  /** Whether the sweep's own seed-0 trials return the numbers already
   *  published from the single-seed runs. If they did not, seed-to-seed
   *  differences would be measuring pipeline drift rather than the seed. */
  reproduces_published_seed0: {
    available: boolean;
    reason?: string;
    canonical_runs?: string;
    n_compared?: number;
    n_mismatched?: number;
    max_abs_diff?: number;
    exact?: boolean;
    models_compared?: string[];
    not_yet_in_sweep?: string[];
  };
  complete: boolean;
}

const stabilityMod = import.meta.glob<{ default: Stability }>(
  "../../public/data/stability.json",
  { eager: true },
);

export const stability: Stability | null =
  Object.values(stabilityMod)[0]?.default ?? null;

/** The decisive comparison is not the spread but its size against the gaps it
 *  would have to cross. A large spread between well-separated models still
 *  cannot reorder them. */
export const spreadRatio = (s: Stability): number =>
  s.min_adjacent_gap / s.max_seed_spread;

/** Sorted best-first on the mean, matching the leaderboard's ordering rule. */
export const stabilityRanked = (s: Stability): StabilityModel[] =>
  [...s.models].sort((a, b) => b.mean - a.mean);

/** Models whose exported vectors lose badly to their own native scorer. The
 *  gap is the site's second finding — the vector table is the lossy part, and
 *  the vector table is what everyone ships. */
export const nativeGap = leaderboard.models
  .filter((m) => m.has_native && m.M6_recall_at_10 != null)
  .map((m) => ({
    model: m.model,
    embedding: m.M6_recall_at_10!,
    native: m.M6_native_recall_at_10!,
    ratio: m.M6_native_recall_at_10! / m.M6_recall_at_10!,
  }))
  .sort((a, b) => b.ratio - a.ratio);
