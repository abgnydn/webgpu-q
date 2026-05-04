// ─────────────────────────────────────────────────────────────
// stats.ts — trimmed statistics for benchmark samples.
//
// Every experiment discards W warmup samples and keeps T trials.
// This file computes median, p10, p90, p99, mean, std, IQR and
// flags the sample as NOISY if std / median > 0.1. The threshold
// matches RESEARCH.md §Timing. Noisy cells block a run from
// publishing numbers without investigation.
// ─────────────────────────────────────────────────────────────

export interface Stats {
  readonly n: number;
  readonly median: number;
  readonly p10: number;
  readonly p90: number;
  readonly p99: number;
  readonly min: number;
  readonly max: number;
  readonly mean: number;
  readonly std: number;
  readonly iqr: number;           // p75 − p25
  readonly cv: number;            // std / mean (coefficient of variation)
  readonly noisy: boolean;        // std/median > 0.1 → investigate
  readonly raw: readonly number[]; // kept for audit
}

/** Percentile via linear interpolation (type 7, numpy / R default). */
export function percentile(sortedAsc: readonly number[], p: number): number {
  const n = sortedAsc.length;
  if (n === 0) return NaN;
  if (n === 1) return sortedAsc[0]!;
  const idx = (p / 100) * (n - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const w = idx - lo;
  return (1 - w) * sortedAsc[lo]! + w * sortedAsc[hi]!;
}

export function stats(samples: readonly number[]): Stats {
  const n = samples.length;
  if (n === 0) throw new Error("stats: empty sample");
  const sorted = [...samples].sort((a, b) => a - b);

  const mean = sorted.reduce((a, b) => a + b, 0) / n;
  let sumSq = 0;
  for (const x of sorted) sumSq += (x - mean) ** 2;
  const std = Math.sqrt(sumSq / Math.max(1, n - 1));

  const median = percentile(sorted, 50);
  const p10 = percentile(sorted, 10);
  const p25 = percentile(sorted, 25);
  const p75 = percentile(sorted, 75);
  const p90 = percentile(sorted, 90);
  const p99 = percentile(sorted, 99);

  return {
    n,
    median,
    p10, p90, p99,
    min: sorted[0]!,
    max: sorted[n - 1]!,
    mean,
    std,
    iqr: p75 - p25,
    cv: mean === 0 ? 0 : std / mean,
    noisy: median === 0 ? false : (std / median) > 0.1,
    raw: sorted,
  };
}

/** Format a Stats block for logging (human-readable, one line). */
export function fmtStats(s: Stats, unit = ""): string {
  const u = unit ? ` ${unit}` : "";
  const flag = s.noisy ? " [NOISY]" : "";
  return `med=${s.median.toFixed(3)}${u} p10=${s.p10.toFixed(3)} p90=${s.p90.toFixed(3)} p99=${s.p99.toFixed(3)} std=${s.std.toFixed(3)} cv=${s.cv.toFixed(3)}${flag}`;
}
