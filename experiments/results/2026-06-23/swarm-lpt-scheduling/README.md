# Cost-aware (LPT) swarm scheduling — the throughput win

**Question:** the swarm's real axis is throughput (N independent molecules). But
`swarm-scaling` found it sub-linear (~59% at 4 tabs) because molecule costs are
uneven and FIFO hands a heavy molecule out late → one tab tails past everyone.
Does **LPT** (longest-processing-time-first — sort the queue heaviest-first) fix it?

**Answer: yes, clearly — the first unambiguous win on this axis**, and the real win
is *bigger* than the model. On a real 4-tab cloud swarm (GitHub Actions, 4 cores)
**LPT ran 1.98× faster than FIFO** (89.5 s → 45.2 s) — see *Empirical* below. The
compute-only sim under-predicted it (~1.2×): +19 efficiency points at 3 tabs, +11 at
4, with LPT reaching the *optimal* makespan (the longest single molecule). The
ceiling then becomes that one indivisible molecule, not the scheduler.

## Method

10-molecule library at cc-pVDZ HF (the `swarm-scaling` library). Each molecule's
real HF cost timed single-threaded; greedy-pull makespan then simulated
(`simulateGreedyMakespan`) for k=1..4 under **FIFO** (library order) vs **LPT**
(ordered by the `chemTileCost` = n⁴ predictor, executed with REAL costs — so this
tests the actual a-priori scheduler, not an oracle). Compute-only ceiling; the
real wall-clock with comms/scheduler overhead is `e2e/swarm-scheduling.spec.ts`.

## Measured per-molecule cost (HF, cc-pVDZ)

| mol | n | HF | | mol | n | HF |
|---|--:|--:|---|---|--:|--:|
| H₂ | 10 | 0.1 s | | H₂O | 25 | 1.8 s |
| LiH | 20 | 0.9 s | | CH₄ | 35 | 5.4 s |
| HF | 20 | 0.9 s | | C₂H₂ | 40 | 17.3 s |
| N₂ | 30 | 7.4 s | | CH₂O | 40 | 17.6 s |
| CO | 30 | 8.7 s | | **C₂H₄** | **50** | **34.0 s** |

Predictor orders LPT `[C₂H₄, C₂H₂, CH₂O, CH₄, N₂, CO, H₂O, LiH, HF, H₂]` — heavy
first. (n⁴ groups same-n molecules, but ranks the genuinely heavy ones correctly.)

## Result

| k tabs | FIFO speedup (eff) | LPT speedup (eff) | makespan FIFO→LPT |
|--:|---|---|---|
| 1 | 1.00× (100%) | 1.00× (100%) | 94.1 s → 94.1 s |
| 2 | 1.92× (96%) | 1.96× (98%) | 49.0 s → 48.1 s |
| **3** | 2.20× (73%) | **2.77× (92%)** | 42.8 s → **34.0 s** |
| **4** | 2.33× (58%) | **2.77× (69%)** | 40.4 s → **34.0 s** |

## Interpretation

1. **LPT wins, big, where it matters.** At 3 tabs it lifts efficiency **73% → 92%**
   (2.20× → 2.77×); at 4 tabs **58% → 69%**. FIFO's 58% at k=4 reproduces the
   project's documented ~59% sub-linear ceiling — LPT is the fix the note called for.
2. **LPT reaches the optimal makespan.** At k ≥ 3, LPT makespan = **34.0 s = C₂H₄'s
   runtime** — the theoretical floor (you can't finish faster than the single
   longest job). FIFO mis-packs and overshoots it (40–43 s).
3. **The new ceiling is the longest *indivisible* molecule, not the scheduler.**
   Beyond 3 tabs, more tabs don't help *this* library: C₂H₄ alone is 34 s and can't
   be split, so the 4th tab idles (that's why k=4 eff 69% < k=3 92%). To push past
   it you'd need a bigger / more-uniform library, or to split the heaviest molecule
   — i.e. single-molecule distribution, which is capped at ~1.3× (see the federated
   (T) result). The two axes meet exactly here.

## Scaling law — how far does throughput go?

The idle-4th-tab cap above is a special case of a general rule. Sweeping tab count
on the real costs, and on a 3× library (a realistic 30-molecule screen):

> **Throughput scales near-linearly under LPT until k ≈ total_work / longest_single_job,
> then the longest INDIVISIBLE molecule floors the makespan.**

| k | 10-mol (total 94 s, longest 34 s → cap k≈2.8) | 30-mol (total 282 s → cap k≈8.3) |
|--:|---|---|
| | FIFO / **LPT** | FIFO / **LPT** |
| 2 | 1.92× / **1.97×** | 1.97× / **2.00×** |
| 4 | 2.33× (58%) / **2.77× (69%)** | 3.32× (83%) / **3.98× (100%)** |
| 6 | 2.70× (45%) / **2.77× (46%)** | 4.33× (72%) / **5.92× (99%)** |
| 8 | 2.76× (35%) / **2.77× (35%)** | 5.23× (65%) / **7.60× (95%)** |

1. **LPT's advantage GROWS with scale.** On the realistic 30-molecule screen, LPT
   holds ~100% efficiency through 5 tabs and **95% at 8 tabs (7.60×)**, while FIFO
   degrades to 65%. The 10-molecule library understated LPT precisely *because* it
   was too small (capped at k≈3 by C₂H₄).
2. **The cap is predictable.** Past k ≈ total/longest, the single longest molecule
   floors the makespan regardless of tab count — the one place the throughput axis
   touches the (capped ~1.3×) single-molecule axis. Want more tabs to help? Add
   molecules (a bigger screen), not tabs.

## Empirical confirmation (GitHub Actions, 4-core runner, 4 tabs)

Everything above is a compute-only model. `e2e/swarm-scheduling.spec.ts` ran the
*same* 10-molecule cc-pVDZ library on a real 4-tab BroadcastChannel swarm (headless
Chromium, `ubuntu-latest`, 4 cores) — FIFO then LPT, asserting identical energies:

```
[sched] 10 molecules, cc-pVDZ HF
[sched] FIFO 89454ms  LPT 45166ms  speedup 1.98x
```

| order | wall (4 tabs) | vs serial (94.1 s sum) |
|---|--:|--:|
| **FIFO** | 89.5 s | 1.05× — *almost no parallelism* |
| **LPT** | 45.2 s | 2.08× |
| **LPT vs FIFO** | | **1.98×** |

**LPT beats FIFO 1.98× — far more than the sim's ~1.2×.** The honest reason the model
*under*-predicted the win: **real FIFO is much worse than the model.** The sim assumed
ideal greedy-pull parallelism for both orders; on real tabs the single-threaded master
blocks its own event loop while it computes a tile, so it can't route pull requests or
hand out work while busy. The library is roughly ascending, so FIFO makes the master
block on the *heavy* molecules last, with the worker tabs already drained and idle →
the swarm collapses to **near-serial (89.5 s ≈ the 94.1 s serial sum, 1.05×)**. LPT
front-loads the heavy molecules: they grind in parallel across all tabs from t=0 and
the light tail overlaps, landing at 45 s — the same ballpark as the sim's 34 s LPT
prediction (the cloud cores are slower than the costs the sim was fit on; the *ratio*
is what transfers). LPT is robust to the master-blocking pathology; FIFO is not.

**Takeaway:** on a real browser swarm the scheduler matters *more* than the
compute-only model says, not less. FIFO + an ascending library ≈ serial; cost-aware
LPT recovers ~2× on 4 tabs. The sim was the conservative case — a rare honest surprise
in the right direction. (CI: run 28010504492, job `swarm-quick`, 6 passed.)

## Caveats

- **Compute-only makespan** — ignores BroadcastChannel comms + scheduler overhead.
  The real wall-clock is the e2e on a quiet multi-core machine; this is the ceiling.
- **LPT is a worst-case/average win, not pointwise** (Graham's 4/3 bound). A
  specific FIFO order can occasionally tie or beat it (`tests/parallel/swarm-lpt`
  pins this) — but a real ascending-ordered library is near-worst-case for FIFO, so
  LPT helps a lot, as measured.
- Single-shot per-molecule timings (~±15% noise); the n⁴ predictor needs only to
  rank, and it does.

## Verdict

**The throughput axis's first clear win — and confirmed on real hardware.** A 4-tab
GitHub Actions swarm ran the 10-molecule library **1.98× faster under LPT than FIFO**
(89.5 s → 45.2 s), *exceeding* the compute-only model because real FIFO collapses to
near-serial (the master blocks its own event loop on late heavy tiles). Cost-aware
(LPT) scheduling is a ~5-line change (`costFn` on `swarmMap`) that buys +11–19
efficiency points on the small library in the sim and, on a realistic 30-molecule
screen, **scales to 8 tabs at 95% efficiency (7.60×) vs FIFO's 65%** — its advantage
grows with scale. It reaches the optimal makespan and cleanly exposes the real limit
(the longest single molecule, capped at k ≈ total/longest). Correctness is untouched
— LPT reorders the queue, not the results (`tests/parallel/swarm-lpt`).
