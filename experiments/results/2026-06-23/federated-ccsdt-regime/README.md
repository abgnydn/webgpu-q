# Federated CCSD(T) — the S/C regime measurement (honest, partial negative)

**Question:** does distributing the (T) triples across the swarm beat the
distributed-MP2 honest negative (single-molecule speedup ~1.10×)?

**Short answer:** yes, but less than hypothesized — and the bottleneck moved.
At H₂O cc-pVDZ, distributing (T) predicts **1.28× on 2 tabs** (vs MP2's 1.10×),
but **C/S = 0.76 < 1**: the (T) grind does *not* dominate the redundant setup,
because the setup is now dominated by **CCSD**, not SCF/DF. My prior — "(T) ≫
setup, C ≫ S" — was **wrong**, and the code comments that claimed it are corrected.

## Method

H₂O cc-pVDZ (n = 25 basis fns, 10 e⁻), frozen-core = 1 (O 1s). Single-threaded,
single-shot, this dev container (Node 22, 4 cores but timing is single-threaded
to keep the S/C ratio low-noise). Same `S` vs `C` decomposition as
`e2e/swarm-mp2-speedup` (where it produced the 1.10× negative).

## Measured

| stage | wall time |
|---|---:|
| ERI build | 2.2 s |
| SCF | 0.1 s |
| **CCSD** | **53.4 s** |
| **(T)** | **40.6 s** (E_(T) = −0.00409 Ha) |

- **S** (redundant per-tab setup = SCF + CCSD) = **53.5 s**
- **C** (splittable (T) grind) = **40.6 s**
- **C/S = 0.76**  (MP2 benzene had C/S ≈ 0.22 → 1.10×)

Predicted single-molecule speedup `(S+C)/(S+C/k)`:

| k tabs | predicted speedup |
|---:|---:|
| 2 | **1.28×** |
| 3 | 1.40× |
| 4 | 1.48× |

## Interpretation (honest)

1. **Distributing (T) beats distributing MP2** — 1.28× vs 1.10× on 2 tabs. The
   feature is a real, if modest, improvement on the swarm's single-molecule axis.
2. **But (T) does NOT dominate at H₂O cc-pVDZ.** C/S = 0.76. The hypothesis that
   "(T) ~100 s ≫ a few-second setup" was wrong on both numbers: (T) is **40 s**
   (frozen-core; the stale `ccsd-t.ts` "~5 min" note predates this), and the
   setup is **53 s — dominated by the redundant CCSD**, not SCF/DF.
3. **The bottleneck moved from SCF/DF (MP2's story) to CCSD.** Each tab rebuilds
   the full CCSD redundantly; that 53 s sits on the critical path of every tab.
4. **The crossover is a size argument.** (T) is O(N⁷), CCSD is O(N⁶), so C/S ∝ N.
   H₂O cc-pVDZ sits just *below* the clean crossover (C/S = 0.76); larger
   molecules push C/S > 1 and the 2-tab speedup toward ~2×. To win at *accessible*
   sizes you'd also distribute (or share) the CCSD — the next frontier.

## Caveats

- **Predicted, not empirical.** These speedups are `(S+C)/(S+C/k)` from the real
  measured S and C (low-noise because single-threaded). The actual 2-tab
  wall-clock — with BroadcastChannel overhead and real parallelism — is
  `e2e/swarm-ccsdt-speedup.spec.ts` on a quiet multi-core machine (the M2 Pro),
  not this shared container.
- Single-shot (no warmup + trials), so the ±10–20% timing noise applies; the
  **C/S ratio** is the robust, structural quantity.
- Frozen-core (T); all-electron would scale both S and C up.

## Verdict

A **partial honest negative**: the federated (T) is correct (Σ slices ==
single-shot < 1e-9, `tests/chemistry/ccsdt-slice`) and beats MP2, but at
browser-accessible sizes the redundant CCSD caps the win at ~1.3×, not the ~2×
implied. The swarm's single-molecule speedup for correlation is real but
size-gated — and the next bottleneck to attack is CCSD, not (T).
