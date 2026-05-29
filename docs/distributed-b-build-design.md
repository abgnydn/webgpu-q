# Distributed B-tensor build (design doc, 2026-05-29)

The current `buildAuxBasisDFCholesky` (master-side) is the last
remaining single-tab memory bottleneck in the swarm-HF stack:

- Master allocates full V tensor (size `n²·n_aux·8`)
- Master allocates full B tensor (same size) during back-substitution
- Master peaks at **V + B ≈ 2 × n²·n_aux·8** during the transition
  (mitigated by `3123505` which frees V right after B is built, so
   the genuine peak is ≈ V_size + B_being_written ≈ 1.5× the final B)

For C₆₀ STO-3G: V = B = 1.82 GB, master peak ≈ 3 GB. M2 Pro (Mac
Chrome ≈ 4 GB WASM heap) survives; Ubuntu CI runner (≈ 2 GB WASM
cap) traps with `Error: unreachable`. Bigger molecules (C₆₀ cc-pVDZ,
nonacene+ STO-3G) exceed Mac too.

## Why partition by aux index P

We already validated bit-exact P-partitioned JK in `9103ed6`:

- γ[P] = Σ_μν B[μν,P]·D[μν]                     (linear in P)
- J[μν] = Σ_P B[μν,P]·γ[P]                      (linear in P)
- X[P,μ,σ] = Σ_λ B[μλ,P]·D[λ,σ]                 (each P independent)
- K[μ,σ] = Σ_P Σ_σ' X[P,μ,σ']·B[σσ',P]           (linear in P)

So splitting B by aux column index P into disjoint ranges and
summing partial (J, K) reproduces the full single-slab result.

**Current swarm-HF** uses this partition for the SCF phase but the
master still builds the full B first, then ships P-slices. The
master-peak problem is unchanged.

**Goal**: have each tab build *only its own* B aux-slice from
scratch, so the swarm aggregate memory replaces the master peak.

## The algorithm

### Setup (single-shot, master tab)

1. Master tab partitions the aux *shell* set into N disjoint subsets
   `aux_T` for tabs T = 0..N-1. Roughly balanced by shell count (or
   primitive count for tighter load balancing).
2. Master broadcasts the orbital shells, the global aux-shell list,
   and each tab's `aux_T` index range to the workers.
3. Master computes M = (P|Q) over the *full* aux basis (small —
   `n_aux²·8` ≈ 50 MB at C₆₀ scale). M is small enough to compute
   on master without distributed memory issues.
4. Master runs pivoted Cholesky on M, gets `L` and `pivots` (an
   array of r ≤ n_aux global aux indices in selection order).
5. Master broadcasts L, pivots, and an "ownership map" of each
   `pivots[k]` → which tab T owns the aux shell containing that
   aux basis function index.

### Per-tab V build (parallel, no communication)

Each tab T:

6. Builds `V_T[μν, P_local]` = (μν|P) for P in `aux_T` only. Size
   per tab ≈ `n²·n_aux/N·8`. For C₆₀ with N=4: 450 MB per tab.
   Uses the existing `eri_3idx_build_slice` kernel adapted to take
   an aux-shell subset (today it takes an orbital μ-slice).

### Sequential back-substitution (serial-by-column, parallel-by-row)

For each global pivot column k = 0..r-1:

7. Let `tab_k` = ownership_map[pivots[k]] = the tab that holds V at
   the aux index `pivots[k]` in its V_T.
8. `tab_k` retrieves the column `V_T[:, P_local(pivots[k])]` — size
   `n²·8` = 720 KB at C₆₀ scale.
9. `tab_k` computes:
   ```
   for each μν:
       B[μν, k] = (V_T[μν, P_local(pivots[k])] - running_sum[μν,k])
                  / L[pivots[k], k]
   ```
   where `running_sum[μν, k] = Σ_{j<k} L[pivots[k], j] · B[μν, j]`.
10. `tab_k` broadcasts B[:, k] to all other tabs via BroadcastChannel
    (one f64 column = 720 KB at C₆₀).
11. All tabs T that own *future* columns k' > k update their
    running sums:
    ```
    for each k' > k owned by T:
        for each μν:
            running_sum[μν, k'] += L[pivots[k'], k] · B[μν, k]
    ```
    This is the only per-step cross-tab dependency.
12. Each tab T also stores B[:, k] in its own B_T iff k is in T's
    output partition (matches the SCF P-slice partition).

### Memory ledger per tab

- `V_T`              : n²·n_aux/N·8   ≈ 450 MB (C₆₀, N=4)
- `B_T` (output)     : n²·n_aux/N·8   ≈ 450 MB
- `running_sum`      : n²·n_aux/N·8   ≈ 450 MB (one per own column)
- `L` (broadcast)    : n_aux²·8       ≈ 50 MB
- workspace          : ~50 MB

**Per-tab peak ≈ 1.4 GB at C₆₀.** Under the Ubuntu 2 GB Chrome cap.

Aggregate across N=4 tabs ≈ 5.6 GB — *worse* than the current
single-master arrangement on aggregate but the relevant constraint
is per-tab, not aggregate. A 16 GB Mac with 4 tabs at 1.4 GB each
has plenty of system RAM left.

### Communication

- One column broadcast per step: r broadcasts × 720 KB = 1.8 GB
  total for C₆₀ (r ≈ 2520). At ~1 GB/s BroadcastChannel bandwidth
  this is ~2 seconds — added to a ~120 s DF-build, negligible.
- Latency: each broadcast is one message-tick (~10-100 µs on same
  machine). 2520 × 50 µs = 125 ms total latency overhead.

## What needs to change

### New TS module: `src/parallel/distributed-df-build.ts`

```typescript
export async function buildDistributedDF_Cholesky(
  orbitalShells: CGShell[],
  auxShells: CGShell[],
  threshold: number,
  workers: BroadcastChannelWorker[],
): Promise<DFResult> { ... }
```

### New WASM kernel: `eri_3idx_build_aux_slice`

Like `eri_3idx_build_slice` but takes an *aux* shell subset instead
of an orbital μ subset. Builds V[:, :, P_local] for the given aux
subset. The arithmetic is symmetric — just a different loop nest.

### Per-tab back-sub orchestration

Coordinator on master tab. Implements the column-loop with column
ownership lookup, broadcast/wait per column, running-sum updates
on each tab via message handler.

### E2E test

Validate distributed B is bit-equivalent to master-built B on a
small molecule (water, benzene) before going to C₆₀-scale.

## Effort estimate

- WASM kernel adaptation: 1-2 hours (Rust)
- Distributed-B-build TS module: 2-3 hours
- Integration with existing swarm-HF SCF: 1 hour
- E2E + bit-equivalence test: 1-2 hours
- Debug + tune: 2-3 hours

**Total ~10 hours of focused work.** Single-session feasible
but substantial. Would unlock:

- Anthracene cc-pVDZ on Ubuntu CI (currently the master build
  works locally, fails in CI)
- C₆₀ cc-pVDZ in browser tabs (currently impossible — would need
  n=600+, B ≈ 8 GB)
- Larger molecules generally: any system where the *per-tab* slice
  fits but the *full B* doesn't.

## Alternatives considered

### Approximate local Cholesky per tab

Each tab does its own Cholesky on M restricted to its aux subset.
Ignores cross-shell terms in M.

- ❌ Loses bit-exactness vs single-tab
- ❌ Accuracy degradation depends on aux shell separations
- ✅ No cross-tab communication during DF build (only during SCF)

Rejected: scientific credibility requires bit-exact match.

### Out-of-core on OPFS

Page B to disk and pull in slices as the SCF needs them.

- ❌ OPFS bandwidth ~100 MB/s, much slower than SAB
- ❌ SCF iter would be I/O-bound, not compute-bound
- ✅ Effectively unlimited B size

Maybe worth doing AFTER per-tab B build if even larger molecules
are needed.

### WGSL GPU build

GPU has fewer memory constraints in some configurations.

- ❌ f32 precision concerns
- ❌ Doesn't address the per-tab cap, just changes where it lives
- ✅ Faster build phase

Independent dimension; would compose with per-tab distribution.
