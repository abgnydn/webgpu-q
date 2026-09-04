import type { CGShell } from "./integrals-cg.js";
import type { DFResult } from "./df.js";
import { eigsymmetric } from "../manybody/dense-eig.js";
import { loadWasm, packShells } from "./df-aux.js";
import { DF_CHOLESKY_TOL } from "./numerical-tolerances.js";

/** Result of {@link buildAuxBasisDFStreaming}: a DF tensor plus the peak number
 *  of f64 V-tensor elements held at once during the build. For the streaming
 *  path that peak is `muBlock·n·n_aux`, NOT `n²·n_aux` — the evidence that the
 *  full 3-index tensor is never materialized on any single tab. */
export interface StreamingDFResult extends DFResult {
  readonly peakVFloats: number;
  /** Total kept modes across ALL tabs (this slice holds `nAux` of them). Lets a
   *  tab know the full B-tensor size (n²·nKeptTotal·8) without ever building it. */
  readonly nKeptTotal: number;
}

/**
 * Streaming, integral-direct, mode-partitioned auxiliary-basis DF build.
 *
 * Fits the same Coulomb metric as {@link buildAuxBasisDF}, but produced so the
 * full V[μν,Q] 3-index tensor is NEVER resident: V is built one μ-block at a
 * time and immediately projected onto the regularized inverse-metric modes
 *
 *     B[μν,i] = Σ_Q V[μν,Q] · U[Q,i] · λ_i^(−1/2)      (i = kept eigen-modes),
 *
 * which is itself a valid DF tensor — Σ_i B[μν,i]B[λσ,i] = (μν|λσ)_fit, because
 * Σ_i U[Q,i]·λ_i^(−1)·U[R,i] = (M⁻¹)[Q,R]. The mode index i is exactly the axis
 * the JK build contracts over, so it is the natural axis to PARTITION across a
 * swarm: pass `modeStart`/`modeEnd` and a tab builds only B[:, :, modeStart:modeEnd]
 * — still without ever holding all of V. Partial JK over disjoint mode ranges
 * sums to the full JK. With the default full mode range this is a single-tab
 * drop-in whose peak V footprint is `muBlock·n·n_aux`.
 *
 * The eri_3idx_build_slice kernel returns only ν ≥ μ pairs, so each block fills
 * the upper triangle of its B rows; the B slice is symmetrized at the end (the
 * B slice is the kept output — full V is not).
 */
export async function buildAuxBasisDFStreaming(
  orbitalShells: readonly CGShell[],
  auxShells?: readonly CGShell[],
  metricRegularization = DF_CHOLESKY_TOL,
  opts: {
    modeStart?: number;
    modeEnd?: number;
    muBlock?: number;
    /** Swarm convenience: build this tab's contiguous share of the kept modes.
     *  Each tab computes the (deterministic) eigendecomposition independently,
     *  so they agree on nKept and tile it without coordination. Overrides
     *  modeStart/modeEnd. */
    partition?: { tab: number; of: number };
  } = {},
): Promise<StreamingDFResult> {
  const mod = await loadWasm();
  const orb = packShells(orbitalShells);
  const aux = auxShells ? packShells(auxShells) : orb;
  const n = orbitalShells.length;
  const nAux = auxShells ? auxShells.length : n;

  // 2-index metric → eigendecomposition (ascending λ, column-major U).
  const M = mod.eri_2idx_build(
    nAux, aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );
  const eig = eigsymmetric(M, nAux);
  // Kept modes: λ_i > reg. U[Q,i] = eig.vectors[i*nAux + Q].
  const keptModes: number[] = [];
  for (let i = 0; i < nAux; i++) if (eig.values[i]! > metricRegularization) keptModes.push(i);
  const nKept = keptModes.length;

  let modeStart = opts.modeStart ?? 0;
  let modeEnd = opts.modeEnd ?? nKept;
  if (opts.partition) {
    const { tab, of } = opts.partition;
    modeStart = Math.floor((tab * nKept) / of);
    modeEnd = Math.floor(((tab + 1) * nKept) / of);
  }
  const mLocal = modeEnd - modeStart;

  // W mode-major: Wcm[m·n_aux + Q] = U[Q, i_m]·λ_{i_m}^(−1/2). Mode-major so
  // each mode's coefficient vector is contiguous in Q for the SIMD dot product.
  const Wcm = new Float64Array(mLocal * nAux);
  for (let m = 0; m < mLocal; m++) {
    const i = keptModes[modeStart + m]!;
    const inv = 1.0 / Math.sqrt(eig.values[i]!);
    const col = i * nAux;
    const wBase = m * nAux;
    for (let Q = 0; Q < nAux; Q++) Wcm[wBase + Q] = eig.vectors[col + Q]! * inv;
  }

  // Output: local mode slice B[(μ·n+ν)·mLocal + m]. Never the full tensor.
  const B = new Float64Array(n * n * mLocal);
  const muBlock = Math.max(1, opts.muBlock ?? 8);
  const Vblk = new Float64Array(muBlock * n * nAux); // transient, reused per block
  const peakVFloats = Vblk.length;

  for (let mu0 = 0; mu0 < n; mu0 += muBlock) {
    const mu1 = Math.min(mu0 + muBlock, n);
    const rows = mu1 - mu0;
    const mus = new Uint32Array(rows);
    for (let r = 0; r < rows; r++) mus[r] = mu0 + r;

    // Build V[μ∈block, ν≥μ, Q] as packed [μ,ν,P,val]; unpack into Vblk.
    Vblk.fill(0, 0, rows * n * nAux);
    const packed = mod.eri_3idx_build_slice(
      mus, n, nAux,
      orb.nPrims, orb.primOff, orb.alpha, orb.c, orb.center, orb.angular,
      aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
    );
    const K = packed.length / 4;
    for (let k = 0; k < K; k++) {
      const base = k * 4;
      const mu = packed[base]! | 0;
      const nu = packed[base + 1]! | 0;
      const P = packed[base + 2]! | 0;
      Vblk[((mu - mu0) * n + nu) * nAux + P] = packed[base + 3]!;
    }

    // Project this block onto the mode slice in WASM (SIMD f64x2 dot products);
    // upper-triangle (ν ≥ μ) only — symmetrized below.
    const Bblk = mod.df_project_block_modes(
      Vblk.subarray(0, rows * n * nAux), Wcm, rows, n, nAux, mLocal, mu0,
    );
    B.set(Bblk, mu0 * n * mLocal);
  }

  // Symmetrize: B[νμ] = B[μν] for μ < ν (B slice is the kept output).
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu + 1; nu < n; nu++) {
      const up = (mu * n + nu) * mLocal;
      const lo = (nu * n + mu) * mLocal;
      for (let m = 0; m < mLocal; m++) B[lo + m] = B[up + m]!;
    }
  }

  return { B, nAux: mLocal, threshold: metricRegularization, n, peakVFloats, nKeptTotal: nKept };
}

/** Result of {@link buildAuxBasisDFStreamingCooperative}: one DF mode-slice per
 *  tab, plus build statistics that substantiate the no-redundancy / never-full-V
 *  claims. */
export interface CooperativeDFResult {
  /** One mode-partitioned DF tensor per tab. In a real swarm each lives on its
   *  own tab; summed, the partial (J,K) reproduce the single-tab build. */
  readonly slices: StreamingDFResult[];
  /** Effective DF rank (sum of slice mode counts). */
  readonly nKept: number;
  /** Peak f64 V elements held at once = one μ-block (`muBlock·n·n_aux`). */
  readonly peakVFloats: number;
  /** Number of eri_3idx_build_slice invocations. Cooperative = ceil(n/muBlock)
   *  (each integral built ONCE and fanned out to all tabs), versus
   *  `nTabs · ceil(n/muBlock)` if every tab built independently. */
  readonly kernelCalls: number;
}

/**
 * Cooperative streaming aux-DF build: the single-machine (CI) engine.
 *
 * Same mode-partitioned fit as {@link buildAuxBasisDFStreaming}, but each V
 * μ-block is built EXACTLY ONCE and immediately projected onto every tab's mode
 * slice, instead of every tab rebuilding all integrals. This removes the N×
 * redundant integral cost of fully-independent per-tab builds — the right
 * trade when the tabs share a machine (a CI runner, multiple same-origin tabs).
 * Only one V μ-block is resident at a time; each tab keeps only its B slice.
 *
 * Returns one DF slice per tab. (In this single-process validator all slices
 * are resident at once — their sum is the full B; the memory DISTRIBUTION is a
 * property of the real per-tab browser deployment, where each tab holds just
 * `slices[t]`.)
 */
export async function buildAuxBasisDFStreamingCooperative(
  orbitalShells: readonly CGShell[],
  auxShells: readonly CGShell[] | undefined,
  nTabs: number,
  metricRegularization = DF_CHOLESKY_TOL,
  muBlock = 8,
): Promise<CooperativeDFResult> {
  const mod = await loadWasm();
  const orb = packShells(orbitalShells);
  const aux = auxShells ? packShells(auxShells) : orb;
  const n = orbitalShells.length;
  const nAux = auxShells ? auxShells.length : n;

  const M = mod.eri_2idx_build(
    nAux, aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );
  const eig = eigsymmetric(M, nAux);
  const keptModes: number[] = [];
  for (let i = 0; i < nAux; i++) if (eig.values[i]! > metricRegularization) keptModes.push(i);
  const nKept = keptModes.length;

  // Contiguous mode partition across tabs; per-tab W[Q,m] = U[Q,mode]·λ^(−1/2).
  const ranges: Array<{ start: number; end: number }> = [];
  for (let t = 0; t < nTabs; t++) {
    ranges.push({
      start: Math.floor((t * nKept) / nTabs),
      end: Math.floor(((t + 1) * nKept) / nTabs),
    });
  }
  const Wt = ranges.map(({ start, end }) => {
    const mLocal = end - start;
    const W = new Float64Array(mLocal * nAux); // mode-major for the SIMD kernel
    for (let m = 0; m < mLocal; m++) {
      const i = keptModes[start + m]!;
      const inv = 1.0 / Math.sqrt(eig.values[i]!);
      const col = i * nAux;
      const wBase = m * nAux;
      for (let Q = 0; Q < nAux; Q++) W[wBase + Q] = eig.vectors[col + Q]! * inv;
    }
    return W;
  });
  const Bt = ranges.map(({ start, end }) => new Float64Array(n * n * (end - start)));

  const muBlk = Math.max(1, muBlock);
  const Vblk = new Float64Array(muBlk * n * nAux);
  const peakVFloats = Vblk.length;
  let kernelCalls = 0;

  for (let mu0 = 0; mu0 < n; mu0 += muBlk) {
    const mu1 = Math.min(mu0 + muBlk, n);
    const rows = mu1 - mu0;
    const mus = new Uint32Array(rows);
    for (let r = 0; r < rows; r++) mus[r] = mu0 + r;

    // Build this V μ-block ONCE (the cooperative win).
    Vblk.fill(0, 0, rows * n * nAux);
    const packed = mod.eri_3idx_build_slice(
      mus, n, nAux,
      orb.nPrims, orb.primOff, orb.alpha, orb.c, orb.center, orb.angular,
      aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
    );
    kernelCalls++;
    const Kp = packed.length / 4;
    for (let k = 0; k < Kp; k++) {
      const base = k * 4;
      const mu = packed[base]! | 0;
      const nu = packed[base + 1]! | 0;
      const P = packed[base + 2]! | 0;
      Vblk[((mu - mu0) * n + nu) * nAux + P] = packed[base + 3]!;
    }

    // Fan out: project this block into every tab's slice in WASM (one V build,
    // reused across all tabs — the cooperative win).
    const vsub = Vblk.subarray(0, rows * n * nAux);
    for (let t = 0; t < ranges.length; t++) {
      const { start, end } = ranges[t]!;
      const mLocal = end - start;
      if (mLocal === 0) continue;
      const Bblk = mod.df_project_block_modes(vsub, Wt[t]!, rows, n, nAux, mLocal, mu0);
      Bt[t]!.set(Bblk, mu0 * n * mLocal);
    }
  }

  // Symmetrize each slice (B[νμ] = B[μν]).
  for (let t = 0; t < ranges.length; t++) {
    const mLocal = ranges[t]!.end - ranges[t]!.start;
    if (mLocal === 0) continue;
    const B = Bt[t]!;
    for (let mu = 0; mu < n; mu++) {
      for (let nu = mu + 1; nu < n; nu++) {
        const up = (mu * n + nu) * mLocal;
        const lo = (nu * n + mu) * mLocal;
        for (let m = 0; m < mLocal; m++) B[lo + m] = B[up + m]!;
      }
    }
  }

  const slices: StreamingDFResult[] = ranges.map((rg, t) => ({
    B: Bt[t]!, nAux: rg.end - rg.start, threshold: metricRegularization, n, peakVFloats,
    nKeptTotal: nKept,
  }));
  return { slices, nKept, peakVFloats, kernelCalls };
}
