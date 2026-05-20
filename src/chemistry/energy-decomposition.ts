// ─────────────────────────────────────────────────────────────
// energy-decomposition.ts — split an HF / UHF total energy into
// its constituent physical contributions:
//
//   E_HF = E_one + ½ E_J − ¼ E_K + V_nn      (closed-shell RHF)
//   E_UHF = E_one + ½ E_J − ½ E_K + V_nn     (open-shell UHF)
//
// where
//   E_one = Σ_μν D_μν · h_μν           (one-electron: T + V_ne)
//   E_J   = Σ_μν D_μν · J_μν           (Coulomb)
//   E_K   = Σ_μν D_μν · K_μν           (HF exchange)
//   V_nn   = Σ_{A<B} Z_A Z_B / R_AB     (nuclear-nuclear repulsion)
//
// The kinetic / nuclear-attraction split inside E_one would require
// the raw T and V_ne AO matrices; currently only their sum h_AO is
// stored. Decomposing E_one further is a follow-up if needed.
//
// For DFT (RKS / UKS) the decomposition becomes
//
//   E_DFT = E_kinetic + E_ne + ½ E_J + E_xc − ½·hfMix·E_K + V_nn
//
// where E_xc is the integrated XC energy. The HF exchange term
// drops out for pure functionals (hfMix = 0).
//
// Useful for: diagnosing where the bulk of the total energy comes
// from, comparing methods (e.g., HF vs DFT have very different
// E_K / E_xc balance), publication-quality energy tables.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { HFResult } from "./hf-scf.js";
import type { UHFResult } from "./uhf-scf.js";

export interface EnergyDecomposition {
  /** Σ D · h_core — one-electron energy (T + V_ne combined). */
  readonly oneElectron: number;
  /** ½ Σ D · J — Coulomb (Hartree) energy. */
  readonly coulomb: number;
  /** Exchange contribution to E_total (already signed: typically negative). */
  readonly exchange: number;
  /** V_nn — nuclear repulsion. */
  readonly nuclearRepulsion: number;
  /** Sum of components — should reconstruct the total SCF energy. */
  readonly total: number;
  /** Discrepancy: total - hfResult.energy. Should be < 1e-8 Ha for a
   *  converged calculation. */
  readonly checkResidual: number;
}

/**
 * Decompose a closed-shell HF energy into its components. Requires
 * the integrals (T, V_ne, ERIs, V_nn) and the converged D and energy.
 */
export function decomposeHFEnergy(
  hf: HFResult,
  integrals: MolecularIntegrals,
): EnergyDecomposition {
  const n = integrals.n;
  const D = hf.D;
  const h = integrals.h_AO;
  const eri = integrals.eri_AO;

  let oneElectron = 0;
  for (let i = 0; i < n * n; i++) oneElectron += D[i]! * h[i]!;

  // J[μν] = Σ_{λσ} D[λσ] · (μν|λσ).  E_J = ½ Σ D · J.
  // K[μν] = Σ_{λσ} D[λσ] · (μλ|νσ).  E_K = ½ Σ D · K.
  let coulombTwice = 0;
  let exchangeTwice = 0;
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      const Dmnu = D[mu * n + nu]!;
      if (Dmnu === 0) continue;
      let J = 0, K = 0;
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          const Dlasi = D[la * n + si]!;
          if (Dlasi === 0) continue;
          J += Dlasi * eri[((mu * n + nu) * n + la) * n + si]!;
          K += Dlasi * eri[((mu * n + la) * n + nu) * n + si]!;
        }
      }
      coulombTwice += Dmnu * J;
      exchangeTwice += Dmnu * K;
    }
  }
  const coulomb = 0.5 * coulombTwice;
  // For RHF the exact-exchange contribution is −½·½·K = −¼·K (since
  // hfMix=1 means full HF and the convention gives −½·E_K in total
  // energy after the ½ in F = h + J - 0.5K and the ½ in E = ½·D(h+F)).
  // Total: E_HF = E_one + ½·E_J − ¼·Σ D·K
  const exchange = -0.25 * exchangeTwice;
  const nuclearRepulsion = integrals.Vnn;
  const total = oneElectron + coulomb + exchange + nuclearRepulsion;
  return {
    oneElectron, coulomb, exchange, nuclearRepulsion, total,
    checkResidual: total - hf.energy,
  };
}

/**
 * Open-shell decomposition (UHF). Uses the full density D = D_α + D_β
 * for one-electron and Coulomb terms; per-spin K(D_σ) for exchange.
 *
 *   E_UHF = Σ D h + ½ Σ D_total J(D_total)
 *         − ½ [Σ D_α K(D_α) + Σ D_β K(D_β)] + V_nn
 */
export function decomposeUHFEnergy(
  uhf: UHFResult,
  integrals: MolecularIntegrals,
): EnergyDecomposition {
  const n = integrals.n;
  const h = integrals.h_AO;
  const eri = integrals.eri_AO;

  // D_total for one-electron / Coulomb.
  const Dtotal = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) {
    Dtotal[i] = uhf.D_alpha[i]! + uhf.D_beta[i]!;
  }

  let oneElectron = 0;
  for (let i = 0; i < n * n; i++) oneElectron += Dtotal[i]! * h[i]!;

  // Coulomb from D_total.
  let coulombTwice = 0;
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      const Dmnu = Dtotal[mu * n + nu]!;
      if (Dmnu === 0) continue;
      let J = 0;
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          J += Dtotal[la * n + si]! * eri[((mu * n + nu) * n + la) * n + si]!;
        }
      }
      coulombTwice += Dmnu * J;
    }
  }
  const coulomb = 0.5 * coulombTwice;

  // Exchange: −½ [Σ D_α K(D_α) + Σ D_β K(D_β)].
  let exchangeContrib = 0;
  for (const Dspin of [uhf.D_alpha, uhf.D_beta]) {
    for (let mu = 0; mu < n; mu++) {
      for (let nu = 0; nu < n; nu++) {
        const Dmnu = Dspin[mu * n + nu]!;
        if (Dmnu === 0) continue;
        let K = 0;
        for (let la = 0; la < n; la++) {
          for (let si = 0; si < n; si++) {
            K += Dspin[la * n + si]! * eri[((mu * n + la) * n + nu) * n + si]!;
          }
        }
        exchangeContrib += Dmnu * K;
      }
    }
  }
  const exchange = -0.5 * exchangeContrib;
  const nuclearRepulsion = integrals.Vnn;
  const total = oneElectron + coulomb + exchange + nuclearRepulsion;
  return {
    oneElectron, coulomb, exchange, nuclearRepulsion, total,
    checkResidual: total - uhf.energy,
  };
}
