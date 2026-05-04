// ─────────────────────────────────────────────────────────────
// pair-density.ts — conditional pair density ρ(r₂ | r₀) for the
// closed-shell H₂ singlet ground state.
//
// |ψ⟩ = c_g |σ_g σ_g⟩ + c_u |σ_u σ_u⟩  (closed-shell singlet)
//
// The two-electron spatial wavefunction (factoring out the spin
// singlet) is:
//   Ψ(1, 2) = c_g σ_g(r₁) σ_g(r₂) + c_u σ_u(r₁) σ_u(r₂)
//
// Spin-summed pair density:
//   ρ₂(r₁, r₂) = 2 |Ψ(1,2)|²
//
// At HF (c_u = 0): ρ₂ = 2 c_g² σ_g(r₁)² σ_g(r₂)², no correlation
// → ρ_cond = σ_g(r₂)², independent of r₁. No visible hole.
//
// At FCI with non-trivial c_u: the cross-term `2 c_g c_u σ_g σ_g
// σ_u σ_u` introduces a *real* exchange-correlation hole that
// tracks r₀. As R grows, c_u grows toward ±1/√2 (Heitler-London
// limit) and the hole becomes dramatic.
// ─────────────────────────────────────────────────────────────

import { phi1s } from "./orbital.js";
import { S_AB } from "../chemistry/integrals.js";

const ANGSTROM_TO_BOHR = 1 / 0.529177210903;

export interface PairGrid {
  positions: Float32Array;
  densities: Float32Array;
  cursor: [number, number, number];
  cursorRho: number;
  extentBohr: number;
  gridSize: number;
  maxDensity: number;
}

export function pairDensityGridFromCoeffs(
  R_angstrom: number,
  cG: number,
  cU: number,
  cursor: readonly [number, number, number],
  gridSize = 48,
  extentBohr?: number,
): PairGrid {
  const R_Bohr = R_angstrom * ANGSTROM_TO_BOHR;
  const extent = extentBohr ?? Math.max(2.5, R_Bohr / 2 + 2);
  const A: [number, number, number] = [0, 0, -R_Bohr / 2];
  const B: [number, number, number] = [0, 0,  R_Bohr / 2];

  const Sab = S_AB(A, B);
  const invNg = 1 / Math.sqrt(2 * (1 + Sab));
  const invNu = 1 / Math.sqrt(2 * (1 - Sab));

  // Cursor MO values + cursor density for the conditional normalization.
  const cphiA = phi1s(cursor[0] - A[0], cursor[1] - A[1], cursor[2] - A[2]);
  const cphiB = phi1s(cursor[0] - B[0], cursor[1] - B[1], cursor[2] - B[2]);
  const sigmaG_c = (cphiA + cphiB) * invNg;
  const sigmaU_c = (cphiA - cphiB) * invNu;
  const psi_c = cG * sigmaG_c;       // partial: c_g σ_g(r₀)
  const psi_cU = cU * sigmaU_c;      // c_u σ_u(r₀)
  const rho_c = 2 * (cG * cG * sigmaG_c * sigmaG_c + cU * cU * sigmaU_c * sigmaU_c);
  const safeRho_c = Math.max(rho_c, 1e-6);

  const N = gridSize;
  const N3 = N * N * N;
  const positions = new Float32Array(N3 * 3);
  const densities = new Float32Array(N3);
  let pi = 0;
  let di = 0;
  let maxD = 0;

  const step = (2 * extent) / N;
  const start = -extent + step / 2;

  for (let ix = 0; ix < N; ix++) {
    const x = start + ix * step;
    for (let iy = 0; iy < N; iy++) {
      const y = start + iy * step;
      for (let iz = 0; iz < N; iz++) {
        const z = start + iz * step;
        positions[pi++] = x;
        positions[pi++] = y;
        positions[pi++] = z;

        const phiA = phi1s(x - A[0], y - A[1], z - A[2]);
        const phiB = phi1s(x - B[0], y - B[1], z - B[2]);
        const sigmaG = (phiA + phiB) * invNg;
        const sigmaU = (phiA - phiB) * invNu;
        // Ψ(r, r₀) for spatial singlet:
        //   c_g σ_g(r) σ_g(r₀) + c_u σ_u(r) σ_u(r₀)
        const Psi = sigmaG * psi_c + sigmaU * psi_cU;
        const rho2 = 2 * Psi * Psi;
        const cond = rho2 / safeRho_c;
        densities[di++] = cond;
        if (cond > maxD) maxD = cond;
      }
    }
  }

  return {
    positions,
    densities,
    cursor: [cursor[0], cursor[1], cursor[2]],
    cursorRho: rho_c,
    extentBohr: extent,
    gridSize: N,
    maxDensity: maxD,
  };
}
