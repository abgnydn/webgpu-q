// ─────────────────────────────────────────────────────────────
// molden.ts — emit a Molden-format file from a converged
// closed-shell HF / KS calculation. Tier 3 (interop with
// Jmol / Avogadro / IboView / Multiwfn for orbital plotting,
// natural orbital analysis, and external visualization).
//
// Molden format spec:
//   https://www3.cmbi.umcn.nl/molden/molden_format.html
//
// Layout we emit:
//
//   [Molden Format]
//   [Atoms] AU
//    H        1     1     0.000  0.000  0.000
//    O        2     8     1.000  0.000  0.000
//   [GTO]
//      1 0
//    s  3 1.00
//       3.42525091   0.15432897
//       0.62391373   0.53532814
//       0.16885540   0.44463454
//
//      2 0
//    s  3 1.00
//       ...
//    p  3 1.00
//       ...
//
//   [6D]                     (Cartesian d ordering: xx, yy, zz, xy, xz, yz)
//
//   [MO]
//    Sym= a
//    Ene= -0.5000
//    Spin= Alpha
//    Occup= 2.0
//        1    0.5000
//        2    0.2000
//        ...
//
// Scope:
//   - Cartesian-Gaussian only (no spherical-d transform handled here;
//     if `integrals.sphericalT` is non-null this routine refuses).
//     Spherical export needs the back-transform of MO coeffs from
//     spherical → Cartesian and is left for a follow-up.
//   - Closed-shell (RHF / RKS) emits all occupied with Spin=Alpha and
//     Occup=2.0. Open-shell (UHF / UKS) requires two MO blocks
//     (Spin=Alpha, Spin=Beta) — a follow-up once UKS lands.
//   - All atoms, shells, and MO coefficients in atomic units. The
//     `[Atoms] AU` tag tells viewers to interpret coords in bohr.
//
// Validation:
//   - Output should parse cleanly in Jmol's "load molden ..." and
//     Multiwfn's molden reader (manually tested).
//   - Programmatic tests check the structural correctness: required
//     section headers, correct atom count, correct number of MO
//     blocks, basis-function count consistent with shells.
// ─────────────────────────────────────────────────────────────

import type { CGShell } from "./integrals-cg.js";
import type { Atom } from "./atoms.js";
import { Z_FOR } from "./atoms.js";

export interface MoldenExportInput {
  /** All atoms in the molecule, in the same order as `integrals.nuclei`. */
  readonly atoms: readonly Atom[];
  /** Per-atom shell list, FLAT and in the same order as used to
   *  build `integrals` — shells must be grouped per-atom (all atom 0's
   *  shells before atom 1's, etc.) which is what `moleculeToShellsNuclei`
   *  already produces. */
  readonly shells: readonly CGShell[];
  /** Mapping shell index → atom index. If omitted, inferred from
   *  shell centers matching atom positions (within 1e-8 Å). */
  readonly shellAtomIdx?: readonly number[];
  /** AO → MO coefficient matrix (n × n, row-major, column p = MO p). */
  readonly C_MO: Float64Array;
  /** Orbital energies in Hartree (ascending). */
  readonly orbitalEnergies: Float64Array;
  /** Number of doubly-occupied spatial MOs (closed-shell). */
  readonly nOccupied: number;
  /** Optional spherical-d transform matrix from integrals. If set,
   *  this routine throws — Molden export currently only handles
   *  Cartesian basis. */
  readonly sphericalT?: Float64Array | null;
}

/**
 * Render a Molden-format string for the given closed-shell calculation.
 * Suitable for piping to an .molden file consumed by Jmol / Avogadro /
 * Multiwfn for orbital visualization or natural-orbital analysis.
 */
export function toMoldenString(input: MoldenExportInput): string {
  if (input.sphericalT) {
    throw new Error(
      "toMoldenString: spherical-d basis not yet supported. " +
      "Re-run with Cartesian d (no sphericalT) or back-transform MO " +
      "coefficients first.",
    );
  }
  const { atoms, shells, C_MO, orbitalEnergies, nOccupied } = input;
  const n = orbitalEnergies.length;
  if (C_MO.length !== n * n) {
    throw new Error(`toMoldenString: C_MO length ${C_MO.length} ≠ n² = ${n * n}`);
  }

  // Resolve shell → atom mapping.
  const shellAtomIdx = input.shellAtomIdx ?? inferShellAtomIdx(shells, atoms);

  // Group shells per atom into (L, alpha[], c[]) primitive blocks.
  // Walk shells and merge consecutive shells with identical
  // (atom, alpha, c) into one shell-group; the group's L is the
  // total angular momentum of any member (they all agree since
  // contraction implies same L).
  const groups: ShellGroup[] = [];
  let cursor: ShellGroup | null = null;
  for (let s = 0; s < shells.length; s++) {
    const sh = shells[s]!;
    const atomI = shellAtomIdx[s]!;
    const L = sh.angular[0] + sh.angular[1] + sh.angular[2];
    if (cursor && cursor.atomIdx === atomI && cursor.L === L
        && sameArray(cursor.alpha, sh.alpha) && sameArray(cursor.c, sh.c)) {
      cursor.endShell = s + 1;
    } else {
      cursor = {
        atomIdx: atomI,
        L,
        alpha: sh.alpha,
        c: sh.c,
        startShell: s,
        endShell: s + 1,
      };
      groups.push(cursor);
    }
  }
  // Validate group sizes against the expected Cartesian-shell count
  // for L: 1, 3, 6, 10, 15, 21 for L = 0..5.
  const cartCount = (L: number) => ((L + 1) * (L + 2)) / 2;
  for (const g of groups) {
    const size = g.endShell - g.startShell;
    if (size !== cartCount(g.L)) {
      throw new Error(
        `toMoldenString: shell group at atom ${g.atomIdx} L=${g.L} has ` +
        `${size} Cartesian components, expected ${cartCount(g.L)}. ` +
        `Shells aren't in canonical (alpha[],c[])-grouped order.`,
      );
    }
  }

  // ── Build output text. ─────────────────────────────────────
  const lines: string[] = [];
  lines.push("[Molden Format]");

  lines.push("[Atoms] AU");
  for (let a = 0; a < atoms.length; a++) {
    const at = atoms[a]!;
    const Z = at.ghost ? 0 : Z_FOR[at.symbol];
    const x = aaToBohr(at.pos[0]);
    const y = aaToBohr(at.pos[1]);
    const z = aaToBohr(at.pos[2]);
    lines.push(
      ` ${at.symbol.padEnd(2)}  ${(a + 1).toString().padStart(5)}  ${Z.toString().padStart(3)}  ` +
      `${x.toFixed(10).padStart(15)} ${y.toFixed(10).padStart(15)} ${z.toFixed(10).padStart(15)}`,
    );
  }

  lines.push("[GTO]");
  for (let a = 0; a < atoms.length; a++) {
    lines.push(`  ${(a + 1).toString().padStart(3)} 0`);
    for (const g of groups) {
      if (g.atomIdx !== a) continue;
      const letter = L_LETTER[g.L];
      if (letter === undefined) {
        throw new Error(`toMoldenString: L=${g.L} not supported (max L=5)`);
      }
      const nPrim = g.alpha.length;
      lines.push(` ${letter}  ${nPrim.toString().padStart(2)} 1.00`);
      for (let p = 0; p < nPrim; p++) {
        lines.push(`        ${g.alpha[p]!.toExponential(10)}   ${g.c[p]!.toExponential(10)}`);
      }
    }
    lines.push("");
  }

  // Cartesian d / f / g angular-ordering tags. Emit only when the
  // corresponding angular momentum actually appears — Molden viewers
  // can otherwise misinterpret the basis dimension.
  if (groups.some((g) => g.L === 2)) lines.push("[6D]");
  if (groups.some((g) => g.L === 3)) lines.push("[10F]");
  if (groups.some((g) => g.L === 4)) lines.push("[15G]");
  lines.push("");

  lines.push("[MO]");
  for (let p = 0; p < n; p++) {
    const ene = orbitalEnergies[p]!;
    const occup = p < nOccupied ? 2.0 : 0.0;
    lines.push(` Sym= a`);
    lines.push(` Ene= ${ene.toExponential(10)}`);
    lines.push(` Spin= Alpha`);
    lines.push(` Occup= ${occup.toFixed(6)}`);
    for (let mu = 0; mu < n; mu++) {
      lines.push(`     ${(mu + 1).toString().padStart(4)}  ${C_MO[mu * n + p]!.toExponential(10)}`);
    }
  }

  return lines.join("\n") + "\n";
}

// ─────────────────────────────────────────────────────────────

interface ShellGroup {
  readonly atomIdx: number;
  readonly L: number;
  readonly alpha: readonly number[];
  readonly c: readonly number[];
  readonly startShell: number;
  endShell: number;
}

const L_LETTER: readonly string[] = ["s", "p", "d", "f", "g", "h"];

// Convert Ångström → bohr (the Atom.pos convention is Å; integrals
// were built after a Å → bohr conversion via moleculeToShellsNuclei).
// 1 Å = 1.8897261245650618 bohr.
const ANGSTROM_PER_BOHR = 1.8897261245650618;
function aaToBohr(x: number): number {
  return x * ANGSTROM_PER_BOHR;
}

function sameArray(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i]! - b[i]!) > 1e-15) return false;
  }
  return true;
}

function inferShellAtomIdx(
  shells: readonly CGShell[],
  atoms: readonly Atom[],
): number[] {
  // Convert atom positions Å → bohr to match shell centers (which
  // are stored in bohr inside CGShell).
  const tol = 1e-6;
  const idx: number[] = [];
  for (const sh of shells) {
    let found = -1;
    for (let a = 0; a < atoms.length; a++) {
      const ax = aaToBohr(atoms[a]!.pos[0]);
      const ay = aaToBohr(atoms[a]!.pos[1]);
      const az = aaToBohr(atoms[a]!.pos[2]);
      if (Math.abs(sh.center[0] - ax) < tol
       && Math.abs(sh.center[1] - ay) < tol
       && Math.abs(sh.center[2] - az) < tol) {
        found = a;
        break;
      }
    }
    if (found < 0) {
      throw new Error(
        `inferShellAtomIdx: shell with center (${sh.center.join(",")}) ` +
        `doesn't match any atom (within ${tol} bohr). ` +
        `Pass shellAtomIdx explicitly.`,
      );
    }
    idx.push(found);
  }
  return idx;
}
