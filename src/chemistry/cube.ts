// ─────────────────────────────────────────────────────────────
// cube.ts — Gaussian Cube file format for density and MO
// isosurface visualization (Jmol / VMD / Avogadro / Multiwfn).
//
// Cube format (Gaussian98 / Gaussian03 standard):
//
//   title line 1                       (anything, typically molecule)
//   title line 2                       (e.g., "density" or "MO N")
//   N_atoms x0 y0 z0                   (number of atoms; grid origin in bohr)
//   Nx dx_x dx_y dx_z                  (x grid count; x-axis step vector)
//   Ny dy_x dy_y dy_z                  (y; positive Nx means cube units are bohr,
//                                       negative would mean Ångström)
//   Nz dz_x dz_y dz_z
//   Z charge x y z                     (one line per atom; charge usually = Z)
//   Z charge x y z
//   ...
//   v[0,0,0] v[0,0,1] v[0,0,2] v[0,0,3] v[0,0,4] v[0,0,5]
//   v[0,0,6] v[0,0,7] ...
//   (6 values per line, "%13.5E" Fortran format; outer index x, middle y,
//    inner z — z varies fastest)
//
// Grid is built around the molecule:
//   - Bounding box = min/max of atom coordinates ± padding
//   - Number of steps along each axis from `opts.stepSize` (default 0.3 bohr)
//   - Axis vectors are aligned with global x/y/z (orthorhombic grid only;
//     general parallelepiped grids are valid Cube but not emitted here)
//
// Two entry points:
//   `densityCube(integrals, shells, D, atoms)` — total electron density
//   `moCube(integrals, shells, C_MO, orbitalIdx, atoms)` — single MO amplitude
// ─────────────────────────────────────────────────────────────

import type { CGShell } from "./integrals-cg.js";
import type { Atom } from "./atoms.js";
import { Z_FOR } from "./atoms.js";

export interface CubeOpts {
  /** Padding (bohr) added around the molecular bounding box. Default 4.0. */
  readonly padding?: number;
  /** Step size (bohr) along each axis. Default 0.3 (~0.16 Å). */
  readonly stepSize?: number;
  /** Optional first title line. */
  readonly title?: string;
}

const ANGSTROM_PER_BOHR = 1.8897261245650618;

function aaToBohr(x: number): number {
  return x * ANGSTROM_PER_BOHR;
}

/**
 * Build a Cube-file string sampling the total electron density
 * ρ(r) = Σ_{μν} D_{μν} φ_μ(r) φ_ν(r) on a regular cubic grid
 * around the molecule.
 */
export function densityCube(
  shells: readonly CGShell[],
  D: Float64Array,
  atoms: readonly Atom[],
  opts: CubeOpts = {},
): string {
  return buildCube(
    shells, atoms, opts,
    (r: readonly [number, number, number]) => evalDensityAtPoint(D, shells, r),
    opts.title ?? `Electron density (rho)`,
    "density",
  );
}

/**
 * Build a Cube-file string sampling a single MO amplitude
 * φ_p(r) = Σ_μ C_MO[μ, p] · φ_μ(r) on a regular cubic grid.
 *
 * Sign of the returned amplitude is meaningful (MOs aren't densities).
 * Viewers render |φ|² isosurfaces by default.
 */
export function moCube(
  shells: readonly CGShell[],
  C_MO: Float64Array,
  orbitalIdx: number,
  atoms: readonly Atom[],
  opts: CubeOpts = {},
): string {
  const n = shells.length;
  if (orbitalIdx < 0 || orbitalIdx >= n) {
    throw new Error(`moCube: orbitalIdx ${orbitalIdx} out of range [0, ${n})`);
  }
  const coeffs = new Float64Array(n);
  for (let mu = 0; mu < n; mu++) coeffs[mu] = C_MO[mu * n + orbitalIdx]!;
  return buildCube(
    shells, atoms, opts,
    (r: readonly [number, number, number]) => evalMOAtPoint(coeffs, shells, r),
    opts.title ?? `MO #${orbitalIdx}`,
    `MO ${orbitalIdx}`,
  );
}

// ─────────────────────────────────────────────────────────────

function buildCube(
  _shells: readonly CGShell[],
  atoms: readonly Atom[],
  opts: CubeOpts,
  fieldAt: (r: readonly [number, number, number]) => number,
  title1: string,
  title2: string,
): string {
  const padding = opts.padding ?? 4.0;
  const step = opts.stepSize ?? 0.3;

  // Bounding box in bohr.
  let xmin = Infinity, ymin = Infinity, zmin = Infinity;
  let xmax = -Infinity, ymax = -Infinity, zmax = -Infinity;
  for (const at of atoms) {
    const x = aaToBohr(at.pos[0]);
    const y = aaToBohr(at.pos[1]);
    const z = aaToBohr(at.pos[2]);
    if (x < xmin) xmin = x; if (x > xmax) xmax = x;
    if (y < ymin) ymin = y; if (y > ymax) ymax = y;
    if (z < zmin) zmin = z; if (z > zmax) zmax = z;
  }
  xmin -= padding; xmax += padding;
  ymin -= padding; ymax += padding;
  zmin -= padding; zmax += padding;

  const Nx = Math.max(2, Math.ceil((xmax - xmin) / step) + 1);
  const Ny = Math.max(2, Math.ceil((ymax - ymin) / step) + 1);
  const Nz = Math.max(2, Math.ceil((zmax - zmin) / step) + 1);

  const lines: string[] = [];
  lines.push(title1);
  lines.push(title2);
  // N_atoms, origin
  lines.push(
    `${atoms.length.toString().padStart(5)} ${fmt(xmin)} ${fmt(ymin)} ${fmt(zmin)}`,
  );
  // Axis vectors (Nx, dx, 0, 0)
  lines.push(`${Nx.toString().padStart(5)} ${fmt(step)} ${fmt(0)} ${fmt(0)}`);
  lines.push(`${Ny.toString().padStart(5)} ${fmt(0)} ${fmt(step)} ${fmt(0)}`);
  lines.push(`${Nz.toString().padStart(5)} ${fmt(0)} ${fmt(0)} ${fmt(step)}`);
  // Atoms.
  for (const at of atoms) {
    const Z = at.ghost ? 0 : Z_FOR[at.symbol];
    lines.push(
      `${Z.toString().padStart(5)} ${fmt(Z)} ${fmt(aaToBohr(at.pos[0]))} ${fmt(aaToBohr(at.pos[1]))} ${fmt(aaToBohr(at.pos[2]))}`,
    );
  }

  // Volumetric data: x outer, y middle, z inner. z varies fastest.
  // Format: 6 values per line, fortran %13.5E.
  let buf: string[] = [];
  let col = 0;
  for (let ix = 0; ix < Nx; ix++) {
    const x = xmin + ix * step;
    for (let iy = 0; iy < Ny; iy++) {
      const y = ymin + iy * step;
      for (let iz = 0; iz < Nz; iz++) {
        const z = zmin + iz * step;
        const v = fieldAt([x, y, z]);
        buf.push(formatExp(v));
        col++;
        if (col === 6) {
          lines.push(buf.join(" "));
          buf = [];
          col = 0;
        }
      }
      // Flush at end of inner row (Gaussian's preferred line-break).
      if (col > 0) {
        lines.push(buf.join(" "));
        buf = [];
        col = 0;
      }
    }
  }
  if (buf.length > 0) lines.push(buf.join(" "));

  return lines.join("\n") + "\n";
}

function fmt(x: number): string {
  return x.toFixed(6).padStart(12);
}
function formatExp(x: number): string {
  // Fortran-style %13.5E, e.g. " 1.23456E+02".
  const sign = x < 0 ? "-" : " ";
  const abs = Math.abs(x);
  if (abs === 0) return " 0.00000E+00";
  const exp = Math.floor(Math.log10(abs));
  const mantissa = abs / Math.pow(10, exp);
  const mStr = mantissa.toFixed(5);
  const eSign = exp < 0 ? "-" : "+";
  const eAbs = Math.abs(exp).toString().padStart(2, "0");
  return `${sign}${mStr}E${eSign}${eAbs}`;
}

// ─────────────────────────────────────────────────────────────
// Single-point CGShell evaluators. Independent of the grid module
// (which uses pre-computed batches for speed); cube emission is
// one-off and visualization-grade, so a per-point loop is fine.

function evalShellAtPoint(
  shell: CGShell,
  r: readonly [number, number, number],
): number {
  const dx = r[0] - shell.center[0];
  const dy = r[1] - shell.center[1];
  const dz = r[2] - shell.center[2];
  const r2 = dx * dx + dy * dy + dz * dz;
  const [ix, iy, iz] = shell.angular;
  const ang = Math.pow(dx, ix) * Math.pow(dy, iy) * Math.pow(dz, iz);
  let radial = 0;
  for (let p = 0; p < shell.alpha.length; p++) {
    const a = shell.alpha[p]!;
    const c = shell.c[p]!;
    // Per-primitive normalization (matches integrals-cg normCG).
    const L = ix + iy + iz;
    const norm = Math.pow(2 * a / Math.PI, 0.75)
      * Math.sqrt(Math.pow(4 * a, L) / (dfOdd(ix) * dfOdd(iy) * dfOdd(iz)));
    radial += norm * c * Math.exp(-a * r2);
  }
  return ang * radial;
}

function dfOdd(n: number): number {
  if (n <= 0) return 1;
  let r = 1;
  for (let k = 1; k <= n; k++) r *= 2 * k - 1;
  return r;
}

function evalDensityAtPoint(
  D: Float64Array,
  shells: readonly CGShell[],
  r: readonly [number, number, number],
): number {
  const n = shells.length;
  const phi = new Float64Array(n);
  for (let mu = 0; mu < n; mu++) phi[mu] = evalShellAtPoint(shells[mu]!, r);
  let rho = 0;
  for (let mu = 0; mu < n; mu++) {
    const phimu = phi[mu]!;
    if (phimu === 0) continue;
    for (let nu = 0; nu < n; nu++) {
      rho += D[mu * n + nu]! * phimu * phi[nu]!;
    }
  }
  return rho;
}

function evalMOAtPoint(
  coeffs: Float64Array,
  shells: readonly CGShell[],
  r: readonly [number, number, number],
): number {
  let phi = 0;
  for (let mu = 0; mu < shells.length; mu++) {
    phi += coeffs[mu]! * evalShellAtPoint(shells[mu]!, r);
  }
  return phi;
}
