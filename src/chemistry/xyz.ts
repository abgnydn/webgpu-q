// ─────────────────────────────────────────────────────────────
// xyz.ts — parse standard XYZ file text into Atom[] / emit Atom[]
// as XYZ.
//
// XYZ format (the de facto standard for molecular geometry):
//
//   <N_atoms>
//   <comment line; often title, charge, multiplicity>
//   <symbol> <x> <y> <z>
//   <symbol> <x> <y> <z>
//   ...
//
// Coordinates are conventionally in Ångström (matches our `Atom.pos`
// convention). Optional 4th column on the second line carries
// arbitrary properties — we ignore.
//
// Supports symbol either as element symbol ("H", "O", "C") or as
// atomic number ("1", "8", "6") — both common in PDB/QM workflows.
//
// Limitations:
//   - Single-frame only. Multi-frame trajectory XYZ (concatenated)
//     would need explicit frame splitting; queued.
//   - Element set restricted to the AtomSymbol union (H, He, Li, Be, C, N,
//     O, F). Throws for unsupported elements with a clear message.
//   - Comment line preserved as opt return; no parsing of charge /
//     multiplicity / properties (those vary too much across codes).
// ─────────────────────────────────────────────────────────────

import { type Atom, type AtomSymbol } from "./atoms.js";

// He included: it is a member of AtomSymbol, carries basis data in atoms.ts, and
// molecules.ts ships a He molecule — omitting it here made toXYZ → parseXYZ fail
// to round-trip a molecule this library provides.
const SYMBOL_BY_Z: Record<number, AtomSymbol> = {
  1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B",
  6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
};
const SUPPORTED_SYMBOLS = new Set<AtomSymbol>([
  "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
]);

export interface XYZParseResult {
  readonly atoms: Atom[];
  /** Free-form comment line from the XYZ (line 2). */
  readonly comment: string;
}

/**
 * Parse a single-frame XYZ string. Returns the atoms (positions in
 * Ångström — same convention as our Atom interface) and the comment
 * line for context.
 */
export function parseXYZ(text: string): XYZParseResult {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  if (lines.length < 2) {
    throw new Error(`parseXYZ: input too short (${lines.length} lines, need ≥ 2 header lines)`);
  }
  const nAtoms = parseInt(lines[0]!.trim(), 10);
  if (!Number.isInteger(nAtoms) || nAtoms <= 0) {
    throw new Error(`parseXYZ: line 1 must be a positive integer N_atoms, got "${lines[0]}"`);
  }
  const comment = lines[1] ?? "";
  if (lines.length < 2 + nAtoms) {
    throw new Error(`parseXYZ: declared ${nAtoms} atoms but only ${lines.length - 2} atom lines provided`);
  }
  const atoms: Atom[] = [];
  for (let i = 0; i < nAtoms; i++) {
    const lineNo = 3 + i;            // 1-indexed for error messages
    const raw = lines[2 + i]!.trim();
    if (raw === "") {
      throw new Error(`parseXYZ: line ${lineNo} is empty`);
    }
    const toks = raw.split(/\s+/);
    if (toks.length < 4) {
      throw new Error(`parseXYZ: line ${lineNo} needs ≥ 4 tokens (symbol, x, y, z), got ${toks.length}`);
    }
    const sym = resolveSymbol(toks[0]!, lineNo);
    const x = parseFloat(toks[1]!);
    const y = parseFloat(toks[2]!);
    const z = parseFloat(toks[3]!);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      throw new Error(`parseXYZ: line ${lineNo} has non-finite coords (${toks[1]}, ${toks[2]}, ${toks[3]})`);
    }
    atoms.push({ symbol: sym, pos: [x, y, z] });
  }
  return { atoms, comment };
}

/**
 * Emit an XYZ string from an Atom[]. Coordinates rendered in Å with
 * 8 decimal places.
 */
export function toXYZ(atoms: readonly Atom[], comment = ""): string {
  const lines: string[] = [];
  lines.push(String(atoms.length));
  lines.push(comment);
  for (const at of atoms) {
    lines.push(
      `${at.symbol.padEnd(2)}  ${at.pos[0].toFixed(8).padStart(15)} ${at.pos[1].toFixed(8).padStart(15)} ${at.pos[2].toFixed(8).padStart(15)}`,
    );
  }
  return lines.join("\n") + "\n";
}

export interface TrajectoryFrame {
  readonly atoms: readonly Atom[];
  /** Optional energy in Hartree (or any unit; written verbatim on
   *  the comment line as "E = <value>"). */
  readonly energy?: number;
  /** Optional comment string for this frame. If both `comment` and
   *  `energy` are given, energy is appended to comment with " | ". */
  readonly comment?: string;
}

/**
 * Emit a multi-frame XYZ trajectory string. Each frame is a full
 * XYZ block back-to-back, the standard convention readable by
 * VMD / Avogadro / OVITO / Jmol's trajectory viewer.
 *
 * Useful for visualizing geom-opt convergence, NEB / IRC reaction
 * paths, MD trajectories.
 */
export function toMultiFrameXYZ(frames: readonly TrajectoryFrame[]): string {
  let out = "";
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i]!;
    let comment = f.comment ?? `frame ${i + 1}`;
    if (f.energy !== undefined) {
      comment += ` | E = ${f.energy.toFixed(10)} Ha`;
    }
    out += toXYZ(f.atoms, comment);
  }
  return out;
}

/**
 * Parse a multi-frame XYZ string into frames. Same format as
 * `toMultiFrameXYZ` produces; tolerates extra blank trailing lines.
 */
export function parseMultiFrameXYZ(text: string): TrajectoryFrame[] {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const frames: TrajectoryFrame[] = [];
  let pos = 0;
  while (pos < lines.length) {
    // Skip blank lines between frames.
    while (pos < lines.length && lines[pos]!.trim() === "") pos++;
    if (pos >= lines.length) break;
    const nAtomsRaw = lines[pos]!.trim();
    const nAtoms = parseInt(nAtomsRaw, 10);
    if (!Number.isInteger(nAtoms) || nAtoms <= 0) {
      throw new Error(`parseMultiFrameXYZ: expected N_atoms at line ${pos + 1}, got "${nAtomsRaw}"`);
    }
    // Reconstruct a single-frame XYZ block.
    const block = lines.slice(pos, pos + 2 + nAtoms).join("\n");
    const single = parseXYZ(block);
    // Parse "E = <value> Ha" from the comment if present.
    let energy: number | undefined;
    const match = single.comment.match(/E\s*=\s*(-?\d+\.?\d*)/);
    if (match) energy = parseFloat(match[1]!);
    frames.push({ atoms: single.atoms, energy, comment: single.comment });
    pos += 2 + nAtoms;
  }
  return frames;
}

// ─────────────────────────────────────────────────────────────

function resolveSymbol(token: string, lineNo: number): AtomSymbol {
  // Try as atomic number first.
  const Z = parseInt(token, 10);
  if (Number.isInteger(Z) && Z > 0) {
    const sym = SYMBOL_BY_Z[Z];
    if (!sym) {
      throw new Error(`parseXYZ: line ${lineNo} atomic number ${Z} is not supported (supported: H, He, Li, Be, B, C, N, O, F, Ne)`);
    }
    return sym;
  }
  // Otherwise treat as symbol — case-normalize and validate.
  const upper = token.charAt(0).toUpperCase() + token.slice(1).toLowerCase();
  if (!SUPPORTED_SYMBOLS.has(upper as AtomSymbol)) {
    throw new Error(`parseXYZ: line ${lineNo} element "${token}" not supported (supported: H, He, Li, Be, B, C, N, O, F, Ne)`);
  }
  return upper as AtomSymbol;
}
