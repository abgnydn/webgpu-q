// File-format parsers for user-supplied geometry input.
// Supported: .xyz (XMol), .pdb (Protein Data Bank), .mol/.sdf (MDL).
// Output: Atom[] in our internal convention (symbol + Ångström position).

import type { Atom, AtomSymbol } from "../chemistry/atoms.js";

// The chemistry stack currently supports periods 1-2 complete (see
// src/chemistry/atoms.ts → AtomSymbol). Imports outside this set are
// rejected with a clear error.
const SUPPORTED_SYMBOLS = new Set<string>([
  "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
]);

export interface ParseResult {
  readonly format: "xyz" | "pdb" | "mol" | "sdf";
  readonly atoms: Atom[];
  /** Comment line / title from the file, if any. */
  readonly title?: string;
}

export function detectFormat(filename: string): ParseResult["format"] | null {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  if (ext === "xyz") return "xyz";
  if (ext === "pdb") return "pdb";
  if (ext === "mol") return "mol";
  if (ext === "sdf") return "sdf";
  return null;
}

export function parseGeometry(text: string, format: ParseResult["format"]): ParseResult {
  switch (format) {
    case "xyz": return parseXYZ(text);
    case "pdb": return parsePDB(text);
    case "mol":
    case "sdf": return parseMOL(text);
  }
}

// ── XYZ ────────────────────────────────────────────────────────
// Line 1: integer N
// Line 2: comment / title (free text)
// Lines 3..N+2: <Symbol>  <X>  <Y>  <Z>  (Ångström)
function parseXYZ(text: string): ParseResult {
  const lines = text.split(/\r?\n/);
  if (lines.length < 3) throw new Error("XYZ: file too short");
  const n = parseInt(lines[0]!.trim(), 10);
  if (!Number.isFinite(n) || n <= 0) {
    throw new Error(`XYZ: first line must be atom count, got "${lines[0]}"`);
  }
  const title = lines[1]?.trim();
  const atoms: Atom[] = [];
  for (let i = 0; i < n; i++) {
    const raw = lines[2 + i];
    if (!raw) throw new Error(`XYZ: missing atom row ${i + 1} of ${n}`);
    const parts = raw.trim().split(/\s+/);
    if (parts.length < 4) {
      throw new Error(`XYZ: row ${i + 1} needs "Symbol X Y Z", got "${raw}"`);
    }
    const symbol = canonicalSymbol(parts[0]!);
    const x = parseFloat(parts[1]!); const y = parseFloat(parts[2]!); const z = parseFloat(parts[3]!);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      throw new Error(`XYZ: row ${i + 1} non-finite coordinate`);
    }
    atoms.push({ symbol, pos: [x, y, z] });
  }
  return { format: "xyz", atoms, ...(title ? { title } : {}) };
}

// ── PDB ────────────────────────────────────────────────────────
// Fixed-column format. We honor only ATOM / HETATM records.
// Cols 31-38: X (Å). 39-46: Y. 47-54: Z. 77-78: element symbol.
// If element field is blank we fall back to inferring from atom name.
function parsePDB(text: string): ParseResult {
  const atoms: Atom[] = [];
  let title: string | undefined;
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("TITLE")) {
      const t = line.slice(10).trim();
      if (t) title = title ? `${title} ${t}` : t;
      continue;
    }
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    if (line.length < 54) continue;
    // Element column 77-78 (PDB v3.30); 1-indexed 77,78 = 0-indexed 76,77.
    let symRaw = line.slice(76, 78).trim();
    if (!symRaw) {
      // Fall back to first 1-2 alpha chars of atom name (cols 13-16).
      const name = line.slice(12, 16).trim();
      symRaw = name.replace(/[^A-Za-z]/g, "").slice(0, 2);
    }
    if (!symRaw) continue;
    const x = parseFloat(line.slice(30, 38));
    const y = parseFloat(line.slice(38, 46));
    const z = parseFloat(line.slice(46, 54));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    atoms.push({ symbol: canonicalSymbol(symRaw), pos: [x, y, z] });
  }
  if (atoms.length === 0) {
    throw new Error("PDB: no ATOM / HETATM records found");
  }
  return { format: "pdb", atoms, ...(title ? { title } : {}) };
}

// ── MOL / SDF (MDL V2000) ──────────────────────────────────────
// Line 1: title; Line 2: program info; Line 3: comment;
// Line 4: counts line "  N  M  …" (atoms, bonds).
// Then N atom lines (cols 1-30: X Y Z floats; cols 31-33: element).
function parseMOL(text: string): ParseResult {
  const lines = text.split(/\r?\n/);
  if (lines.length < 5) throw new Error("MOL/SDF: file too short");
  const title = lines[0]?.trim() || undefined;
  const counts = lines[3] ?? "";
  if (counts.length < 6) throw new Error(`MOL/SDF: bad counts line "${counts}"`);
  const n = parseInt(counts.slice(0, 3).trim(), 10);
  if (!Number.isFinite(n) || n <= 0) {
    throw new Error(`MOL/SDF: atom count parse failed from "${counts}"`);
  }
  const atoms: Atom[] = [];
  for (let i = 0; i < n; i++) {
    const raw = lines[4 + i];
    if (!raw) throw new Error(`MOL/SDF: missing atom row ${i + 1} of ${n}`);
    if (raw.length < 34) {
      throw new Error(`MOL/SDF: row ${i + 1} too short: "${raw}"`);
    }
    const x = parseFloat(raw.slice(0, 10));
    const y = parseFloat(raw.slice(10, 20));
    const z = parseFloat(raw.slice(20, 30));
    const symRaw = raw.slice(31, 34).trim();
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      throw new Error(`MOL/SDF: row ${i + 1} non-finite coordinate`);
    }
    atoms.push({ symbol: canonicalSymbol(symRaw), pos: [x, y, z] });
  }
  return { format: "mol", atoms, ...(title ? { title } : {}) };
}

// ── Symbol normalization ───────────────────────────────────────
function canonicalSymbol(raw: string): AtomSymbol {
  // "C" → "C", "ca" → "Ca", "12C" → "C", "Cα" → "C".
  const clean = raw.replace(/[^A-Za-z]/g, "");
  if (!clean) throw new Error(`Unrecognized atom symbol "${raw}"`);
  const norm = clean.slice(0, 1).toUpperCase() + clean.slice(1, 2).toLowerCase();
  // Try the 2-char form first; if not in the supported set, fall back to 1-char.
  if (SUPPORTED_SYMBOLS.has(norm)) return norm as AtomSymbol;
  const single = clean.slice(0, 1).toUpperCase();
  if (SUPPORTED_SYMBOLS.has(single)) return single as AtomSymbol;
  throw new Error(`Unsupported atom symbol "${raw}" (parsed "${norm}"). ` +
    `Supported: ${Array.from(SUPPORTED_SYMBOLS).join(", ")}`);
}
