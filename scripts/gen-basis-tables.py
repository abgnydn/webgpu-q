"""Gate 0.2 — generate TypeScript basis-set constants from PySCF's own tables.

docs/RUN-PLAN-24H-ELEMENTS.md forbids hand-transcribing ~100 blocks of
decimal digits for 10 new elements x 3 basis sets. This script reads
`pyscf.gto.basis.load` and emits constants in the exact shape
`src/chemistry/integrals.ts` already uses:

    export const STO3G_C_2P = {
      alpha: [2.9412494, 0.6834831, 0.2222899] as const,
      c: STO3G_L_2P_C,
    };

It also runs the trust gate: the SAME code path is diffed, name by name
and number by number, against the constants currently exported by
integrals.ts (dumped via scripts/dump-basis-constants.ts). The generator
is only trusted for Na-Ar once it reproduces the known-clean cells
(C/N/O/F in sto-3g + cc-pvdz, C in aug-cc-pvdz) to ~1e-6 relative.

Usage:
    npx --yes tsx scripts/dump-basis-constants.ts > /tmp/ours.json
    cd ~/dev/ml-research && uv run python \
        /path/to/webgpu-q/scripts/gen-basis-tables.py \
        --verify /tmp/ours.json --out /tmp/generated

Flags:
    --out DIR       write <basis>.ts fragments + all.ts into DIR
    --verify FILE   diff against a dump-basis-constants.ts JSON
    --structure     print the per-element shell-structure table
    --elements ...  restrict the element list (default: all 18)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

from pyscf import gto

# ── configuration ────────────────────────────────────────────────

ELEMENTS = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
]

BASES = ["sto-3g", "cc-pvdz", "aug-cc-pvdz"]

TS_PREFIX = {"sto-3g": "STO3G", "cc-pvdz": "CCPVDZ", "aug-cc-pvdz": "AUG_CCPVDZ"}

LSYM = "SPDFGH"

# Shell labels, per (basis, row), in PySCF's own ordering within each l.
# Reproduces the convention already in integrals.ts exactly:
#   H  cc-pVDZ : 1S, 2S, 2P
#   C  cc-pVDZ : 1S, 2S, 2S_P, 2P, 2P_P, 3D    (trailing _P = "prime")
# and extends it to row 3, where cc-pVDZ is [4s,3p,1d].
#
# Row key: 1 = H/He, 2 = Li..Ne, 3 = Na..Ar.
LABELS = {
    ("sto-3g", 1): {0: ["1S"]},
    ("sto-3g", 2): {0: ["1S", "2S"], 1: ["2P"]},
    ("sto-3g", 3): {0: ["1S", "2S", "3S"], 1: ["2P", "3P"]},
    ("cc-pvdz", 1): {0: ["1S", "2S"], 1: ["2P"]},
    ("cc-pvdz", 2): {0: ["1S", "2S", "2S_P"], 1: ["2P", "2P_P"], 2: ["3D"]},
    ("cc-pvdz", 3): {0: ["1S", "2S", "3S", "3S_P"], 1: ["2P", "3P", "3P_P"], 2: ["3D"]},
}

# STO-3G contraction coefficients are universal across the whole row set:
# only the exponents change from atom to atom. integrals.ts already
# factors the first three out as shared constants; the M-shell pair is
# new (row 3) and has to be emitted.
SHARED_STO3G = OrderedDict([
    ("1S", ("STO3G_S_C", True)),      # (const name, already exists in integrals.ts)
    ("2S", ("STO3G_L_2S_C", True)),
    ("2P", ("STO3G_L_2P_C", True)),
    ("3S", ("STO3G_M_3S_C", False)),
    ("3P", ("STO3G_M_3P_C", False)),
])

SHARED_TOL = 1e-6


def row_of(symbol: str) -> int:
    z = gto.charge(symbol)
    if z <= 2:
        return 1
    if z <= 10:
        return 2
    return 3


# ── PySCF -> flat contracted functions ───────────────────────────

def contracted_functions(basis: str, symbol: str):
    """[(l, alphas, coeffs)] in PySCF order.

    General contractions are expanded column by column: each coefficient
    column over a shared exponent block is one contracted function.
    Primitives with a zero coefficient in that column are dropped, which
    is the convention integrals.ts stores (only the non-zero span).
    """
    out = []
    for shell in gto.basis.load(basis, symbol):
        l = shell[0]
        rows = shell[1:]
        ncol = len(rows[0]) - 1
        for col in range(ncol):
            pairs = [(r[0], r[col + 1]) for r in rows if r[col + 1] != 0.0]
            if not pairs:
                continue
            out.append((l, tuple(a for a, _ in pairs), tuple(c for _, c in pairs)))
    return out


def same(x, y, tol=1e-10):
    return len(x) == len(y) and all(
        abs(a - b) <= tol * max(1.0, abs(b)) for a, b in zip(x, y)
    )


def diffuse_functions(symbol: str):
    """aug-cc-pVDZ minus cc-pVDZ, per l. Exactly one extra per l."""
    base = contracted_functions("cc-pvdz", symbol)
    aug = contracted_functions("aug-cc-pvdz", symbol)
    remaining = list(aug)
    for f in base:
        for i, g in enumerate(remaining):
            if f[0] == g[0] and same(f[1], g[1]) and same(f[2], g[2]):
                remaining.pop(i)
                break
        else:
            raise RuntimeError(
                f"{symbol}: cc-pVDZ function l={f[0]} alpha={f[1]} "
                "has no counterpart in aug-cc-pVDZ"
            )
    for l, alphas, _ in remaining:
        if len(alphas) != 1:
            raise RuntimeError(
                f"{symbol}: aug diffuse l={l} is not a single primitive: {alphas}"
            )
    return remaining


# ── naming ───────────────────────────────────────────────────────

def named_shells(basis: str, symbol: str):
    """OrderedDict TS_CONST_NAME -> (l, alphas, coeffs)."""
    prefix = TS_PREFIX[basis]
    el = symbol.upper()
    out = OrderedDict()

    if basis == "aug-cc-pvdz":
        for l, alphas, coeffs in diffuse_functions(symbol):
            out[f"{prefix}_{el}_DIFFUSE_{LSYM[l]}"] = (l, alphas, coeffs)
        return out

    labels = LABELS[(basis, row_of(symbol))]
    seen = {}
    for l, alphas, coeffs in contracted_functions(basis, symbol):
        i = seen.get(l, 0)
        seen[l] = i + 1
        table = labels.get(l)
        if table is None or i >= len(table):
            raise RuntimeError(
                f"{basis} {symbol}: unexpected function #{i} with l={l}; "
                "the LABELS table needs extending"
            )
        out[f"{prefix}_{el}_{table[i]}"] = (l, alphas, coeffs)
    return out


def all_shells(elements):
    """(basis, symbol) -> OrderedDict of named shells."""
    return {
        (b, s): named_shells(b, s)
        for b in BASES
        for s in elements
    }


# ── STO-3G shared-coefficient factoring ──────────────────────────

def shared_sto3g_vectors(elements):
    """label -> coefficient tuple, verified identical across elements.

    Returns only the labels where every element that has that shell
    agrees to SHARED_TOL. Anything else stays inline.
    """
    seen = {}
    reject = set()
    for s in elements:
        for name, (_l, _a, c) in named_shells("sto-3g", s).items():
            label = name.split("_", 2)[2]
            if label not in SHARED_STO3G:
                continue
            if label in seen:
                if not same(c, seen[label], SHARED_TOL):
                    reject.add(label)
            else:
                seen[label] = c
    return {k: v for k, v in seen.items() if k not in reject}


# ── TypeScript emission ──────────────────────────────────────────

def fmt(x: float) -> str:
    """Shortest round-tripping literal. Python's repr is exactly that,
    and every form it produces (incl. 1.7016e-05) is valid TS."""
    return repr(float(x))


def fmt_array(values, indent: str) -> str:
    body = ", ".join(fmt(v) for v in values)
    if len(body) <= 72:
        return f"[{body}] as const"
    lines, cur = [], []
    for v in values:
        cur.append(fmt(v))
        if len(", ".join(cur)) > 64:
            lines.append(", ".join(cur) + ",")
            cur = []
    if cur:
        lines.append(", ".join(cur) + ",")
    inner = "\n".join(indent + "  " + ln for ln in lines)
    return "[\n" + inner + "\n" + indent + "] as const"


def emit_shared_block(elements) -> str:
    shared = shared_sto3g_vectors(elements)
    new = [(lbl, shared[lbl]) for lbl, (_n, exists) in SHARED_STO3G.items()
           if lbl in shared and not exists]
    if not new:
        return ""
    lines = [
        "// ── STO-3G shared contraction coefficients (M-shell, row 3) ──",
        "// STO-3G's radial form is universal: within a shell type only the",
        "// exponents change from atom to atom. integrals.ts already factors",
        "// out STO3G_S_C / STO3G_L_2S_C / STO3G_L_2P_C; these are the 3s/3p",
        "// (M-shell) counterparts, needed by Na-Ar. Verified identical across",
        f"// all row-3 elements to {SHARED_TOL:g} relative.",
        "",
    ]
    for label, vec in new:
        name = SHARED_STO3G[label][0]
        lines.append(f"const {name} = {fmt_array(vec, '')};")
    lines.append("")
    return "\n".join(lines)


def emit_element(basis: str, symbol: str, shared) -> str:
    lines = [f"// ── {symbol} {basis} ──"]
    for name, (_l, alphas, coeffs) in named_shells(basis, symbol).items():
        label = name.split("_", 2)[2] if basis != "aug-cc-pvdz" else None
        cref = None
        if basis == "sto-3g" and label in shared and same(coeffs, shared[label], SHARED_TOL):
            cref = SHARED_STO3G[label][0]
        lines.append(f"export const {name} = {{")
        lines.append(f"  alpha: {fmt_array(alphas, '  ')},")
        lines.append(f"  c: {cref if cref else fmt_array(coeffs, '  ')},")
        lines.append("};")
    lines.append("")
    return "\n".join(lines)


def emit(elements, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    shared = shared_sto3g_vectors(elements)
    import pyscf
    header = (
        f"// GENERATED by scripts/gen-basis-tables.py from PySCF {pyscf.__version__}.\n"
        "// Source: pyscf.gto.basis.load. Do not hand-edit.\n"
        "// Requires already in scope (integrals.ts defines them):\n"
        "//   STO3G_S_C, STO3G_L_2S_C, STO3G_L_2P_C\n"
    )
    combined = [header, emit_shared_block(elements)]
    for basis in BASES:
        chunk = [header, ""]
        if basis == "sto-3g":
            chunk.append(emit_shared_block(elements))
        for s in elements:
            chunk.append(emit_element(basis, s, shared))
            combined.append(emit_element(basis, s, shared))
        path = os.path.join(outdir, TS_PREFIX[basis].lower() + ".ts")
        with open(path, "w") as fh:
            fh.write("\n".join(chunk))
    with open(os.path.join(outdir, "all.ts"), "w") as fh:
        fh.write("\n".join(combined))
    return outdir


# ── trust gate ───────────────────────────────────────────────────

def reldev(a, b):
    """max_i |a_i - b_i| / max(|b_i|, tiny). None if lengths differ."""
    if len(a) != len(b):
        return None
    worst = 0.0
    for x, y in zip(a, b):
        denom = abs(y) if abs(y) > 0 else 1.0
        worst = max(worst, abs(x - y) / denom)
    return worst


def verify(dump_path: str, elements):
    with open(dump_path) as fh:
        ours = json.load(fh)

    rows = []
    for basis in BASES:
        for s in elements:
            gen = named_shells(basis, s)
            present = [n for n in gen if n in ours]
            if not present:
                continue
            worst = 0.0
            worst_at = ""
            struct = []
            for name in present:
                _l, alphas, coeffs = gen[name]
                da = reldev(ours[name]["alpha"], alphas)
                dc = reldev(ours[name]["c"], coeffs)
                if da is None or dc is None:
                    struct.append(
                        f"{name}: len(ours)={len(ours[name]['alpha'])}/"
                        f"{len(ours[name]['c'])} vs pyscf={len(alphas)}/{len(coeffs)}"
                    )
                    continue
                for tag, d in (("alpha", da), ("c", dc)):
                    if d > worst:
                        worst, worst_at = d, f"{name}.{tag}"
            missing = [n for n in gen if n not in ours]
            rows.append({
                "basis": basis, "element": s,
                "checked": len(present), "generated": len(gen),
                "missing_in_repo": missing,
                "struct_mismatch": struct,
                "max_rel": worst, "at": worst_at,
            })

    width = max(len(r["at"]) for r in rows) if rows else 10
    print(f"{'basis':12s} {'el':3s} {'cells':>6s} {'max rel dev':>12s}  "
          f"{'worst entry':{width}s}  note")
    print("-" * (44 + width + 8))
    for r in rows:
        note = ""
        if r["struct_mismatch"]:
            note = "STRUCT: " + "; ".join(r["struct_mismatch"])
        elif r["missing_in_repo"]:
            note = f"{len(r['missing_in_repo'])} not in repo: " + \
                   ",".join(n.split("_", 2)[2] for n in r["missing_in_repo"])
        print(f"{r['basis']:12s} {r['element']:3s} "
              f"{r['checked']:3d}/{r['generated']:<2d} {r['max_rel']:12.3e}  "
              f"{r['at']:{width}s}  {note}")

    gate = [("sto-3g", e) for e in ("C", "N", "O", "F")] + \
           [("cc-pvdz", e) for e in ("C", "N", "O", "F")] + \
           [("aug-cc-pvdz", "C")]
    print("\nTRUST GATE (known-clean cells, bar = 1e-6 relative):")
    failures = 0
    for basis, el in gate:
        r = next((x for x in rows if x["basis"] == basis and x["element"] == el), None)
        if r is None:
            print(f"  MISSING {basis:12s} {el}")
            failures += 1
            continue
        ok = (not r["struct_mismatch"]) and r["max_rel"] <= 1e-6
        failures += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'} {basis:12s} {el:2s}  "
              f"max rel dev {r['max_rel']:.3e} at {r['at'] or '-'}")
    print("\nGATE 0.2:", "PASS" if failures == 0
          else f"FAIL — {failures} clean cell(s) disagree")
    return rows, failures


# ── shell structure report ───────────────────────────────────────

def structure(elements):
    for basis in BASES:
        print(f"\n=== {basis} ===")
        print(f"{'el':3s} {'s':>2s} {'p':>2s} {'d':>2s}  shells (label:nprim)")
        for s in elements:
            gen = named_shells(basis, s)
            cnt = {}
            desc = []
            for name, (l, alphas, _c) in gen.items():
                cnt[l] = cnt.get(l, 0) + 1
                desc.append(f"{name.split('_', 2)[2]}:{len(alphas)}")
            print(f"{s:3s} {cnt.get(0,0):2d} {cnt.get(1,0):2d} {cnt.get(2,0):2d}  "
                  + " ".join(desc))


# ── main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", metavar="DIR")
    ap.add_argument("--verify", metavar="JSON")
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--elements", nargs="*", default=ELEMENTS)
    args = ap.parse_args()

    elements = args.elements
    rc = 0

    if args.structure:
        structure(elements)

    if args.verify:
        print("\n=== trust gate: generator vs integrals.ts ===")
        _rows, failures = verify(args.verify, elements)
        rc = 1 if failures else 0

    if args.out:
        path = emit(elements, args.out)
        print(f"\nwrote TypeScript constants to {path}/")

    if not (args.out or args.verify or args.structure):
        ap.error("nothing to do: pass --out, --verify and/or --structure")

    return rc


if __name__ == "__main__":
    sys.exit(main())
