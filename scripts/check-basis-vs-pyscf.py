"""Gate 0.2 — prove PySCF's basis tables reproduce our hand-transcribed constants.

The 24h element-expansion run (docs/RUN-PLAN-24H-ELEMENTS.md) generates
TypeScript basis constants from PySCF's basis library instead of
hand-transcribing ~100 blocks of decimal digits. That generator is only
trustworthy if it reproduces the EIGHT elements already in the repo,
which were transcribed independently from EMSL/Pople tables.

This script compares, per (element, basis), the multiset of contracted
shells (exponents + coefficients) exported by src/chemistry/integrals.ts
against pyscf.gto.basis.load. Structural match => the generator can be
trusted for S, P, Cl.

It is now also a standing CI gate (`npm run check:basis`, job `basis-digits`
in .github/workflows/ci.yml): it is the only artifact that checks the basis
DIGITS at the source, rather than checking an energy that happens to come out
close. Every element in `BasisName`'s coverage is compared, and a constant
that no rule classifies is a failure — otherwise a newly-added element could
be silently exempt from the gate.

Usage:
    npx --yes tsx scripts/dump-basis-constants.ts > /tmp/ours.json
    uv run python scripts/check-basis-vs-pyscf.py /tmp/ours.json
"""
import json
import sys
from collections import defaultdict

from pyscf import gto

# Longest prefix first — AUG_CCPVDZ_ must win over CCPVDZ_.
BASIS_PREFIXES = [
    ("AUG_CCPVDZ_", "aug-cc-pvdz"),
    ("CCPVDZ_", "cc-pvdz"),
    ("STO3G_", "sto-3g"),
]
# Periods 1-3, H through Ar — the full coverage claimed in LIMITATIONS.md §1.
# `classify` matches the longest element token first, so CL/AL/AR/NA/NE/MG/SI
# can't be shadowed by C/A/N/M/S.
ELEMENTS = [
    "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
    "NA", "MG", "AL", "SI", "P", "S", "CL", "AR",
]
TOL = 1e-7

# (basis, element) cells where webgpu-q DELIBERATELY differs from the standard
# table. Excused only when the diff is EXACTLY the documented one — the expected
# extra/missing shells are matched by exponent, so a bad digit anywhere else in
# the same cell still fails the gate.
ALLOWED_DEVIATIONS = {
    # webgpu-q's STO-3G Li is s-only (1s + 2s); standard STO-3G Li also carries
    # a 2p L-shell. Intentional, and mirrored on the reference side by
    # LI_S_ONLY_STO3G_PYSCF in scripts/run-pyscf-reference.py so LiH cells stay
    # apples-to-apples.
    ("sto-3g", "Li"): {
        "reason": "s-only Li (no 2p L-shell) — see scripts/run-pyscf-reference.py",
        "ours_only": (),
        "pyscf_only": ((0.6362897, 0.1478601, 0.0480887),),
    },
}


def classify(name):
    """NAME -> (basis, element) or None."""
    for prefix, basis in BASIS_PREFIXES:
        if name.startswith(prefix):
            rest = name[len(prefix):]
            for el in sorted(ELEMENTS, key=len, reverse=True):
                if rest == el or rest.startswith(el + "_"):
                    return basis, el.capitalize()
            return None
    return None


def pyscf_shells(basis, element):
    """PySCF basis -> list of (alpha tuple, coeff tuple), general
    contractions expanded column-by-column, zero-coefficient
    primitives dropped (our tables store only the non-zero span)."""
    raw = gto.basis.load(basis, element)
    out = []
    for shell in raw:
        rows = shell[1:]           # [[exp, c0, c1, ...], ...]
        ncol = len(rows[0]) - 1
        for col in range(ncol):
            pairs = [(r[0], r[col + 1]) for r in rows if abs(r[col + 1]) > 0.0]
            if not pairs:
                continue
            out.append((tuple(a for a, _ in pairs), tuple(c for _, c in pairs)))
    return out


def close(x, y):
    return len(x) == len(y) and all(abs(a - b) <= TOL * max(1.0, abs(a)) for a, b in zip(x, y))


def deviation_is_allowed(basis, el, ours_only, theirs_only):
    """True iff this cell's diff is exactly the documented intentional one."""
    allowed = ALLOWED_DEVIATIONS.get((basis, el))
    if allowed is None:
        return None
    def same(got, want):
        return len(got) == len(want) and all(close(a, w) for (a, _), w in zip(got, want))
    if same(ours_only, allowed["ours_only"]) and same(theirs_only, allowed["pyscf_only"]):
        return allowed["reason"]
    return None


def match(ours, theirs):
    """Greedy multiset match. Returns (matched, ours_only, theirs_only)."""
    remaining = list(theirs)
    matched, ours_only = [], []
    for a, c in ours:
        for i, (a2, c2) in enumerate(remaining):
            if close(a, a2) and close(c, c2):
                matched.append((a, c))
                remaining.pop(i)
                break
        else:
            ours_only.append((a, c))
    return matched, ours_only, remaining


def main():
    with open(sys.argv[1]) as fh:
        dumped = json.load(fh)

    by_key = defaultdict(list)
    unclassified = []
    for name, val in dumped.items():
        key = classify(name)
        if key is None:
            unclassified.append(name)
            continue
        by_key[key].append((name, tuple(val["alpha"]), tuple(val["c"])))

    failures = 0
    # An unclassified constant is a HOLE in the gate, not a note: it means a
    # shell exported by integrals.ts is compared against nothing at all.
    if unclassified:
        print(f"FAIL {len(unclassified)} constant(s) not classified — add the element to "
              f"ELEMENTS (or the prefix to BASIS_PREFIXES): {unclassified}\n")
        failures += 1

    excused = 0
    for basis in ("sto-3g", "cc-pvdz", "aug-cc-pvdz"):
        for el in [e.capitalize() for e in ELEMENTS]:
            if basis == "aug-cc-pvdz":
                # Our aug tables store ONLY the diffuse additions; the
                # full aug set is cc-pVDZ + diffuse.
                ours_raw = by_key.get(("cc-pvdz", el), []) + by_key.get((basis, el), [])
            else:
                ours_raw = by_key.get((basis, el), [])
            if not ours_raw:
                continue
            ours = [(a, c) for _, a, c in ours_raw]
            try:
                theirs = pyscf_shells(basis, el)
            except Exception as exc:                       # noqa: BLE001
                print(f"FAIL {basis:12s} {el:2s}  PySCF load error: {exc}")
                failures += 1
                continue
            matched, ours_only, theirs_only = match(ours, theirs)
            ok = not ours_only and not theirs_only
            reason = None if ok else deviation_is_allowed(basis, el, ours_only, theirs_only)
            status = "OK  " if ok else ("KNOWN" if reason else "FAIL")
            print(f"{status:5s}{basis:12s} {el:2s}  matched {len(matched):2d}/{len(ours):2d}"
                  f"  ours_only={len(ours_only)} pyscf_only={len(theirs_only)}"
                  + (f"  — intentional: {reason}" if reason else ""))
            if reason:
                excused += 1
            elif not ok:
                failures += 1
                for a, c in ours_only:
                    print(f"       ours only : alpha={a}\n                   c={c}")
                for a, c in theirs_only:
                    print(f"       pyscf only: alpha={a}\n                   c={c}")

    print()
    print("GATE 0.2:", f"PASS — PySCF reproduces our tables ({excused} intentional deviation(s) excused)"
          if failures == 0
          else f"FAIL — {failures} (basis, element) cell(s) disagree")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
