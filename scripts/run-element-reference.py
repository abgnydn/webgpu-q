"""Element-validation reference — PySCF side.

Emits, for one small molecule per supported element, the RHF energy under
BOTH d-conventions:

  spherical  (PySCF default, 5d)  <-> computeMolecularIntegrals(..., {spherical:true})
  cartesian  (mol.cart = True, 6d) <-> computeMolecularIntegrals(...)   [default path]

Gate 0.1 of docs/RUN-PLAN-24H-ELEMENTS.md: each webgpu-q code path must be
compared against its MATCHING convention. Comparing the Cartesian path to
default (spherical) PySCF produces a ~0.34 mHa phantom error on every
element that carries d functions -- a basis-set convention difference, not
a code bug (see tests/chemistry/ccpvdz-spherical.test.ts).

Geometries are emitted alongside the energies so the TypeScript side
consumes the exact same coordinates -- no double transcription.

Usage:
    cd ~/dev/ml-research && uv run python \
        /path/to/webgpu-q/scripts/run-element-reference.py --out /tmp/ref.json
"""
import argparse
import json
import math

from pyscf import gto, scf

SCF_TOL = 1e-12


def tetrahedral(bond):
    a = bond / math.sqrt(3.0)
    return [(a, a, a), (a, -a, -a), (-a, a, -a), (-a, -a, a)]


def bent(bond, angle_deg):
    half = math.radians(angle_deg / 2.0)
    x, z = bond * math.sin(half), bond * math.cos(half)
    return [(x, 0.0, z), (-x, 0.0, z)]


def pyramidal(bond, angle_deg):
    """3 equivalent bonds, given X-H distance and H-X-H angle."""
    ha = math.radians(angle_deg / 2.0)
    # in-plane radius r and height h satisfying the H-X-H angle
    r = bond * math.sin(ha) / math.sin(math.radians(60.0))
    h = math.sqrt(max(bond * bond - r * r, 0.0))
    return [(r * math.cos(math.radians(90 + 120 * k)),
             r * math.sin(math.radians(90 + 120 * k)), -h) for k in range(3)]


# element -> (name, atoms). One molecule per element; hydrides preferred
# over bare atoms so two-center integrals are actually exercised.
MOLECULES = {
    "H":  ("H2",   [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]),
    "He": ("He",   [("He", (0.0, 0.0, 0.0))]),
    "Li": ("LiH",  [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5949))]),
    "Be": ("BeH2", [("Be", (0.0, 0.0, 0.0)),
                    ("H", (0.0, 0.0, 1.3264)), ("H", (0.0, 0.0, -1.3264))]),
    "B":  ("BH3",  [("B", (0.0, 0.0, 0.0))] +
                   [("H", (1.1900 * math.cos(math.radians(90 + 120 * k)),
                           1.1900 * math.sin(math.radians(90 + 120 * k)), 0.0))
                    for k in range(3)]),
    "C":  ("CH4",  [("C", (0.0, 0.0, 0.0))] +
                   [("H", p) for p in tetrahedral(1.0870)]),
    "Ne": ("Ne",   [("Ne", (0.0, 0.0, 0.0))]),
    "N":  ("NH3",  [("N", (0.0, 0.0, 0.0))] +
                   [("H", p) for p in pyramidal(1.0124, 106.67)]),
    "O":  ("H2O",  [("O", (0.0, 0.0, 0.0))] +
                   [("H", p) for p in bent(0.9572, 104.52)]),
    "F":  ("HF",   [("F", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.9168))]),
    # Third row. Hydrides wherever one exists, so two-center integrals
    # are actually exercised; bare atoms only for the noble gas.
    "Na": ("NaH",  [("Na", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8874))]),
    "Mg": ("MgH2", [("Mg", (0.0, 0.0, 0.0)),
                    ("H", (0.0, 0.0, 1.7297)), ("H", (0.0, 0.0, -1.7297))]),
    "Al": ("AlH3", [("Al", (0.0, 0.0, 0.0))] +
                   [("H", (1.5840 * math.cos(math.radians(90 + 120 * k)),
                           1.5840 * math.sin(math.radians(90 + 120 * k)), 0.0))
                    for k in range(3)]),
    "Si": ("SiH4", [("Si", (0.0, 0.0, 0.0))] +
                   [("H", p) for p in tetrahedral(1.4798)]),
    "P":  ("PH3",  [("P", (0.0, 0.0, 0.0))] +
                   [("H", p) for p in pyramidal(1.4200, 93.5)]),
    "S":  ("H2S",  [("S", (0.0, 0.0, 0.0))] +
                   [("H", p) for p in bent(1.3356, 92.11)]),
    "Cl": ("HCl",  [("Cl", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.2746))]),
    "Ar": ("Ar",   [("Ar", (0.0, 0.0, 0.0))]),
}

BASES = ["sto-3g", "cc-pvdz", "aug-cc-pvdz"]

# webgpu-q uses an s-only STO-3G for lithium (documented; see
# scripts/run-pyscf-reference.py). Feed PySCF the matching basis so the
# comparison stays apples-to-apples.
LI_S_ONLY_STO3G = """
Li    S
     16.1195750              0.15432897
      2.9362007              0.53532814
      0.7946505              0.44463454
Li    S
      0.6362897             -0.09996723
      0.1478601              0.39951283
      0.0480887              0.70011547
"""


def basis_for(symbols, basis):
    if basis == "sto-3g" and "Li" in symbols:
        return {s: (gto.basis.parse(LI_S_ONLY_STO3G) if s == "Li" else "sto-3g")
                for s in set(symbols)}
    return basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = []
    for el, (name, atoms) in MOLECULES.items():
        symbols = [a for a, _ in atoms]
        for basis in BASES:
            for conv, cart in (("spherical", False), ("cartesian", True)):
                row = {
                    "element": el, "molecule": name, "basis": basis,
                    "convention": conv,
                    "atoms": [{"symbol": s, "pos": list(p)} for s, p in atoms],
                }
                try:
                    mol = gto.M(atom=[[s, tuple(p)] for s, p in atoms],
                                basis=basis_for(symbols, basis),
                                unit="Angstrom", verbose=0)
                    mol.cart = cart
                    mol.build()
                    mf = scf.RHF(mol)
                    mf.conv_tol = SCF_TOL
                    e = mf.kernel()
                    if not mf.converged:
                        row.update(ok=False, note="SCF did not converge")
                    else:
                        row.update(ok=True, E_HF=float(e), nao=int(mol.nao_nr()))
                except Exception as exc:                    # noqa: BLE001
                    row.update(ok=False, note=str(exc))
                results.append(row)
                status = f"{row.get('E_HF', float('nan')):.10f}" if row.get("ok") else "FAIL"
                print(f"{name:5s} {basis:12s} {conv:9s} {status}")

    with open(args.out, "w") as fh:
        json.dump({"scf_tol": SCF_TOL, "rows": results}, fh, indent=2)
    print(f"\nwrote {args.out} ({len(results)} rows)")


if __name__ == "__main__":
    main()
