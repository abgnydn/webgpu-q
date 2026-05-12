#!/usr/bin/env python3
"""
E34 PySCF reference run.

Produces the PySCF side of the wall-clock-vs-pyscf comparison.
Runs the same 4 molecules × 2 basis sets × {HF, MP2, CCSD, CCSD(T)}
as experiments/level-6-chemistry/E34-wallclock-vs-pyscf.ts with
identical convergence thresholds, emitting matching JSON schema.

Usage:
    pip install pyscf==2.13.0
    python3 scripts/run-pyscf-reference.py > pyscf-reference.json

Honest scope:
    - "Wall-clock" is what `time.perf_counter()` reports around the
      call. PySCF amortizes integral construction; we time as PySCF
      naturally orders the calls.
    - No GPU PySCF here. To compare against gpu4pyscf, install
      gpu4pyscf and re-run with `--engine gpu4pyscf`.
    - Convergence thresholds match webgpu-q: SCF 1e-10, CCSD 1e-10.

Reference: PySCF 2.13.0 (April 2026).
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone

try:
    from pyscf import __version__ as pyscf_version
    from pyscf import gto, scf, mp, cc
except ImportError:
    print("ERROR: pyscf not installed. Run: pip install pyscf==2.13.0", file=sys.stderr)
    sys.exit(1)

SCF_TOL = 1e-10
CCSD_TOL = 1e-10

# Same geometries as experiments/level-6-chemistry/E34-wallclock-vs-pyscf.ts.
MOLECULES = [
    {
        "name": "H₂",
        "atom": "H 0 0 0; H 0 0 0.7414",
        "n_electrons": 2,
    },
    {
        "name": "LiH",
        "atom": "Li 0 0 0; H 0 0 1.595",
        "n_electrons": 4,
    },
    {
        "name": "BeH₂",
        "atom": "Be 0 0 0; H 0 0 -1.34; H 0 0 1.34",
        "n_electrons": 6,
    },
    {
        "name": "H₂O",
        # 104.52°, 0.9572 Å — matches webgpu-q E34
        "atom": "O 0 0 0; H 0.75695 0 0.58588; H -0.75695 0 0.58588",
        "n_electrons": 10,
    },
]

BASES = ["sto-3g", "cc-pvdz"]


# Honest scope: webgpu-q's STO-3G Li is currently "s-only" — just 1s + 2s,
# missing the 2p L-shell that standard STO-3G includes. See
# src/chemistry/integrals.ts and LIMITATIONS.md. To make LiH cells truly
# apples-to-apples, we pass the matching s-only Li basis to PySCF.
#
# Exponents and coefficients copied from src/chemistry/integrals.ts so the
# two sides agree to the last digit.
LI_S_ONLY_STO3G_PYSCF = """
Li    S
     16.1195750             0.15432897
      2.9362007              0.53532814
      0.7946505              0.44463454
Li    S
      0.6362897              -0.09996723
      0.1478601              0.39951283
      0.0480887              0.70011547
"""


def basis_for(molecule, basis):
    """Map (molecule, basis) → PySCF basis spec.

    For STO-3G with Li present, return a dict that uses our s-only Li
    basis (matching webgpu-q exactly) and standard STO-3G for the rest.
    For everything else, return the plain basis string.
    """
    symbols = {a.split()[0] for a in molecule["atom"].replace(";", " ").split() if a and not a[0].isdigit() and a not in ("-", "+")}
    has_li = "Li" in symbols
    if basis == "sto-3g" and has_li:
        return {sym: gto.basis.parse(LI_S_ONLY_STO3G_PYSCF) if sym == "Li" else "sto-3g"
                for sym in symbols if sym in {"H", "Li", "Be", "C", "N", "O"}}
    return basis


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def row(molecule, basis, method, seconds, energy_Ha, success, notes=None):
    # Emit `null` (not `NaN`) for missing energies so the JSON is
    # parseable by strict JSON consumers (downstream TS renderer).
    if energy_Ha is None:
        energy_out = None
    else:
        e = float(energy_Ha)
        energy_out = None if (e != e) else e  # NaN check via e != e
    r = {
        "molecule": molecule,
        "basis": basis,
        "method": method,
        "seconds": seconds,
        "energy_Ha": energy_out,
        "success": bool(success),
    }
    if notes is not None:
        r["notes"] = notes
    return r


def run_one(molecule, basis):
    """Run HF → MP2 → CCSD → CCSD(T) on one (molecule, basis) cell."""
    rows = []
    mol = gto.M(
        atom=molecule["atom"],
        basis=basis_for(molecule, basis),
        unit="Angstrom",
        symmetry=False,
        verbose=0,
    )

    # HF
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_TOL
    try:
        e_hf, t_hf = timed(lambda: mf.kernel())
    except Exception as e:
        rows.append(row(molecule["name"], basis, "HF", 0, float("nan"), False, str(e)))
        return rows
    rows.append(row(molecule["name"], basis, "HF", t_hf, e_hf, mf.converged,
                    None if mf.converged else "SCF did not converge"))
    if not mf.converged:
        return rows

    # MP2
    try:
        mp2 = mp.MP2(mf)
        e_mp2_corr, t_mp2 = timed(lambda: mp2.kernel())
        # mp2.kernel returns (E_corr, T2). e_hf + E_corr is total.
        e_mp2_total = e_hf + e_mp2_corr[0] if isinstance(e_mp2_corr, tuple) else e_hf + e_mp2_corr
        rows.append(row(molecule["name"], basis, "MP2", t_mp2, e_mp2_total, True))
    except Exception as e:
        rows.append(row(molecule["name"], basis, "MP2", 0, float("nan"), False, str(e)))

    # CCSD
    mycc = cc.CCSD(mf)
    mycc.conv_tol = CCSD_TOL
    try:
        (e_ccsd_corr, _, _), t_ccsd = timed(lambda: mycc.kernel())
        e_ccsd_total = e_hf + e_ccsd_corr
        rows.append(row(molecule["name"], basis, "CCSD", t_ccsd, e_ccsd_total, mycc.converged,
                        None if mycc.converged else "CCSD did not converge"))
        if not mycc.converged:
            return rows
    except Exception as e:
        rows.append(row(molecule["name"], basis, "CCSD", 0, float("nan"), False, str(e)))
        return rows

    # CCSD(T) — PySCF computes both CPU and (with gpu4pyscf) GPU via the
    # same ccsd_t() entry point. We label this as the canonical reference.
    run_cpu_t = basis == "sto-3g" or molecule["name"] == "H₂"
    if run_cpu_t:
        try:
            e_t_corr, t_t = timed(lambda: mycc.ccsd_t())
            e_t_total = e_ccsd_total + e_t_corr
            rows.append(row(molecule["name"], basis, "CCSD(T)-CPU", t_t, e_t_total, True))
        except Exception as e:
            rows.append(row(molecule["name"], basis, "CCSD(T)-CPU", 0, float("nan"), False, str(e)))
    else:
        rows.append(row(
            molecule["name"], basis, "CCSD(T)-CPU", 0, float("nan"), True,
            "skipped — match webgpu-q E34 (CPU (T) at cc-pVDZ is minutes; GPU only)",
        ))

    # No native GPU CCSD(T) in stock PySCF — gpu4pyscf required.
    rows.append(row(
        molecule["name"], basis, "CCSD(T)-GPU", 0, float("nan"), True,
        "not run — install gpu4pyscf and re-run for GPU comparison",
    ))

    return rows


def main():
    parser = argparse.ArgumentParser(description="PySCF reference run for webgpu-q E34")
    parser.add_argument("--out", default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    rows = []
    for mol in MOLECULES:
        for basis in BASES:
            print(f"[E34-pyscf] {mol['name']} / {basis} ...", file=sys.stderr)
            try:
                rows.extend(run_one(mol, basis))
            except Exception as e:
                print(f"  cell failed: {e}", file=sys.stderr)
                rows.append(row(mol["name"], basis, "HF", 0, float("nan"), False, f"cell-level: {e}"))

    artifact = {
        "meta": {
            "protocol": "E34-wallclock-vs-pyscf-reference",
            "hypothesis": "PySCF reference side of E34 wall-clock comparison. Same molecules, same basis, same convergence thresholds (SCF 1e-10, CCSD 1e-10) as webgpu-q E34.",
            "passBar": "Every method converges OR is explicitly skipped.",
            "seed": "E34_PYSCF_REFERENCE",
            "warmup": 0,
            "trials": 1,
        },
        "env": {
            "runtime": f"python {platform.python_version()}",
            "pyscf_version": pyscf_version,
            "platform": platform.platform(),
            "arch": platform.machine(),
            "hostname": socket.gethostname(),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "method": "PySCF reference: scf.RHF → mp.MP2 → cc.CCSD → ccsd_t() (CPU only, no gpu4pyscf in this run)",
        },
        "rows": rows,
        "status": "pass" if all(r["success"] for r in rows) else "fail",
        "diagnosis": f"Generated {len(rows)} reference rows. Merge with webgpu-q E34 artifact for side-by-side comparison.",
    }

    out_str = json.dumps(artifact, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out_str)
        print(f"[E34-pyscf] wrote {args.out}", file=sys.stderr)
    else:
        print(out_str)


if __name__ == "__main__":
    main()
