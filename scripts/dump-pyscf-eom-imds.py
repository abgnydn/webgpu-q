#!/usr/bin/env python3
"""Dump PySCF EOM-CCSD intermediates for LiH STO-3G as JSON.

This is the verifier for src/chemistry/eom-ccsd-ported.ts (next-session
port). For each PySCF intermediate (Fvv, Foo, Fov, woOoO, woVoO, ...),
emit a flat Float64 array and the shape. The TS port can then compare
its computed intermediate to this JSON element-by-element.

Usage:
    /tmp/pyscf-venv/bin/python scripts/dump-pyscf-eom-imds.py \\
        > experiments/results/2026-05-13/level-6/E36-pyscf-imds-lih.json

Honest scope:
    - LiH STO-3G with the s-only Li basis (matches webgpu-q; see
      LIMITATIONS.md). PySCF basis dict matches the same primitives as
      src/chemistry/integrals.ts.
    - We invoke EOMEESinglet's make_imds rather than re-implementing.
    - SCF tol 1e-10, CCSD tol 1e-10.
"""
from __future__ import annotations

import json
import sys
import time
import socket
import platform
from datetime import datetime, timezone

import numpy as np
try:
    from pyscf import __version__ as pyscf_version
    from pyscf import gto, scf, cc
    from pyscf.cc.eom_rccsd import EOMEESinglet
except ImportError:
    print("ERROR: pyscf not installed. Run: pip install pyscf==2.13.0", file=sys.stderr)
    sys.exit(1)


# Matched s-only Li basis from src/chemistry/integrals.ts.
LI_S_ONLY_STO3G = """
Li    S
     16.1195750             0.15432897
      2.9362007              0.53532814
      0.7946505              0.44463454
Li    S
      0.6362897              -0.09996723
      0.1478601              0.39951283
      0.0480887              0.70011547
"""


def serialize_tensor(name: str, arr: np.ndarray) -> dict:
    """Flatten a NumPy array to row-major list of floats + shape."""
    return {
        "name": name,
        "shape": list(arr.shape),
        "data": [float(x) for x in arr.flatten()],
    }


def main():
    mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.595",
        basis={"Li": gto.basis.parse(LI_S_ONLY_STO3G), "H": "sto-3g"},
        unit="Angstrom",
        symmetry=False,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    t0 = time.perf_counter()
    e_hf = mf.kernel()
    t_hf = time.perf_counter() - t0
    print(f"[E36-imds] HF = {e_hf:.10f} Ha ({t_hf*1000:.1f} ms)", file=sys.stderr)

    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-10
    t0 = time.perf_counter()
    e_corr, t1, t2 = mycc.kernel()
    t_ccsd = time.perf_counter() - t0
    print(f"[E36-imds] CCSD E_corr = {e_corr:.10f} Ha ({t_ccsd*1000:.1f} ms)", file=sys.stderr)

    eom = EOMEESinglet(mycc)
    t0 = time.perf_counter()
    imds = eom.make_imds()
    t_imds = time.perf_counter() - t0
    print(f"[E36-imds] make_imds OK ({t_imds*1000:.1f} ms)", file=sys.stderr)

    # Collect every intermediate the singlet matvec reads. Triplet uses
    # the same set, so this dump is sufficient for both.
    tensors = {}
    for name in [
        "Fvv", "Foo", "Fov",
        "woOoO", "woVoO", "wvOvV",
        "woVVo", "woVvO", "woOoV",
    ]:
        arr = np.asarray(getattr(imds, name))
        tensors[name] = serialize_tensor(name, arr)
        print(f"[E36-imds]   {name}: shape={arr.shape}, "
              f"||·||∞={np.max(np.abs(arr)):.4e}", file=sys.stderr)

    # Also dump t1, t2 (so the TS port can verify it's using the same
    # amplitudes), and the spatial-orbital ERIs needed by the matvec.
    tensors["t1"] = serialize_tensor("t1", np.asarray(mycc.t1))
    tensors["t2"] = serialize_tensor("t2", np.asarray(mycc.t2))

    eris = mycc.ao2mo()
    nocc = mycc.nocc
    nvir = mol.nao - nocc
    eris_ovov = np.asarray(eris.ovov).reshape(nocc, nvir, nocc, nvir)
    tensors["eris_ovov"] = serialize_tensor("eris_ovov", eris_ovov)
    try:
        eris_ovvv = np.asarray(eris.get_ovvv(slice(None))).reshape(nocc, nvir, nvir, nvir)
        tensors["eris_ovvv"] = serialize_tensor("eris_ovvv", eris_ovvv)
    except Exception as e:
        print(f"[E36-imds]   eris_ovvv: unavailable ({e})", file=sys.stderr)

    artifact = {
        "meta": {
            "protocol": "E36-pyscf-eom-imds-dump-lih",
            "hypothesis": "Reference values for every PySCF EOM-CCSD "
                          "intermediate, evaluated on LiH STO-3G with s-only "
                          "Li basis matching webgpu-q. The TS port in "
                          "src/chemistry/eom-ccsd-ported.ts compares its "
                          "computed intermediates element-by-element to "
                          "these tensors.",
            "passBar": "Each tensor computed by makeIMDsPorted matches the "
                       "corresponding tensor here to 10⁻⁹ Ha element-wise.",
            "seed": "E36_PYSCF_IMDS",
            "warmup": 0,
            "trials": 1,
        },
        "env": {
            "runtime": f"python {platform.python_version()}",
            "pyscf_version": pyscf_version,
            "platform": platform.platform(),
            "arch": platform.machine(),
            "hostname": socket.gethostname(),
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "molecule": "LiH (Li 0 0 0; H 0 0 1.595)",
            "basis": "STO-3G (s-only Li, matched to webgpu-q)",
            "nocc": int(nocc),
            "nvir": int(nvir),
            "e_hf_Ha": float(e_hf),
            "e_ccsd_corr_Ha": float(e_corr),
        },
        "tensors": tensors,
    }

    json.dump(artifact, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
