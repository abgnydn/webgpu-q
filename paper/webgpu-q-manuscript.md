# Quantum chemistry in a browser tab — webgpu-q

**Ahmet Barış Günaydın** · Independent researcher · `github.com/abgnydn/webgpu-q`

> **Canonical manuscript.** The full, current paper is the LaTeX source
> [`main.tex`](./main.tex) and its rendered PDF [`main.pdf`](./main.pdf), with
> figures `fig-*.pdf` and references [`refs.bib`](./refs.bib). This file is a
> readable abstract only; the earlier long-form Markdown draft was retired so
> there is a single source of truth (it had drifted from the LaTeX on several
> numbers).

## Abstract

We present **webgpu-q**, an electronic-structure stack — Hartree–Fock
(RHF/UHF), MP2, CCSD, CCSD(T), density-functional theory across the LDA/GGA/
hybrid ladder, and EE/IP/EA-EOM-CCSD — that runs entirely inside a web browser
with no installation, no server, and no CUDA. Open Computational Chemistry
(OCC) has shown Hartree–Fock and DFT running client-side in the browser; we are
not aware of a prior in-browser implementation of the *correlated post-Hartree–
Fock* hierarchy. Numerical hot paths (electron-repulsion integrals, the
auxiliary-basis density-fitting B-tensor, the Fock build) are hand-written in
Rust compiled to WebAssembly with SIMD128; the methods are ported from PySCF
with attribution, so the contribution is the delivery mechanism, the SIMD
kernels, and the distribution layer. We introduce a **browser-tab swarm**: the
density-fitted Fock build is partitioned by auxiliary index across N same-origin
tabs that coordinate over `BroadcastChannel`/`SharedArrayBuffer`, and the
partial Coulomb/exchange matrices sum to reproduce the single-tab build to a
relative error of order 10⁻¹⁵. The swarm raises the single-tab memory ceiling
and scales Hartree–Fock to **C₆₀ (300 basis functions, STO-3G)** on a 16 GB
consumer laptop, distributing the 1.82 GB three-index tensor across four tabs at
454 MB each. Energies are validated against PySCF — bit-for-bit where the
algorithm is identical, and to ≤ 0.5 mHa (HF; ≤ 0.1 mHa with spherical-d) /
≤ 0.25 mHa (CCSD(T) vs FCI) otherwise. Speed is workload-dependent: faster than
native suites on small systems (no interpreter startup), but up to ~two orders
of magnitude slower on large correlated calculations where tuned BLAS dominates.
The contribution is not speed but delivery — full quantum chemistry reachable
from a URL.

---

*See [`main.pdf`](./main.pdf) for the complete manuscript (methods, the full
swarm protocol, validation, the acene→C₆₀ scaling results, honest limitations,
reproducibility, and figures). Archived at Zenodo, concept DOI
[10.5281/zenodo.20494382](https://doi.org/10.5281/zenodo.20494382).*
