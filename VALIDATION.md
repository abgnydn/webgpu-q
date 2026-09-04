# Validation

What "validated" means for `webgpu-q`, by method, with the oracles and
tolerances published in `BENCHMARKS.md` and `CLAUDE.md`. This is the
link to post when asked "validated by who".

Last updated: 2026-09-04.

---

## Per-method check table

| method | oracle | tolerance | note |
|---|---|---|---|
| HF (general) | PySCF 2.13.0 | ≤ 0.5 mHa | enforced across the mini-molecule ladder |
| HF (H₂O cc-pVDZ spherical-d) | PySCF 2.13.0 | ≤ 0.1 mHa | headline precision case |
| MP2 | PySCF 2.13.0 | ≤ 2 mHa (H₂O STO-3G regression cell) | tolerance covers geometry/convention spread |
| CCSD | FCI / PySCF | ≥ 95% correlation capture (~99% typical on H₂O/CH₄) | enforced |
| CCSD(T) | FCI | ≤ 0.25 mHa | sub-chemical accuracy (chemical accuracy = 1.594 mHa) |
| EE-EOM-CCSD | explicit H̄ projection on LiH | ~5×10⁻¹³ Ha element-wise | brute-force oracle, NSO=6, T̂² ≠ 0 |
| IP-EOM-CCSD | explicit H̄ projection on LiH | ~5×10⁻¹³ Ha element-wise | brute-force oracle |
| EA-EOM-CCSD | explicit H̄ projection on LiH | ~5×10⁻¹³ Ha element-wise | brute-force oracle |
| DFT (RKS/UKS) | PySCF 2.13.0 exact/DF | H₂O LDA 0.07 mHa / B3LYP5 0.02 mHa | runRHFAuto/runRKSAuto provenance |
| Gradients (HF/DFT) | finite difference / PySCF | HF analytical exact; DFT grid residual ~1×10⁻³ Ha/Bohr | see `LIMITATIONS.md` §3 |

Literature context: EOM-CCSD singlet single-excitations are typically
0.1–0.2 eV (~3.7–7.4 mHa) vs FCI, with doubly-excited states up to ~1 eV.
Our H₂ STO-3G 10⁻⁵ Ha exactness is an algorithmic-precision check
(T̂² = 0 for two-electron systems), not a real-system method claim.

---

## LiH brute-force verifiers

Three independent explicit Fock-space projections on LiH (NSO=6,
T̂² ≠ 0) — the permanent acceptance gate after any σ-vector change:

1. `tests/chemistry/eom-ccsd-bruteforce-lih.test.ts`
2. `tests/chemistry/ip-eom-ccsd-bruteforce-lih.test.ts`
3. `tests/chemistry/ea-eom-ccsd-bruteforce-lih.test.ts`

Each asserts `max|M_mine − M_exact| < 1×10⁻¹⁰` Ha over the full σ-matrix.
Measured residuals after the PySCF ports are ~5×10⁻¹³ Ha.

---

## Corrected failure mode: EA-EOM-CCSD stage-32e patch

`src/chemistry/ea-eom-ccsd.ts` once carried an empirical
`+½·E_corr·R₂` σ_2 diagonal patch (stage-32e). It was a textbook
"Correction = −Error" curve-fit: the patch was derived from the H₂
brute-force diff and then confirmed by the same diff. A 2026-06-16
audit added a multi-electron LiH oracle and measured the patch ~1 mHa
wrong. The patch was deleted and σ_2 was replaced by a direct port of
PySCF `eom_gccsd.eaccsd_matvec`. See `RESEARCH_STANDARDS.md` §7a and
`CLAUDE.md` "Honest negatives / open work".

---

## CI gates

| trigger | workflow | what it runs | gating? |
|---|---|---|---|
| PR / push | `ci.yml` | lint, typecheck, `npm run test:fast`, build | yes |
| nightly / dispatch | `ci.yml` | `npm run test:slow` (PHASE_E5 opt-in slow cells) | yes |
| PR / push | `gpu-smoke.yml` | SwiftShader landing smoke | no (`continue-on-error`) |
| PR / push | `gpu-smoke.yml` | `naga` WGSL shader compile | yes |
| weekly / dispatch | `perf-track.yml` | vitest perf tracking | no (tracking-only) |
| PR (swarm paths) / push / nightly | `swarm-benches.yml` | CPU-only `swarm-quick` specs | yes |

WebGPU-dependent Playwright specs (`npm run test:e2e`) run nowhere in
CI because hosted runners expose no adapter. They are reproduced
locally.

---

## Honest negatives

Not every result is a pass. Known limitations, retracted claims, and
`fail`-status artifacts are collected in `LIMITATIONS.md`. They are
part of the validation record, not exceptions to it.

---

## Links

- Research discipline: `RESEARCH_STANDARDS.md` §7a (porting acceptance gate)
- Dissemination enforcement: [publishing-estate](https://github.com/abgnydn/publishing-estate)
- Known limits and retracted claims: `LIMITATIONS.md`
- Benchmark roadmap and standardized sets: `BENCHMARKS.md`
