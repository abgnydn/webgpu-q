<div align="center">

<img src="./public/readme-hero.svg" alt="webgpu-q — quantum chemistry and many-body physics in a browser tab" width="100%"/>

<br/>

<a href="https://webgpu-q.vercel.app"><img alt="Launch" src="https://img.shields.io/badge/%E2%96%B6%20LAUNCH-webgpu--q.vercel.app-22d3ee?style=for-the-badge&labelColor=0b1224"/></a>
&nbsp;
<a href="https://webgpu-q.vercel.app/molecule.html"><img alt="SI Report" src="https://img.shields.io/badge/H%E2%82%82O%20%C2%B7%20SI%20REPORT-%2Fmolecule.html-c084fc?style=for-the-badge&labelColor=0b1224"/></a>
&nbsp;
<a href="https://webgpu-q.vercel.app/experiments/"><img alt="Research dashboard" src="https://img.shields.io/badge/RESEARCH%20DASH-%2Fexperiments-34d399?style=for-the-badge&labelColor=0b1224"/></a>

<br/><br/>

<img alt="version" src="https://img.shields.io/badge/v0.9.1-0ea5e9?style=flat-square&labelColor=0b1224"/>
<img alt="license" src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square&labelColor=0b1224"/>
<img alt="tests" src="https://img.shields.io/badge/tests-CI%20green-22c55e?style=flat-square&labelColor=0b1224"/>
<img alt="typescript" src="https://img.shields.io/badge/typescript-strict-3178c6?style=flat-square&labelColor=0b1224"/>
<img alt="webgpu" src="https://img.shields.io/badge/WebGPU-required-ff7849?style=flat-square&labelColor=0b1224"/>
<img alt="install-free" src="https://img.shields.io/badge/install-0%20bytes-eab308?style=flat-square&labelColor=0b1224"/>
<a href="https://doi.org/10.5281/zenodo.20494383"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20494383-3b82f6?style=flat-square&labelColor=0b1224"/></a>

</div>

<br/>

<table align="center" border="0">
<tr>
<td align="center" width="800">

**A browser-native quantum chemistry sandbox.** Open a URL and get HF · UHF · DFT (RKS+UKS) · MP2 · CCSD · CCSD(T) · EOM-CCSD · TDDFT α(ω) · C₆ dispersion on real molecules — with WebGPU acceleration for the (T) bottleneck. No install. No backend. No CUDA.

</td>
</tr>
</table>

<br/>

<table align="center" border="0">
<tr>
<td valign="top" width="33%">

**What this is**

- Browser-native quantum chemistry engine, all methods ported from PySCF and brute-force verified to ≤ 1e-10 Ha element-wise.
- WebGPU systems demonstration — a 110-line WGSL kernel makes CCSD(T)/cc-pVDZ browser-feasible (13.8× median speedup over CPU TypeScript, 5w+20t harness; p10 28×, p90 10×, noisy).
- Distributed-compute substrate — `swarmMap` primitive over BroadcastChannel (verified) and WebRTC (cross-machine path; chem-energy kernel works in same-machine multi-tab, cross-machine unverified e2e).
- Teaching / reproducibility / methodology platform — URL-as-citation, drag-import (XYZ/PDB/MOL/SDF), in-browser Pyodide REPL with "Compare to PySCF" button.

</td>
<td valign="top" width="33%">

**What this isn't**

- **Not a PySCF replacement.** PySCF runs 10-100× faster, scales to 100× larger molecules, supports 10× more methods, and has 25 years of validation. If you're a working computational chemist, install PySCF.
- **Not a production research tool for ≥ 30-atom molecules.** Largest benchmarked system is BeH₂/cc-pVDZ; nothing at the porphyrin / metal-complex / enzyme-active-site scale that real research chemistry needs.
- **Not a CUDA competitor.** WebGPU is the lowest-common-denominator GPU API; raw throughput is 2-5× behind well-tuned CUDA today (gap closing as the WebGPU spec evolves — f16 shader extension, subgroups, tensor-core intrinsics).
- **Not novel chemistry.** Every method here was already in PySCF before we ported it. The contribution is the substrate, not the algorithms.

</td>
<td valign="top" width="33%">

**Who it's for**

- **Chemistry educators.** Quantum methods are an opaque pipeline ("HF then MP2 then CCSD…"). webgpu-q makes every intermediate inspectable, computed in real time, with no install barrier. Drop a PDB file, run HF, see the orbitals.
- **WebGPU / systems researchers.** A real O(n⁷) kernel benchmarked end-to-end in the browser. Hardware-vs-API throughput gap can be tracked over time as the spec evolves.
- **AI-paired-development practitioners.** The repo is a worked case study in *porting* (vs re-deriving) textbook methods — see `RESEARCH_STANDARDS.md` §7a and the `MEMORY.md` triad on porting acceptance gates, symbol collisions, and the diagnostic loop trap.
- **Anyone who wants reproducible chemistry.** Calculations are URLs; results are shareable via auto-run links + IndexedDB history.

</td>
</tr>
</table>

<br/>

---

<h3 align="center">📸 &nbsp; See it</h3>

<table align="center">
<tr>
<td align="center" width="33%">
<a href="https://webgpu-q.vercel.app"><img src="./public/screenshots/landing.png" width="100%" alt="Landing page"/></a>
<br/><sub><b>landing</b> · what the engine is + run-anywhere CTAs</sub>
</td>
<td align="center" width="33%">
<a href="https://webgpu-q.vercel.app/molecule.html"><img src="./public/screenshots/hyperscope.png" width="100%" alt="Molecule SI report"/></a>
<br/><sub><b>/molecule.html</b> · H₂O SI report — properties, spectra, gradients</sub>
</td>
<td align="center" width="33%">
<a href="https://webgpu-q.vercel.app/experiments/"><img src="./public/screenshots/experiments.png" width="100%" alt="Experiment dashboard"/></a>
<br/><sub><b>/experiments/</b> · research dashboard (E1–E33, JSON artifacts)</sub>
</td>
</tr>
</table>

<br/>

---

<h3 align="center">📊 &nbsp; The numbers <sub><sup>(single source of truth — see bottom of file)</sup></sub></h3>

<div align="center">

<img src="./public/readme-numbers.svg" alt="Key numbers: 553 tests, 39.3× CCSD(T) on GPU, 10⁻⁵ Ha EOM-CCSD vs FCI, 7×10⁻¹⁴ Ha DF-HF, 1.35×10⁻¹¹ GPU↔CPU, 4.18× fusion, F ≥ 0.999999 statevector, N=128 MPS" width="100%"/>

</div>

<br/>

---

<h3 align="center">🧪 &nbsp; What's inside</h3>

<div align="center">

<img src="./public/readme-capabilities.svg" alt="Capability map: 7 modules — ground state, correlation, excited states, properties, geometry, density fitting, many-body simulation" width="100%"/>

</div>

<br/>

---

<h3 align="center">⚡ &nbsp; How fast — honestly, both directions</h3>

<p align="center"><sub>The headline 39.3× on CCSD(T) is real. So is the fact that PySCF/NumPy is 480× faster than us on CCSD at cc-pVDZ. Both numbers come from <a href="./experiments/results/2026-05-12/level-6/E34-comparison.md">the same comparison run</a>, against PySCF 2.13.0 on identical inputs.</sub></p>

<div align="center">

<img src="./public/readme-perf.svg" alt="Performance comparison vs PySCF — 198.6s CPU to 5.05s GPU on H₂O cc-pVDZ CCSD(T) (39.3× speedup), plus an honest two-column where-we-win / where-we-lose summary from E34" width="100%"/>

</div>

<sub align="center"><sub>↑ Single-run measurements (not warmup+trials harness). Energy agreement ≤ 10⁻⁴ Ha on all 19 comparable cells (well below chemical accuracy of 1.594 mHa). Where we win: no Python startup, HF up to medium systems, GPU CCSD(T) at cc-pVDZ. Where we lose: CPU MP2 / CCSD at production basis where NumPy / BLAS dominates. Full data: <a href="./experiments/results/2026-05-12/level-6/E34-comparison.md">E34-comparison.md</a>.</sub></sub>

<br/>

---

<h3 align="center">🆚 &nbsp; Completeness scorecard</h3>

<p align="center"><sub>Every method PySCF 2.13 / ORCA 6.1 / Psi4 1.10 ship, our status against it, and the roadmap tier for every gap.<br/>No "we don't do that" — every missing capability has a planned slot.</sub></p>

<div align="center">

<img src="./public/readme-matrix.svg" alt="Completeness scorecard: 50+ methods across 12 sections (mean field, correlation, multireference, excited states, properties, geometry, basis, solvent, acceleration, relativistic, periodic, platform) with shipped/Tier 3/Tier 4/out-of-scope status per row" width="100%"/>

</div>

<br/>

---

<h3 align="center">🔬 &nbsp; Validation matrix</h3>

<div align="center">

<img src="./public/readme-validation.svg" alt="Validation matrix: every layer cross-checked against PySCF, libxc, ITensor, brute-force projection, or experiment, with residuals disclosed" width="100%"/>

</div>

<br/>

---

<h3 align="center">🪜 &nbsp; The research ladder</h3>

<div align="center">

<img src="./public/readme-ladder.svg" alt="6-level research ladder: statevector, MPS/DMRG, kernel fusion (shipped foundation), WebRTC swarm, hardware verify (deferred), quantum chemistry (flagship)" width="100%"/>

</div>

<br/>

---

<h3 align="center">⏱️ &nbsp; 60-second demo</h3>

```bash
git clone https://github.com/abgnydn/webgpu-q && cd webgpu-q
npm install
npm run dev          # http://localhost:5175
                     # /molecule.html → H₂O SI report
                     # /experiments/  → research dashboard
```

```bash
npm run test         # vitest unit/integration · CI green
npm run typecheck    # tsc --noEmit, strict + noUncheckedIndexedAccess
npm run test:e2e     # Playwright · headless WebGPU Chromium
```

```ts
// Or one-call from molecule to a full property report
import { molecules, quickReport, uvVisSpectrum, toMoldenString } from "./src/chemistry";

const report  = quickReport(molecules.h2o, { addD2: true, addStaticAlpha: true });
const uvvis   = uvVisSpectrum(molecules.h2o, { method: "b3lyp5" });
const molden  = toMoldenString({ atoms: molecules.h2o, /* ... */ });
// → energy, dipole, charges, bond orders, NOON, ⟨S²⟩, multireference
//   verdict, static α, D2 dispersion, UV-vis spectrum + peaks, Molden file.
//   All in a Chrome tab.
```

```ts
// Or piece by piece if you want control
import { runRHFSCF, runMP2, runCCSD, runCCSDT_GPU, runEOMCCSD } from "./src/chemistry";

const hf      = runRHFSCF(integrals, nElectrons);
const mp2     = runMP2(hf, integrals);
const ccsd    = runCCSD(hf, integrals);
const t       = await runCCSDT_GPU(ccsd, hf, integrals, device);  // 39× on cc-pVDZ
const excited = runEOMCCSD(ccsd, integrals, hf);
```

<br/>

---

<h3 align="center">🧱 &nbsp; Architecture · URL → silicon</h3>

<div align="center">

<img src="./public/readme-architecture.svg" alt="8-layer architecture stack from URL through dashboard, research harness, chemistry modules, numerical core, WGSL shaders, WebGPU API, down to GPU silicon" width="100%"/>

</div>

<br/>

<table align="center">
<tr>
<td valign="top" width="50%">

**Research harness** · `experiments/lib/`

- `runner.ts` — `timedRun` with forced GPU sync (read-after-submit on a tiny buffer)
- `seeds.ts` — named deterministic seeds (no `Math.random()`)
- `env.ts` — captures adapter info, limits, SHA, UTC
- `fidelity.ts` — `F = |⟨ψ_ref|ψ_test⟩|²`, not max\|Δp\|
- `stats.ts` — median, p10/p90/p99, IQR

</td>
<td valign="top" width="50%">

**Discipline (non-negotiable)**

- 5 warmup + 20 trials per measurement
- Pass bar: `F ≥ 1 − 10⁻⁵` (f32 GPU paths)
- `std/median > 0.1` → `status: "noisy"`
- Honest negatives **committed** as JSON with diagnosis
- vitest + Playwright e2e · CI green · TS strict + `noUncheckedIndexedAccess`

</td>
</tr>
</table>

<br/>

---

<h3 align="center">📚 &nbsp; Method catalog</h3>

<details>
<summary><b>Ground-state electronic structure</b> · HF · UHF · DFT · MP2</summary>
<br/>

| method | notes |
|---|---|
| RHF SCF | DIIS, frozen-core, spherical-d, f/g/h, level-shift |
| UHF SCF | open-shell, stacked α+β DIIS, ⟨S²⟩ check, level-shift |
| RKS-DFT (LDA, GGA, hybrid) | LDA · BVWN5 · BLYP · B3VWN5 · B3LYP5; Becke grid, Lebedev |
| UKS-DFT (LDA, GGA, hybrid) | full functional ladder, spin-polarized XC kernel, ⟨S²⟩ |
| MP2 · DF-MP2 | spin-orbital + B-tensor reformulation |
| Cholesky DF (CD-DF) | rank-3 B-tensor, threshold-controlled |
| HF / DFT analytical ∇ | Pulay 1969, 8-fold ERI loop, Schwarz screening |

</details>

<details>
<summary><b>Correlation & excited states</b> · CCSD · CCSD(T) · EOM-CCSD · CIS · TDDFT</summary>
<br/>

| method | notes |
|---|---|
| CCSD (RHF) | Stanton-Bartlett, antisym spin-orbital + frozen-core |
| UCCSD (UHF) | shared `ccsdIterate` core, 3-block ERI |
| CCSD(T) CPU | per-triple, FCI-validated ≤ 0.25 mHa, frozen-core via Set |
| **CCSD(T) GPU** | **39.3× on H₂O cc-pVDZ**, f32→f64 reduce |
| UCCSD(T) | open-shell perturbative triples, frozen-core via Set |
| EE-EOM-CCSD | Stanton-Bartlett σ + stage-32c diagonal patch, Davidson |
| IP-EOM-CCSD | R₁ exact (brute-force); R₂ open |
| EA-EOM-CCSD | R₁ + R₂ patched to exact (stage 32e) |
| CIS · TDA · TDDFT (Casida) | full functional ladder, triplet via spin-pol, Davidson |
| Counterpoise / BSSE | HF / MP2 / CCSD / UHF / UCCSD / RKS / UKS + optional D2 add-on |
| Oscillator strengths | f = (2/3)·ω·|μ|², R₁·μ AO→MO transform |
| Spin classifier | singlet/triplet/spin-flip weight per root |

</details>

<details>
<summary><b>Properties & spectroscopy</b></summary>
<br/>

| property | notes |
|---|---|
| Dipole μ | AO→MO transform, RHF + post-HF densities |
| Polarizability α(0) | finite field + analytical CPHF (RHF + UHF) |
| Polarizability α(ω) | TDHF + TDDFT + open-shell UHF-TDHF response |
| Polarizability α(iω) | imaginary-axis response for Casimir-Polder |
| C₆ van-der-Waals coefficients | Casimir-Polder integral; HF/UHF/DFT references |
| Hyperpolarizability β | 3D finite-field stencil |
| Mulliken populations | charges + spin-density resolved (closed + open shell) |
| Wiberg / Mayer bond orders | + atomic valences, Lewis-multiplicity inference |
| Natural orbital occupations | NOON, multi-reference diagnostic |
| D2 dispersion correction | Grimme JCC 2006, energy + analytical gradient |
| Multireference verdict | T1/D1/⟨S²⟩/NOON aggregator with cutoff flags |
| TRK sum rule | oscillator-strength conservation check |
| Foster-Boys / Pipek-Mezey | orbital localization (Boys 1960 + PM 1989) |
| Energy decomposition | one-electron + Coulomb + exchange + V_nn breakdown |
| Coordination numbers | smooth Grimme D3 CN for chemistry-aware features |
| Coulomb matrix descriptor | permutation-invariant ML feature |
| Molecular formula / graph | Hill convention, adjacency, connected components |
| RMSD + Kabsch alignment | optimal rotation between geometries |
| Standard orientation | COM + principal-axes alignment |
| Rotational constants | A/B/C in cm⁻¹ and GHz |
| Multi-frame XYZ trajectory | for geom-opt / NEB visualization |
| Pre-built molecule library | h2o, ch4, nh3, beh2, hf, etc. |
| `quickReport(atoms)` | one-call full property report |
| `uvVisSpectrum(atoms)` | atoms → excitations + broadened spectrum + peaks |
| Harmonic ω | mass-weighted Hessian by finite diff |
| IR intensities | dμ/dQ along normal modes |
| Raman activities | Placzek invariants from α(Q) |
| Thermo (Sackur-Tetrode + RR + HO) | H₂O entropy 45.06 vs expt 45.1 |
| Koopmans / ΔSCF / EOM IPs | H₂O: 10.65 / 8.36 / **12.03** eV (expt 12.62) |
| Koopmans / EOM EAs | H₂O: −16.48 / **−16.37** eV |
| Molden orbital export | Cartesian Gaussian basis, Jmol/Avogadro/Multiwfn |
| Gaussian Cube export | density + MO isosurfaces, VMD/Jmol/Multiwfn |
| QCSchema JSON export | MolSSI standard, QCEngine/QCFractal/cclib consumable |
| XYZ format I/O | parse / emit standard geometry files |

</details>

<details>
<summary><b>Geometry & basis sets</b></summary>
<br/>

| feature | notes |
|---|---|
| BFGS geom-opt | analytical HF + DFT gradients |
| Lebedev angular grids | 2.6× point reduction at better accuracy |
| STO-3G | H, He, Li, Be, C, N, O, F (full first + second period) |
| 6-31G* | available |
| cc-pVDZ | H, He, Li, Be, C, N, O, F; CCSD(T) on H₂O in 5 s (GPU) |
| aug-cc-pVDZ | H, He, Li, Be, C, N, O, F (diffuse functions wired) |
| Spherical-d | sphd shell (Tier 1 bundle) |
| f / g / h orbitals | Cartesian integrals + transform |
| Schwarz integral screening | 8-fold canonical loop |

</details>

<details>
<summary><b>Many-body simulation</b> · statevector · MPS · DMRG · kernel fusion</summary>
<br/>

| level | notes |
|---|---|
| L1 statevector | f32 vec2 amplitudes, N/2 threads/gate |
| L1 controlled-U | N/4 threads, only control=1 touched |
| L2 MPS | canonical form, Jacobi complex SVD |
| L2 TEBD | `_canonicalizeBond(q)` invariant before two-site |
| DMRG | Lanczos + MPO, ITensor cross-checked N=8 |
| L3 fusion Tier B/C | 4.18× headline (Tier C, 8×8) |
| L3 fusion Tier D | documented honest negative (plateau) |
| Phase 6 GPU MPS | χ ≤ 64 |

</details>

<br/>

---

<h3 align="center">🔬 &nbsp; For researchers</h3>

<table align="center">
<tr>
<td valign="top" width="33%">

**📖 How to cite**

See [`CITATION.cff`](./CITATION.cff). For papers:

> Günaydın, A.B. (2026). _webgpu-q v0.9.1_. Zenodo. https://doi.org/10.5281/zenodo.20494383

Archived on Zenodo — DOI [10.5281/zenodo.20494383](https://doi.org/10.5281/zenodo.20494383) (concept DOI, resolves to the latest version).

</td>
<td valign="top" width="33%">

**⚠️ [Limitations](./LIMITATIONS.md)**

Honest, single-page list of what we **cannot** do, what is **untested**, and what is **known broken** — system size ceilings, browser/vendor matrix, SCF failure modes, missing output formats, precision disclosures.

</td>
<td valign="top" width="33%">

**📊 [Benchmarks queue](./BENCHMARKS.md)**

Standardized sets we've run vs. queued: GMTKN55, Thiel/QUEST, W4-11, S66, HEAT-345, SIE4x4, wall-clock vs PySCF / gpu4pyscf, cross-vendor parity.

</td>
</tr>
<tr>
<td valign="top">

**🛠️ [Contributing](./CONTRIBUTING.md) · 🔁 [Migration](./MIGRATION.md) · 📐 [Research standards](./RESEARCH_STANDARDS.md)**

15 canonical principles ([`RESEARCH_STANDARDS.md`](./RESEARCH_STANDARDS.md), mirrored in sibling [webgpu-dna](https://github.com/abgnydn/webgpu-dna)). Validation discipline (5w + 20t, fidelity pass bar, honest negatives committed). **Migration policy**: hand-write only the WebGPU/WGSL/browser layer; port chemistry methods (HF, CCSD, EOM-CCSD, DFT functionals…) from PySCF / libxc with attribution. See [`LICENSE-PYSCF`](./LICENSE-PYSCF) for the upstream license.

</td>
<td valign="top">

**📐 [Modern standards audit](./CLAUDE.md#modern-reference-standards-audited-2026-05)**

Every claim mapped to current literature — GMTKN55 functional rankings, EOM-CCSD literature accuracy bars, chemical accuracy bar (1 kcal/mol = 1.594 mHa), AFQMC beyond-CCSD(T), WebGPU subgroups status.

</td>
<td valign="top">

**🤝 [Code of conduct](./CODE_OF_CONDUCT.md)**

Contributor Covenant 2.1. Report concerns to [hi@barisgunaydin.com](mailto:hi@barisgunaydin.com).

</td>
</tr>
</table>

<br/>

---

<h3 align="center">🌐 &nbsp; Companion projects</h3>

<table align="center">
<tr>
<td align="left" width="50%">

- **[kernelfusion.dev](https://kernelfusion.dev)** — umbrella theory site, two preprints
- **[gpubench.dev](https://gpubench.dev)** — WebGPU bench harness, 592+ devices
- **[webgpudna.com](https://webgpudna.com)** — Geant4-DNA radiobiology port (sibling repo)

</td>
<td align="left" width="50%">

- **[zerotvm.com](https://zerotvm.com)** — Phi-3-mini in the browser, no compiler
- **[neuropulse.live](https://neuropulse.live)** — live 3.8B-param transformer visualization
- **[barisgunaydin.com](https://barisgunaydin.com)** — author site, project hub

</td>
</tr>
</table>

<br/>

---

<h3 align="center">📜 &nbsp; Key numbers — single source of truth</h3>

<details>
<summary>Click to expand · edit here when stages move forward</summary>
<br/>

> Anywhere a number appears above, it traces back to this table. Update the entry below, then rebuild the SVG hero (`public/readme-hero.svg`) if a top-line number changed.
>
> **How to read the precision numbers.** Anything tighter than 1.6 mHa (= 1 kcal/mol = chemical accuracy) is a *software regression assertion*, not a chemistry result. We diff GPU/CPU paths at 10⁻¹⁰ Ha to catch porting bugs, not because chemistry cares at that scale (basis-set incompleteness and functional choice dwarf any algorithmic difference by 6+ orders of magnitude). When comparing webgpu-q's chemistry numbers to PySCF or experiment, the only line that matters is the |ΔE| vs reference at ≥ 1 mHa.
>
> **How to read the speedup numbers.** `CCSD_T_SPEEDUP = 39.3×` is single-run on M2 Pro vs our own single-threaded CPU TypeScript. Not vs PySCF wall-clock. Not vs GPU4PySCF on CUDA. Not yet through the warmup+20-trials harness (gap #4 — open). The number gives an order of magnitude; the apples-to-apples vs production chemistry stacks is open work.

| symbol | value | context |
|---|---|---|
| `TESTS` | **553** | vitest unit + integration, all green |
| `CHEMISTRY_TESTS` | **437** | chemistry subset (1 skipped: opt-in cc-pVDZ CCSD(T)) |
| `E2E_SPECS` | **14** | Playwright headless WebGPU (CCSD(T), EOM, UV-vis, wallclock-vs-PySCF, levels 1/2/3/6, smoke tests) |
| `CCSD_T_SPEEDUP_MEDIAN` | **13.8×** | H₂O · cc-pVDZ · M2 Pro · 5 warmup + 20 trials · vs our CPU TypeScript · NOISY (std/median = 42%) |
| `CCSD_T_SPEEDUP_P10` | **28.4×** | best-case across the 20 trials (was historically reported as "39×" — that was a single lucky run) |
| `CCSD_T_SPEEDUP_P90` | **10.1×** | worst-case across the 20 trials |
| `CCSD_T_GPU_TIME_MEDIAN` | **8.4 s** | median per-call GPU time over 20 trials |
| `CCSD_T_CPU_TIME` | **116.4 s** | single CPU run on the same machine (CPU isn't run in trials — too slow) |
| `CCSD_T_GPU_DELTA` | **1.06×10⁻⁹ Ha** | regression assertion only — 6 orders below chemical accuracy (1.6 mHa); validates the GPU port matches CPU, not the method |
| `WIN_HF_H2_STO3G` | **105×** | E34 vs PySCF 2.13.0 · no-startup advantage |
| `WIN_CCSD_LIH_STO3G` | **40×** | E34 vs PySCF 2.13.0 · small-system advantage |
| `LOSS_CCSD_H2O_CCPVDZ` | **480× slower** | E34 vs PySCF 2.13.0 · BLAS gap (NumPy wins) |
| `LOSS_MP2_H2O_CCPVDZ` | **136× slower** | E34 vs PySCF 2.13.0 · BLAS gap |
| `E34_ENERGY_MAX_DELTA` | **1.0×10⁻⁴ Ha** | max &#124;ΔE&#124; vs PySCF over 19 cells · below chemical accuracy |
| `E34_ENERGY_MEAN_DELTA` | **8.1×10⁻⁶ Ha** | mean &#124;ΔE&#124; vs PySCF over 19 cells |
| `EOM_CCSD_PRECISION` | **10⁻⁵ Ha** | H₂ STO-3G · post-32c patch · 2-electron only |
| `EOM_CCSD_LIH_TRIPLET_GAP` | **7 meV** | E35 vs PySCF · 4-electron triplet · effectively exact |
| `EOM_CCSD_LIH_SINGLET_GAP` | **~0.27 eV** | E35 vs PySCF · post-32k sign-fix · within literature EOM-CCSD ↔ FCI bar (~0.1–0.2 eV) |
| `EOM_CCSD_H2O_SINGLET_GAP` | **~1.9 eV** | E35 vs PySCF · 10-e⁻ system · remaining missing T-dressings · PySCF port closes |
| `IP_EOM_H2O` | **12.03 eV** | expt 12.62 |
| `EA_EOM_H2O` | **−16.37 eV** | STO-3G (unbound) |
| `DF_HF_PRECISION` | **7×10⁻¹⁴ Ha** | DF-HF vs direct HF regression assertion (engineering, not chemistry-relevant) |
| `DF_MP2_PRECISION` | **0 Ha** | DF-MP2 vs direct MP2 regression assertion at τ=10⁻¹⁰ |
| `FUSION_HEADLINE` | **4.18×** | Tier C · 8×8 cascade |
| `STATEVECTOR_FIDELITY` | **F ≥ 0.999999** | f32 GPU vs f64 CPU |
| `MPS_N_MAX` | **128** | TFIM/Heisenberg, χ ≤ 32, browser |
| `MPS_CHI_MAX` | **64** | Phase 6 GPU MPS |
| `H2O_ENTROPY` | **45.06 cal/(mol·K)** | expt 45.1 |
| `STAGES_SHIPPED` | **through v0.8.0** | **WASM + aux-DF release**: 50× HF speedup on benzene cc-pVDZ (841 s → 16.8 s) via WASM hot-path stack — branch-free `prim_eri`, pair-table cache, r_aux buffer pooling, SIMD JK kernel, parallel WASM ERI build via Web Workers. **Auxiliary-basis density fitting** with pivoted Cholesky now works for arbitrary organic molecules at sub-mHa accuracy without external jkfit basis tables (49 μHa on benzene). `skipERI`/`skipOAO` integral flags eliminate ~100 s of dead work per HF on benzene. WGSL JK kernel shipped as research artifact (faster as isolated kernel, 3.3× slower in SCF due to f32 precision). Prior through v0.7.0: EE/IP-EOM-CCSD PySCF-ported, browser-platform stack (Web Workers, Pyodide REPL, drag-import, PWA + IndexedDB) |
| `LIVE_URL` | **webgpu-q.vercel.app** | production |

</details>

<br/>

---

<div align="center">

<sub>
MIT · Built with WebGPU, TypeScript strict, vitest, Playwright<br/>
Author <a href="https://github.com/abgnydn">@abgnydn</a> · <a href="mailto:hi@barisgunaydin.com">hi@barisgunaydin.com</a>
</sub>

<br/><br/>

<a href="https://webgpu-q.vercel.app">
<img src="https://img.shields.io/badge/%E2%96%B6%20OPEN%20IN%20BROWSER-webgpu--q.vercel.app-22d3ee?style=for-the-badge&labelColor=0b1224"/>
</a>

</div>
