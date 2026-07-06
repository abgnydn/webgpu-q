<div align="center">

<a href="https://webgpu-q.vercel.app"><img src="./public/demo.gif" alt="webgpu-q computing live — a wavefunction field and a real Hartree–Fock SCF converging to the ground state, in a browser tab" width="820"/></a>

<h1>webgpu-q</h1>

**Real quantum chemistry, in a browser tab.** Open a URL and run the actual thing — Hartree–Fock, DFT, MP2, CCSD, CCSD(T), EOM-CCSD — computed on your own machine, in the tab. No install. No backend. No CUDA. Every number cross-checked against PySCF.

<a href="https://webgpu-q.vercel.app"><img alt="Launch" src="https://img.shields.io/badge/%E2%96%B6%20LAUNCH-webgpu--q.vercel.app-22d3ee?style=for-the-badge&labelColor=0b1224"/></a>
&nbsp;
<a href="https://webgpu-q.vercel.app/screening.html"><img alt="Live screen" src="https://img.shields.io/badge/LIVE%20SCREEN-%2Fscreening.html-f472b6?style=for-the-badge&labelColor=0b1224"/></a>
&nbsp;
<a href="https://webgpu-q.vercel.app/experiments/"><img alt="Research dashboard" src="https://img.shields.io/badge/RESEARCH%20DASH-%2Fexperiments-34d399?style=for-the-badge&labelColor=0b1224"/></a>

<img alt="version" src="https://img.shields.io/badge/v0.12.0-0ea5e9?style=flat-square&labelColor=0b1224"/>
<img alt="license" src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square&labelColor=0b1224"/>
<img alt="tests" src="https://img.shields.io/badge/tests-CI%20green-22c55e?style=flat-square&labelColor=0b1224"/>
<img alt="webgpu" src="https://img.shields.io/badge/WebGPU-required-ff7849?style=flat-square&labelColor=0b1224"/>
<a href="https://doi.org/10.5281/zenodo.20494382"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20494382-3b82f6?style=flat-square&labelColor=0b1224"/></a>

</div>

<br/>

<table align="center" border="0">
<tr>
<td valign="top" width="33%">

**What this is**

- A browser-native electronic-structure engine. Methods ported from PySCF and brute-force verified to ≤ 1e-10 Ha element-wise.
- A WebGPU systems demo — a ~110-line WGSL kernel makes CCSD(T)/cc-pVDZ browser-feasible (~14× median over CPU TypeScript; noisy).
- A distributed-compute substrate — a browser-tab "swarm" that splits a Hartree–Fock build across tabs/machines (N=2 cross-machine HF verified, energy matches single-machine to 5.7e-14).
- A teaching / reproducibility platform — every calculation is a shareable URL; drag-import XYZ/PDB/MOL/SDF; in-browser "Compare to PySCF" REPL.

</td>
<td valign="top" width="33%">

**What this isn't**

- **Not a PySCF replacement.** PySCF is 10–100× faster, scales 100× larger, has 25 years of validation. If you do production chemistry, use it.
- **Not for big systems.** Benchmarked up to benzene/naphthalene HF and H₂O cc-pVDZ CCSD(T) — nothing at porphyrin / metal-complex / enzyme scale.
- **Not a CUDA competitor.** WebGPU has no f64 and is ~2–5× behind tuned CUDA today.
- **Not novel chemistry.** Every method was in PySCF first. The contribution is the *substrate*, not the algorithms.

</td>
<td valign="top" width="33%">

**Who it's for**

- **Educators** — every intermediate of the HF→MP2→CCSD pipeline is inspectable, in real time, with zero install.
- **WebGPU / systems researchers** — a real O(n⁷) kernel benchmarked end-to-end in the browser.
- **Reproducibility** — calculations are URLs; results shareable via auto-run links + IndexedDB history.

</td>
</tr>
</table>

<br/>

---

<h3 align="center">See it</h3>

<table align="center">
<tr>
<td align="center" width="33%">
<a href="https://webgpu-q.vercel.app/screening.html"><img src="./public/screenshots/screening.png" width="100%" alt="Live molecular screen"/></a>
<br/><sub><b>/screening.html</b> · rank a library by HOMO–LUMO gap, live HF per molecule</sub>
</td>
<td align="center" width="33%">
<a href="https://webgpu-q.vercel.app/molecule.html"><img src="./public/screenshots/molecule.png" width="100%" alt="Molecule SI report"/></a>
<br/><sub><b>/molecule.html</b> · full chemistry-paper SI for a small molecule</sub>
</td>
<td align="center" width="33%">
<a href="https://webgpu-q.vercel.app/experiments/"><img src="./public/screenshots/experiments.png" width="100%" alt="Research dashboard"/></a>
<br/><sub><b>/experiments/</b> · the research ladder, live runner + JSON artifacts</sub>
</td>
</tr>
</table>

<br/>

---

<h3 align="center">How fast — honestly, both directions</h3>

All vs **PySCF 2.13.0 on identical inputs** ([E34 run](./experiments/results/2026-05-12/level-6/E34-comparison.md)); energy agreement ≤ 10⁻⁴ Ha on all 19 comparable cells (well below the 1.594 mHa chemical-accuracy bar).

| | system | result |
|---|---|---|
| 🟢 **win** | HF · H₂ · STO-3G | **105×** faster (no Python startup) |
| 🟢 **win** | CCSD · LiH · STO-3G | **40×** faster (small-system advantage) |
| 🟢 **win** | CCSD(T) · H₂O · cc-pVDZ · GPU | **~14× median** vs our CPU TypeScript (39× best run; std/median ≈ 42%, *noisy* — [LIMITATIONS](./LIMITATIONS.md)) |
| 🔴 **loss** | CCSD · H₂O · cc-pVDZ | **480× slower** (NumPy/BLAS dominates) |
| 🔴 **loss** | MP2 · H₂O · cc-pVDZ | **136× slower** (BLAS gap) |

The honest takeaway: we win on no-startup + small systems; we lose badly at production basis where BLAS rules. **The CPU rows (HF/MP2/CCSD wins *and* losses) are real wall-clock vs PySCF 2.13.0 on identical inputs** — the E34 comparison, [reproduced 2026-07-06](./experiments/results/2026-07-06/level-6/E34-pyscf.json) with bit-identical energies. The **one** self-referential number is the **CCSD(T)-GPU ~14×**, which is vs our *own* CPU TypeScript — and here's the honest reason it *stays* that way: **there is no accessible GPU-CCSD(T) reference to compare against.** `gpu4pyscf` (the obvious GPU PySCF) supports SCF/DFT/gradients/Hessian/TDDFT and lists MP2/CCSD as *experimental* — **it has no GPU CCSD(T) at all** ([verified 2026-07-06](https://github.com/pyscf/gpu4pyscf)). A GPU-vs-GPU (T) comparison would need TeraChem (commercial) or a research CC-on-GPU code (ByteQC, Psi4 `gpu_dfcc`) — a different undertaking, tracked in [BENCHMARKS](./BENCHMARKS.md), not a quick `pip install`.

<br/>

---

<h3 align="center">Validated against ground truth</h3>

| layer | cross-checked against | residual |
|---|---|---|
| HF (RHF/UHF, DIIS, spherical-d) | PySCF | ≤ 0.5 mHa (≤ 0.1 mHa cc-pVDZ sphd) |
| MP2 · FCI | PySCF · analytic | ≤ 0.76 mHa (CH₄ FCI) |
| CCSD(T) | FCI | ≤ 0.25 mHa |
| EE / IP / EA-EOM-CCSD | explicit H̄ = e⁻ᵀHeᵀ projection | < 1e-10 Ha element-wise (EA: < 5e-13 on multi-electron LiH) |
| DF-HF · DF-MP2 | direct 4-index ERI | ~7×10⁻¹⁴ Ha (engineering assertion) |
| statevector (GPU f32) | f64 CPU reference | F ≥ 0.999999 |
| MPS / DMRG | ITensor · Pfeuty/Bethe | f64 at N=8 · 1/N limits |
| swarm HF (cross-machine) | single-machine | 5.7e-14 Ha |

> **Reading the precision numbers:** anything tighter than 1.6 mHa (= 1 kcal/mol) is a *software regression assertion* (does the GPU/DF port match the CPU/direct path?), **not** a chemistry result — basis-set incompleteness dwarfs it by 6+ orders. The EOM ports are faithful to PySCF's `eom_gccsd` / `eaccsd_matvec`, not re-derived; the brute-force diffs prove the implementation, not method accuracy on real systems.

<br/>

---

<h3 align="center">60-second demo</h3>

```bash
git clone https://github.com/abgnydn/webgpu-q && cd webgpu-q
npm install
npm run dev          # http://localhost:5175  ·  /screening.html  ·  /molecule.html  ·  /experiments/
npm run test         # vitest · CI green        npm run typecheck   # tsc strict
```

```ts
// one call: molecule → full property report (energy, dipole, charges, bond orders,
// NOON, ⟨S²⟩, multireference verdict, α, D2, UV-vis, Molden) — all in a tab
import { molecules, quickReport, uvVisSpectrum } from "./src/chemistry";
const report = quickReport(molecules.h2o, { addD2: true, addStaticAlpha: true });
const uvvis  = uvVisSpectrum(molecules.h2o, { method: "b3lyp5" });

// or piece by piece
import { runRHFSCF, runMP2, runCCSD, runCCSDT_GPU, runEOMCCSD } from "./src/chemistry";
const hf  = runRHFSCF(integrals, nElectrons);
const t   = await runCCSDT_GPU(runCCSD(hf, integrals), hf, integrals, device); // ~14× median (noisy)
const eom = runEOMCCSD(runCCSD(hf, integrals), integrals, hf);
```

<br/>

---

<h3 align="center">What's inside</h3>

**Ground state** HF · UHF · RKS/UKS-DFT (LDA/GGA/hybrid ladder) · MP2 · DF-MP2 · aux-basis f64 density fitting · analytical HF/DFT gradients · BFGS geom-opt
**Correlation & excited states** CCSD · UCCSD · CCSD(T) (CPU + WebGPU) · EE/IP/EA-EOM-CCSD · CIS · TDA · TDDFT (singlet + triplet) · oscillator strengths
**Properties** dipole · α(0)/α(ω)/α(iω) · C₆ · β · Mulliken · Wiberg/Mayer · NOON + multireference verdict · D2 dispersion · IR · Raman · UV-vis · thermochemistry · IPs/EAs
**I/O** Molden · Gaussian Cube · QCSchema · XYZ · drag-import PDB/MOL/SDF
**Many-body** GPU statevector · MPS/DMRG (ITensor-checked) · kernel fusion (4.22×)

<details>
<summary><b>Full method catalog</b> (click)</summary>
<br/>

| ground state | notes |
|---|---|
| RHF / UHF SCF | DIIS, frozen-core, spherical-d, f/g/h, level-shift, ⟨S²⟩ |
| RKS / UKS-DFT | LDA · BVWN5 · BLYP · B3VWN5 · B3LYP5; Becke + Lebedev grids |
| MP2 · DF-MP2 | spin-orbital + B-tensor; aux-basis f64 DF, no 4-index ERI |
| HF/DFT gradients | analytical Pulay 1969, Schwarz screening, BFGS / L-BFGS |

| correlation / excited | notes |
|---|---|
| CCSD · UCCSD | Stanton-Bartlett, antisymmetrized spin-orbital, frozen-core |
| CCSD(T) CPU + **GPU** | FCI-validated ≤ 0.25 mHa; GPU ~14× median (noisy), f32→f64 reduce |
| EE / IP / EA-EOM-CCSD | PySCF `eom_gccsd` / `eaccsd_matvec` ports; brute-force < 1e-10 Ha |
| CIS · TDA · TDDFT | Casida, full functional ladder, triplet via spin-pol, Davidson |

| properties | notes |
|---|---|
| α(0)/α(ω)/α(iω) · C₆ · β | CPHF + TDHF + TDDFT; Casimir-Polder; finite-field β |
| Mulliken · Wiberg/Mayer · NOON | charges/spin, bond orders, multireference verdict (T1/D1/⟨S²⟩) |
| IR · Raman · thermo | mass-weighted Hessian; Placzek; Sackur-Tetrode (H₂O S = 45.06 vs expt 45.1) |
| IPs / EAs | Koopmans / ΔSCF / EOM (H₂O EOM IP 12.03 eV vs expt 12.62) |

| basis / many-body | notes |
|---|---|
| STO-3G · 6-31G* · cc-pVDZ · aug-cc-pVDZ | first + second period; spherical-d; f/g/h |
| statevector · MPS · DMRG | GPU f32 (F ≥ 0.999999); Jacobi SVD + canonical TEBD; Lanczos + MPO |
| kernel fusion | 4.22× (Tier C, 8×8 cascade); Tier D 3.78× plateau = documented honest negative |

</details>

<br/>

---

<h3 align="center">Research discipline</h3>

<table align="center">
<tr>
<td valign="top" width="50%">

**Harness** · `experiments/lib/`

- `runner.ts` — `timedRun` with forced GPU sync (read-after-submit)
- `seeds.ts` — named deterministic seeds, no `Math.random()`
- `fidelity.ts` — `F = |⟨ψ_ref|ψ_test⟩|²`, not max\|Δp\|
- `env.ts` / `stats.ts` — adapter/SHA capture · median, p10/p90/p99

</td>
<td valign="top" width="50%">

**Non-negotiables**

- 5 warmup + 20 trials per measurement
- pass bar `F ≥ 1 − 10⁻⁵`; `std/median > 0.1` → `status: "noisy"`
- honest negatives **committed** as JSON with a diagnosis
- vitest + Playwright e2e · CI green · TS strict + `noUncheckedIndexedAccess`

</td>
</tr>
</table>

<br/>

---

<h3 align="center">For researchers</h3>

- **📖 Cite** — [`CITATION.cff`](./CITATION.cff). Concept DOI [10.5281/zenodo.20494382](https://doi.org/10.5281/zenodo.20494382) (resolves to latest; each release also gets a version DOI).
- **⚠️ [Limitations](./LIMITATIONS.md)** — what we *cannot* do, what's *untested*, what's *known broken*: size ceilings, vendor matrix, SCF failure modes, precision disclosures.
- **📊 [Benchmarks queue](./BENCHMARKS.md)** — run vs queued: GMTKN55, QUEST, W4-11, S66, wall-clock vs PySCF/gpu4pyscf.
- **🛠️ [Contributing](./CONTRIBUTING.md) · 🔁 [Migration](./MIGRATION.md) · 📐 [Research standards](./RESEARCH_STANDARDS.md)** — port chemistry from PySCF/libxc with attribution ([`LICENSE-PYSCF`](./LICENSE-PYSCF)); hand-write only the WebGPU/WGSL/browser layer.
- **🤝 [Code of conduct](./CODE_OF_CONDUCT.md)** — Contributor Covenant 2.1.

<details>
<summary><b>Key numbers — single source of truth</b> (click)</summary>
<br/>

| symbol | value | context |
|---|---|---|
| `CCSD_T_SPEEDUP_MEDIAN` | **13.8×** | H₂O cc-pVDZ · 5w+20t · vs our CPU TS · NOISY (std/median 42%) |
| `CCSD_T_SPEEDUP_P10 / P90` | **28.4× / 10.1×** | best / worst across 20 trials (39× once-headlined was a single lucky run, retired) |
| `CCSD_T_GPU / CPU_TIME` | **8.4 s / 116.4 s** | median GPU per-call vs single CPU run, M2 Pro |
| `WIN_HF_H2_STO3G` | **105×** | E34 vs PySCF 2.13.0 (no-startup) |
| `WIN_CCSD_LIH_STO3G` | **40×** | E34 vs PySCF 2.13.0 (small system) |
| `LOSS_CCSD_H2O_CCPVDZ` | **480× slower** | E34 vs PySCF 2.13.0 (BLAS gap) |
| `LOSS_MP2_H2O_CCPVDZ` | **136× slower** | E34 vs PySCF 2.13.0 (BLAS gap) |
| `E34_ENERGY_MAX_DELTA` | **1.0×10⁻⁴ Ha** | max \|ΔE\| vs PySCF over 19 cells (sub-chemical) |
| `EOM_BRUTEFORCE_DIFF_LIH` | **< 1e-10 Ha** | EE/IP σ vs explicit H̄, element-wise; PySCF-ported, no patches |
| `EA_EOM_BRUTEFORCE_DIFF_LIH` | **< 5e-13 Ha** | EA-EOM σ vs exact H̄ (multi-electron LiH); `eaccsd_matvec` port |
| `DF_HF_PRECISION` | **7×10⁻¹⁴ Ha** | DF-HF vs direct (engineering assertion) |
| `FUSION_HEADLINE` | **4.22×** | Tier C · 8×8 cascade |
| `STATEVECTOR_FIDELITY` | **F ≥ 0.999999** | f32 GPU vs f64 CPU |
| `MPS_N_MAX / CHI_MAX` | **128 / 64** | TFIM/Heisenberg in browser · Phase 6 GPU MPS |
| `H2O_ENTROPY` | **45.06 cal/(mol·K)** | vs expt 45.1 |
| `STAGES_SHIPPED` | **v0.12.0** | full site/landing redesign (live wavefunction hero, cohesive spectral system across all pages); both manuscripts corrected + every citation verified |

</details>

<br/>

---

<h3 align="center">Companion projects</h3>

[kernelfusion.dev](https://kernelfusion.dev) (umbrella · kernel fusion) ·
[gpubench.dev](https://gpubench.dev) (WebGPU bench, 592+ devices) ·
[webgpudna.com](https://webgpudna.com) (Geant4-DNA in the browser) ·
[zerotvm.com](https://zerotvm.com) (Phi-3 in the browser) ·
[neuropulse.live](https://neuropulse.live) (live transformer activations) ·
[barisgunaydin.com](https://barisgunaydin.com) (author)

<br/>

<div align="center">
<sub>MIT · WebGPU · TypeScript strict · vitest · Playwright · by <a href="https://github.com/abgnydn">@abgnydn</a> · <a href="mailto:hi@barisgunaydin.com">hi@barisgunaydin.com</a></sub>
<br/><br/>
<a href="https://webgpu-q.vercel.app"><img src="https://img.shields.io/badge/%E2%96%B6%20OPEN%20IN%20BROWSER-webgpu--q.vercel.app-22d3ee?style=for-the-badge&labelColor=0b1224"/></a>
</div>
