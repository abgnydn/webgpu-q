# Limitations

What webgpu-q **cannot** do, what it has **not yet been tested on**,
and what is **known broken** — all in one place. Updated 2026-05.

Honesty is more credible than completeness. We list every limitation a
reviewer or chemist would discover anyway.

---

## 1. System size ceilings

### Tested and confirmed working

| basis | system | NSO | wall time | notes |
|---|---|---|---|---|
| STO-3G | H₂, LiH, BeH₂, H₂O, CH₄ | ≤ 18 | sub-second | full pipeline incl. CCSD(T) + EOM |
| 6-31G* | H₂O, BeH₂ | ≤ 28 | seconds | spot-checked |
| cc-pVDZ | H₂O | 48 | 5.05 s (GPU CCSD(T)) | headline case |
| aug-cc-pVDZ | H₂O | 64 | minutes | diffuse functions wired |

### Tested and confirmed working but unbenchmarked

- cc-pVDZ CCSD(T) on CH₄, BeH₂, LiH — methods work; not in CI on every commit.
- STO-3G EOM-CCSD on systems up to ~12 occupied SOs — algorithm scales as
  O(N⁶) and gets slow above that.

### Not tested

- **cc-pVTZ on anything** — would push past ~ 100 SOs; no measurement.
- **Benzene / pyridine / formaldehyde** at any basis above STO-3G.
- **Transition metals** of any kind (no ECP / pseudopotential support yet).
- **Heavy elements (Z > 18)** — no scalar-relativistic correction.

### Hard ceilings

- **WebGPU buffer max ~4 GB per buffer** on Chrome / desktop, less on
  Firefox. A naive (i,j,k,a,b,c)-laid-out f32 partial-sum buffer for
  CCSD(T) at ~100 SOs is already 100⁶ · 4 = a lot — we'd need to tile,
  which is unimplemented.
- **Single-tab heap ceiling** of ~4 GB on Chrome desktop. JavaScript-side
  arrays for ERIs at large basis hit this before WebGPU does.

### Basis-set atom coverage

| basis | atoms wired |
|---|---|
| STO-3G | H, Li, Be, C, N, O, F |
| 6-31G* | H, C, N, O (spot-checked) |
| **cc-pVDZ** | **H, Li, Be, C, N, O, F** (full first-row coverage — Tier 3 shipped 2026-05) |
| aug-cc-pVDZ | **H, Li, Be, C, N, O, F** (diffuse tables for full first row wired 2026-05) |

LiH / BeH₂ / CH₄ / NH₃ / HF now first-class cc-pVDZ targets.
Verified by `tests/chemistry/ccpvdz-firstrow.test.ts` — each
molecule's cc-pVDZ HF energy converges and lies variationally
below its STO-3G counterpart. Tolerance is ~10 mHa vs PySCF 2.13.0
reference values (loose, because the test is verifying basis
wiring not SCF precision; the existing H₂O cc-pVDZ tests cover
the precision case to 35 µHa).

Aug-cc-pVDZ diffuse functions for Li, Be, C, N, F → ~30 minutes
each from the same EMSL source, queued as a follow-up if needed
for anions or excited-state work on those systems.

---

## 2. Browser / GPU vendor matrix

| browser | GPU | status |
|---|---|---|
| Chromium / Chrome | Apple M2 Pro | **primary test target** · everything works |
| Chromium / Chrome | NVIDIA discrete | untested |
| Chromium / Chrome | AMD discrete | untested |
| Chromium / Chrome | Intel iGPU | untested |
| Firefox (Nightly WebGPU) | M2 Pro | untested |
| Safari (Tech Preview WebGPU) | M2 Pro | untested |
| Edge | any | likely OK (Chromium engine) |
| Mobile (any) | any | unlikely to work — buffer limits + power |

We **assume** WebGPU 1.0 conformance and the published `requiredLimits`.
Adapters that don't report the expected `maxStorageBufferBindingSize`
will silently truncate large dispatches.

---

## 3. Known SCF / CC failure modes

- **HF doesn't converge** — we return `converged: false` and stop. No
  level-shift, no damping, no quadratically-convergent second-order SCF
  fallback. Stretched bonds and near-degenerate HOMO-LUMO gaps fail.
- **CCSD T-amplitude divergence** — for systems with multi-reference
  character (broken bonds, biradicals), the spin-orbital CCSD residual
  doesn't converge. We catch and report; no DIIS for amplitudes (only
  for the Fock matrix in SCF).
- **EOM-CCSD with R₂ dominant states** — algorithm runs and returns
  energies, but the literature accuracy for doubly-excited states is
  ~1 eV vs FCI (vs ~0.1–0.2 eV for singly-excited). Don't trust
  shake-up / two-electron states from this implementation.
- **DFT grid quadrature instability** — Becke-partitioned grid weights
  derivatives have a ~10⁻³ Ha/Bohr translational-invariance residual.
  Documented in CLAUDE.md. Don't trust DFT gradients beyond that scale.
- **TDA-DFT / DFT-gradient with spherical-d** — refuses with a clear
  error today; proper fix is Cartesian → spherical transform on grid
  values. Documented.

---

## 4. Missing features that researchers expect

| missing | impact | roadmap |
|---|---|---|
| Counterpoise / BSSE (HF + MP2 + CCSD + UHF + UCCSD) | **Shipped 2026-05**. `runCounterpoise(atoms, fragments, basis, hfOpts, method, ccsdOpts, uhfOpts)` in `src/chemistry/counterpoise.ts`. Boys-Bernardi: per-fragment energy in the full dimer basis via ghost atoms (`ghost?: boolean` flag on `Atom` — basis present, Z=0, no electrons). `method` selects HF / MP2 / CCSD / UHF / UCCSD; the ghost-atom plumbing passes through to the post-HF / post-UHF correlation step unchanged. For open-shell methods, each `CounterpoiseFragment` carries an explicit `spin: { nAlpha, nBeta }`; supermolecule spin is the sum. Returns supermolecule energy, per-fragment energies in dimer + bare basis, uncorrected + CP interaction energies, and `bsseCorrection = ΔE_CP − ΔE ≥ 0`. Verified by 11 tests in `counterpoise.test.ts` (7 closed-shell on H₂...H₂; 4 open-shell on H↑+H↑ at 5 Å, Li doublet + H↑, H₂ singlet + H↑ via UCCSD, and the missing-spin error path). | shipped ✓ |
| Frozen-core in CCSD(T) | **Audited 2026-05** — CCSD(T) frozen-1s on H₂O / CH₄ lies above all-electron by < 30 mHa, (T) stays negative. Verified by `tests/chemistry/frozen-core-audit.test.ts`. | shipped ✓ |
| Frozen-core in UCCSD(T) | **Shipped 2026-05** — `runUCCSDT(uccsd, uhf, integrals, { nFrozenCore })`. `ccsdtFromSO` refactored to accept `frozenOccSO: ReadonlySet<number>` (non-contiguous). RHF interleaved → contiguous {0..2·nFC−1}; UCCSD all-α-first → {0..nFC−1} ∪ {nα..nα+nFC−1}. Closed-shell H₂O frozen-1s: UCCSD(T) matches RHF-CCSD(T) frozen-1s to 1e-7. Skips frozen i/j/k outer loops → 2-3× speedup vs the previous `nFrozenSO=0` workaround (which relied on T1/T2 being zero at those positions). | shipped ✓ |
| Frozen-core in UCCSD | **Audited + fixed 2026-05**. Audit found UCCSD's frozen-core was freezing α-occupied SOs [0, nFrozenSO) instead of α + β of the lowest k spatials (SO-ordering mismatch with ccsdIterate's contiguous-frozen contract). Fix: ccsdIterate now accepts `ReadonlySet<number>` of frozen SO indices; UCCSD constructs the correct interleaved (α-spatial-s, β-spatial-s) set. Verified by `frozen-core-audit.test.ts` — UCCSD frozen-core on closed-shell H₂O now matches RHF-CCSD frozen-core to 1e-6 Ha. | shipped ✓ |
| Frozen-core in EE-EOM-CCSD | **Audited + shipped 2026-05**. `runEOMCCSD` accepts `nFrozenCore`. Implementation: packed (singles + antisym doubles) basis restricted to occupied indices ≥ 2·nFrozenCore; σ-equation internals unchanged (R_1 / R_2 are zero at frozen indices, so frozen-index contributions to the inner sums vanish automatically). Verified by `frozen-core-audit.test.ts` — H₂O STO-3G frozen-1s lowest 3 EOM excitations are real-positive, ordered, and shift by < 100 mHa from all-electron. Dense + Davidson paths both honor frozen-core. | shipped ✓ |
| Frozen-core in IP-EOM / EA-EOM-CCSD | **Audited + shipped 2026-05**. Mirror of the EE-EOM-CCSD pattern: packed (1h + antisym 2h1p) basis in IP-EOM and (1p + antisym 1h2p) basis in EA-EOM restricted to occupied indices ≥ 2·nFrozenCore. σ-equations unchanged (R_1/R_2 zero at frozen indices). H₂O frozen-1s lowest IP / EA shift by < 100 mHa from all-electron and remain real-positive (IPs) / real (EAs). Verified by `ip-ea-eom-extensions.test.ts`. | shipped ✓ |
| Davidson eigensolver in IP-EOM / EA-EOM-CCSD | **Shipped 2026-05**. `useDavidson: true` option on both `runIPEOMCCSD` and `runEAEOMCCSD`. Davidson + dense agree on k=3 lowest IPs/EAs to ≤ 1e-4 Ha on H₂O STO-3G. Same diagonal preconditioner pattern as EE-EOM: −ε_i (IP R_1), −ε_i − ε_j + ε_a (IP R_2), ε_a (EA R_1), −ε_i + ε_a + ε_b (EA R_2). | shipped ✓ |
| ⟨S²⟩ post-UCCSD spin diagnostics | **Shipped 2026-05**. `UCCSDResult` exposes `s2Reference` (passed through from UHF), `s2T2Correction` (Chen-Schlegel 1994 first-order T2 correction: `−Σ_{i_α j_β a_α b_β} |T2|²`), and `s2Approx = s2Reference + s2T2Correction`. Verified on H₂ singlet, Li doublet, H₂O singlet. Truncated-Hylleraas approximation; full UCCSD ⟨S²⟩ (T1·T2 + T2² + higher terms) is a follow-up if needed for publishable values. | shipped ✓ (diagnostic-grade) |
| CCSD T1 + D1 diagnostics | **Shipped 2026-05**. CCSDResult exposes `t1Diagnostic` (Lee-Taylor 1989: ‖T1‖_F/√N_e; multi-reference bar ~0.02) and `d1Diagnostic` (Janssen-Nielsen 1998: σ_max(T1) via power iteration; bar ~0.05). Diagnostic flags for whether CCSD is reliable on the input system. Verified by `ccsd-diagnostics.test.ts`. | shipped ✓ |
| Analytical static polarizability (CPHF) | **Shipped 2026-05**. `cphfPolarizability(hf, integrals, shells)` in `src/chemistry/cphf.ts`. Closed-shell RHF only. Builds the (A+B) orbital Hessian on the OV space, solves 3 dense linear systems (one per Cartesian) via Gauss-Jordan, contracts with dipole-OV matrix elements: `α_μν = 4 · Σ_ai X^μ_ai · F^ν_ai`. Verified against the finite-field implementation on H₂O HF/STO-3G to < 1% on the isotropic α and < 5% per diagonal. NMR shielding via CPHF on magnetic perturbation is also a mechanical extension. | shipped ✓ |
| Frequency-dependent polarizability α(ω) (TDHF / RPA) | **Shipped 2026-05**. `tdhfPolarizability(hf, integrals, shells, ω)` in `src/chemistry/tdhf.ts`. Closed-shell RHF only. Solves the symmetric RPA response equation `[(A−B)(A+B) − ω²·I] X = (A−B)·F` and contracts `α(ω)_μν = 4·X^μ·F^ν`. At ω → 0 reproduces `cphfPolarizability` to 1e-7. Monotonic dispersion below the first pole verified on H₂O STO-3G. | shipped ✓ |
| Open-shell TDHF α(ω) for radicals | **Shipped 2026-05**. `uhfTdhfPolarizability(uhf, integrals, shells, ω)` and `uhfTdhfPolarizabilityImag(...)` in `src/chemistry/uhf-tdhf.ts`. UHF reference, 4-spin-block (A±B) on the combined (α-OV + β-OV) space. Refactored `uhf-cphf.ts` to expose `buildUHFCPHFHessians` (cross-spin block of A−B = 0 since cross-spin A = B; same-spin block carries exchange). Validation: static limit matches `uhfCphfPolarizability` to 1e-7; closed-shell limit (n_α=n_β H₂O) matches RHF TDHF α(ω) to 1e-7 at ω = 0.1 Ha; α(iω) for Li doublet cc-pVDZ decreases monotonically. | shipped ✓ |
| TDDFT α(ω) frequency response (closed-shell DFT) | **Shipped 2026-05**. `tddftPolarizability(integrals, hf, shells, opts, ω)` in `src/chemistry/tddft-response.ts`. Closed-shell RKS reference; reuses the exported `buildTDABlocks` from `tda-dft.ts` to get A and B (including the XC kernel) for any supported functional (hf, lda-svwn, bvwn5, blyp, b3vwn5, b3lyp5). Solves the same symmetric RPA response `[(A−B)(A+B) − ω²·I] X = (A−B)·F` and contracts `α(ω)_μν = 4·X·F`. Validation: `method="hf"` reproduces `tdhfPolarizability` to ≤ 1e-7; B3LYP5 static α(0) on H₂O STO-3G positive + symmetric + in physical range; B3LYP5 dispersion monotonic below the first TDDFT singlet pole. DFT-based dynamic α(ω) is the de facto standard for molecular property work — typical errors on α(0) drop from ~20% (HF) to ~5% (B3LYP) vs CCSD/cc-pVTZ. | shipped ✓ |
| UKS-CPHF static α with XC kernel | **Shipped 2026-05 (LSDA only)** — `uksCphfPolarizability(uks, integrals, shells, opts)` in `src/chemistry/uks-cphf.ts`. Builds the UKS 4-spin-block (A+B) on (α-OV + β-OV) with Coulomb 2(ai\|bj)_σσ' + LSDA XC kernel `2·⟨ai\|f^σσ'\|bj⟩` (factor 2 from the orbital-rotation density-response convention that also gives the 2(ai\|bj) Coulomb factor). XC kernel f^σσ' computed via central FD on `evalXC_LSDA` (new `functional-spin-kernel.ts` module). Solves 3-RHS Gauss-Jordan. Closed-shell H₂O UKS-LSDA static α matches RKS-TDDFT@ω=0 to ≤ 1e-3 per tensor element / 1e-4 isotropic. Li doublet cc-pVDZ produces real, symmetric, positive α. GGA + hybrid refused this stage (needs spin-polarized GGA kernel + scaled HF exchange in (A+B)). | shipped ✓ (LSDA) |
| UKS-TDDFT α(ω) frequency response | **Shipped 2026-05 (LSDA only)** — `uksTddftPolarizability(uks, integrals, shells, ω, opts)` and `uksTddftPolarizabilityImag(...)` in `src/chemistry/uks-tdhf.ts`. Closes the {RHF, UHF, RKS, UKS} × {static α, α(ω), α(iω), C₆} matrix (12/12 cells). For UKS-LSDA the (A−B) matrix simplifies to a diagonal of orbital-energy gaps (Coulomb cancels in A−B; LSDA kernel is symmetric so cancels too; no HF exchange at hfMix=0), making the response solve well-conditioned. Validated: static limit ω=0 matches `uksCphfPolarizability` to 1e-7; closed-shell H₂O UKS-LSDA @ ω=0.1 matches RKS-TDDFT @ ω=0.1 to 1e-3; Li doublet α(iω) monotonically decreasing. | shipped ✓ (LSDA) |
| C₆ van-der-Waals dispersion coefficients | **Shipped 2026-05**. `c6Coefficient(hfA, ..., hfB, ...)` (closed-shell convenience) and `c6CoefficientGeneral(srcA, srcB, opts?)` (any combination of RHF / UHF / RKS-DFT / UKS-DFT references via an `AlphaImagSource` discriminated union with kinds "rhf" / "uhf" / "rks" / "uks"). All four α(iω) routines wired in: `tdhfPolarizabilityImag`, `uhfTdhfPolarizabilityImag`, `tddftPolarizabilityImag`, `uksTddftPolarizabilityImag`. Casimir-Polder integral via Gauss-Legendre quadrature (default 16 points, Golub-Welsch nodes); converges to ≤ 2% from N=8 → N=32. Symmetric C₆(A,B) = C₆(B,A) to 1e-10. All four source kinds round-trip cleanly: closed-shell-via-UHF (n_α=n_β) matches RHF to 1e-6; DFT route same order of magnitude as HF on H₂ STO-3G; Li doublet self-C₆ finite + positive via UHF, RKS, and UKS routes. Absolute magnitudes need aug-cc-pVDZ. **The {reference} × {static α, α(ω), α(iω), C₆} 4×4 = 16-cell matrix is now 16/16 closed** (counting UKS LSDA-only with hybrid/GGA queued). | shipped ✓ |
| Level-shift / damping | **Shipped 2026-05** for both RHF (`hf-scf.ts` `HFOpts.levelShift`) and RKS DFT (`rks-scf.ts` `RKSOpts.levelShift`). Lifts virtual KS / Fock eigenvalues by `levelShift · Σ_{p≥nOcc} c'_p ⊗ c'_p^T` in the orthogonal basis each iteration; strips the shift from reported virtual `orbitalEnergies` on return so Koopmans IPs / EAs stay physical. Verified by `dft-level-shift.test.ts`: H₂O/B3LYP5 and BeH₂/LDA converge to the same energy ±1e-7 vs no-shift. Useful for stretched-bond / near-degenerate SCF cases where the unshifted HOMO-LUMO gap is too small. | shipped ✓ |
| Open-shell DFT (UKS) SCF | **Shipped 2026-05** — `runUKSDFT(integrals, nAlpha, nBeta, nucleiSymbols, opts)` in `src/chemistry/dft/uks-scf.ts`. Mirror of `rks-scf.ts` on spin-resolved densities D_α / D_β with separate Fock matrices F_σ = h + J(D_total) + V_xc^σ − hfMix · K(D_σ) (no 0.5 factor — matches uhf-scf hybrid convention; differs from the 0.5·K(D_total) factor that's correct for RKS where D_RKS = 2·D_α). Uses `evalXC_LSDA` (LDA) or `evalXC_GGA_spin` (GGA + hybrid) from functional-spin.ts. Stacked-spin DIIS, symmetry-breaking initial guess (0.01 Ha default), per-spin level shift. ⟨S²⟩ via Mayer formula (same as UHF). Validated across the full functional ladder: closed-shell H₂O matches RKS to ≤ 1e-5 Ha for lda-svwn / bvwn5 / blyp / b3vwn5 / b3lyp5 (all within SCF convergence noise of each other). H atom doublet ⟨S²⟩ = 0.75 exactly; Li doublet cc-pVDZ converges with non-zero spin density; Li BLYP converges. UKS-TDDFT response α(ω) is a downstream follow-up to close the polarizability matrix. | shipped ✓ |
| Quadratically-convergent SCF | Hard cases bail out | Tier 3 |
| Molden / HDF5 / Cube output | **Shipped 2026-05 (Molden + Cube)** — `toMoldenString({...})` in `src/chemistry/molden.ts` (Cartesian-Gaussian basis only) and `densityCube` / `moCube` in `src/chemistry/cube.ts`. Cube sampling is on an orthorhombic grid built around the molecular bounding box (default 4-bohr padding, 0.3-bohr step), emitted in Gaussian98 / Gaussian03 standard format. `densityCube(shells, D, atoms)` for total electron density; `moCube(shells, C_MO, orbitalIdx, atoms)` for a single MO amplitude (signed, viewers render |φ|²). Validated: H₂ HF/STO-3G density Cube integrates to ≈ 2 e⁻ (1.5–2.3 a.u. window for the test grid); H₂ HOMO σ_g MO Cube is positive-only with ∫|φ|² ≈ 1. Unblocks isosurface plotting in Jmol / VMD / Avogadro / Multiwfn. HDF5 still queued. | shipped ✓ (Molden + Cube) |
| Natural orbital occupation numbers (NOON) | **Shipped 2026-05** — `naturalOrbitalOccupations(D, S, opts?)` in `src/chemistry/natural-orbitals.ts`. Diagonalizes D in the Löwdin-orthogonal metric (D̃ = S^(1/2)·D·S^(1/2)) and returns occupation numbers (sorted descending, 0 ≤ n_p ≤ 2), natural-orbital coefficients in the AO basis, and "strongly occupied" / "active" counts (thresholds 1.95 / 0.05 by default). Validated on closed-shell H₂O HF (NOONs = {2,2,2,2,2,0,0} ± 1e-10, ΣN = 10) and Be UHF (NOONs = {2,2,0,0,0} for no-symm-break; broken-symmetry tested too — Σ N = N_electrons preserved). Useful multi-reference diagnostic on UHF/UKS spin-contaminated systems or correlated 1-PDMs (MP2/CCSD relaxed densities — those require relaxed-density build first, queued). | shipped ✓ |
| ccData / QC-Schema compatibility | **Shipped 2026-05 (QCSchema)** — `toQCSchemaClosedShell` and `toQCSchemaOpenShell` in `src/chemistry/qcschema.ts`. Emit a `qcschema_output` v1 AtomicResult JSON with the standard `molecule` / `model` / `properties` / `return_result` / `success` / `extras` layout. Properties block includes `scf_total_energy`, `nuclear_repulsion_energy`, `scf_correlation_energy` (+ method-tagged `mp2_correlation_energy` / `ccsd_correlation_energy` when applicable), `calcinfo_*` (natom / nbasis / nmo / nalpha / nbeta), `molecular_multiplicity` (open-shell), `scf_s_squared` (open-shell). Validated by 4/4 tests covering closed-shell HF / CCSD method tag / open-shell H atom UHF / extras preservation. Consumable by QCEngine, QCFractal, QCArchive, cclib, ASE workflows. cclib parser support follows separately. | shipped ✓ (QCSchema) |
| FAIR / Zenodo DOIs per release | Citations point at GitHub tag, not DOI | Tier 3 — set up CI workflow |
| Aux-basis density fitting | We have CD-DF but not JKFIT / RIFIT integral path | Tier 3 — needs 3-index ERI routine |
| DFT dispersion correction | **Shipped 2026-05 (D2)** — `dispersionD2(atoms, opts)` in `src/chemistry/dispersion-d2.ts`. Grimme JCC 2006 atomic-pairwise C6/R⁶ with Fermi damping and functional-specific s6 (BLYP=1.20, B3LYP5=1.05, B3VWN5=1.05, BVWN5=1.20, LDA=1.05 default). Tabulated C6 and R_R for H, He, Li, Be, C, N, O, F. Atomic-units throughout. O(n_atoms²) cost — microseconds for typical molecules. Validated on He-He, H₂-H₂ pair, C-C pair, CH₄, ghost-atom cases. 8/8 tests green. D3 (Grimme 2010, atom-environment-dependent C8/C9) is a follow-up. Gradients ∂E_disp/∂R_A are mechanical extension; queued. | shipped ✓ (D2) |
| Davidson / Krylov eigensolver | **Shipped 2026-05**. Block Davidson (`src/manybody/davidson.ts`) wired into EE/IP/EA-EOM-CCSD (commits 6b9a96f, a35baeb) and now also CIS / TDA (this commit) behind `useDavidson: true`. Dense + Davidson agree on k roots to ≤ 1e-6 Ha across all paths. Unblocks large-basis CIS/TDDFT + bigger EOM-CCSD systems. | shipped ✓ |
| Multi-node parallel | One tab, one GPU | Tier 4 — substrate is Phase D WebRTC |
| Periodic boundary conditions | No solids, no surfaces | Tier 4 |
| Spin-orbit coupling / X2C / DKH | No heavy elements | Tier 4 |
| QM/MM | No biomolecules | Tier 4 |
| Anharmonic VPT2 | Harmonic only | Tier 4 |

---

## 5. Honest precision disclosures (carried from CLAUDE.md)

- **CCSD(T) GPU 39.3×** is a **single-run measurement** on M2 Pro. Not
  through the warmup+20-trials harness. The correctness (|Δ| = 2.4×10⁻¹⁰
  Ha) is reproducible; the specific 39.3× number is ±20% on different
  hardware and ±10% run-to-run.
- **EOM-CCSD ≡ FCI at 10⁻⁵ Ha** is **algorithmic precision on H₂
  STO-3G only**, where T̂² = 0 makes EOM-CCSD = FCI by construction
  (2-electron limit). E35 cross-validation against PySCF EOM-CCSD
  on LiH / BeH₂ / H₂O / NH₃ / CH₄ STO-3G is more nuanced than the
  first cut suggested. The gap is **not uniform across spin sectors**:
  - **Triplet excitations agree well**: LiH lowest triplet matches
    PySCF to **7 meV**, BeH₂ degenerate triplet matches to 1.3 meV.
    H₂O / NH₃ / CH₄ triplets show ~0.5–1.0 eV gap (worsening with
    system size).
  - **Singlet excitations show a consistent ~2–3 eV gap** across
    LiH, BeH₂, H₂O, NH₃, CH₄.
  - **HF + CCSD energies agree to 10⁻⁷ Ha** throughout.
  This pattern (triplets mostly correct, singlets systematically off)
  is now **isolated to one missing term in σ_1**. The LiH brute-force
  diagnostic (`tests/chemistry/eom-ccsd-bruteforce-lih.test.ts`)
  builds H̄ = e⁻ᵀ̂ H eᵀ̂ explicitly in the 64-state 4-electron Fock
  space, projects onto the (R_1 + antisym R_2) basis, and diagonalizes.
  CCSD energy matches FCI exactly on LiH STO-3G (CCSD = FCI for this
  system). The exact M_exact projection matches PySCF EOM-CCSD on
  all triplets AND singlets — so PySCF is correct.

  Our σ-equation matches M_exact on triplets to 7 meV but disagrees
  on singlets by ~2.57 eV in opposite directions. After extending
  the LiH brute-force test to diff the FULL 14×14 M_mine vs M_exact
  element-by-element (not just R_1×R_1), the bug structure is now:

  | block | max \|Δ\| | nature |
  |-------|----------:|--------|
  | R_1 × R_1 | 0.53 eV  | diagonal patch artifact (cosmetic) |
  | R_1 × R_2 | 5.84 eV  | cross-coupling — major |
  | R_2 × R_1 | 4.04 eV  | cross-coupling — major |
  | R_2 × R_2 | 7.26 eV  | self-coupling — dominant bug |

  The R_1×R_1 off-diagonal couplings ARE correct (initial hypothesis
  about missing ⟨iα jβ ‖ aα bβ⟩·R_1 was wrong — that coupling is
  in W_mbej and the diff confirms it). The singlet eigenvalue gap
  flows from R_2 contamination via R_1 ↔ R_2 mixing, not from R_1
  itself. The dominant offending entry is
  [R₂[0<3,0<1], R₂[0<1,0<1]] = 7.26 eV — an R_2 ↔ R_2 coupling
  between two doubles sharing the (a=0, b=1) virtual pair but
  different occupied pairs. That kind of coupling flows through
  Σ_mn W̄_mnij R_2[m,n,a,b] in σ_2 — so our W_mnij contraction or
  the W̄_mnij intermediate itself is the next thing to audit.

  Scope: was a real σ_2 bug, partially closed via a sign correction.

  Tested-and-rejected hypotheses (2026-05-13):
  - "Stage 32c patches over-correct on multi-electron — revert them
    and see if singlets improve." Reverting made the LiH lowest
    triplet WORSE (7 meV → 540 meV gap) and did NOT shrink singlets.
    Patches restored.

  Stage 32i: diagnostic basis-ordering correction — the prior R_2 ×
  R_2 "off-diagonal 7.26 eV bug" was diagnostic permutation noise,
  not a real bug. After correction the R_2 × R_2 off-diagonals went
  to ~10⁻¹⁵ Ha.

  Stage 32k (the actual fix): the σ_1 ← R_2 W̄_amef term had a
  sign-flip. Code used `+½ ⟨ma||ef⟩` where Stanton-Bartlett 1993
  Eq 41 requires `+½ ⟨am||ef⟩` (= −½ ⟨ma||ef⟩ by antisymmetry).
  One-line fix:
    V(m, a+VO, e+VO, f+VO)  →  V(a+VO, m, e+VO, f+VO)
  RESULT: LiH singlet gap collapsed 2.57 eV → **0.27 eV** (10×
  better), within the literature EOM-CCSD ↔ FCI accuracy bar
  (0.1–0.2 eV per Stanton-Bartlett). Triplet 6.77 meV (essentially
  exact).

  Stages 32j, 32l: added T1·T1 + linear-T1 dressings on W̄_abej,
  W̄_mbij, W̄_mnie, W̄_amef per Crawford-Schaefer 2000. Each closed
  ~10–25% of the remaining gap on bigger systems.

  Current state:
  - LiH STO-3G: triplets exact (7 meV); singlets 0.27 eV (method
    precision limit reached).
  - BeH₂: triplets within 0.1 eV; singlets not separately tested.
  - H₂O / NH₃ / CH₄: singlets still ~0.5–1.9 eV off — more
    structural T-dressings missing (PySCF's woVoO has ~8 dressings;
    we have 3-4 of them).

  Closure path: the PySCF port (MIGRATION.md) lands all remaining
  T-dressings at once. The brute-force diagnostic
  (`tests/chemistry/eom-ccsd-bruteforce-lih.test.ts`) is the
  permanent verifier — every fix attempt makes the M_mine − M_exact
  diff shrink or doesn't.
  See `experiments/results/2026-05-13/level-6/E35-comparison.md`.
- **IP-EOM-CCSD R₂ satellites** have a known **~2 Ha (~60 eV)
  over-count** on H₂ STO-3G. Documented in `ip-eom-ccsd.ts`. Affects
  R₂-dominated Auger / shake-up states only; physical lowest IPs are
  validated exact against brute-force.
- **Stage 30 eigenvectors** for degenerate eigenvalues set the
  zero-denominator entry to 0, picking one representative from the
  degenerate eigenspace. The 3 returned vectors are individually
  unit-normalized but NOT guaranteed mutually orthogonal. Callers that
  need an orthonormal degenerate basis must Gram-Schmidt explicitly.
- **DF-HF / DF-MP2 machine-precision** is validated on **STO-3G**
  (H₂O, BeH₂). cc-pVDZ DF is expected to be equally clean by
  construction (same Cholesky algorithm) but not separately
  benchmarked.

---

## 6. Software engineering

- **No CI vendor-matrix** — GitHub Actions runs Linux Chromium only.
  Apple Silicon, Windows, AMD, NVIDIA, Intel — manual spot checks.
- **No mutation testing** — `npm run test` runs the full vitest suite
  green, but we don't measure whether it would catch real mutations.
- **No formal coverage metric** — we know which files have tests; we
  don't track lines/branches.
- **No fuzz testing** of the integral / SCF / CC paths.
- **No memory safety audit** of WGSL kernels beyond manual review.
  WebGPU bounds-checks shader memory accesses by spec, so the blast
  radius is bounded, but undefined behavior is still possible.

---

## 7. Not in scope (now or ever)

- **Force fields (MM)** — this is an electronic-structure engine.
- **Molecular dynamics propagators** (Verlet, RESPA, etc.) — out of
  scope unless used by BOMD/NAMD, which is Tier 4.
- **Docking / drug discovery** — wrong layer of abstraction.
- **Crystal structure prediction** — needs periodic + force fields.
- **4-component Dirac with QED corrections** — specialist territory
  (DIRAC, BERTHA, BAGEL).

---

## When in doubt

Run a smaller analog through PySCF or ORCA first, then check ours
matches before relying on the result. Issue a GitHub issue with the
input + both outputs and we'll add a regression test.
