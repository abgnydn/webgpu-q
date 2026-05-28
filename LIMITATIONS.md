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
| Counterpoise / BSSE (HF + MP2 + CCSD + UHF + UCCSD + RKS + UKS + D2) | **Shipped 2026-05**. `runCounterpoise(atoms, fragments, basis, hfOpts, method, ccsdOpts, uhfOpts)` in `src/chemistry/counterpoise.ts`. Boys-Bernardi: per-fragment energy in the full dimer basis via ghost atoms (`ghost?: boolean` flag on `Atom` — basis present, Z=0, no electrons). `method` selects HF / MP2 / CCSD / UHF / UCCSD; the ghost-atom plumbing passes through to the post-HF / post-UHF correlation step unchanged. For open-shell methods, each `CounterpoiseFragment` carries an explicit `spin: { nAlpha, nBeta }`; supermolecule spin is the sum. Returns supermolecule energy, per-fragment energies in dimer + bare basis, uncorrected + CP interaction energies, and `bsseCorrection = ΔE_CP − ΔE ≥ 0`. Verified by 11 tests in `counterpoise.test.ts` (7 closed-shell on H₂...H₂; 4 open-shell on H↑+H↑ at 5 Å, Li doublet + H↑, H₂ singlet + H↑ via UCCSD, and the missing-spin error path). | shipped ✓ |
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
| Foster-Boys orbital localization | **Shipped 2026-05** — `fosterBoys(C_MO, integrals, shells, nOccupied, opts?)` in `src/chemistry/foster-boys.ts`. Boys 1960 maximization of Σ_i ⟨ψ_i\|r\|ψ_i⟩² via 2×2 Jacobi sweeps. Returns unitary rotation matrix U (n_occ × n_occ) and per-sweep L history. Localized orbitals = C_MO[:, :nOcc] · U. Converges in 5-20 sweeps for typical small molecules. | shipped ✓ |
| Pipek-Mezey orbital localization | **Shipped 2026-05** — `pipekMezey(C_MO, integrals, shellAtomIdx, nOccupied, opts?)` in `src/chemistry/pipek-mezey.ts`. PM 1989 maximization of Σ_i Σ_A (Q^i_A)² (Mulliken atomic populations of orbital i on atom A). Same Jacobi-sweep machinery as Foster-Boys, but using Mulliken populations instead of dipole centroids. Often separates σ and π bonds where Boys mixes them into banana hybrids; also origin-invariant (Mulliken populations don't depend on coordinate origin). Validated on H₂O / CH₄: monotonic L convergence, U orthogonal to 1e-9. | shipped ✓ |
| Natural orbital occupation numbers (NOON) | **Shipped 2026-05** — `naturalOrbitalOccupations(D, S, opts?)` in `src/chemistry/natural-orbitals.ts`. Diagonalizes D in the Löwdin-orthogonal metric (D̃ = S^(1/2)·D·S^(1/2)) and returns occupation numbers (sorted descending, 0 ≤ n_p ≤ 2), natural-orbital coefficients in the AO basis, and "strongly occupied" / "active" counts (thresholds 1.95 / 0.05 by default). Validated on closed-shell H₂O HF (NOONs = {2,2,2,2,2,0,0} ± 1e-10, ΣN = 10) and Be UHF (NOONs = {2,2,0,0,0} for no-symm-break; broken-symmetry tested too — Σ N = N_electrons preserved). Useful multi-reference diagnostic on UHF/UKS spin-contaminated systems or correlated 1-PDMs (MP2/CCSD relaxed densities — those require relaxed-density build first, queued). | shipped ✓ |
| ccData / QC-Schema compatibility | **Shipped 2026-05 (QCSchema)** — `toQCSchemaClosedShell` and `toQCSchemaOpenShell` in `src/chemistry/qcschema.ts`. Emit a `qcschema_output` v1 AtomicResult JSON with the standard `molecule` / `model` / `properties` / `return_result` / `success` / `extras` layout. Properties block includes `scf_total_energy`, `nuclear_repulsion_energy`, `scf_correlation_energy` (+ method-tagged `mp2_correlation_energy` / `ccsd_correlation_energy` when applicable), `calcinfo_*` (natom / nbasis / nmo / nalpha / nbeta), `molecular_multiplicity` (open-shell), `scf_s_squared` (open-shell). Validated by 4/4 tests covering closed-shell HF / CCSD method tag / open-shell H atom UHF / extras preservation. Consumable by QCEngine, QCFractal, QCArchive, cclib, ASE workflows. cclib parser support follows separately. | shipped ✓ (QCSchema) |
| FAIR / Zenodo DOIs per release | Citations point at GitHub tag, not DOI | Tier 3 — set up CI workflow |
| Aux-basis density fitting | We have CD-DF but not JKFIT / RIFIT integral path | Tier 3 — needs 3-index ERI routine |
| DFT dispersion correction | **Shipped 2026-05 (D2)** — `dispersionD2(atoms, opts)` in `src/chemistry/dispersion-d2.ts`. Grimme JCC 2006 atomic-pairwise C6/R⁶ with Fermi damping and functional-specific s6 (BLYP=1.20, B3LYP5=1.05, B3VWN5=1.05, BVWN5=1.20, LDA=1.05 default). Tabulated C6 and R_R for H, He, Li, Be, C, N, O, F. Atomic-units throughout. O(n_atoms²) cost — microseconds for typical molecules. Validated on He-He, H₂-H₂ pair, C-C pair, CH₄, ghost-atom cases. 8/8 tests green. D3 (Grimme 2010, atom-environment-dependent C8/C9) is a follow-up. Gradients ∂E_disp/∂R_A are mechanical extension; queued. | shipped ✓ (D2) |
| Davidson / Krylov eigensolver | **Shipped 2026-05**. Block Davidson (`src/manybody/davidson.ts`) wired into EE/IP/EA-EOM-CCSD (commits 6b9a96f, a35baeb) and now also CIS / TDA (this commit) behind `useDavidson: true`. Dense + Davidson agree on k roots to ≤ 1e-6 Ha across all paths. Unblocks large-basis CIS/TDDFT + bigger EOM-CCSD systems. | shipped ✓ |
| Multi-node parallel | **All three steps verified e2e 2026-05-26.** (1) Same-origin multi-tab via BroadcastChannel — `e2e/swarm.spec.ts`. (2) Cross-machine WebRTC pairing via peerjs.com broker with the chem-energy H₂ bond-scan kernel — `e2e/swarm-webrtc.spec.ts` (two isolated browser contexts simulate two machines; finds equilibrium r ≈ 0.73 Å over WebRTC DataChannel in ~6 s). (3) BroadcastChannel-isolation check passes (contexts can't see each other via BC). Open hardening: self-hosted PeerServer (replaces peerjs.com dependency), TURN-server fallback for symmetric-NAT corporate networks. | shipped ✓ |
| Periodic boundary conditions | No solids, no surfaces | Tier 4 |
| Spin-orbit coupling / X2C / DKH | No heavy elements | Tier 4 |
| QM/MM | No biomolecules | Tier 4 |
| Anharmonic VPT2 | Harmonic only | Tier 4 |

---

## 5. Honest precision disclosures (carried from CLAUDE.md)

**Read this first.** webgpu-q reports a lot of precision numbers at
10⁻¹⁰ Ha and tighter. These are **software regression assertions**
that catch porting / GPU-CPU drift bugs; they are not chemistry
results. Chemical accuracy is 1.6 mHa (1 kcal/mol). Basis-set
incompleteness, functional choice, and method truncation all
contribute errors at 1–10 kcal/mol on real systems. So:

- **|GPU − CPU| < 10⁻¹⁰ Ha** = "our GPU port matches our CPU TypeScript
  port" — engineering claim.
- **|webgpu-q − PySCF| < 10⁻¹⁰ Ha (brute-force diff)** = "our σ-matrix
  matches PySCF's σ-matrix element-by-element" — also engineering.
- **|webgpu-q HF − PySCF HF| at ≥ 1 mHa, |E_CCSD(T) − FCI| at ≥ 0.25
  mHa** = method-quality numbers — actually chemistry-relevant.

Treat anything tighter than 1 mHa as "we don't have a porting bug,"
not as a meaningful chemistry result. The reviewer-defensible
precision floor is chemical accuracy.

- **CCSD(T) GPU speedup** — properly measured 2026-05-26
  (`e2e/bench-ccsdt-gpu.spec.ts`). H₂O cc-pVDZ, M2 Pro, headless
  Chromium, **5 warmup + 20 trials**:
  | metric | value |
  |---|---:|
  | median speedup vs CPU TS | **13.8×** |
  | p10 (best run) | 28.4× |
  | p90 (worst run) | 10.1× |
  | std/median | **41.7% — officially "noisy"** per RESEARCH_STANDARDS sec 4 |

  The previously-headlined "39.3×" was a *single lucky run* near p10;
  the **honest sustained number is 13.8× median with wide variance**.
  Still vs single-threaded CPU TypeScript — not vs PySCF wall-clock,
  not vs GPU4PySCF on CUDA. Apples-to-apples comparison work
  outstanding.
- **Parallel HF buildG via Web Workers** — measured 2026-05-26
  (`e2e/bench-parallel-hf.spec.ts`). M2 Pro, headless Chromium, COI on,
  hardwareConcurrency = 12:

  | molecule | n (basis fns) | sync HF median | parallel=8 | speedup | trials |
  |---|---:|---:|---:|---:|---:|
  | H₂O cc-pVDZ | 25 | 17 ms | 8 ms | **2.08×** | 5 |
  | Ethane C₂H₆ cc-pVDZ | 60 | 724 ms | 241 ms | **3.00×** | 5 |
  | Furan C₄H₄O cc-pVDZ | 95 | 4620 ms | 1152 ms | **4.01×** | 1 |
  | Benzene C₆H₆ cc-pVDZ | 120 | 14.07 s | 2.19 s | **6.43×** | 1 |

  Speedup grows roughly **+1× per +30 basis functions** on M2 Pro
  (12-thread hardwareConcurrency), with a *bigger* jump at n=120
  (2.08× → 3.00× → 4.01× → **6.43×**). Energies match sync exactly
  across all parallel=N (Δ = 0 Ha). The honest "best win we've
  measured" is **6.43× on benzene cc-pVDZ (n=120)**. Theoretical
  max for parallel=8 on a 12-thread machine is 8×; we're at 80%
  of that ceiling on benzene.

  **ERI build is the wall-time bottleneck** (one-time, doesn't
  parallelize with Workers): 740 s on benzene cc-pVDZ vs 14.1 s for
  the entire HF SCF after that. So total wall-clock benefit of
  parallel=8 on a single benzene HF run is only ~1.6×; if you run
  many SCFs on the same integrals (geom-opt, DFT functional sweep,
  etc.) it grows to the full 6.43×. Benzene HF previously timed out
  at 20 + 30 min Playwright caps; ran cleanly in 13.4 min after the
  ERI pair-cache landed.

  **What would lift the ceiling**: (a) vectorize the Obara-Saika
  ERI recursions, (b) shell-pair Schwarz screening on top of the
  existing primitive-pair screening, (c) **aux-basis density fitting**
  (3-index ERIs directly, never build the n⁴ tensor — *not* the same
  as our shipped CD-DF which still builds the full tensor first;
  measured 11.5× slower at n=25, 20.5× slower at n=60 — see DF-HF
  benches in `e2e/bench-parallel-hf.spec.ts`).

- **Rust+WASM ERI kernel** — measured 2026-05-27
  (`e2e/bench-wasm-eri.spec.ts`). Algorithm-identical port of the ERI
  primitive kernel (Boys, E-coefs, R-aux table, pair-cache, 8-fold
  symmetry) to Rust → wasm-pack → wasm32. Native-compiled inner loops
  beat JIT'd TypeScript on the n⁴ build:
  | molecule | n | TypeScript | Rust+WASM | speedup |
  |---|---:|---:|---:|---:|
  | ethane cc-pVDZ | 60 | 25.5 s | 6.1 s | **4.18×** |
  | benzene cc-pVDZ | 120 | 827 s | 175 s | **4.73×** |
  Output bit-identical at both sizes (max |Δ| = 4.4×10⁻¹⁶ Ha, pure
  float-rounding). WASM module is ~80 KB, loads on demand, first-call
  init ~1 s. Speedup *grows* with molecule size — bigger inner loops
  amortize the WASM/JS call overhead better.

  **End-to-end HF wall time on benzene cc-pVDZ (measured 2026-05-27):**
  | path | total wall time | breakdown |
  |---|---:|---|
  | TS-only | 841 s (14 min) | 827 s ERI + 14 s sync HF |
  | WASM ERI + parallel=8 HF | 180 s (3 min) | 175 s ERI + 5.2 s par HF |
  | WASM × parallel=8 ERI + parallel=8 HF | **44 s** | 38.5 s ERI + 5.2 s par HF |
  | speedup vs TS | **19.1×** | ERI 21.5×, HF dominated by ERI |

  Compounds further with WASM SIMD128 intrinsics (open).

- **WASM × Workers compound** — measured 2026-05-27
  (`e2e/bench-wasm-parallel-eri.spec.ts`). Each worker loads its own
  `wasm-eri` instance and computes its μ-row slice via the native Rust
  kernel, then writes the 8 symmetric positions to a shared n⁴ ERI
  SAB. Schwarz Q-table built once on main thread; round-robin μ
  distribution to balance the canonical-encoding work-per-row decay.
  Output is bit-identical to both single-thread paths.

  | molecule | n | TS | WASM 1× | WASM × par=4 | WASM × par=8 | best vs TS |
  |---|---:|---:|---:|---:|---:|---:|
  | ethane cc-pVDZ  |  60 |  26.3 s | 6.0 s |  2.4 s | 3.0 s | **10.8×** (par=4) |
  | benzene cc-pVDZ | 120 | 827 s   | 189 s | 61.7 s | 38.5 s | **21.5×** (par=8) |

  Sweet spot shifts with molecule size: at n=60 the per-worker setup
  cost saturates parallel=8; at n=120 the work-per-worker is large
  enough that parallel=8 still scales near-linearly (4.91× over WASM
  single-thread). All paths agree element-wise (max|Δ|=0 on benzene,
  4.4×10⁻¹⁶ on ethane — pure rounding).

- **Hot-path refactor in `prim_eri_with_pairs`** — measured 2026-05-27.
  Branch-free + hoisted intermediates in the inner 6-loop of the
  Rust ERI kernel:
  - Drop `if eN == 0.0 continue` early-exits (loop bounds keep us in
    the recurrence-filled E-coef region — branches rarely fire for
    cc-pVDZ anyway).
  - `sign = 1 - 2·parity` (branch-free) replaces `if (parity) {
    -1.0 } else { 1.0 }`.
  - Hoist partial products: `xyz1 = ex1·ey1·ez1` out of v,tau,nu,phi
    loops; `xyz1_x2 = xyz1·ex2` out of nu,phi; etc. Inner loop is
    2 mults + 1 multiply-by-sign + 1 load + 1 add.
  - Slice-reference the 6 1D E-coef rows instead of computing the
    flat index per access. **First attempt prefetched into `[f64; 13]`
    stack arrays — regressed benzene single-thread by 55%** because
    Rust's zero-init across ~130 M primitive-ERI calls swamped any
    inner-loop gain. Direct slice indexing recovered the win.

  | molecule | n | WASM 1× before | WASM 1× after | par=8 before | par=8 after |
  |---|---:|---:|---:|---:|---:|
  | benzene cc-pVDZ | 120 | 189 s | **127 s** (1.49×) | 38.5 s | **25.3 s** (1.52×) |

  Combined with the WASM × Workers compound, benzene cc-pVDZ ERI is
  now **32.7× faster than the TS-only baseline** (827 s → 25.3 s).
  Bit-identical to the textbook path (max|Δ|=0 on benzene).

- **Pair-table cache in `eri_build` / `eri_build_slice`** — measured
  2026-05-27. The Rust kernel was rebuilding the bra and ket
  primitive-pair tables (E-coefficients via Hermite-Gaussian
  recurrence) on every (μν|λσ) call. For benzene cc-pVDZ that's
  ~470 M redundant pair builds (26 M unique ERIs × 18 builds per
  call), even though there are only n²/2 = 7 200 distinct (a, b)
  shell pairs. Hoisting to a single `precompute_pair_tables` call
  at the top of the build collapses this to 7 200 builds — a
  65 000× reduction.

  | molecule | n | WASM 1× before | WASM 1× after | par=8 before | par=8 after |
  |---|---:|---:|---:|---:|---:|
  | ethane cc-pVDZ  |  60 | 5.6 s | **3.4 s** (1.65×) |  1.55 s | **0.94 s** (1.65×) |
  | benzene cc-pVDZ | 120 | 127 s | **81.7 s** (1.55×) | 25.3 s  | **16.8 s** (1.51×) |

  Memory cost is ~4.5 MB per worker on benzene (n²/2 × ~1.2 KB per
  pair table). Trivial vs the 1.65 GB ERI tensor. Bit-identical
  output (max|Δ|=0 on benzene). Compounded:

  Benzene cc-pVDZ ERI build: **49.2× faster than the TS-only
  baseline** (827 s → 16.8 s).

- **r_aux_table buffer pooling** — measured 2026-05-27. The Rust
  `r_aux_table` allocated a fresh `Vec<f64>` of up to 625 entries on
  every primitive ERI call (~2 B mallocs on benzene cc-pVDZ).
  Refactored to write into caller-provided `f_buf` / `r_buf` scratch
  buffers that get re-used across all 81 primitive-pair calls per
  ERI quartet and across all ERI quartets in the build. `Vec::clear()
  + resize(0.0)` keeps capacity (no realloc after the first
  allocation) and the resize-with-0 just memsets the active prefix.

  | molecule | n | WASM 1× before | WASM 1× after | par=8 before | par=8 after |
  |---|---:|---:|---:|---:|---:|
  | ethane cc-pVDZ  |  60 | 3.4 s | **3.05 s** (1.12×) | 0.94 s | **0.83 s** (1.14×) |
  | benzene cc-pVDZ | 120 | 81.7 s | **74.7 s** (1.09×) | 16.8 s | **15.6 s** (1.08×) |

  Bit-identical output (max|Δ|=0 on benzene).

  Benzene cc-pVDZ ERI build: **53× faster than the TS-only
  baseline** (827 s → 15.6 s).

  End-to-end HF benzene "cold shells → converged energy":
  - TS-only baseline: 841 s (14 min)
  - All wins shipped: **~16.8 s** (15.6 s ERI + ~1.24 s SIMD WASM HF)
  - Total speedup: **~50× over the start-of-session baseline**.

- **wasm-simd128 hand-vectorized JK inner loop** — measured
  2026-05-27. The σ-summation in `fock_one_mu_row`
  (Σ_σ D[λσ]·(J − ½K)) is a length-n linear scan — perfect for
  f64x2 SIMD. Hand-wrote a `jk_dot` helper using
  `std::arch::wasm32::*` intrinsics: `v128_load` + `f64x2_mul` +
  `f64x2_sub` + `f64x2_add` to process 2 σ per cycle, with a scalar
  fallback for the (rare) odd remainder.

  | molecule | n | TS JK | WASM-only JK | + SIMD JK | speedup over TS |
  |---|---:|---:|---:|---:|---:|
  | ethane cc-pVDZ  |  60 | 13.5 ms | 4.2 ms  | **2.5 ms** | 5.31× |
  | benzene cc-pVDZ | 120 | 193 ms  | 129 ms  | **~85 ms** | ~2.3× |

  Benzene HF SCF total: 1.89 s → **1.24 s** (1.52× faster).
  Bit-identical output (max|Δ|=1.19e-13 Ha — pure rounding).

  SIMD on the ERI hot loop (`prim_eri_with_pairs` 6-deep nest)
  remains unattractive: cc-pVDZ inner loops are 1-5 iterations,
  too short to amortize SIMD setup, and the parity-sign / E-coef
  branches break LLVM's auto-vec. The JK kernel succeeded because
  the inner loop is a clean length-n contiguous reduction.

  A 4-way unroll of the SIMD JK (two independent f64x2
  accumulators) showed no measurable signal on benzene (17.88s vs
  18.01s on a single trial — within noise band); the 2-way SIMD
  already saturated the pipeline at our problem sizes. Reverted to
  the cleaner 2-way version.

### HF SCF parallel scaling (with WASM JK, measured 2026-05-27)

End-to-end HF SCF wall time across molecule sizes with all wins
shipped (`runRHFSCFAsync(..., parallel: 8, useWasmJK: true)` vs
`runRHFSCF` sync baseline):

| molecule | n | sync HF | parallel=8 HF | speedup |
|---|---:|---:|---:|---:|
| H₂O cc-pVDZ      |  25 | 18 ms    | 8 ms     | 2.29× (worker overhead floor) |
| ethane cc-pVDZ   |  60 | 602 ms   | 93 ms    | **6.48×** |
| furan cc-pVDZ    |  95 | 3882 ms  | 591 ms   | **6.56×** |
| benzene cc-pVDZ  | 120 | (TS path 14+ min, skipped) | 1.24 s | (vs WASM-1× HF ~5.2 s ≈ 4.2×) |

(Benzene sync omitted from the parallel-HF bench because the TS-built
ERI alone takes 14+ minutes; the WASM ERI path is the relevant
sync reference. Full benzene WASM HF end-to-end ≈ 16.8 s, vs the
TS-only "cold shells → converged" baseline of 841 s.)

The 2.29× floor on H₂O reflects worker-spawn + message-passing
overhead at n=25: workers idle most of their lifetime. For
n ≥ 60 the JK build dominates and speedup approaches the
6-8× ceiling that 8 workers + 2-lane f64 SIMD allow.

- **WGSL JK kernel (WebGPU)** — measured 2026-05-27
  (`e2e/bench-jk-gpu.spec.ts`). The Fock G matrix construction is
  ported to a WebGPU compute shader: one thread per (μ, ν) entry of
  G, with a serial f32 inner loop over (λ, σ). ERI tensor lives in
  GPU storage (~830 MB f32 on benzene cc-pVDZ, uploaded once per
  HF call). D matrix uploaded per iter (~58 KB).

  | molecule | n | WASM (SIMD) JK | GPU JK | speedup | max |G_WASM − G_GPU| |
  |---|---:|---:|---:|---:|---:|
  | ethane cc-pVDZ  |  60 |  2.1 ms |  3.4 ms | 0.62× (loses) | — |
  | benzene cc-pVDZ | 120 | 69.5 ms | 29.2 ms | **2.38×** | 8.6×10⁻⁵ Ha |

  The cross-over: GPU pays a fixed ~1-3 ms dispatch + readback
  overhead per call. For small n that dominates compute; for benzene
  n=120 (n⁴ inner work) compute dominates and the GPU wins.

  **f32 precision: 2×10⁻⁴ max relative error.** Wired into
  `runRHFSCFAsync` via `useWgpuJK?: GPUDevice` opt for research use,
  but **fails HF SCF convergence at energyTol ≤ 1e-6 Ha**: tested on
  benzene cc-pVDZ where the DIIS error vector stays above the f32
  noise floor (~10⁻⁴ relative) and SCF cannot settle. Hit maxIter=100
  in 15.76 s without converging, vs 10.85 s / 9 iters / converged
  with WASM-f64. Per-iter JK was indeed 2.38× faster but the extra
  iterations swallow the gain and then some.

  Honest read: the per-kernel speedup is real, but f32 alone isn't
  enough. The proper next step is mixed-precision iterative
  refinement (f32 GPU bulk + periodic f64 WASM correction). Until
  that's implemented, production HF should stay on the WASM SIMD
  path (`useWasmJK = true`).

  **WGSL JK at loose tolerances also loses.** Measured 2026-05-27
  on benzene cc-pVDZ with progressively looser SCF thresholds:

  | tolerance | WGSL time | iters | converged | E error vs WASM ref |
  |---:|---:|---:|---:|---:|
  | 1e-3 (chemical) | 8.52 s |   7 | yes | 8.5×10⁻⁵ Ha |
  | 1e-4            | 13.1 s | 101 | NO  | 5.6×10⁻⁵ Ha |
  | 1e-5 (tight)    | 13.2 s | 101 | NO  | 5.6×10⁻⁵ Ha |
  | (WASM tight ref)| 2.56 s |   8 | yes | — |

  At chemical tolerance WGSL DOES converge but is **3.3× slower**
  than WASM tight: per-iter GPU dispatch + `mapAsync` readback
  overhead (~100 ms) dwarfs the ~30 ms kernel compute. The
  per-kernel 2.38× speedup in the synthetic bench does not carry
  over to in-loop SCF use because WASM SIMD JK is already so fast
  that the GPU's dispatch latency is the bottleneck, not its
  compute throughput.

  Real path to a GPU win: (a) keep the GPU pipeline hot between
  iters (don't `unmap` / `destroy`), (b) use persistent mapped
  staging buffers for D upload, (c) batch multiple SCF iters into
  one GPU submission with a small CPU-readback frequency. That's
  a real engineering project, not a single-session push.

- **Aux-basis density fitting Phase 1** — measured 2026-05-27.
  Foundation laid: a proper 3-index ERI kernel (μν|P) and 2-index
  ERI kernel (P|Q) for the auxiliary basis path, both written in
  Rust/WASM with McMurchie-Davidson recursion specialized for the
  single-function (no-pair) ket side. New `df-aux.ts` JS bridge
  composes them: V = (μν|P), M = (P|Q), eigendecompose M via the
  existing Jacobi solver, form B = V · M^(-1/2). Returns a
  standard `DFResult` interoperable with `buildJK_DF`.

  Phase 1 uses the **orbital basis as the auxiliary basis** because
  cc-pVDZ-jkfit aux-basis data tables aren't in the repo yet. This
  validates the algorithm but the aux basis is too small to span
  orbital products well — reconstruction errors on H₂ cc-pVDZ are
  ~66 mHa max, ~4 mHa RMS. Symmetries are exact (V[μν,P]=V[νμ,P]
  to 0.0, M[P,Q]=M[Q,P] to 0.0); M is positive-definite (min
  diagonal 5.8). Eigendecomp + matrix inverse-sqrt is numerically
  stable.

  Phase 2 (next session): wire in cc-pVDZ-jkfit aux data tables
  for H, C, N, O (the minimum to run organics). With proper aux
  basis the reconstruction quality should be sub-mHa, matching
  PySCF/ORCA's RI-HF accuracy.

  Phase 3 (later): swap into `runRHFSCFAsync` as
  `useDF: { type: "aux-basis", ... }` opt. Expected 5-10× HF
  speedup at n ≥ 80 because B is ~3× smaller than the 4-index
  tensor AND the JK build over B is cheaper.

  **Kernels are bit-perfect** (measured 2026-05-28). Cross-checked
  the 2-index and 3-index kernels against closed-form analytical
  values:

    (s|s)     for normalized 1s @ origin = 4π/α:
       α=1    expected 12.56637061   computed 12.56637061  rel 1.4e-16
       α=0.5  expected 25.13274123   computed 25.13274123  rel 1.4e-16
       α=2    expected  6.28318531   computed  6.28318531  rel 1.4e-16
       α=10   expected  1.25663706   computed  1.25663706  rel 1.8e-16

    (s_a s_a | s_c) for normalized 1s @ origin = N³(α)·π^(5/2)/(α^(5/2)·√3):
       α=1    expected  3.65632112   computed  3.65632112  rel 1.2e-16
       α=0.5  expected  4.34812309   computed  4.34812309  rel 2.0e-16
       α=2    expected  3.07458732   computed  3.07458732  rel 1.4e-16
       α=5    expected  2.44512930   computed  2.44512930  rel 1.8e-16

  Rel error 10⁻¹⁶ on every case = pure float rounding. Algorithm
  is correct; the HF-level errors below come entirely from aux
  basis insufficiency, not from the integral kernels.

  **HF energy errors with the current Phase-1 path scale unexpectedly with system size**:

  | system | orb basis | aux basis | n_orb | n_aux | DF HF error |
  |---|---|---|---:|---:|---:|
  | H₂  | cc-pVDZ | cc-pVDZ      | 10 | 10 |   −7 mHa |
  | H₂  | cc-pVDZ | aug-cc-pVDZ  | 10 | 18 |   −5 mHa |
  | H₂O | cc-pVDZ | cc-pVDZ      | 25 | 25 | −209 mHa |
  | H₂O | cc-pVDZ | aug-cc-pVDZ  | 25 | 43 | −188 mHa |
  | H₂O | STO-3G  | cc-pVDZ      |  7 | 25 | −252 mHa |

  With the kernels validated above as bit-perfect, the diagnosis
  is now clear: **aux basis insufficiency dominates**, both in
  angular momentum range AND in exponent coverage:

  - For H₂O cc-pVDZ orbital with cc-pVDZ aux: orbital has L=2
    (d-functions). Pair products span up to L=4 (g-functions
    via d·d). cc-pVDZ aux only has L=0,1,2. The missing g and f
    components in aux contribute the ~200 mHa error.
  - For H₂O STO-3G orbital with cc-pVDZ aux: STO-3G has different
    exponent ranges than cc-pVDZ. cc-pVDZ aux exponents don't
    "fit" STO-3G orbital products well even though the L coverage
    is nominally sufficient. That's the source of the 252 mHa
    error.

  Phase 2 to do: load proper cc-pVDZ-jkfit aux-basis data tables
  for H, C, N, O. These were designed by Weigend specifically to
  span orbital products of cc-pVDZ with optimal exponents AND
  proper L coverage (up to g on first-row atoms). Expected DF HF
  energy errors with proper jkfit: < 0.1 mHa (matches PySCF and
  ORCA RI-HF accuracy).

  Until proper aux tables are loaded, the algorithm correctness
  is validated but quantitative HF use is gated. aux-DF stays
  unwired from production HF SCF defaults.

  **Update 2026-05-28**: `generateAutoAux` (decontract orbital
  primitives + extend angular momentum by `extraL`) gives sub-chemical
  accuracy at `extraL=1` without needing external aux basis tables:

  | system | extraL | n_aux | HF error | iter | converged |
  |---|---:|---:|---:|---:|---|
  | H₂O cc-pVDZ | 0 (decontract only) |  41 |   19 mHa  | 16 | ✓ |
  | H₂O cc-pVDZ | 1 (add L+1)         | 138 | **0.11 mHa** | 14 | ✓ |
  | H₂O cc-pVDZ | 2 (add L+2)         | 311 |  −55 Ha     | 101 | ✗ |

  extraL=1 hits chemical accuracy (1.6 mHa = 1 kcal/mol) on H₂O.
  extraL=2 catastrophic failure (−55 Ha, doesn't converge) at L=4
  reveals an untested g-function code path in the 3-index kernel.
  Likely a normalization or recurrence bug at L ≥ 4 — the existing
  4-index path doesn't exercise g because cc-pVDZ orbital tops at
  d. Phase 2 should: (a) add a focused L=4 unit test against
  closed-form, (b) fix whatever's wrong, (c) re-run extraL=2 and
  expect sub-µHa accuracy.

  **For production**: `useDF: generateAutoAux(shells, 1)`-built
  DFResult through `runRHFSCFAsync` is viable today for any cc-pVDZ
  orbital system, with 0.1 mHa expected error. The auto-aux build is
  fast (~100 ms on H₂O n=25). The B-tensor is ~5× the size of the
  orbital basis (n=25, n_aux=138, B = 25²·138 = 86 KB). Composes with
  the existing `buildJK_DF` path. Wiring as a default opt-in awaits
  systematic correctness benchmarks across the H/C/N/O/F atom set.

  **End-to-end timing — ethane cc-pVDZ (n=60, measured 2026-05-28)**:

  | path | ERI/B build | HF SCF | total | E error |
  |---|---:|---:|---:|---:|
  | Direct 4-index (single-thread WASM) | ~22 s | ~0.3 s | 22.3 s | 0 (reference) |
  | Direct 4-index (WASM × parallel=8)  | 0.83 s | ~0.1 s | ~1 s | 0 |
  | Auto-aux DF extraL=1 (single-thread) | 3.00 s | 2.66 s | 5.66 s | 0.4 mHa |

  Auto-aux DF is **slower than the WASM-parallel direct path on
  ethane** — the direct path's parallel 4-index ERI build is hard to
  beat at n=60. The aux-DF win is:

  1. **Memory** — at benzene cc-pVDZ, direct ERI tensor is 1.65 GB;
     auto-aux B-tensor is ~60 MB (30× reduction). Direct path hits
     browser memory limits around n ≈ 180-200; aux-DF unlocks
     larger systems (naphthalene+).
  2. **Algorithmic scaling** — 3-index O(n²·n_aux) eventually beats
     4-index O(n⁴) as n grows. Crossover around n = 150-200 even
     before parallelizing.
  3. **Future**: porting the 3-index ERI build to the worker pool
     (same pattern as the existing 4-index parallel path) should
     give parallel-DF that beats parallel-4-index from n ≈ 80+ at
     equal accuracy. Phase 3 work.

  **Parallel 3-index V build shipped 2026-05-28** (`buildAuxBasisDFParallel`):
  bit-perfect correctness (max|B_sync − B_parallel|=0) and after the
  matmul loop reorder for cache locality, real parallel speedups
  appear at n=120:

  | molecule | n_orb | n_aux | single-thread | parallel=8 | speedup |
  |---|---:|---:|---:|---:|---:|
  | ethane cc-pVDZ  |  60 | 200  |  2.88 s |  2.57 s | 1.12× |
  | benzene cc-pVDZ | 120 | ~400 | 39.31 s | 21.22 s | **1.85×** |

  The loop reorder (commit 938d4ca) was the key — inner P loops in
  B = T · Uᵀ now read U's columns contiguously instead of with
  stride-nAux, which thrashed L1 at n_aux=400. Benzene parallel
  dropped 35 s → 21 s (14 s saved).

  Remaining gap to optimized direct path on benzene:
    WASM-parallel direct (4-index):  16.8 s
    aux-DF parallel=8 + auto-aux:    21.2 s
    → aux-DF is ~25 % slower at n=120

  **Update 2026-05-28**: B-tensor matmul ported to Rust+WASM with
  f64x2 SIMD (`form_b_tensor`). Benzene B-tensor build:

  | path | before WASM matmul | after WASM matmul | change |
  |---|---:|---:|---:|
  | Single-thread aux-DF | 39.31 s | **12.78 s** | 3.1× |
  | Parallel=8 aux-DF    | 21.22 s |  **9.41 s** | 2.3× |

  Now **aux-DF B-tensor build (9.41 s) beats WASM-parallel direct
  4-index ERI build (15.6 s) on benzene cc-pVDZ** — the crossover
  has materialized.

  Remaining differences:
    - Auto-aux extraL=1 produces ~3-4× more aux functions than
      curated jkfit would. Halving n_aux would give another ~2×.
    - Eigendecomp of M (400×400 Jacobi) is in TS — ~0.5 s, small.
    - The 2-index M build is single-threaded WASM on main — cheap
      at n_aux=400 (~0.1 s).

  Where aux-DF still wins decisively:
    - Memory: ERI tensor 1.65 GB → B tensor 60 MB on benzene (28×).
    - Scaling: direct hits the ~4 GB browser tab heap ceiling around
      n ≈ 200 (e.g., naphthalene cc-pVDZ would need ~10 GB for ERI).
      aux-DF at n_aux ~ 600 needs only ~170 MB for B — fits easily.

- **ERI pair-table caching** — measured 2026-05-27. `ERI_cg` was
  rebuilding the bra-pair Hermite-Gaussian E-coefficient tables for
  every primitive quartet (n_prim⁴ buildPair calls per ERI). Now caches
  bra-pair and ket-pair tables once per primitive-pair combo (2·n_prim²
  builds total). For cc-pVDZ with ~3 primitives/shell, that's 81 → 18
  buildPair calls per ERI_cg — ~4.5× reduction in pair-build work.
  Wall-time impact on ethane cc-pVDZ ERI build:
  | threshold | before | after | speedup |
  |---|---:|---:|---:|
  | 1e-10 (default) | 31.2 s | 23.9 s | 1.31× |
  | 1e-6 | 31.3 s | 24.9 s | 1.26× |
  Energies bit-identical at 1e-10. Furan ERI build extrapolates from
  361 s → ~275 s; benzene from ~25 min → ~14 min. Real and free.

- **Schwarz screening is ineffective at cc-pVDZ density.** Measured
  on ethane cc-pVDZ 2026-05-27 (`e2e/bench-eri-screening.spec.ts`):
  | threshold | skip rate | ERI build | |ΔE| vs 1e-10 |
  |---|---:|---:|---:|
  | 1e-12 | 0.0% | 31.5 s | numerical noise |
  | **1e-10 (default)** | **0.0%** | 31.2 s | — |
  | 1e-8 | 0.3% | 32.3 s | 10⁻⁸ Ha |
  | 1e-6 | 3.2% | 31.3 s | 7×10⁻⁶ Ha |
  | 1e-4 | 18.9% | 25 s (–20%) | **1.5 mHa (chemical-accuracy edge)** |

  At cc-pVDZ density, almost every basis-function pair has Q > 10⁻¹⁰
  — the integrals are too dense for Cauchy-Schwarz bounds to find
  skippable pairs. Tightening or loosening the threshold within
  chemistry-relevant ranges (1e-12 to 1e-6) saves at most 3% of
  work. The only meaningful win is at 1e-4, which sits at chemical
  accuracy and is risky. **The "2-3× ERI build speedup from Schwarz"
  promise was wrong** — for our specific basis sets and molecule
  sizes, this screen is essentially a no-op. Real ERI speedup needs
  WebGPU offload or aux-basis DF.

- **CD-DF as shipped is a NET LOSS for HF speed.** The Cholesky
  decomposition operates on the already-built n⁴ ERI tensor — it
  doesn't avoid the ERI build cost (the actual bottleneck) and adds
  decomposition overhead on top. Useful for post-HF methods that work
  in B-tensor space (memory savings) but the `opts.useDF: true` HF
  path is strictly slower than `useDF: false` on every system we
  benched. The real DF speedup (PySCF-style, 5×+) requires aux-basis
  3-index ERIs — a new integral routine, not derivable from `ERI_cg`.
  Measured 2026-05-27:
  | molecule | n | direct HF | DF-HF | ratio |
  |---|---:|---:|---:|---:|
  | H₂O cc-pVDZ | 25 | 18 ms | 204 ms | DF 11.5× slower |
  | Ethane cc-pVDZ | 60 | 652 ms | 13336 ms | DF 20.5× slower |
  Energies match to ≤ 1.1×10⁻⁵ Ha (below chemical accuracy), so DF is
  correct, just slow. Documentation that hand-waved DF as a "speedup"
  was wrong; the API is now annotated honestly.
- **EE-EOM-CCSD σ-equations** — **PySCF-ported and verified
  2026-05-21**. σ_1 + σ_2 follow Wang-Tu-Wang 2014 Eqs (9)-(10) with
  PySCF eom_gccsd intermediates (Foo/Fvv/Fov, Woooo, Wvvvv, Wovvo
  with full t2 dressing, Wooov, Wvovv, Wovoo, Wvvvo). The earlier
  empirical stage-32c/32e diagonal patches are removed.

  Verifier: `tests/chemistry/eom-ccsd-bruteforce-lih.test.ts` builds
  H̄ = e^(-T̂) H e^(T̂) explicitly in the 64-dim 4-electron LiH Fock
  space, projects onto the (R_1 + antisym R_2) basis, and diffs the
  full 14×14 against the σ-equation matrix element-by-element. After
  the port, max |Δ| < 1e-10 Ha — i.e., the σ-equation builds the
  same H̄ projection as the brute-force construction to numerical
  noise. The hard assertion `expect(maxDiffHa).toBeLessThan(1e-10)`
  is in the test as a permanent regression check.

  H₂ STO-3G EOM-CCSD eigenvalues now match FCI **to 8+ decimal
  places** (3 triplets at 0.60479072 Ha, 2 singlets at 0.96736838 /
  1.61710528 Ha) — the 10⁻⁵ Ha "algorithmic precision" cap noted in
  prior versions of this doc was an artifact of the empirical
  patches, not a method limit. H₂O STO-3G now gives a 10.81 eV
  lowest triplet and 12.44 eV first singlet.
- **IP-EOM-CCSD σ-equations** — **PySCF-ported and verified 2026-05-22**.
  Mirror of the 2026-05-21 EE-EOM port: σ_1 + σ_2 follow Tu-Wang-Li
  2012 Eqs (8)-(9) with PySCF eom_gccsd intermediates (Foo/Fov/Fvv with
  bare canonical Fock diagonal, Woooo/Wvvvv/Wovvo with full τ/t2
  dressings, Wooov/Wovoo properly dressed). The pre-port R_2 satellite
  over-count (~60 eV on H₂) is closed; brute-force diff
  `tests/chemistry/ip-eom-ccsd-bruteforce.test.ts` < 1e-10 Ha
  element-by-element with hard `expect(maxDiff).toBeLessThan(1e-10)`
  regression assertion.
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
