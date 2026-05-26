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
| Multi-node parallel | **Same-machine multi-tab via BroadcastChannel verified e2e (Phase D Step 1).** WebRTCTransport (Phase D Step 2) ships via peerjs.com's free public broker — **cross-machine swarming has NOT been verified in e2e for the chem-energy kernel**; only the BroadcastChannel transport is exercised by `e2e/swarm.spec.ts`. Symmetric-NAT corporate networks may fail without TURN (not provided). Self-hosted PeerServer + TURN-server fallback + a real cross-machine e2e are outstanding hardening work. The protocol layer is transport-agnostic, so the upgrade path is a drop-in. | partial — same-machine ✓, cross-machine unverified |
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
- **Parallel HF buildG via Web Workers** — measured for the first time
  2026-05-26 (`e2e/bench-parallel-hf.spec.ts`). H₂O cc-pVDZ on M2 Pro,
  headless Chromium, COI on, 5 trials:
  | config | median wall | vs sync |
  |---|---:|---:|
  | sync | 17 ms | 1.0× |
  | parallel=2 | 13 ms | 1.31× |
  | parallel=4 | 10 ms | **1.71×** (best) |
  | parallel=8 | 12 ms | 1.47× (overhead drag) |

  Earlier docs hand-waved a "4–8× win" for Web Workers; the honest
  measured number on n=25 cc-pVDZ is **1.7×**, with the win flattening
  at parallel=8 because spawn + SAB-message overhead starts to dominate
  the per-iteration JK cost. The speedup is real but modest. Bigger
  molecules (n ≥ 50) would amortize the worker overhead better;
  benching at that scale is open work.
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
