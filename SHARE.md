# Share kit — webgpu-q

Assets (this folder): `demo.mp4` (990 KB, upload to posts) · `demo.gif` (4.9 MB, README) ·
`public/og-image.png` (link-preview card). Live: https://webgpu-q.vercel.app ·
Code: https://github.com/abgnydn/webgpu-q

---

## Show HN

**Title:** Show HN: webgpu-q – real quantum chemistry (HF/DFT/CCSD(T)) in a browser tab

**Body:**

webgpu-q runs real electronic-structure calculations — Hartree–Fock, DFT, MP2,
CCSD, CCSD(T), and EOM-CCSD — entirely in a browser tab. WebAssembly does the
math; WebGPU accelerates the CCSD(T) bottleneck. No install, no server, no CUDA.

Open the page and a real Hartree–Fock SCF converges in front of you (watch the
energy fall to the ground state). In the hyperscope you can drag a molecule's
bond length and the wavefunction is recomputed every frame.

Honest about scope: the methods are ported from PySCF and validated against it —
CCSD(T) is sub-mHa vs FCI, and EOM-CCSD matches a brute-force H̄ projection to
<1e-10 Ha element-wise. It is **not** a PySCF replacement: PySCF is 10–100×
faster and scales to far larger systems. The point isn't to beat it — it's that
a real wavefunction calculation now ships as a URL. Good for teaching (every
intermediate is inspectable), reproducibility (a calculation is a shareable
link), and as a systems demo of how far the browser has come.

There's also a browser-tab "swarm" that distributes a Hartree–Fock build across
multiple tabs/machines, and the whole thing is open source (MIT + Apache for the
ported parts), validated in CI.

Live: https://webgpu-q.vercel.app · Code: https://github.com/abgnydn/webgpu-q

Built solo (independent researcher, AI-paired). Genuinely interested in where it
breaks — fire away.

---

## r/comp_chem  (they will scrutinize — lead with validation + limits)

**Title:** I put HF / DFT / CCSD(T) / EOM-CCSD in a browser tab (WebAssembly + WebGPU), validated against PySCF

**Body:**

Weekend-project-turned-bigger: a browser-native electronic-structure stack.
Everything runs client-side — HF, UHF, RKS/UKS DFT (LDA/GGA/hybrids), MP2,
CCSD, CCSD(T), EE/IP/EA-EOM-CCSD, TDDFT, geometry opt, IR/Raman, polarizabilities.

I'm posting here because I want it torn apart by people who'd know. The honest
validation surface:
- HF vs PySCF ≤ 0.5 mHa (≤ 0.1 mHa cc-pVDZ spherical-d)
- CCSD(T) ≤ 0.25 mHa vs FCI
- EOM-CCSD σ matched element-wise to an explicit H̄ = e⁻ᵀHeᵀ projection to
  <1e-10 Ha (the methods are ported from PySCF's eom_gccsd, not re-derived)
- density fitting bit-exact-ish vs direct ERI; aux-basis f64 DF for larger systems

Honest limits: minimal/medium bases only (largest benchmarked is cc-pVDZ on small
molecules); it's far slower and far smaller-scale than PySCF; minimal-basis HF
screening overestimates HOMO–LUMO gaps and can hit artifacts (flagged in the UI).
Not a research workhorse — a teaching / reproducibility / "serious science as a
URL" tool, and a WebGPU systems demonstration (the CCSD(T) kernel is ~14× on GPU).

Live (open it, it computes immediately): https://webgpu-q.vercel.app
Code + validation tests: https://github.com/abgnydn/webgpu-q

Where would you expect it to break first? What would make it actually useful to you?

---

## X / Twitter / Bluesky  (short)

Real quantum chemistry — Hartree–Fock, DFT, CCSD(T), EOM-CCSD — running in a
browser tab. WebAssembly + WebGPU, no install, no server. Open the link and a
real SCF converges in front of you; every number is cross-checked against PySCF.

🔗 https://webgpu-q.vercel.app  (open source)

---

## Notes on framing (keep it honest)
- Lead with what it IS and that it's validated; never claim it beats PySCF.
- "Ported from PySCF, validated" is a credibility signal, not a weakness — say it.
- The wow is "serious science ships as a URL," not raw speed.
- Invite critique; the comp-chem crowd rewards humility + real numbers.
