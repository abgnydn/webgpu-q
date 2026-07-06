# E34 PySCF reference — reproduction (2026-07-06)

Re-ran `scripts/run-pyscf-reference.py` with **PySCF 2.13.0** pinned (via
`uv run --with pyscf==2.13.0`), same M2 Pro, same molecules × basis × methods
as the original 2026-05-12 E34 reference. Purpose: confirm the apples-to-apples
CPU comparison in [`../../2026-05-12/level-6/E34-comparison.md`](../../2026-05-12/level-6/E34-comparison.md)
still holds, and correct the README framing that mislabeled it "open work."

**Result — the CPU comparison reproduces.**

- **Energies: bit-identical** to the 2026-05-12 run (|ΔE| = 0.0 on every cell —
  same PySCF version, deterministic).
- **Wall-clock: within run-to-run noise, marginally faster** on the warm machine
  (newer scipy). Representative cc-pVDZ CCSD wall-clock, old → new:
  H₂O 86.75 → 67.50 ms · LiH 76.40 → 60.64 ms · BeH₂ 58.29 → 42.94 ms.
  (PySCF got a little faster, so if anything our production-basis *losses* are
  slightly larger than the headline 480×/136× — the honest direction.)

**What this does and doesn't settle.**

- ✅ The CPU wins **and** losses in the README table are genuine wall-clock vs
  PySCF 2.13.0 on identical inputs — not self-referential. The old footnote
  ("speedups are vs our own CPU baseline … open work") was wrong for the CPU rows.
- ⚠️ The **webgpu-q side** of E34 is still the 2026-05-12 in-browser measurement
  (chemistry unchanged on this branch, so still valid; a fresh in-browser E34 run
  is a cheap follow-up to re-timestamp it).
- 🔴 Still genuinely open: the **CCSD(T)-GPU** number is vs our own CPU TypeScript.
  A same-hardware **gpu4pyscf** (CUDA) comparison needs an NVIDIA GPU — runnable on
  Modal/Colab per the `tools/modal/webgpu_t4_probe.py` precedent. That, not the CPU
  comparison, is the remaining apples-to-apples work.

Fresh PySCF artifact: [`E34-pyscf.json`](./E34-pyscf.json).
