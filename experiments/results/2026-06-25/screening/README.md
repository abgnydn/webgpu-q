# Screening campaign — validation + exhaustive aza-chromophore discovery

**First committed result artifact for the screening track.** The e2e specs
(`screening-validation-polyene`, `screening-discovery-azahexatriene`,
`screening-discovery-scaled`) *define* this campaign but run in the browser and
never recorded a result. This runs the identical chemistry in Node (the
`chem-energy` kernel is engine-independent — WASM here, WASM/GPU in the tab) and
records the ranking. Runner: `tests/screening/screening-campaign.test.ts`
(gated behind `SCREENING_CAMPAIGN=1`); data: `screening-campaign.json`.

**Descriptor:** HOMO–LUMO gap (eV). **Method:** RHF / STO-3G, exact ERI.
**Determinism:** fixed library + deterministic SCF, no RNG. **Status: pass.**

## Bar ① — validation (does the descriptor track a KNOWN trend?)

An all-trans polyene's gap must shrink as the conjugated chain grows (the
particle-in-a-longer-box effect — why long conjugated dyes are colored).

| polyene | C=C | gap (eV) |
|---|--:|--:|
| ethylene | 1 | 17.42 |
| butadiene | 2 | 13.77 |
| hexatriene | 3 | 11.98 |
| octatetraene | 4 | 10.94 |
| decapentaene | 5 | 10.27 |

**Monotonic decrease, 7.14 eV drop.** The descriptor reproduces the textbook
trend → the screening filter is trustworthy. (HF/STO-3G overestimates *absolute*
gaps; the *trend* — what screening ranks on — is robust.)

## Bar ② — discovery (a BLIND exhaustive screen)

Every placement of ≤3 nitrogens on a 6-atom conjugated chain = **42 isomers**,
all isoelectronic (44 e⁻), so gap differences are purely structural. No human
candidate bias — screen the space, let champions emerge. **Question:** do the
best chromophores share the **azo (N=N)** motif behind most commercial dyes?

**42/42 converged. Gap range 2.39 – 12.11 eV.** Top 15 (smallest gap = most
red-shifted / best visible absorber):

| # | isomer | gap (eV) | azo (N=N) |
|--:|---|--:|:--:|
| 1 | 2,3,5-aza | 2.39 | — |
| 2 | 2,4,5-aza | 2.39 | — |
| 3 | 3,4-aza | 7.61 | ✓ |
| 4 | 3,4,5-aza | 7.71 | ✓ |
| 5 | 2,3,4-aza | 7.71 | ✓ |
| 6 | 1,2,3-aza | 11.20 | ✓ |
| 7 | 4,5,6-aza | 11.20 | ✓ |
| 8 | 1,3,4-aza | 11.61 | ✓ |
| 9 | 3,4,6-aza | 11.61 | ✓ |
| 10 | 1,2-aza | 11.83 | ✓ |
| 11 | 5,6-aza | 11.83 | ✓ |
| 12 | 1,6-aza | 11.86 | — |
| 13 | 1,2,6-aza | 11.86 | ✓ |
| 14 | 1,5,6-aza | 11.86 | ✓ |
| 15 | 1,2,4-aza | 11.86 | ✓ |

**Lead signal: the azo (N=N) motif is 36% of the library but 80% of the top 15
→ 2.24× enriched.** The screen tracks real dye chemistry.

**Honest trap (the discipline working):** the two smallest-gap hits
(2,3,5- and 2,4,5-aza, **2.39 eV**) are *non-azo* and sit ~3× below the azo pack
(~7.6 eV) — anomalously small for minimal-basis HF, the classic fingerprint of an
**RHF instability** (near-diradical character). They are **likely artifacts, not
trusted leads** — re-examine with UHF / a multireference method. A screen
surfaces both leads *and* traps; you scrutinize the top, never trust the single
best number blindly. HF/STO-3G ⇒ a **triage shortlist**, not proven hits.

## Systems note — why the swarm exists, and where LPT does/doesn't help

Single-threaded in Node this campaign took **1043 s (~17 min)** — exact-ERI HF on
n ≈ 36–62 is genuinely heavy (decapentaene n=62 alone is 107 s). The same
42-isomer screen runs in **~194 s across two GPU-assisted browser tabs**
(`screening-discovery-scaled` e2e). That gap *is* the case for distribution.

But note **which** kind of screen this is: the 42 aza isomers are a **uniform**
library — all 6-atom, all isoelectronic, so every tile costs ≈ the same. Uniform
work is embarrassingly parallel and **LPT scheduling = FIFO** (nothing to reorder;
any scheduler scales ~linearly). The cost-aware **LPT** win (1.98× — see
`experiments/results/2026-06-23/swarm-lpt-scheduling/`) is for the *opposite*
regime: a **size-diverse** library (H₂ → C₂H₄) where one heavy molecule tails the
batch. The two screening regimes are complementary:

| library | tile costs | scheduler that matters |
|---|---|---|
| this aza screen (uniform) | ≈ equal | any (FIFO ≈ LPT), ~linear scaling |
| diverse molecule batch | uneven | **LPT** (1.98×, the merged result) |

## Verdict

The screening track's first recorded result: the descriptor **passes validation**
(known polyene trend reproduced) and, on a **blind 42-isomer exhaustive screen**,
**enriches 2.24× for the real azo dye motif** while honestly flagging two RHF-
instability traps at the very top. Engine-independent (Node ≡ browser); the ~17 min
single-thread cost is the swarm's reason to exist, with LPT the lever for the
*diverse* sibling regime.
