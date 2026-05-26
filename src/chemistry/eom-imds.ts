// ─────────────────────────────────────────────────────────────
// eom-imds.ts — EOM-CCSD intermediates (PySCF gintermediates.py).
//
// Shared module imported by both production EOM-CCSD modules
// (eom-ccsd.ts, ip-eom-ccsd.ts) AND their brute-force regression
// tests, so the σ-equation intermediates are guaranteed to be
// identical across all four sites.
//
// All intermediates differ from the cc_ counterparts in ccsd.ts in
// well-defined ways (the bare canonical Fock diagonal in Foo/Fvv, the
// full τ/t2 dressings in Woooo/Wvvvv/Wovvo, the full PySCF builds for
// Wooov/Wvovv/Wovoo/Wvvvo). See [[port-dont-rederive]] +
// [[symbol-collision-porting]] in memory.
//
// Refs:
//   - PySCF eom_gccsd.py (Wang-Tu-Wang 2014 Eqs 9-10 for EE,
//     Tu-Wang-Li 2012 Eqs 8-9 for IP)
//   - PySCF gintermediates.py for the intermediate formulas
//
// Apache 2.0 ported portion (LICENSE-PYSCF at repo root).
// ─────────────────────────────────────────────────────────────

export interface EOMIntermediates {
  /** F_ov[m,e] = f_me + Σ_nf t_nf ⟨mn||ef⟩. Canonical RHF: f_me = 0. */
  readonly Fov: Float64Array;
  /** F_vv[a,e]: includes ε_a δ_ae diagonal, T1 + τ̃ dressings, ½ t1·Fov term. */
  readonly Fvv: Float64Array;
  /** F_oo[m,i]: includes ε_i δ_mi diagonal, T1 + τ̃ dressings, ½ t1·Fov term. */
  readonly Foo: Float64Array;
  /** W_oooo[m,n,i,j] with ½·τ (not the cc_W's ¼·τ). */
  readonly Woooo: Float64Array;
  /** W_vvvv[a,b,e,f] with ½·τ. */
  readonly Wvvvv: Float64Array;
  /** W_ovvo[m,b,e,j] with full t2·oovv (not the cc_W's ½·t2). */
  readonly Wovvo: Float64Array;
  /** W_ooov[m,n,i,e] = ⟨mn||ie⟩ + Σ_f t_if ⟨mn||fe⟩. */
  readonly Wooov: Float64Array;
  /** W_vovv[a,m,e,f] = ⟨am||ef⟩ − Σ_n t_na ⟨nm||ef⟩. */
  readonly Wvovv: Float64Array;
  /** W_ovoo[m,b,i,j] — full PySCF build. Heavily dressed. */
  readonly Wovoo: Float64Array;
  /** W_vvvo[a,b,e,i] — full PySCF build. Heavily dressed. */
  readonly Wvvvo: Float64Array;
}

export interface EOMIntermediatesOpts {
  /** Only build the intermediates IP-EOM needs (skip Wvvvv + Wvvvo). */
  readonly subset?: "ip" | "ee";
}

/**
 * Build EOM-CCSD intermediates from T1, T2, antisym spin-orbital ERIs
 * and the canonical-orbital energies. All inputs are flat row-major
 * Float64Arrays in the SO basis (P = 2p + σ).
 *
 * `eri[((P*NSO+Q)*NSO+R)*NSO+S]` is ⟨PQ||RS⟩ in physicist's antisym
 * notation. `tau_t = τ̃` (T2 + ½·T1·T1 antisymmetrized).
 * `tau = τ`  (T2 + 1·T1·T1 antisymmetrized).
 */
export function buildEOMIntermediates(
  T1: Float64Array, T2: Float64Array, eri: Float64Array,
  tau_t: Float64Array, tau: Float64Array,
  eps: Float64Array, NOCC: number, NVIRT: number, NSO: number,
  opts: EOMIntermediatesOpts = {},
): EOMIntermediates {
  const VO = NOCC;
  const V = (P: number, Q: number, R: number, S: number): number =>
    eri[((P * NSO + Q) * NSO + R) * NSO + S]!;

  // ── Fov ──────────────────────────────────────────────────────
  const Fov = new Float64Array(NOCC * NVIRT);
  for (let m = 0; m < NOCC; m++) {
    for (let e = 0; e < NVIRT; e++) {
      let s = 0;
      for (let n = 0; n < NOCC; n++) {
        for (let f = 0; f < NVIRT; f++) {
          s += T1[n * NVIRT + f]! * V(m, n, e + VO, f + VO);
        }
      }
      Fov[m * NVIRT + e] = s;
    }
  }

  // ── Fvv ──────────────────────────────────────────────────────
  const Fvv = new Float64Array(NVIRT * NVIRT);
  for (let a = 0; a < NVIRT; a++) {
    for (let e = 0; e < NVIRT; e++) {
      let s = (a === e) ? eps[a + VO]! : 0;
      for (let m = 0; m < NOCC; m++) {
        for (let f = 0; f < NVIRT; f++) {
          s += T1[m * NVIRT + f]! * V(a + VO, m, e + VO, f + VO);
        }
      }
      for (let m = 0; m < NOCC; m++) {
        for (let n = 0; n < NOCC; n++) {
          for (let f = 0; f < NVIRT; f++) {
            s -= 0.5 * tau_t[((m * NOCC + n) * NVIRT + a) * NVIRT + f]! *
                       V(m, n, e + VO, f + VO);
          }
        }
      }
      for (let m = 0; m < NOCC; m++) {
        s -= 0.5 * T1[m * NVIRT + a]! * Fov[m * NVIRT + e]!;
      }
      Fvv[a * NVIRT + e] = s;
    }
  }

  // ── Foo ──────────────────────────────────────────────────────
  const Foo = new Float64Array(NOCC * NOCC);
  for (let m = 0; m < NOCC; m++) {
    for (let i = 0; i < NOCC; i++) {
      let s = (m === i) ? eps[m]! : 0;
      for (let n = 0; n < NOCC; n++) {
        for (let e = 0; e < NVIRT; e++) {
          s += T1[n * NVIRT + e]! * V(m, n, i, e + VO);
        }
      }
      for (let n = 0; n < NOCC; n++) {
        for (let e = 0; e < NVIRT; e++) {
          for (let f = 0; f < NVIRT; f++) {
            s += 0.5 * tau_t[((i * NOCC + n) * NVIRT + e) * NVIRT + f]! *
                       V(m, n, e + VO, f + VO);
          }
        }
      }
      for (let e = 0; e < NVIRT; e++) {
        s += 0.5 * T1[i * NVIRT + e]! * Fov[m * NVIRT + e]!;
      }
      Foo[m * NOCC + i] = s;
    }
  }

  // ── Woooo ────────────────────────────────────────────────────
  const Woooo = new Float64Array(NOCC * NOCC * NOCC * NOCC);
  for (let m = 0; m < NOCC; m++) {
    for (let n = 0; n < NOCC; n++) {
      for (let i = 0; i < NOCC; i++) {
        for (let j = 0; j < NOCC; j++) {
          let s = V(m, n, i, j);
          for (let e = 0; e < NVIRT; e++) {
            s += T1[j * NVIRT + e]! * V(m, n, i, e + VO);
            s -= T1[i * NVIRT + e]! * V(m, n, j, e + VO);
          }
          for (let e = 0; e < NVIRT; e++) {
            for (let f = 0; f < NVIRT; f++) {
              s += 0.5 * tau[((i * NOCC + j) * NVIRT + e) * NVIRT + f]! *
                         V(m, n, e + VO, f + VO);
            }
          }
          Woooo[((m * NOCC + n) * NOCC + i) * NOCC + j] = s;
        }
      }
    }
  }

  // ── Wvvvv (EE only) ──────────────────────────────────────────
  const Wvvvv = opts.subset === "ip"
    ? new Float64Array(0)
    : new Float64Array(NVIRT * NVIRT * NVIRT * NVIRT);
  if (opts.subset !== "ip") {
    for (let a = 0; a < NVIRT; a++) {
      for (let b = 0; b < NVIRT; b++) {
        for (let e = 0; e < NVIRT; e++) {
          for (let f = 0; f < NVIRT; f++) {
            let s = V(a + VO, b + VO, e + VO, f + VO);
            for (let m = 0; m < NOCC; m++) {
              s -= T1[m * NVIRT + b]! * V(a + VO, m, e + VO, f + VO);
              s += T1[m * NVIRT + a]! * V(b + VO, m, e + VO, f + VO);
            }
            for (let m = 0; m < NOCC; m++) {
              for (let n = 0; n < NOCC; n++) {
                s += 0.5 * tau[((m * NOCC + n) * NVIRT + a) * NVIRT + b]! *
                           V(m, n, e + VO, f + VO);
              }
            }
            Wvvvv[((a * NVIRT + b) * NVIRT + e) * NVIRT + f] = s;
          }
        }
      }
    }
  }

  // ── Wovvo ────────────────────────────────────────────────────
  const Wovvo = new Float64Array(NOCC * NVIRT * NVIRT * NOCC);
  for (let m = 0; m < NOCC; m++) {
    for (let b = 0; b < NVIRT; b++) {
      for (let e = 0; e < NVIRT; e++) {
        for (let j = 0; j < NOCC; j++) {
          let s = V(m, b + VO, e + VO, j);
          for (let f = 0; f < NVIRT; f++) {
            s += T1[j * NVIRT + f]! * V(m, b + VO, e + VO, f + VO);
          }
          for (let n = 0; n < NOCC; n++) {
            s -= T1[n * NVIRT + b]! * V(m, n, e + VO, j);
          }
          for (let n = 0; n < NOCC; n++) {
            for (let f = 0; f < NVIRT; f++) {
              const t2v = T2[((j * NOCC + n) * NVIRT + f) * NVIRT + b]!;
              const t1p = T1[j * NVIRT + f]! * T1[n * NVIRT + b]!;
              s -= (t2v + t1p) * V(m, n, e + VO, f + VO);
            }
          }
          Wovvo[((m * NVIRT + b) * NVIRT + e) * NOCC + j] = s;
        }
      }
    }
  }

  // ── Wooov ────────────────────────────────────────────────────
  const Wooov = new Float64Array(NOCC * NOCC * NOCC * NVIRT);
  for (let m = 0; m < NOCC; m++) {
    for (let n = 0; n < NOCC; n++) {
      for (let i = 0; i < NOCC; i++) {
        for (let e = 0; e < NVIRT; e++) {
          let s = V(m, n, i, e + VO);
          for (let f = 0; f < NVIRT; f++) {
            s += T1[i * NVIRT + f]! * V(m, n, f + VO, e + VO);
          }
          Wooov[((m * NOCC + n) * NOCC + i) * NVIRT + e] = s;
        }
      }
    }
  }

  // ── Wvovv (EE only) ──────────────────────────────────────────
  const Wvovv = opts.subset === "ip"
    ? new Float64Array(0)
    : new Float64Array(NVIRT * NOCC * NVIRT * NVIRT);
  if (opts.subset !== "ip") {
    for (let a = 0; a < NVIRT; a++) {
      for (let m = 0; m < NOCC; m++) {
        for (let e = 0; e < NVIRT; e++) {
          for (let f = 0; f < NVIRT; f++) {
            let s = V(a + VO, m, e + VO, f + VO);
            for (let n = 0; n < NOCC; n++) {
              s -= T1[n * NVIRT + a]! * V(n, m, e + VO, f + VO);
            }
            Wvovv[((a * NOCC + m) * NVIRT + e) * NVIRT + f] = s;
          }
        }
      }
    }
  }

  // ── Wovoo ────────────────────────────────────────────────────
  const Wovoo = new Float64Array(NOCC * NVIRT * NOCC * NOCC);
  for (let m = 0; m < NOCC; m++) {
    for (let b = 0; b < NVIRT; b++) {
      for (let i = 0; i < NOCC; i++) {
        for (let j = 0; j < NOCC; j++) {
          let s = V(m, b + VO, i, j);
          for (let e = 0; e < NVIRT; e++) {
            s -= Fov[m * NVIRT + e]! *
                 T2[((i * NOCC + j) * NVIRT + b) * NVIRT + e]!;
          }
          for (let n = 0; n < NOCC; n++) {
            s -= T1[n * NVIRT + b]! *
                 Woooo[((m * NOCC + n) * NOCC + i) * NOCC + j]!;
          }
          for (let e = 0; e < NVIRT; e++) {
            for (let f = 0; f < NVIRT; f++) {
              s += 0.5 * V(m, b + VO, e + VO, f + VO) *
                         tau[((i * NOCC + j) * NVIRT + e) * NVIRT + f]!;
            }
          }
          for (let n = 0; n < NOCC; n++) {
            for (let e = 0; e < NVIRT; e++) {
              s += V(m, n, i, e + VO) *
                   T2[((j * NOCC + n) * NVIRT + b) * NVIRT + e]!;
              s -= V(m, n, j, e + VO) *
                   T2[((i * NOCC + n) * NVIRT + b) * NVIRT + e]!;
            }
          }
          for (let e = 0; e < NVIRT; e++) {
            s += T1[i * NVIRT + e]! * V(m, b + VO, e + VO, j);
            s -= T1[j * NVIRT + e]! * V(m, b + VO, e + VO, i);
          }
          for (let n = 0; n < NOCC; n++) {
            for (let e = 0; e < NVIRT; e++) {
              for (let f = 0; f < NVIRT; f++) {
                const tvj = T2[((n * NOCC + j) * NVIRT + b) * NVIRT + f]!;
                const tvi = T2[((n * NOCC + i) * NVIRT + b) * NVIRT + f]!;
                const eV = V(m, n, e + VO, f + VO);
                s -= T1[i * NVIRT + e]! * tvj * eV;
                s += T1[j * NVIRT + e]! * tvi * eV;
              }
            }
          }
          Wovoo[((m * NVIRT + b) * NOCC + i) * NOCC + j] = s;
        }
      }
    }
  }

  // ── Wvvvo (EE only) ──────────────────────────────────────────
  const Wvvvo = opts.subset === "ip"
    ? new Float64Array(0)
    : new Float64Array(NVIRT * NVIRT * NVIRT * NOCC);
  if (opts.subset !== "ip") {
    for (let a = 0; a < NVIRT; a++) {
      for (let b = 0; b < NVIRT; b++) {
        for (let e = 0; e < NVIRT; e++) {
          for (let i = 0; i < NOCC; i++) {
            let s = V(a + VO, b + VO, e + VO, i);
            for (let m = 0; m < NOCC; m++) {
              for (let n = 0; n < NOCC; n++) {
                s += 0.5 * V(m, n, e + VO, i) *
                           tau[((m * NOCC + n) * NVIRT + a) * NVIRT + b]!;
              }
            }
            for (let m = 0; m < NOCC; m++) {
              s -= Fov[m * NVIRT + e]! *
                   T2[((m * NOCC + i) * NVIRT + a) * NVIRT + b]!;
            }
            for (let m = 0; m < NOCC; m++) {
              for (let f = 0; f < NVIRT; f++) {
                s -= V(m, b + VO, e + VO, f + VO) *
                     T2[((m * NOCC + i) * NVIRT + a) * NVIRT + f]!;
                s += V(m, a + VO, e + VO, f + VO) *
                     T2[((m * NOCC + i) * NVIRT + b) * NVIRT + f]!;
              }
            }
            for (let m = 0; m < NOCC; m++) {
              s -= T1[m * NVIRT + a]! * V(m, b + VO, e + VO, i);
              s += T1[m * NVIRT + b]! * V(m, a + VO, e + VO, i);
            }
            for (let m = 0; m < NOCC; m++) {
              for (let n = 0; n < NOCC; n++) {
                for (let f = 0; f < NVIRT; f++) {
                  const eV = V(m, n, e + VO, f + VO);
                  const tva = T2[((n * NOCC + i) * NVIRT + b) * NVIRT + f]!;
                  const tvb = T2[((n * NOCC + i) * NVIRT + a) * NVIRT + f]!;
                  s += T1[m * NVIRT + a]! * tva * eV;
                  s -= T1[m * NVIRT + b]! * tvb * eV;
                }
              }
            }
            for (let f = 0; f < NVIRT; f++) {
              s += Wvvvv[((a * NVIRT + b) * NVIRT + e) * NVIRT + f]! *
                   T1[i * NVIRT + f]!;
            }
            Wvvvo[((a * NVIRT + b) * NVIRT + e) * NOCC + i] = s;
          }
        }
      }
    }
  }

  return { Fov, Fvv, Foo, Woooo, Wvvvv, Wovvo, Wooov, Wvovv, Wovoo, Wvvvo };
}
