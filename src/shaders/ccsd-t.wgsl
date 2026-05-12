// ─────────────────────────────────────────────────────────────
// ccsd-t.wgsl — perturbative-triples (T) correction on the GPU.
//
// Parallel decomposition: 1 thread per (i, j, k) occupied triple.
// Each thread sums over all (a, b, c) virtual triples internally
// and writes a single f32 partial sum to partial_sums[triple_idx].
// CPU reduces in f64 for final precision.
//
// f32 precision is intentional — see TS dispatch wrapper docstring
// for the precision argument: per-(i,j,k) partial sums are bounded
// in magnitude (µHa to mHa range), so f32 storage gives sub-µHa
// reduction error after f64 CPU sum.
//
// Layout matches the CPU reference exactly:
//   eri:  NSO⁴ row-major as eri[((P*NSO+Q)*NSO+R)*NSO+S]
//   T1:   NOCC × NVIRT row-major
//   T2:   NOCC² × NVIRT² row-major as T2[((i*NOCC+j)*NVIRT+a)*NVIRT+b]
//   eps:  NSO orbital energies (occ first, then virtual)
//
// Dim packed in uniform: { NSO, NOCC, NVIRT, nFrozenSO }.
// ─────────────────────────────────────────────────────────────

struct Dims {
  NSO: u32,
  NOCC: u32,
  NVIRT: u32,
  nFrozenSO: u32,
};

@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> eri: array<f32>;
@group(0) @binding(2) var<storage, read> T1: array<f32>;
@group(0) @binding(3) var<storage, read> T2: array<f32>;
@group(0) @binding(4) var<storage, read> eps: array<f32>;
@group(0) @binding(5) var<storage, read_write> partial_sums: array<f32>;

// ⟨PQ||RS⟩ accessor.
fn eri_at(P: u32, Q: u32, R: u32, S: u32) -> f32 {
  return eri[((P * d.NSO + Q) * d.NSO + R) * d.NSO + S];
}

// T2[((i*NOCC+j)*NVIRT+a)*NVIRT+b]
fn t2_at(i: u32, j: u32, a: u32, b: u32) -> f32 {
  return T2[((i * d.NOCC + j) * d.NVIRT + a) * d.NVIRT + b];
}

// W_base(i,j,k,a,b,c) = Σ_e ⟨ie||bc⟩ t_jk^ae  −  Σ_m ⟨jk||am⟩ t_im^bc
fn W_base(i: u32, j: u32, k: u32, a: u32, b: u32, c: u32) -> f32 {
  let A = a + d.NOCC;
  let B = b + d.NOCC;
  let C = c + d.NOCC;
  var s: f32 = 0.0;
  for (var e: u32 = 0u; e < d.NVIRT; e = e + 1u) {
    let E = e + d.NOCC;
    s = s + eri_at(i, E, B, C) * t2_at(j, k, a, e);
  }
  for (var m: u32 = 0u; m < d.NOCC; m = m + 1u) {
    s = s - eri_at(j, k, A, m) * t2_at(i, m, b, c);
  }
  return s;
}

// V_base(i,j,k,a,b,c) = t_i^a · ⟨jk||bc⟩
fn V_base(i: u32, j: u32, k: u32, a: u32, b: u32, c: u32) -> f32 {
  return T1[i * d.NVIRT + a] * eri_at(j, k, b + d.NOCC, c + d.NOCC);
}

// P(i/jk) P(a/bc) signed-9-perm sum of fn.
// (i,j,k) → +(i,j,k) −(j,i,k) −(k,j,i)
// (a,b,c) → +(a,b,c) −(b,a,c) −(c,b,a)
fn perm9_W(i: u32, j: u32, k: u32, a: u32, b: u32, c: u32) -> f32 {
  return W_base(i, j, k, a, b, c) - W_base(i, j, k, b, a, c) - W_base(i, j, k, c, b, a)
       - W_base(j, i, k, a, b, c) + W_base(j, i, k, b, a, c) + W_base(j, i, k, c, b, a)
       - W_base(k, j, i, a, b, c) + W_base(k, j, i, b, a, c) + W_base(k, j, i, c, b, a);
}

fn perm9_V(i: u32, j: u32, k: u32, a: u32, b: u32, c: u32) -> f32 {
  return V_base(i, j, k, a, b, c) - V_base(i, j, k, b, a, c) - V_base(i, j, k, c, b, a)
       - V_base(j, i, k, a, b, c) + V_base(j, i, k, b, a, c) + V_base(j, i, k, c, b, a)
       - V_base(k, j, i, a, b, c) + V_base(k, j, i, b, a, c) + V_base(k, j, i, c, b, a);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let triple_idx = gid.x;
  let NOCC2 = d.NOCC * d.NOCC;
  let total_triples = d.NOCC * NOCC2;
  if (triple_idx >= total_triples) { return; }

  // Decode (i, j, k) from triple_idx.
  let i = triple_idx / NOCC2;
  let jk = triple_idx - i * NOCC2;
  let j = jk / d.NOCC;
  let k = jk - j * d.NOCC;

  // Frozen-core skip.
  if (i < d.nFrozenSO || j < d.nFrozenSO || k < d.nFrozenSO) {
    partial_sums[triple_idx] = 0.0;
    return;
  }

  let ei = eps[i];
  let ej = eps[j];
  let ek = eps[k];

  var E_local: f32 = 0.0;
  for (var a: u32 = 0u; a < d.NVIRT; a = a + 1u) {
    let ea = eps[a + d.NOCC];
    for (var b: u32 = 0u; b < d.NVIRT; b = b + 1u) {
      let eb = eps[b + d.NOCC];
      for (var c: u32 = 0u; c < d.NVIRT; c = c + 1u) {
        let ec = eps[c + d.NOCC];
        let D = ei + ej + ek - ea - eb - ec;
        if (abs(D) < 1e-14) { continue; }
        let W = perm9_W(i, j, k, a, b, c);
        let V = perm9_V(i, j, k, a, b, c);
        E_local = E_local + (W * W / 36.0 + W * V / 4.0) / D;
      }
    }
  }
  partial_sums[triple_idx] = E_local;
}
