// ─────────────────────────────────────────────────────────────
// sigma-sort.wgsl — single-workgroup insertion sort over σ +
// truncation renormalization scalar.
//
// After SVD + col-norms the merged matrix has σ in arbitrary
// order (Jacobi rotates rather than orders). Phase 5+ wants to
// stay GPU-resident across the whole two-site update — eliminating
// the host-side sort + readback that was the per-gate sync point.
//
// Phase A (truncation renorm): when kKeep < n, dropping σ_{kKeep..}
// shrinks the merged matrix's Frobenius norm and the chain leaks
// probability mass. Phase 4b/5 was correct for no-truncation
// (unitary gates preserve ‖M‖²_F automatically), but kKeep < n
// loses (sumAll − sumKept). To restore the *pre-SVD* Frobenius
// norm — which equals ‖ψ‖² when the chain is canonical and
// otherwise equals whatever the GPU MPS was carrying — we want
//   σ_kept → σ_kept · √(sumAll / sumKept)
// applied to whichever child tensor carries σ (T_{q+1} left-
// canonical, T_q right-canonical). When kKeep = n, sumKept =
// sumAll and the factor is 1.0 — the no-truncation hot path
// stays a one-multiply no-op.
//
// Output:
//   perm[j]   = source column index of the j-th-largest σ
//   renorm[0] = √(sumAll / sumKept)   (1.0 if sumKept ≤ 0)
//
// Single thread, insertion sort, n ≤ 64. ~2000 ops at peak —
// cheap compared to the SVD that just ran.
// ─────────────────────────────────────────────────────────────

struct Params {
  n: u32,
  kKeep: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       sigma:  array<f32>;
@group(0) @binding(1) var<storage, read_write> perm:   array<u32>;
@group(0) @binding(2) var<uniform>             params: Params;
@group(0) @binding(3) var<storage, read_write> renorm: array<f32>;

@compute @workgroup_size(1)
fn sort(@builtin(local_invocation_id) lid: vec3<u32>) {
  let n = params.n;
  // Initialize perm = identity.
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    perm[i] = i;
  }
  // Insertion sort, descending by sigma.
  for (var i: u32 = 1u; i < n; i = i + 1u) {
    let key = perm[i];
    let kv = sigma[key];
    var j: i32 = i32(i) - 1;
    loop {
      if (j < 0) { break; }
      if (sigma[perm[u32(j)]] >= kv) { break; }
      perm[u32(j) + 1u] = perm[u32(j)];
      j = j - 1;
    }
    perm[u32(j + 1)] = key;
  }
  // Truncation renorm: √(sumAll / sumKept). For kKeep = n this is 1.0
  // (no-truncation hot path). For kKeep < n it restores the merged
  // matrix's pre-truncation Frobenius norm, preserving whatever state
  // norm the chain was carrying before the gate. kKeep is clamped to
  // n by the host but we belt-and-brace here.
  let kCap = min(params.kKeep, n);
  var sumKept: f32 = 0.0;
  var sumAll: f32 = 0.0;
  for (var k: u32 = 0u; k < n; k = k + 1u) {
    let s = sigma[perm[k]];
    let s2 = s * s;
    sumAll = sumAll + s2;
    if (k < kCap) {
      sumKept = sumKept + s2;
    }
  }
  if (sumKept > 0.0) {
    renorm[0] = sqrt(sumAll / sumKept);
  } else {
    renorm[0] = 1.0;
  }
  // Debug telemetry — exposed for spot-checks via readback.
  renorm[1] = sumKept;
  renorm[2] = f32(params.kKeep);
  renorm[3] = f32(params.n);
}
