// ─────────────────────────────────────────────────────────────
// sigma-sort.wgsl — single-workgroup insertion sort over σ.
//
// After SVD + col-norms the merged matrix has σ in arbitrary
// order (Jacobi rotates rather than orders). Phase 5+ wants to
// stay GPU-resident across the whole two-site update — eliminating
// the host-side sort + readback that was the per-gate sync point.
//
// Output: perm[j] = source column index of the j-th-largest σ.
// extractU and buildTj already accept a perm buffer, so this
// drops in without any kernel-side changes downstream.
//
// Single thread, insertion sort, n ≤ 32. ~1000 ops at peak —
// cheap compared to the SVD that just ran.
// ─────────────────────────────────────────────────────────────

struct Params {
  n: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       sigma: array<f32>;
@group(0) @binding(1) var<storage, read_write> perm:  array<u32>;
@group(0) @binding(2) var<uniform>             params: Params;

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
}
