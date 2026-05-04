// ─────────────────────────────────────────────────────────────
// mps-two-site-extract-u.wgsl — write a permuted, normalized
// column slice of A_post into T_q'.
//
// After Jacobi SVD, A_post has orthogonal columns:
//   A_post[r, c] = σ[c] · U[r, c]
// We want T_q'[r, c'] = U[r, perm[c']] for c' < kKeep, which is
//   T_q'[r, c'] = A_post[r, perm[c']] / σ[perm[c']]
//
// One thread per output cell.
// ─────────────────────────────────────────────────────────────

struct Params {
  rowsOut:  u32,    // chiL · 2 (= rows of T_q' = rows of A_post)
  kKeep:    u32,    // cols of T_q' (= chiR of T_q')
  uCols:    u32,    // n (cols of A_post)
  _pad:     u32,
};

@group(0) @binding(0) var<storage, read>       A:     array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       sigma: array<f32>;
@group(0) @binding(2) var<storage, read>       perm:  array<u32>;
@group(0) @binding(3) var<storage, read_write> Tq:    array<vec2<f32>>;
@group(0) @binding(4) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn extract_u(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  let total = params.rowsOut * params.kKeep;
  if (tid >= total) { return; }
  let row = tid / params.kKeep;
  let col = tid % params.kKeep;
  let src = perm[col];
  let s = sigma[src];
  let inv = select(0.0, 1.0 / s, s > 1e-12);
  let v = A[row * params.uCols + src];
  Tq[row * params.kKeep + col] = vec2<f32>(v.x * inv, v.y * inv);
}
