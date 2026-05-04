// ─────────────────────────────────────────────────────────────
// mps-two-site-extract-uS.wgsl — write A_perm (= U·σ post-Jacobi)
// directly into T_q, without dividing by σ.
//
// Companion to extract-u, used by the right-canonical pipeline:
// the residual (U·σ) flows LEFT into T_q while T_{q+1} becomes
// the right-canonical isometry V^H. extract-u does the opposite
// (T_q ← U, T_{q+1} ← σ·V^H).
//
// One thread per output cell.
// ─────────────────────────────────────────────────────────────

struct Params {
  rowsOut:  u32,    // chiL · 2 (= rows of T_q' = rows of A)
  kKeep:    u32,    // cols of T_q'
  uCols:    u32,    // n (cols of A)
  _pad:     u32,
};

@group(0) @binding(0) var<storage, read>       A:    array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       perm: array<u32>;
@group(0) @binding(2) var<storage, read_write> Tq:   array<vec2<f32>>;
@group(0) @binding(3) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn extract_uS(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  let total = params.rowsOut * params.kKeep;
  if (tid >= total) { return; }
  let row = tid / params.kKeep;
  let col = tid % params.kKeep;
  let src = perm[col];
  // A_post[row, src] = σ[src] · U[row, src] — write directly, no normalization.
  Tq[row * params.kKeep + col] = A[row * params.uCols + src];
}
