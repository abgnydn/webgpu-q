// ─────────────────────────────────────────────────────────────
// column-norms.wgsl — given an n × n complex matrix A in row-
// major (re, im) layout, compute σ_i = ‖A[:, i]‖. Used after
// the in-place Jacobi SVD: the post-rotation A has orthogonal
// columns whose norms ARE the singular values.
//
// One thread per column. n ≤ 64 in the call sites we use, so a
// single workgroup is plenty.
// ─────────────────────────────────────────────────────────────

struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       A:     array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> sigma: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

@compute @workgroup_size(32)
fn col_norms(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  if (col >= params.n) { return; }
  var s: f32 = 0.0;
  for (var row: u32 = 0u; row < params.n; row = row + 1u) {
    let a = A[row * params.n + col];
    s = s + a.x * a.x + a.y * a.y;
  }
  sigma[col] = sqrt(s);
}
