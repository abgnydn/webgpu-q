// ─────────────────────────────────────────────────────────────
// complex-matmul.wgsl — GPU matrix multiplication for complex
// matrices stored as `array<vec2<f32>>` (each vec2 = [re, im]).
//
// Computes C[i, k] = Σ_j A[i, j] · B[j, k] for matrices in
// row-major layout. Caller passes (rows, inner, cols) via the
// uniform; one thread per output element.
//
// Used as the first GPU primitive in the MPS port: tensor
// contractions at every two-site update need this.
// ─────────────────────────────────────────────────────────────

struct Params {
  rows:  u32,    // rows of A (= rows of C)
  inner: u32,    // cols of A = rows of B
  cols:  u32,    // cols of B (= cols of C)
  _pad:  u32,
};

@group(0) @binding(0) var<storage, read> A: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(8, 8)
fn cmatmul(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let k = gid.y;
  if (i >= params.rows || k >= params.cols) { return; }

  var s = vec2<f32>(0.0, 0.0);
  for (var j = 0u; j < params.inner; j = j + 1u) {
    let a = A[i * params.inner + j];
    let b = B[j * params.cols + k];
    s = s + cmul(a, b);
  }
  C[i * params.cols + k] = s;
}
