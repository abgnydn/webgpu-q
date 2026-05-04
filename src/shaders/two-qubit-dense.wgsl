// ─────────────────────────────────────────────────────────────
// Dense 4×4 two-qubit gate kernel.
//
// Applies an arbitrary 4×4 complex unitary U to a qubit pair
// (qLo, qHi) — used for L3 brick-wall fusion where one layer's
// (single_q ⊗ single_{q+1}) · CNOT(q, q+1) is collapsed into a
// single 4×4 matrix on the CPU side and dispatched once.
//
// Basis ordering inside the 4-vector: index j ∈ {0,1,2,3}
// encodes (s_qHi, s_qLo) as j = 2·s_qHi + s_qLo. The matrix is
// row-major: U[i*4 + j] is the (i, j) entry, with i = output
// state, j = input state. So |out⟩ = U |in⟩.
//
// Thread layout: one thread per amplitude block (4 amps each).
// N_blocks = 2^(n − 2). Each thread reads 4 amps, multiplies by
// the 4×4 matrix in registers, writes 4 amps. Memory traffic
// matches the unfused two-qubit kernel; the win comes from
// folding 2 single-qubit gates + 1 CNOT into a single dispatch
// (3× dispatch reduction per pair per layer).
// ─────────────────────────────────────────────────────────────

struct Params {
  // Matrix entries m_{ij} for i, j ∈ {0..3}. Row-major.
  m00: vec2<f32>, m01: vec2<f32>, m02: vec2<f32>, m03: vec2<f32>,
  m10: vec2<f32>, m11: vec2<f32>, m12: vec2<f32>, m13: vec2<f32>,
  m20: vec2<f32>, m21: vec2<f32>, m22: vec2<f32>, m23: vec2<f32>,
  m30: vec2<f32>, m31: vec2<f32>, m32: vec2<f32>, m33: vec2<f32>,
  qLo: u32,
  qHi: u32,
  n_qubits: u32,
  // 2D dispatch stride: thread index = gid.y * stride_inv + gid.x.
  stride_inv: u32,
};

@group(0) @binding(0) var<storage, read_write> psi: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256)
fn apply(@builtin(global_invocation_id) gid: vec3<u32>) {
  let block = gid.y * params.stride_inv + gid.x;
  let n_blocks = 1u << (params.n_qubits - 2u);
  if (block >= n_blocks) { return; }

  // Reconstruct base index by inserting zero bits at positions
  // qLo and qHi. Caller guarantees qLo < qHi.
  let mLo = (1u << params.qLo) - 1u;
  let bLo = (block & mLo) | ((block & ~mLo) << 1u);
  let mHi = (1u << params.qHi) - 1u;
  let i_base = (bLo & mHi) | ((bLo & ~mHi) << 1u);

  let i00 = i_base;
  let i01 = i_base | (1u << params.qLo);
  let i10 = i_base | (1u << params.qHi);
  let i11 = i01 | (1u << params.qHi);

  let a00 = psi[i00];
  let a01 = psi[i01];
  let a10 = psi[i10];
  let a11 = psi[i11];

  psi[i00] = cmul(params.m00, a00) + cmul(params.m01, a01)
           + cmul(params.m02, a10) + cmul(params.m03, a11);
  psi[i01] = cmul(params.m10, a00) + cmul(params.m11, a01)
           + cmul(params.m12, a10) + cmul(params.m13, a11);
  psi[i10] = cmul(params.m20, a00) + cmul(params.m21, a01)
           + cmul(params.m22, a10) + cmul(params.m23, a11);
  psi[i11] = cmul(params.m30, a00) + cmul(params.m31, a01)
           + cmul(params.m32, a10) + cmul(params.m33, a11);
}
