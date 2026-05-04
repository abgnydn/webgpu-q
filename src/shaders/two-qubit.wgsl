// ─────────────────────────────────────────────────────────────
// controlled-U (two-qubit) gate kernel
//
// Applies a 2×2 complex matrix M to the (target) qubit of every
// basis state that has (control) qubit = 1.
//
// Thread layout: N states total. Each thread inspects one state
// index i. If i has control bit = 1 AND target bit = 0, then i
// and j = i | (1<<target) are the pair we touch. Otherwise the
// thread exits — N/2 threads actually do work, but the scheduling
// is straight-line and the branch is coherent within a warp for
// contiguous control/target bits.
// ─────────────────────────────────────────────────────────────

struct Params {
  m00: vec2<f32>,
  m01: vec2<f32>,
  m10: vec2<f32>,
  m11: vec2<f32>,
  control: u32,
  tgt: u32,
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
  let i = gid.y * params.stride_inv + gid.x;
  let n_states = 1u << params.n_qubits;
  if (i >= n_states) { return; }

  let c_bit = (i >> params.control) & 1u;
  let t_bit = (i >> params.tgt)  & 1u;
  if (c_bit != 1u || t_bit != 0u) { return; }

  let j = i | (1u << params.tgt);

  let a = psi[i];
  let b = psi[j];

  psi[i] = cmul(params.m00, a) + cmul(params.m01, b);
  psi[j] = cmul(params.m10, a) + cmul(params.m11, b);
}
