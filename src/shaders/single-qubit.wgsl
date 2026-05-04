// ─────────────────────────────────────────────────────────────
// single-qubit gate kernel
//
// For qubit q, the statevector splits into N/2 pairs:
//   (|...0...⟩_q, |...1...⟩_q)
// where all other qubit bits match. Each thread handles one
// pair and applies the 2×2 complex matrix M.
//
// Indexing: thread t in [0, N/2) maps to pair (i, j) where
//   mask_low  = (1 << q) - 1
//   i = (t & ~mask_low) << 1 | (t & mask_low)   // bit q of i = 0
//   j = i | (1 << q)                            // bit q of j = 1
// ─────────────────────────────────────────────────────────────

struct Params {
  m00: vec2<f32>,   // row 0 col 0 (real, imag)
  m01: vec2<f32>,
  m10: vec2<f32>,
  m11: vec2<f32>,
  qubit: u32,
  n_qubits: u32,
  // For 2D workgroup dispatch: thread index reconstructed as
  //   pair_idx = gid.y * stride_inv + gid.x
  // where stride_inv = dispatch_wg_x * WG_SIZE. At N ≥ 21 we exceed
  // maxComputeWorkgroupsPerDimension (65535) in a single dim.
  stride_inv: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> psi: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256)
fn apply(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair_idx = gid.y * params.stride_inv + gid.x;
  let n_pairs = 1u << (params.n_qubits - 1u);
  if (pair_idx >= n_pairs) { return; }

  let q = params.qubit;
  let mask_low = (1u << q) - 1u;
  let i = ((pair_idx & ~mask_low) << 1u) | (pair_idx & mask_low);
  let j = i | (1u << q);

  let a = psi[i];
  let b = psi[j];

  psi[i] = cmul(params.m00, a) + cmul(params.m01, b);
  psi[j] = cmul(params.m10, a) + cmul(params.m11, b);
}
