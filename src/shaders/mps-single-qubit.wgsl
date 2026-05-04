// ─────────────────────────────────────────────────────────────
// mps-single-qubit.wgsl — single-qubit gate on a GPU-resident
// MPS tensor. The tensor T at site q has shape (χ_L · 2, χ_R)
// stored row-major as `array<vec2<f32>>`. Index layout:
//   T[l · 2 + s, r]   →   buffer offset (l · 2 + s) · χ_R + r
//
// We apply U (2×2 complex) on the physical leg s ∈ {0, 1}:
//   T'[l, s', r] = Σ_s U[s', s] · T[l, s, r]
//
// One thread per (l, r) pair. Each thread reads two elements
// (s = 0 and s = 1), multiplies through U, writes both back.
// Total threads = χ_L · χ_R.
// ─────────────────────────────────────────────────────────────

struct Params {
  chiL:   u32,
  chiR:   u32,
  m00: vec2<f32>, m01: vec2<f32>,
  m10: vec2<f32>, m11: vec2<f32>,
};

@group(0) @binding(0) var<storage, read_write> T: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn single_qubit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  let total = params.chiL * params.chiR;
  if (tid >= total) { return; }
  let l = tid / params.chiR;
  let r = tid % params.chiR;
  let i0 = (l * 2u + 0u) * params.chiR + r;
  let i1 = (l * 2u + 1u) * params.chiR + r;
  let a = T[i0];
  let b = T[i1];
  T[i0] = cmul(params.m00, a) + cmul(params.m01, b);
  T[i1] = cmul(params.m10, a) + cmul(params.m11, b);
}
