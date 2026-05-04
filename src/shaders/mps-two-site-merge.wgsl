// ─────────────────────────────────────────────────────────────
// mps-two-site-merge.wgsl — build the merged matrix
//   M[l·2+s1', s2'·χR + r] = Σ_{s1,s2,b} G[s1'·2+s2', s1·2+s2]
//                                       · T_q[l·2+s1, b]
//                                       · T_{q+1}[b·2+s2, r]
// for a two-site gate G acting on sites (q, q+1).
//
// One thread per (l, r) pair handles the full 4×4 inner gate
// after a per-thread bond contraction. M ends up as a
// (χ_L · 2) × (2 · χ_R) matrix in row-major (re, im) layout —
// the same shape the SVD kernel expects.
// ─────────────────────────────────────────────────────────────

struct Params {
  chiL:    u32,
  chiBond: u32,
  chiR:    u32,
  // Output column stride. Defaults to 2·chiR (tight packing) when caller
  // wants a (chiL·2) × (2·chiR) matrix; can be set to nMax = max(chiL·2,
  // 2·chiR) so the merged data lands in the top-left submatrix of a
  // padded nMax × nMax square that the SVD kernel can consume.
  mStride: u32,
};

@group(0) @binding(0) var<storage, read>       Tq: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       Tj: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read>       G:  array<vec2<f32>>;     // 16 entries (4×4)
@group(0) @binding(3) var<storage, read_write> M:  array<vec2<f32>>;
@group(0) @binding(4) var<uniform>             params: Params;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn merge(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  let total = params.chiL * params.chiR;
  if (tid >= total) { return; }
  let l = tid / params.chiR;
  let r = tid % params.chiR;

  // Pre-contraction: pre[s1*2 + s2] = Σ_b Tq[l*2+s1, b] · Tj[b*2+s2, r].
  var pre: array<vec2<f32>, 4>;
  for (var s1: u32 = 0u; s1 < 2u; s1 = s1 + 1u) {
    for (var s2: u32 = 0u; s2 < 2u; s2 = s2 + 1u) {
      var sum = vec2<f32>(0.0, 0.0);
      for (var b: u32 = 0u; b < params.chiBond; b = b + 1u) {
        let tq = Tq[(l * 2u + s1) * params.chiBond + b];
        let tj = Tj[(b * 2u + s2) * params.chiR + r];
        sum = sum + cmul(tq, tj);
      }
      pre[s1 * 2u + s2] = sum;
    }
  }

  // Post-gate: post[ip] = Σ_i G[ip*4 + i] · pre[i]; write to M.
  for (var ip: u32 = 0u; ip < 4u; ip = ip + 1u) {
    var post = vec2<f32>(0.0, 0.0);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      post = post + cmul(G[ip * 4u + i], pre[i]);
    }
    let s1p = ip / 2u;
    let s2p = ip & 1u;
    let row = l * 2u + s1p;
    let col = s2p * params.chiR + r;
    M[row * params.mStride + col] = post;
  }
}
