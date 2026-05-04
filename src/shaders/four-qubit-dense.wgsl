// ─────────────────────────────────────────────────────────────
// Dense 16×16 four-qubit gate kernel (L3 Tier D).
//
// Applies an arbitrary 16×16 complex unitary U to a qubit
// quadruple (qLo, qMid1, qMid2, qHi) with strict ordering
// qLo < qMid1 < qMid2 < qHi. One dispatch reads/writes the full
// statevector, with each thread processing a 16-amp block.
//
// Basis ordering inside the 16-vector: index j ∈ {0..15} encodes
// (s_qHi, s_qMid2, s_qMid1, s_qLo) as
//   j = 8·s_qHi + 4·s_qMid2 + 2·s_qMid1 + s_qLo.
// Matrix is row-major: U[r*16 + c] is the (r, c) entry.
//
// Uniform packing: 256 complex entries are packed into 128
// vec4<f32>, each vec4 holding two complex pairs (xy, zw).
//
// Thread layout: one thread per amplitude block (16 amps each).
// N_blocks = 2^(n − 4). Tier D widens the tile further than
// Tier C: a 7-op cascade (4 singles + 3 cascading CNOTs) folds
// into a single 16×16 dispatch — 7× dispatch reduction per tile
// per layer.
// ─────────────────────────────────────────────────────────────

struct Params {
  // 256 complex entries, packed two-per-vec4: pair k holds U[2k] in xy
  // and U[2k+1] in zw. flat = r*16+c, pair = flat>>1, lane = flat&1.
  matrix: array<vec4<f32>, 128>,
  qLo: u32,
  qMid1: u32,
  qMid2: u32,
  qHi: u32,
  n_qubits: u32,
  stride_inv: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> psi: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn get_m(flat: u32) -> vec2<f32> {
  let v = params.matrix[flat >> 1u];
  return select(v.xy, v.zw, (flat & 1u) == 1u);
}

@compute @workgroup_size(256)
fn apply(@builtin(global_invocation_id) gid: vec3<u32>) {
  let block = gid.y * params.stride_inv + gid.x;
  let n_blocks = 1u << (params.n_qubits - 4u);
  if (block >= n_blocks) { return; }

  // Reconstruct base index by inserting zero bits at qLo < qMid1 < qMid2 < qHi.
  let mLo = (1u << params.qLo) - 1u;
  let bLo = (block & mLo) | ((block & ~mLo) << 1u);
  let mMid1 = (1u << params.qMid1) - 1u;
  let bMid1 = (bLo & mMid1) | ((bLo & ~mMid1) << 1u);
  let mMid2 = (1u << params.qMid2) - 1u;
  let bMid2 = (bMid1 & mMid2) | ((bMid1 & ~mMid2) << 1u);
  let mHi = (1u << params.qHi) - 1u;
  let i_base = (bMid2 & mHi) | ((bMid2 & ~mHi) << 1u);

  let bitLo = 1u << params.qLo;
  let bitMid1 = 1u << params.qMid1;
  let bitMid2 = 1u << params.qMid2;
  let bitHi = 1u << params.qHi;

  // Read 16 amps; index encoding: j = 8*s_qHi + 4*s_qMid2 + 2*s_qMid1 + s_qLo.
  var amps: array<vec2<f32>, 16>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let idx = i_base
      | (select(0u, bitLo, (i & 1u) != 0u))
      | (select(0u, bitMid1, (i & 2u) != 0u))
      | (select(0u, bitMid2, (i & 4u) != 0u))
      | (select(0u, bitHi, (i & 8u) != 0u));
    amps[i] = psi[idx];
  }

  // Apply 16×16 in registers: out[r] = Σ_c U[r,c] · amps[c]. We compute
  // every output then write — no aliasing since reads completed above.
  var out: array<vec2<f32>, 16>;
  for (var r: u32 = 0u; r < 16u; r = r + 1u) {
    var s = vec2<f32>(0.0, 0.0);
    for (var c: u32 = 0u; c < 16u; c = c + 1u) {
      s = s + cmul(get_m(r * 16u + c), amps[c]);
    }
    out[r] = s;
  }
  for (var r: u32 = 0u; r < 16u; r = r + 1u) {
    let idx = i_base
      | (select(0u, bitLo, (r & 1u) != 0u))
      | (select(0u, bitMid1, (r & 2u) != 0u))
      | (select(0u, bitMid2, (r & 4u) != 0u))
      | (select(0u, bitHi, (r & 8u) != 0u));
    psi[idx] = out[r];
  }
}
