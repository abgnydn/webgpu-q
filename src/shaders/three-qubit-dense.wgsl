// ─────────────────────────────────────────────────────────────
// Dense 8×8 three-qubit gate kernel (L3 Tier C).
//
// Applies an arbitrary 8×8 complex unitary U to a qubit triple
// (qLo, qMid, qHi), qLo < qMid < qHi. One dispatch reads/writes
// the full statevector, with each thread processing an 8-amp
// block.
//
// Basis ordering inside the 8-vector: index j ∈ {0..7} encodes
// (s_qHi, s_qMid, s_qLo) as j = 4·s_qHi + 2·s_qMid + s_qLo.
// Matrix is row-major: U[r*8 + c] is the (r, c) entry — output
// row r, input column c, so |out⟩ = U |in⟩.
//
// Uniform packing: 64 complex entries are packed into 32 vec4<f32>,
// each vec4 holding two complex pairs (xy, zw). This avoids the
// 16-byte uniform-array stride padding that array<vec2<f32>, 64>
// would force.
//
// Thread layout: one thread per amplitude block (8 amps each).
// N_blocks = 2^(n − 3). The win over 4×4 (Tier B) is that wider
// 3-qubit windows can fuse 5 ops (3 singles + 2 CNOTs in a
// cascade) into one dispatch, giving a 5× dispatch reduction
// per tile per layer.
// ─────────────────────────────────────────────────────────────

struct Params {
  // 64 complex entries, packed two-per-vec4: pair k holds U[2k] in xy
  // and U[2k+1] in zw. flat = r*8+c, pair = flat>>1, lane = flat&1.
  matrix: array<vec4<f32>, 32>,
  qLo: u32,
  qMid: u32,
  qHi: u32,
  n_qubits: u32,
  stride_inv: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
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
  let n_blocks = 1u << (params.n_qubits - 3u);
  if (block >= n_blocks) { return; }

  // Reconstruct the base index by inserting zero bits at qLo, qMid, qHi.
  // Caller guarantees qLo < qMid < qHi.
  let mLo = (1u << params.qLo) - 1u;
  let bLo = (block & mLo) | ((block & ~mLo) << 1u);
  let mMid = (1u << params.qMid) - 1u;
  let bMid = (bLo & mMid) | ((bLo & ~mMid) << 1u);
  let mHi = (1u << params.qHi) - 1u;
  let i_base = (bMid & mHi) | ((bMid & ~mHi) << 1u);

  let bitLo = 1u << params.qLo;
  let bitMid = 1u << params.qMid;
  let bitHi = 1u << params.qHi;

  // Read 8 amplitudes; index encoding: j = 4*s_qHi + 2*s_qMid + s_qLo.
  var amps: array<vec2<f32>, 8>;
  amps[0] = psi[i_base];
  amps[1] = psi[i_base | bitLo];
  amps[2] = psi[i_base | bitMid];
  amps[3] = psi[i_base | bitLo | bitMid];
  amps[4] = psi[i_base | bitHi];
  amps[5] = psi[i_base | bitLo | bitHi];
  amps[6] = psi[i_base | bitMid | bitHi];
  amps[7] = psi[i_base | bitLo | bitMid | bitHi];

  // Apply 8×8 in registers: out[r] = Σ_c U[r,c] · amps[c].
  var out: array<vec2<f32>, 8>;
  for (var r: u32 = 0u; r < 8u; r = r + 1u) {
    var s = vec2<f32>(0.0, 0.0);
    for (var c: u32 = 0u; c < 8u; c = c + 1u) {
      s = s + cmul(get_m(r * 8u + c), amps[c]);
    }
    out[r] = s;
  }

  psi[i_base]                          = out[0];
  psi[i_base | bitLo]                  = out[1];
  psi[i_base | bitMid]                 = out[2];
  psi[i_base | bitLo | bitMid]         = out[3];
  psi[i_base | bitHi]                  = out[4];
  psi[i_base | bitLo | bitHi]          = out[5];
  psi[i_base | bitMid | bitHi]         = out[6];
  psi[i_base | bitLo | bitMid | bitHi] = out[7];
}
