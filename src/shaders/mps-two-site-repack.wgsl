// ─────────────────────────────────────────────────────────────
// mps-two-site-repack.wgsl — given the truncation result from
// the SVD step, write the new T_q and T_{q+1} tensor buffers.
//
// Input: U (chiL·2 × n), Σ-scaled-and-permuted Vh rows packed
// into a buffer with k rows × (2·χR) cols.
//
//   T_q'[l·2 + s1, c] = U[l·2 + s1, perm[c]]   for c < kKeep
//   T_{q+1}'[j·2 + s2, r] = (σ[perm[j]] / renorm) · Vh[perm[j], s2·χR + r]
//
// Both shapes change: T_q' has chiR := kKeep, T_{q+1}' has
// chiL := kKeep.
//
// Two entry points: copy_u (for T_q) and scale_vh (for T_{q+1}).
// Both single-pass, one thread per output element.
// ─────────────────────────────────────────────────────────────

struct CopyParams {
  chiLOut:  u32,    // chiL · 2 of T_q' (= rows of U)
  kKeep:    u32,    // new chiR of T_q'
  uCols:    u32,    // n (cols of U)
  _pad:     u32,
};

@group(0) @binding(0) var<storage, read>       U:    array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       perm: array<u32>;
@group(0) @binding(2) var<storage, read_write> Tq:   array<vec2<f32>>;
@group(0) @binding(3) var<uniform>             cparams: CopyParams;

@compute @workgroup_size(64)
fn copy_u(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  let total = cparams.chiLOut * cparams.kKeep;
  if (tid >= total) { return; }
  let row = tid / cparams.kKeep;
  let col = tid % cparams.kKeep;
  let src = perm[col];
  Tq[row * cparams.kKeep + col] = U[row * cparams.uCols + src];
}

struct ScaleParams {
  kKeep:    u32,    // new chiL · 2 of T_{q+1}' (= 2 · kKeep)
  chiROut:  u32,    // chiR of T_{q+1}'
  vhCols:   u32,    // 2 · chiR (cols of Vh)
  _pad:     u32,
};

@group(0) @binding(0) var<storage, read>       Vh:     array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       sigma:  array<f32>;       // length kKeep, already permuted + renormalized
@group(0) @binding(2) var<storage, read>       perm2:  array<u32>;
@group(0) @binding(3) var<storage, read_write> Tj:     array<vec2<f32>>;
@group(0) @binding(4) var<uniform>             sparams: ScaleParams;

@compute @workgroup_size(64)
fn scale_vh(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  let kKeep = sparams.kKeep;
  let chiR  = sparams.chiROut;
  let total = kKeep * 2u * chiR;
  if (tid >= total) { return; }
  // Decompose tid = j · (2·chiR) + s2 · chiR + r
  let kw = 2u * chiR;
  let j = tid / kw;
  let withinK = tid - j * kw;
  let s2 = withinK / chiR;
  let r = withinK - s2 * chiR;
  let srcRow = perm2[j];
  let v = Vh[srcRow * sparams.vhCols + s2 * chiR + r];
  let s = sigma[j];
  Tj[(j * 2u + s2) * chiR + r] = vec2<f32>(s * v.x, s * v.y);
}
