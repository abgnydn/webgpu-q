// ─────────────────────────────────────────────────────────────
// mps-two-site-build-vh.wgsl — write V^H (perm-sorted) into
// T_{q+1}', without σ scaling.
//
// Right-canonical pipeline companion: T_{q+1}' = V^H reshape, a
// right-canonical isometry (Σ_{s2,r} T_{q+1}[b, s2, r] · conj(T_{q+1}[b', s2, r]) = δ_{bb'}).
//
// MPS convention:
//   T_{q+1}'[j · 2 + s2, r] = Vh[perm[j], s2·χR + r]
//
// where Vh = conj(V^T), so Vh[i, c] = conj(V[c, i]). Substituting:
//   T_{q+1}'[j · 2 + s2, r] = conj(V[s2·χR + r, perm[j]])
//
// One thread per output cell.
// ─────────────────────────────────────────────────────────────

struct Params {
  kKeep:        u32,
  chiROut:      u32,    // χ_R of T_{q+1}'
  vRowCols:     u32,    // n = cols of V
  _pad:         u32,
};

@group(0) @binding(0) var<storage, read>       V:    array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       perm: array<u32>;
@group(0) @binding(2) var<storage, read_write> Tj:   array<vec2<f32>>;
@group(0) @binding(3) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn build_vh(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  let kKeep = params.kKeep;
  let chiR  = params.chiROut;
  let total = kKeep * 2u * chiR;
  if (tid >= total) { return; }
  let kw = 2u * chiR;
  let j = tid / kw;
  let withinK = tid - j * kw;
  let s2 = withinK / chiR;
  let r = withinK - s2 * chiR;
  let srcRow = perm[j];
  let v = V[(s2 * chiR + r) * params.vRowCols + srcRow];
  // Conjugate as part of forming Vh from V.
  Tj[(j * 2u + s2) * chiR + r] = vec2<f32>(v.x, -v.y);
}
