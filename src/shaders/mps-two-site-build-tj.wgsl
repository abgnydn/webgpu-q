// ─────────────────────────────────────────────────────────────
// mps-two-site-build-tj.wgsl — build T_{q+1}' from V (the SVD's
// right singular vectors) and the singular values.
//
// MPS convention (Phase 5+, no host-side renorm):
//   T_{q+1}'[j · 2 + s2, r] = σ[perm[j]] · Vh[perm[j], s2·χR + r]
//
// where Vh = conj(V^T), so Vh[i, c] = conj(V[c, i]).
//
// Phase 5+ change: we read σ via perm directly from the raw
// `sigma` storage buffer (length n), gathering inline. This
// removes the host-side σ-scaled-array construction that used
// to require a CPU readback per gate.
//
// One thread per output cell.
// ─────────────────────────────────────────────────────────────

struct Params {
  kKeep:        u32,
  chiROut:      u32,    // χ_R of T_{q+1}'
  vRowCols:     u32,    // n = cols of V (V is n×n stored row-major)
  _pad:         u32,
};

@group(0) @binding(0) var<storage, read>       V:     array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       sigma: array<f32>;      // length n (raw)
@group(0) @binding(2) var<storage, read>       perm:  array<u32>;
@group(0) @binding(3) var<storage, read_write> Tj:    array<vec2<f32>>;
@group(0) @binding(4) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn build_tj(@builtin(global_invocation_id) gid: vec3<u32>) {
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
  let v = V[(s2 * chiR + r) * params.vRowCols + srcRow];   // V[s2·χR + r, perm[j]]
  let s = sigma[srcRow];
  // Conjugate as part of forming Vh from V.
  Tj[(j * 2u + s2) * chiR + r] = vec2<f32>(s * v.x, -s * v.y);
}
