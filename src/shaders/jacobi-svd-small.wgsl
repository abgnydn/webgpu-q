// ─────────────────────────────────────────────────────────────
// jacobi-svd-small.wgsl — single-workgroup one-sided complex
// Jacobi SVD for square matrices with n ≤ 32.
//
// Algorithm: load A and identity V into workgroup-shared memory,
// run cyclic Jacobi sweeps entirely in shared memory, write A
// (now with orthogonal columns) and V back. The host extracts
// σ_i = ‖A[:,i]‖ and U[:,i] = A[:,i] / σ_i.
//
// One thread per row. Per (p, q) pair: parallel reduction on
// (Re⟨p,q⟩, Im⟨p,q⟩, ‖p‖², ‖q‖²), thread 0 broadcasts via the
// reduction buffer, every thread applies phase + Jacobi rotation
// to its own row of cols p and q.
//
// Storage budget: sA = sV = 24×24 vec2<f32> = 4.6 KB each, plus
// 32 vec4 reduction buf = 0.5 KB. Total ≈ 9.7 KB workgroup mem.
// Fits the WebGPU 16 KB required-minimum with headroom. (32×32
// would land at 16.5 KB and overflow the spec floor — Phase 1C
// will tile larger matrices across multiple workgroups.)
// ─────────────────────────────────────────────────────────────

const MAX_N: u32 = 24u;

struct Params {
  n: u32,
  n_sweeps: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> A: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> V: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> sA: array<vec2<f32>, 576>;    // up to 24×24
var<workgroup> sV: array<vec2<f32>, 576>;
var<workgroup> sRed: array<vec4<f32>, 32>;

@compute @workgroup_size(32)
fn jacobi_svd(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let n = params.n;
  let TOL: f32 = 1e-7;

  // Load A from global → shared; init V = I.
  for (var i = tid; i < n * n; i = i + 32u) {
    sA[i] = A[i];
    let row = i / n;
    let col = i % n;
    sV[i] = select(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), row == col);
  }
  workgroupBarrier();

  for (var sweep = 0u; sweep < params.n_sweeps; sweep = sweep + 1u) {
    for (var p = 0u; p < n; p = p + 1u) {
      for (var q = p + 1u; q < n; q = q + 1u) {
        // Per-thread partial: this thread's row contribution to
        //   d = Σ_i conj(A[i, p]) · A[i, q]
        //   app = Σ_i |A[i, p]|², aqq = Σ_i |A[i, q]|²
        var partial = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (tid < n) {
          let ap = sA[tid * n + p];
          let aq = sA[tid * n + q];
          // conj(ap) · aq = (ap.x − i ap.y)(aq.x + i aq.y)
          //              = (ap.x aq.x + ap.y aq.y) + i (ap.x aq.y − ap.y aq.x)
          let d_re = ap.x * aq.x + ap.y * aq.y;
          let d_im = ap.x * aq.y - ap.y * aq.x;
          let app_l = ap.x * ap.x + ap.y * ap.y;
          let aqq_l = aq.x * aq.x + aq.y * aq.y;
          partial = vec4<f32>(d_re, d_im, app_l, aqq_l);
        }
        sRed[tid] = partial;
        workgroupBarrier();

        // Tree reduction (32 → 1).
        if (tid < 16u) { sRed[tid] = sRed[tid] + sRed[tid + 16u]; }
        workgroupBarrier();
        if (tid < 8u)  { sRed[tid] = sRed[tid] + sRed[tid + 8u]; }
        workgroupBarrier();
        if (tid < 4u)  { sRed[tid] = sRed[tid] + sRed[tid + 4u]; }
        workgroupBarrier();
        if (tid < 2u)  { sRed[tid] = sRed[tid] + sRed[tid + 2u]; }
        workgroupBarrier();
        if (tid < 1u)  { sRed[tid] = sRed[tid] + sRed[tid + 1u]; }
        workgroupBarrier();

        let d_re = sRed[0].x;
        let d_im = sRed[0].y;
        let app  = sRed[0].z;
        let aqq  = sRed[0].w;
        let mag  = sqrt(d_re * d_re + d_im * d_im);

        // Skip rotation if either column is zero or the cross overlap
        // is at numerical noise relative to column norms.
        let do_rot = (app > 0.0) && (aqq > 0.0) && (mag > TOL * sqrt(max(app, aqq)));

        if (do_rot) {
          // ── 1. Phase-align col q so that ⟨p, q⟩ becomes real positive ──
          let zr = d_re / mag;
          let zi = d_im / mag;
          // Multiply col q by conj(z) = (zr, −zi).
          //   (zr − i zi)(aq.x + i aq.y) = (zr·aq.x + zi·aq.y) + i (zr·aq.y − zi·aq.x)
          if (tid < n) {
            let aq = sA[tid * n + q];
            sA[tid * n + q] = vec2<f32>(zr * aq.x + zi * aq.y,
                                        zr * aq.y - zi * aq.x);
            let vq = sV[tid * n + q];
            sV[tid * n + q] = vec2<f32>(zr * vq.x + zi * vq.y,
                                        zr * vq.y - zi * vq.x);
          }

          // ── 2. Real Jacobi rotation on (p, q) ──
          let tau = (aqq - app) / (2.0 * mag);
          var t_j: f32;
          if (tau >= 0.0) {
            t_j = 1.0 / (tau + sqrt(1.0 + tau * tau));
          } else {
            t_j = -1.0 / (-tau + sqrt(1.0 + tau * tau));
          }
          let c = 1.0 / sqrt(1.0 + t_j * t_j);
          let s_j = t_j * c;

          if (tid < n) {
            let ap = sA[tid * n + p];
            let aq = sA[tid * n + q];
            sA[tid * n + p] = vec2<f32>(c * ap.x - s_j * aq.x,
                                        c * ap.y - s_j * aq.y);
            sA[tid * n + q] = vec2<f32>(s_j * ap.x + c * aq.x,
                                        s_j * ap.y + c * aq.y);
            let vp = sV[tid * n + p];
            let vq = sV[tid * n + q];
            sV[tid * n + p] = vec2<f32>(c * vp.x - s_j * vq.x,
                                        c * vp.y - s_j * vq.y);
            sV[tid * n + q] = vec2<f32>(s_j * vp.x + c * vq.x,
                                        s_j * vp.y + c * vq.y);
          }
        }
        // No inter-pair barrier needed: each thread only mutated cells
        // in its own row (different addresses across threads), and the
        // next iteration's first action is a global write to sRed[tid]
        // (also per-thread). Reduction barriers below handle the rest.
      }
    }
  }

  workgroupBarrier();

  // Write back.
  for (var i = tid; i < n * n; i = i + 32u) {
    A[i] = sA[i];
    V[i] = sV[i];
  }
}
