// ─────────────────────────────────────────────────────────────
// jacobi-svd-storage.wgsl — Phase 6 v1: storage-mode complex
// Jacobi SVD for n ≤ 64.
//
// Where small/medium/large hold the entire matrix in workgroup
// shared memory (and so cap at n=24/32/48 by storage budget),
// this kernel keeps A and V in global storage and shuttles only
// the 2 active columns of each (p, q) rotation into shared. The
// per-rotation traffic is 2·n·8 B (≈1 KB at n=64) but amortized
// over the rotation's reduction + apply pass.
//
// Workgroup storage budget: 4 columns of n=64 vec2<f32> + a 32-
// element reduction buffer = 4·64·8 + 32·16 = 2.5 KB. Comfortably
// inside the 16 KB spec floor — works on every adapter.
//
// WGSL uniform-control-flow rule forbids `continue` or `if` blocks
// containing barriers. Instead, "skip rotation" cases collapse to
// an identity rotation via `select` on the params — every thread
// runs the same code path, the writeback is always executed.
// ─────────────────────────────────────────────────────────────

const MAX_PAIR_ROWS: u32 = 64u;

struct Params {
  n: u32,
  n_sweeps: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> A: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> V: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> sP:  array<vec2<f32>, 64>;
var<workgroup> sQ:  array<vec2<f32>, 64>;
var<workgroup> sVp: array<vec2<f32>, 64>;
var<workgroup> sVq: array<vec2<f32>, 64>;
var<workgroup> sRed: array<vec4<f32>, 32>;

@compute @workgroup_size(32)
fn jacobi_svd(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let n = params.n;
  let TOL: f32 = 1e-7;

  // Initialize V = I directly in global storage.
  for (var i = tid; i < n * n; i = i + 32u) {
    let row = i / n;
    let col = i % n;
    V[i] = select(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), row == col);
  }
  workgroupBarrier();

  for (var sweep = 0u; sweep < params.n_sweeps; sweep = sweep + 1u) {
    for (var p = 0u; p < n; p = p + 1u) {
      for (var q = p + 1u; q < n; q = q + 1u) {
        // Load cols p, q of A and V into shared.
        for (var i = tid; i < n; i = i + 32u) {
          sP[i]  = A[i * n + p];
          sQ[i]  = A[i * n + q];
          sVp[i] = V[i * n + p];
          sVq[i] = V[i * n + q];
        }
        workgroupBarrier();

        // Reductions: d = Σ_i conj(P[i]) · Q[i], ‖P‖², ‖Q‖²
        var d_re: f32 = 0.0;
        var d_im: f32 = 0.0;
        var app: f32 = 0.0;
        var aqq: f32 = 0.0;
        for (var i = tid; i < n; i = i + 32u) {
          let ap = sP[i];
          let aq = sQ[i];
          d_re = d_re + ap.x * aq.x + ap.y * aq.y;
          d_im = d_im + ap.x * aq.y - ap.y * aq.x;
          app = app + ap.x * ap.x + ap.y * ap.y;
          aqq = aqq + aq.x * aq.x + aq.y * aq.y;
        }
        sRed[tid] = vec4<f32>(d_re, d_im, app, aqq);
        workgroupBarrier();

        var stride: u32 = 16u;
        loop {
          if (stride == 0u) { break; }
          if (tid < stride) { sRed[tid] = sRed[tid] + sRed[tid + stride]; }
          workgroupBarrier();
          stride = stride >> 1u;
        }
        let red = sRed[0];

        // Decide whether the rotation is meaningful. When invalid we
        // fall through to an identity rotation (zr=1, zi=0, cj=1, sj=0)
        // so every thread runs the same code path and the writeback at
        // the bottom is always executed under uniform control flow.
        let mag = sqrt(red.x * red.x + red.y * red.y);
        let normRef = sqrt(max(red.z, red.w));
        let valid = (red.z > 0.0) && (red.w > 0.0) && (mag > TOL * normRef);

        // Phase-align factors (z = e^{−iφ}). Default to z = 1 when invalid.
        let safeMag = select(1.0, mag, valid);
        let zr = select(1.0, red.x / safeMag, valid);
        let zi = select(0.0, -red.y / safeMag, valid);

        // Apply phase-align to col q (real-equivalent rotation:
        // multiply by (zr, zi) interpreted as (cos, sin) of the phase).
        for (var i = tid; i < n; i = i + 32u) {
          let aq = sQ[i];
          sQ[i] = vec2<f32>(aq.x * zr - aq.y * zi, aq.x * zi + aq.y * zr);
          let vq = sVq[i];
          sVq[i] = vec2<f32>(vq.x * zr - vq.y * zi, vq.x * zi + vq.y * zr);
        }
        workgroupBarrier();

        // Real Jacobi rotation params; default to (cj, sj) = (1, 0)
        // (identity) when invalid so app/aqq don't matter.
        let den = red.z - red.w;
        let two_mag_safe = select(1.0, 2.0 * mag, valid);
        let theta = den / two_mag_safe;
        let signTheta = select(-1.0, 1.0, theta >= 0.0);
        let t_raw = signTheta / (abs(theta) + sqrt(theta * theta + 1.0));
        let cj_raw = 1.0 / sqrt(1.0 + t_raw * t_raw);
        let sj_raw = t_raw * cj_raw;
        let cj = select(1.0, cj_raw, valid);
        let sj = select(0.0, sj_raw, valid);

        for (var i = tid; i < n; i = i + 32u) {
          let ap = sP[i];
          let aq = sQ[i];
          sP[i] = vec2<f32>(cj * ap.x - sj * aq.x, cj * ap.y - sj * aq.y);
          sQ[i] = vec2<f32>(sj * ap.x + cj * aq.x, sj * ap.y + cj * aq.y);
          let vp = sVp[i];
          let vq = sVq[i];
          sVp[i] = vec2<f32>(cj * vp.x - sj * vq.x, cj * vp.y - sj * vq.y);
          sVq[i] = vec2<f32>(sj * vp.x + cj * vq.x, sj * vp.y + cj * vq.y);
        }
        workgroupBarrier();

        // Writeback (always).
        for (var i = tid; i < n; i = i + 32u) {
          A[i * n + p] = sP[i];
          A[i * n + q] = sQ[i];
          V[i * n + p] = sVp[i];
          V[i * n + q] = sVq[i];
        }
        workgroupBarrier();
      }
    }
  }
}
