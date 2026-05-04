// ─────────────────────────────────────────────────────────────
// jacobi-svd-large.wgsl — Phase 6 v0: same one-sided complex
// Jacobi SVD as small/medium, but sized for n ≤ 48.
//
// Total workgroup storage: 2 · 48² · 8 + 32 · 16 = 37 KB. Above
// the WebGPU 16 KB spec floor and the typical 32 KB Apple Metal-3
// envelope; some desktop / mobile drivers expose 48–64 KB and
// will accept this kernel. Pipeline creation falls back to the
// medium (n ≤ 32) or small (n ≤ 24) kernel when the adapter
// can't host this one.
//
// Lifting n ≤ 48 means χ ≤ 24 in MPS two-site updates (since the
// merged matrix is 2χ × 2χ for square bonds), expanding the
// regime where GPU MPS stays GPU-resident end-to-end.
// ─────────────────────────────────────────────────────────────

const MAX_N: u32 = 48u;

struct Params {
  n: u32,
  n_sweeps: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> A: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> V: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> sA: array<vec2<f32>, 2304>;   // up to 48×48
var<workgroup> sV: array<vec2<f32>, 2304>;
var<workgroup> sRed: array<vec4<f32>, 32>;

@compute @workgroup_size(32)
fn jacobi_svd(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let n = params.n;
  let TOL: f32 = 1e-7;

  // Load A → shared; init V = I.
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
        // Per-thread partial reduction over rows for d = Σ_i conj(A[i,p]) · A[i,q]
        // and ‖p‖², ‖q‖².
        var d_re: f32 = 0.0;
        var d_im: f32 = 0.0;
        var app: f32 = 0.0;
        var aqq: f32 = 0.0;
        for (var i = tid; i < n; i = i + 32u) {
          let ap = sA[i * n + p];
          let aq = sA[i * n + q];
          // conj(ap) · aq
          d_re = d_re + ap.x * aq.x + ap.y * aq.y;
          d_im = d_im + ap.x * aq.y - ap.y * aq.x;
          app = app + ap.x * ap.x + ap.y * ap.y;
          aqq = aqq + aq.x * aq.x + aq.y * aq.y;
        }
        sRed[tid] = vec4<f32>(d_re, d_im, app, aqq);
        workgroupBarrier();

        // Tree-reduce.
        var stride: u32 = 16u;
        loop {
          if (stride == 0u) { break; }
          if (tid < stride) {
            sRed[tid] = sRed[tid] + sRed[tid + stride];
          }
          workgroupBarrier();
          stride = stride >> 1u;
        }
        let red = sRed[0];

        if (!(red.z > 0.0)) { continue; }
        if (!(red.w > 0.0)) { continue; }

        let mag = sqrt(red.x * red.x + red.y * red.y);
        let normRef = sqrt(max(red.z, red.w));
        if (mag <= TOL * normRef) { continue; }

        // Phase-align column q so ⟨p, q⟩ becomes real.
        let zr = red.x / mag;
        let zi = -red.y / mag;
        if (zr != zr || zi != zi) { continue; }
        for (var i = tid; i < n; i = i + 32u) {
          let aq = sA[i * n + q];
          sA[i * n + q] = vec2<f32>(aq.x * zr - aq.y * zi, aq.x * zi + aq.y * zr);
          let vq = sV[i * n + q];
          sV[i * n + q] = vec2<f32>(vq.x * zr - vq.y * zi, vq.x * zi + vq.y * zr);
        }
        workgroupBarrier();

        // Real Jacobi rotation: tan(2θ) = 2 mag / (app - aqq).
        let den = red.z - red.w;
        let two_mag = 2.0 * mag;
        var t: f32;
        if (abs(den) < 1e-30 * abs(two_mag)) {
          t = select(-1.0, 1.0, two_mag >= 0.0);
        } else {
          let theta = den / two_mag;
          let s = select(-1.0, 1.0, theta >= 0.0);
          t = s / (abs(theta) + sqrt(theta * theta + 1.0));
        }
        let cj = 1.0 / sqrt(1.0 + t * t);
        let sj = t * cj;
        if (cj != cj || sj != sj) { continue; }

        // Apply rotation to cols p, q of both A and V.
        for (var i = tid; i < n; i = i + 32u) {
          let ap = sA[i * n + p];
          let aq = sA[i * n + q];
          sA[i * n + p] = vec2<f32>(cj * ap.x - sj * aq.x, cj * ap.y - sj * aq.y);
          sA[i * n + q] = vec2<f32>(sj * ap.x + cj * aq.x, sj * ap.y + cj * aq.y);
          let vp = sV[i * n + p];
          let vq = sV[i * n + q];
          sV[i * n + p] = vec2<f32>(cj * vp.x - sj * vq.x, cj * vp.y - sj * vq.y);
          sV[i * n + q] = vec2<f32>(sj * vp.x + cj * vq.x, sj * vp.y + cj * vq.y);
        }
        workgroupBarrier();
      }
    }
  }

  // Write back.
  for (var i = tid; i < n * n; i = i + 32u) {
    A[i] = sA[i];
    V[i] = sV[i];
  }
}
