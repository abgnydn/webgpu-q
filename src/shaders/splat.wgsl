// ─────────────────────────────────────────────────────────────
// splat.wgsl — additive Gaussian-blob "splat" renderer for a
// 3D scalar field sampled on a regular grid.
//
// Each input element is one (position, scalar) tuple. The vertex
// stage emits a 6-vertex camera-aligned billboard quad per
// instance, sized by the scalar value. The fragment stage paints
// a soft Gaussian profile with alpha proportional to scalar.
//
// Color is a perceptually-uniform blue→cyan→white-hot ramp; this
// reads as "electron cloud" without being noisy. Override the
// ramp via the `tint` uniform if you want HOMO/LUMO contrast.
// ─────────────────────────────────────────────────────────────

struct Camera {
  vp: mat4x4<f32>,
  right: vec3<f32>,    _p1: f32,
  up: vec3<f32>,       _p2: f32,
  /** Density divisor for size scaling — typically the grid's max ρ. */
  densityScale: f32,
  /** Splat half-size at unit density (world units). */
  splatSize: f32,
  /** Alpha multiplier; lower = more transparent cloud. */
  alpha: f32,
  _p3: f32,
  /** Color tint for the additive ramp. */
  tint: vec3<f32>,
  _p4: f32,
};

struct Splat {
  position: vec3<f32>,
  density: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;

struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) density: f32,
};

const QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>( 1.0,  1.0),
  vec2<f32>(-1.0,  1.0),
);

@vertex
fn vs(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VOut {
  let s = splats[ii];

  // Reject near-zero densities by emitting a degenerate quad behind
  // the camera. Cheaper than a CPU-side culling pass for moderate N.
  if (s.density < 0.0005 * cam.densityScale) {
    var dead: VOut;
    dead.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    dead.uv = vec2<f32>(0.0, 0.0);
    dead.density = 0.0;
    return dead;
  }

  let offset = QUAD[vi];
  // Size grows mildly with density so high-ρ regions glow brighter,
  // not just larger — pow(ρ, 0.35) keeps the dynamic range readable.
  let normalized = clamp(s.density / cam.densityScale, 0.0, 1.0);
  let size = cam.splatSize * (0.4 + 0.8 * pow(normalized, 0.35));

  let world = s.position + cam.right * offset.x * size + cam.up * offset.y * size;

  var out: VOut;
  out.clip = cam.vp * vec4<f32>(world, 1.0);
  out.uv = offset;
  out.density = normalized;
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  // Gaussian falloff inside the quad.
  let falloff = exp(-r2 * 3.0);

  // Blue → cyan → white ramp, biased to make low-ρ regions visible.
  let t = pow(in.density, 0.5);
  let cold = vec3<f32>(0.10, 0.25, 0.85);
  let warm = vec3<f32>(0.95, 0.95, 1.00);
  let color = mix(cold, warm, t) * cam.tint;

  let a = falloff * cam.alpha * (0.15 + 0.85 * in.density);
  return vec4<f32>(color * a, a);
}
