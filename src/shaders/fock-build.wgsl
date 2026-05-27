// fock-build.wgsl — Fock matrix JK build on the GPU.
//
//   G[μ, ν] = Σ_{λ, σ} D[λ, σ] · ( (μν|λσ) − ½ (μλ|νσ) )
//
// One thread per (μ, ν) entry of G. The Σ over (λ, σ) runs in a serial
// inner loop with f32 accumulator. ERI tensor is stored as a flat
// row-major f32 n⁴ buffer; D and G are flat n² f32 buffers.
//
// f32 precision: f32 has ~7 decimal digits. The inner sum has n²=14400
// terms on benzene; in worst case the sum can drift by ~10⁻⁵ × |max(J)|
// from f64 truth. HF SCF tolerance is typically 1e-8 Ha; for cc-pVDZ
// the rolled-up Fock entries stay within f32 noise of the WASM-f64 path
// (verified empirically). If a target system needs tighter convergence,
// use the WASM JK path instead.

struct Dims {
    n: u32,
    n2: u32,
    n3: u32,
    _padding: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> eri: array<f32>;
@group(0) @binding(2) var<storage, read> d: array<f32>;
@group(0) @binding(3) var<storage, read_write> g: array<f32>;

@compute @workgroup_size(8, 8)
fn buildG(@builtin(global_invocation_id) id: vec3<u32>) {
    let mu = id.x;
    let nu = id.y;
    let n = dims.n;
    if (mu >= n || nu >= n) { return; }
    let n2 = dims.n2;
    let n3 = dims.n3;
    let mu_n3 = mu * n3;
    let nu_n2 = nu * n2;
    let nu_n = nu * n;
    var s: f32 = 0.0;
    for (var la: u32 = 0u; la < n; la = la + 1u) {
        let d_la_base = la * n;
        let j_base = mu_n3 + nu_n2 + la * n;
        let k_base = mu_n3 + la * n2 + nu_n;
        for (var si: u32 = 0u; si < n; si = si + 1u) {
            let dls = d[d_la_base + si];
            let j_val = eri[j_base + si];
            let k_val = eri[k_base + si];
            s = s + dls * (j_val - 0.5 * k_val);
        }
    }
    g[mu * n + nu] = s;
}
