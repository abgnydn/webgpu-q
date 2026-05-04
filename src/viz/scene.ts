// ─────────────────────────────────────────────────────────────
// scene.ts — WebGPU rendering pipeline for the splat viewer.
//
// Owns the device, the canvas context, the splat pipeline, the
// uniform / storage buffers, and the orbit camera. Call
// `updateGrid` to push a new density field; call `render` once
// per frame.
//
// Splat layout: each splat is 16 bytes (vec3 position + f32
// density). The storage buffer is sized once at init based on
// the maximum grid count we expect — CPU-side updates rewrite
// it in place each time the bond slider moves.
// ─────────────────────────────────────────────────────────────

import splatSrc from "../shaders/splat.wgsl?raw";
import {
  mat4LookAt,
  mat4Mul,
  mat4Perspective,
  type Mat4,
  type Vec3,
} from "./math3d.js";
/** Minimal interface the splat renderer needs from any scalar field. */
export interface SplatGrid {
  positions: Float32Array;
  densities: Float32Array;
  maxDensity: number;
  gridSize: number;
}

export interface SceneInit {
  canvas: HTMLCanvasElement;
  /** Largest grid count we'll ever upload; storage buffer sized accordingly. */
  maxSplats: number;
}

export class Scene {
  readonly device: GPUDevice;
  readonly context: GPUCanvasContext;
  readonly format: GPUTextureFormat;
  private readonly pipeline: GPURenderPipeline;
  private readonly camUniform: GPUBuffer;
  private readonly splatBuf: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private currentSplats = 0;
  private maxSplats: number;

  // Camera state — orbit around target
  private azimuth = 0.0;          // radians, 0 = +x looking
  private elevation = 0.3;        // radians
  private distance = 6.0;         // Bohr
  private target: Vec3 = [0, 0, 0];

  // Render parameters
  splatSize = 0.18;               // Bohr, scaled inside shader
  alpha = 0.4;
  densityScale = 1.0;
  tint: Vec3 = [1, 1, 1];

  static fromDevice(opts: { device: GPUDevice; canvas: HTMLCanvasElement; maxSplats: number }): Scene {
    return new Scene(opts);
  }

  private constructor(opts: { device: GPUDevice; canvas: HTMLCanvasElement; maxSplats: number }) {
    this.device = opts.device;
    this.maxSplats = opts.maxSplats;
    const ctx = opts.canvas.getContext("webgpu");
    if (!ctx) throw new Error("canvas.getContext('webgpu') returned null");
    this.context = ctx;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({
      device: opts.device,
      format: this.format,
      alphaMode: "premultiplied",
    });

    const mod = opts.device.createShaderModule({ label: "splat", code: splatSrc });
    this.pipeline = opts.device.createRenderPipeline({
      label: "splat-pipeline",
      layout: "auto",
      vertex: { module: mod, entryPoint: "vs" },
      fragment: {
        module: mod,
        entryPoint: "fs",
        targets: [{
          format: this.format,
          // Pre-multiplied additive blending — cheap "glow" look.
          blend: {
            color: { srcFactor: "one", dstFactor: "one", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
          },
        }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
    });

    // Camera UBO: 64 (vp) + 16 (right + pad) + 16 (up + pad) + 16 (sizes) + 16 (tint + pad) = 128 B.
    this.camUniform = opts.device.createBuffer({
      label: "splat-camera",
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Splat SSBO: 16 B per splat.
    this.splatBuf = opts.device.createBuffer({
      label: "splat-data",
      size: opts.maxSplats * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.bindGroup = opts.device.createBindGroup({
      label: "splat-bind",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camUniform } },
        { binding: 1, resource: { buffer: this.splatBuf } },
      ],
    });
  }

  static async init(opts: SceneInit): Promise<Scene> {
    if (!navigator.gpu) throw new Error("WebGPU not available");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no GPU adapter");
    const device = await adapter.requestDevice({});
    return new Scene({ device, canvas: opts.canvas, maxSplats: opts.maxSplats });
  }

  /** Upload a density grid. Repacks (position, density) into a tight 16-B layout. */
  updateGrid(grid: SplatGrid): void {
    const N = grid.positions.length / 3;
    if (N > this.maxSplats) {
      throw new Error(`grid too large: ${N} > ${this.maxSplats}`);
    }
    const packed = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) {
      packed[i * 4]     = grid.positions[i * 3]!;
      packed[i * 4 + 1] = grid.positions[i * 3 + 1]!;
      packed[i * 4 + 2] = grid.positions[i * 3 + 2]!;
      packed[i * 4 + 3] = grid.densities[i]!;
    }
    this.device.queue.writeBuffer(this.splatBuf, 0, packed);
    this.currentSplats = N;
    this.densityScale = grid.maxDensity;
  }

  // ── Camera controls ─────────────────────────────────────
  orbitBy(dAz: number, dEl: number): void {
    this.azimuth += dAz;
    this.elevation = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, this.elevation + dEl));
  }
  zoomBy(factor: number): void {
    this.distance = Math.max(1.5, Math.min(40, this.distance * factor));
  }
  setTarget(t: Vec3): void {
    this.target = t;
  }

  private viewMatrix(): Mat4 {
    const ce = Math.cos(this.elevation);
    const eye: Vec3 = [
      this.target[0] + this.distance * ce * Math.cos(this.azimuth),
      this.target[1] + this.distance * Math.sin(this.elevation),
      this.target[2] + this.distance * ce * Math.sin(this.azimuth),
    ];
    return mat4LookAt(eye, this.target, [0, 1, 0]);
  }

  private cameraRightUp(): { right: Vec3; up: Vec3 } {
    const view = this.viewMatrix();
    // For a column-major lookAt result, row 0 of the rotation block = right, row 1 = up.
    // Equivalently: M[0,4,8]=right.xyz; M[1,5,9]=up.xyz.
    const right: Vec3 = [view[0]!, view[4]!, view[8]!];
    const up: Vec3 = [view[1]!, view[5]!, view[9]!];
    return { right, up };
  }

  private writeCameraUniform(aspect: number): void {
    const view = this.viewMatrix();
    const proj = mat4Perspective(45 * Math.PI / 180, aspect, 0.1, 200);
    const vp = mat4Mul(proj, view);
    const { right, up } = this.cameraRightUp();

    const ub = new ArrayBuffer(128);
    const f32 = new Float32Array(ub);
    f32.set(vp, 0);                 // 0..63
    f32.set(right, 16);              // 64..71
    f32[19] = 0;                     // pad
    f32.set(up, 20);                 // 80..87
    f32[23] = 0;                     // pad
    f32[24] = this.densityScale;
    f32[25] = this.splatSize;
    f32[26] = this.alpha;
    f32[27] = 0;                     // pad
    f32.set(this.tint, 28);          // tint
    f32[31] = 0;                     // pad
    this.device.queue.writeBuffer(this.camUniform, 0, ub);
  }

  render(): void {
    const canvas = this.context.canvas as HTMLCanvasElement;
    const aspect = canvas.width / canvas.height;
    this.writeCameraUniform(aspect);

    const view = this.context.getCurrentTexture().createView();
    const enc = this.device.createCommandEncoder({ label: "splat-frame" });
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view,
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: 0.02, g: 0.02, b: 0.04, a: 1 },
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(6, this.currentSplats);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
