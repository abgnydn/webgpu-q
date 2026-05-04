// ─────────────────────────────────────────────────────────────
// buffer-pool.ts — recycle GPU buffers across calls.
//
// The Phase 1A/1B/1C primitives all created and destroyed
// buffers per dispatch. That's ~1-3 ms of overhead from
// `createBuffer + writeBuffer + copyBufferToBuffer + mapAsync +
// unmap + destroy` on Apple Metal-3, regardless of the actual
// compute work. For MPS evolution we issue 100+ SVDs per
// circuit, so amortizing buffer setup is a 5-10× win.
//
// API: acquire(size, usage) → returns a GPUBuffer at least that
// size with the requested usage flags; release(buf) returns it
// to the pool. Buffers are bucketed by usage; sizes round up to
// powers of two so a 224-byte uniform happily reuses a 256-byte
// slot from a previous call. There's no time-based eviction —
// the pool just lives for the page's lifetime.
// ─────────────────────────────────────────────────────────────

interface PooledBuffer {
  buffer: GPUBuffer;
  size: number;
  usage: GPUBufferUsageFlags;
}

function ceilPow2(x: number): number {
  if (x <= 16) return 16;
  let n = 16;
  while (n < x) n *= 2;
  return n;
}

export class BufferPool {
  /** Free pool keyed by `${usage}:${roundedSize}`. */
  private readonly free: Map<string, PooledBuffer[]> = new Map();
  /** Set of buffers currently in use; lets `release` validate. */
  private readonly inUse: Set<GPUBuffer> = new Set();

  constructor(private readonly device: GPUDevice) {}

  /** Get a buffer with at least `bytes` bytes and the given usage flags. */
  acquire(bytes: number, usage: GPUBufferUsageFlags, label?: string): GPUBuffer {
    const size = ceilPow2(Math.max(bytes, 16));
    const key = `${usage}:${size}`;
    const stack = this.free.get(key);
    if (stack && stack.length > 0) {
      const top = stack.pop()!;
      this.inUse.add(top.buffer);
      return top.buffer;
    }
    const buf = this.device.createBuffer({
      label: label ?? `pool-${key}`,
      size,
      usage,
    });
    this.inUse.add(buf);
    return buf;
  }

  /** Return a buffer to the pool. The buffer's contents are now
   *  considered garbage and will be overwritten on next acquire. */
  release(buf: GPUBuffer): void {
    if (!this.inUse.has(buf)) return;
    this.inUse.delete(buf);
    const key = `${(buf as unknown as { usage: number }).usage}:${buf.size}`;
    let stack = this.free.get(key);
    if (!stack) {
      stack = [];
      this.free.set(key, stack);
    }
    stack.push({ buffer: buf, size: buf.size, usage: (buf as unknown as { usage: number }).usage });
  }

  /** Destroy every buffer in the pool. Call before disposing the device. */
  destroyAll(): void {
    for (const stack of this.free.values()) for (const b of stack) b.buffer.destroy();
    for (const b of this.inUse) b.destroy();
    this.free.clear();
    this.inUse.clear();
  }

  /** Diagnostic: number of free buffers + total bytes. */
  stats(): { freeCount: number; freeBytes: number } {
    let count = 0, bytes = 0;
    for (const stack of this.free.values()) for (const b of stack) {
      count++; bytes += b.size;
    }
    return { freeCount: count, freeBytes: bytes };
  }
}

const poolByDevice: WeakMap<GPUDevice, BufferPool> = new WeakMap();

/** Per-device singleton pool. Used by matmul.ts and jacobi-svd.ts. */
export function poolFor(device: GPUDevice): BufferPool {
  let p = poolByDevice.get(device);
  if (!p) {
    p = new BufferPool(device);
    poolByDevice.set(device, p);
  }
  return p;
}
