// Wire codec for swarm messages — Lever 3 ("talk less data").
//
// The distributed-SCF path broadcasts a density matrix (and gathers J/K) every
// iteration. Sent as JSON arrays of f64 that was ~21 MB at n=1000 — bandwidth-
// bound, and over a free MQTT hub it doesn't even fit one message. This codec
// keeps the message LOSSLESS (full f64, exact round-trip) but ships large numeric
// arrays as binary f64 + gzip: measured 21 MB → 65 KB (~330×) on a realistic
// (structured) density, vs 182 KB for gzip-of-JSON. f64 is preserved, so the
// swarm's bit-exact (1e-12) agreement with single-machine survives.
//
// Self-describing & backward-compatible: small messages go as plain JSON strings;
// large ones go as gzip (magic bytes 1f 8b), so a peer detects the format on the
// wire. If CompressionStream is unavailable, everything degrades to plain JSON.

const COMPRESS_THRESHOLD = 1024; // JSON bytes; below this, plain string (gzip not worth it)
const ARRAY_THRESHOLD = 64;      // numeric arrays ≥ this go to a binary f64 blob

const hasCompression = typeof CompressionStream !== "undefined" && typeof DecompressionStream !== "undefined";

async function gzip(data: Uint8Array): Promise<Uint8Array> {
  const cs = new CompressionStream("gzip");
  const ab = await new Response(new Blob([data as BlobPart]).stream().pipeThrough(cs)).arrayBuffer();
  return new Uint8Array(ab);
}
async function gunzip(data: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream("gzip");
  const ab = await new Response(new Blob([data as BlobPart]).stream().pipeThrough(ds)).arrayBuffer();
  return new Uint8Array(ab);
}

// Replace each large numeric array with a placeholder {__b, __n} and push its
// data to `blobs`. Everything else passes through unchanged.
function extractArrays(obj: unknown, blobs: Float64Array[]): unknown {
  if (Array.isArray(obj)) {
    if (obj.length >= ARRAY_THRESHOLD && obj.every((x) => typeof x === "number")) {
      const idx = blobs.length;
      blobs.push(Float64Array.from(obj as number[]));
      return { __b: idx, __n: obj.length };
    }
    return obj.map((v) => extractArrays(v, blobs));
  }
  if (obj && typeof obj === "object") {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(obj as Record<string, unknown>)) out[k] = extractArrays((obj as Record<string, unknown>)[k], blobs);
    return out;
  }
  return obj;
}
function restoreArrays(obj: unknown, blobs: Float64Array[]): unknown {
  if (Array.isArray(obj)) return obj.map((v) => restoreArrays(v, blobs));
  if (obj && typeof obj === "object") {
    const o = obj as Record<string, unknown>;
    if (typeof o.__b === "number" && typeof o.__n === "number") return Array.from(blobs[o.__b as number]!);
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(o)) out[k] = restoreArrays(o[k], blobs);
    return out;
  }
  return obj;
}

/** Encode a message for the wire. Small → JSON string; large → binary-f64 frame +
 *  gzip (Uint8Array). Lossless either way. */
export async function encodeMessage(msg: unknown): Promise<string | Uint8Array> {
  const json = JSON.stringify(msg);
  if (!hasCompression || json.length < COMPRESS_THRESHOLD) return json;

  const blobs: Float64Array[] = [];
  const stripped = extractArrays(msg, blobs);
  const header = new TextEncoder().encode(JSON.stringify(stripped));

  let total = 4 + header.length;
  for (const b of blobs) total += 4 + b.byteLength;
  const frame = new Uint8Array(total);
  const dv = new DataView(frame.buffer);
  let off = 0;
  dv.setUint32(off, header.length, true); off += 4;
  frame.set(header, off); off += header.length;
  for (const b of blobs) {
    dv.setUint32(off, b.byteLength, true); off += 4;
    frame.set(new Uint8Array(b.buffer, b.byteOffset, b.byteLength), off); off += b.byteLength;
  }
  return gzip(frame);
}

/** Decode a wire payload (always bytes off MQTT). Detects gzip (1f 8b) → binary
 *  frame; otherwise plain JSON. Returns the original message, f64 intact. */
export async function decodeMessage(payload: Uint8Array): Promise<unknown> {
  if (payload.length >= 2 && payload[0] === 0x1f && payload[1] === 0x8b) {
    const frame = await gunzip(payload);
    const dv = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
    let off = 0;
    const headerLen = dv.getUint32(off, true); off += 4;
    const header = JSON.parse(new TextDecoder().decode(frame.subarray(off, off + headerLen))); off += headerLen;
    const blobs: Float64Array[] = [];
    while (off < frame.length) {
      const len = dv.getUint32(off, true); off += 4;
      const f = new Float64Array(len / 8);            // fresh, 8-byte aligned
      new Uint8Array(f.buffer).set(frame.subarray(off, off + len)); off += len;
      blobs.push(f);
    }
    return restoreArrays(header, blobs);
  }
  return JSON.parse(new TextDecoder().decode(payload));
}
