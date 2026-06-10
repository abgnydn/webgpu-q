import { describe, expect, test } from "vitest";
import { encodeMessage, decodeMessage } from "../../src/parallel/swarm/wire-codec.js";

// Lever 3: the wire codec must be LOSSLESS (full f64 round-trip — the swarm's
// bit-exact 1e-12 agreement depends on it) AND much smaller than JSON for the
// big density/J/K messages.

const enc = (s: string | Uint8Array): Uint8Array =>
  typeof s === "string" ? new TextEncoder().encode(s) : s;

describe("swarm wire codec (gzip binary-f64)", () => {
  test("small message → plain JSON string, round-trips", async () => {
    const msg = { from: "peer-a", type: "hello" };
    const wire = await encodeMessage(msg);
    expect(typeof wire).toBe("string"); // below threshold → not gzipped
    const back = await decodeMessage(enc(wire));
    expect(back).toEqual(msg);
  });

  test("large density message → gzip binary, LOSSLESS f64 round-trip + big shrink", async () => {
    // structured density proxy (n=300 → 90k doubles), like the real broadcast
    const n = 300;
    const D = new Array<number>(n * n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        D[i * n + j] = Math.exp(-0.08 * Math.abs(i - j)) * (i === j ? 2 : 1) + (i * 31 + j) * 1e-13;
    const msg = { from: "peer-a", type: "D", payload: { D } };

    const json = JSON.stringify(msg);
    const wire = await encodeMessage(msg);
    expect(wire).toBeInstanceOf(Uint8Array);                 // gzipped binary
    const wireBytes = (wire as Uint8Array).byteLength;
    // Shrink is DATA-DEPENDENT: a perfectly smooth density gzips ~300×, but a real
    // one has entropy in its low mantissa bits (below convergence precision) that
    // gzip can't touch, so the realistic win is ~3-10×. The reliable, entropy-
    // independent part is JSON→binary-f64 (~2.5×); gzip adds the rest. Assert the
    // modest floor; the lossless round-trip below is the actual requirement.
    expect(wireBytes).toBeLessThan(json.length / 3);         // ≥3× smaller than JSON (real density)

    const back = await decodeMessage(wire as Uint8Array) as { from: string; type: string; payload: { D: number[] } };
    expect(back.from).toBe("peer-a");
    expect(back.type).toBe("D");
    expect(back.payload.D.length).toBe(D.length);
    // BIT-EXACT f64 — every element identical (this is what keeps 1e-12).
    for (let k = 0; k < D.length; k++) expect(back.payload.D[k]).toBe(D[k]);
    // eslint-disable-next-line no-console
    console.log(`[codec] D n=${n}: JSON ${(json.length / 1024).toFixed(0)}KB → wire ${(wireBytes / 1024).toFixed(0)}KB (${(json.length / wireBytes).toFixed(0)}x), lossless`);
  });

  test("J/K partial (two arrays) round-trips losslessly", async () => {
    const N = 64 * 64;
    const J = Array.from({ length: N }, (_, i) => Math.sin(i) * 0.5);
    const K = Array.from({ length: N }, (_, i) => Math.cos(i) * 0.25);
    const msg = { from: "w1", type: "jk-partial", to: "master", payload: { J, K } };
    const wire = await encodeMessage(msg);
    const back = await decodeMessage(enc(wire)) as { payload: { J: number[]; K: number[] } };
    for (let k = 0; k < N; k++) {
      expect(back.payload.J[k]).toBe(J[k]);
      expect(back.payload.K[k]).toBe(K[k]);
    }
  });
});
