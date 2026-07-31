import { test, expect } from "@playwright/test";

// Validate the gzip-binary wire path (Lever 3) end-to-end over the REAL broker:
// a message big enough to trigger compression must arrive LOSSLESS (bit-exact
// f64) and as binary (mqtt.js carries Uint8Array, not just strings). Skips if the
// broker is unreachable.

const ROOM = `wgq-big-${Date.now().toString(36)}`;

test.describe("RelayTransport — gzip-binary big message round-trips losslessly", () => {
  test("a big f64 density survives the broker bit-exact", async ({ browser }) => {
    test.setTimeout(120 * 1000);
    const ctx = await browser.newContext();
    const rx = await ctx.newPage();
    const tx = await ctx.newPage();
    rx.on("pageerror", (e) => console.error(`[rx:pageerror] ${e.message}`));
    tx.on("pageerror", (e) => console.error(`[tx:pageerror] ${e.message}`));
    rx.on("console", (m) => { if (m.text().startsWith("[big]")) console.log(m.text()); });
    await rx.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    await tx.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    // Receiver: open + stash any "bigD" message it gets.
    const rxConn = await rx.evaluate(async (room) => {
      const { RelayTransport } = await import("/src/parallel/swarm/relay-transport.ts" as string);
      const w = window as unknown as { __got: { n: number; sum: number; first: number; last: number } | null };
      w.__got = null;
      const t = new RelayTransport({ room, id: "rx" });
      t.onMessage((m: { type: string; payload?: { D: number[] } }) => {
        if (m.type !== "bigD" || !m.payload) return;
        const D = m.payload.D;
        let sum = 0; for (let i = 0; i < D.length; i++) sum += D[i]!;
        w.__got = { n: D.length, sum, first: D[0]!, last: D[D.length - 1]! };
      });
      try { await t.openAsync(); return true; } catch { return false; }
    }, ROOM);
    test.skip(!rxConn, "broker unreachable");

    // Sender: build a distinctive big f64 density (n=200 → 40k doubles → gzip path).
    const sent = await tx.evaluate(async (room) => {
      const { RelayTransport } = await import("/src/parallel/swarm/relay-transport.ts" as string);
      const t = new RelayTransport({ room, id: "tx" });
      try { await t.openAsync(); } catch { return null; }
      await new Promise((r) => setTimeout(r, 1500)); // let rx subscribe
      const n = 200, D = new Array<number>(n * n);
      for (let i = 0; i < D.length; i++) D[i] = Math.sin(i * 0.5) * 0.5 + i * 1e-12; // f64 low bits matter
      let sum = 0; for (let i = 0; i < D.length; i++) sum += D[i]!;
      // send a few times (in case rx subscribed late)
      for (let k = 0; k < 4; k++) { t.send({ type: "bigD", payload: { D } }); await new Promise((r) => setTimeout(r, 800)); }
      return { n: D.length, sum, first: D[0]!, last: D[D.length - 1]! };
    }, ROOM);
    expect(sent).not.toBeNull();

    // Receiver should have the bit-exact same density.
    const got = await rx.evaluate(async () => {
      const w = window as unknown as { __got: { n: number; sum: number; first: number; last: number } | null };
      for (let i = 0; i < 40 && !w.__got; i++) await new Promise((r) => setTimeout(r, 250));
      return w.__got;
    });
    await rx.evaluate(() => { console.log("[big] received:", JSON.stringify((window as unknown as { __got: unknown }).__got)); });

    expect(got, "no bigD message arrived").not.toBeNull();
    expect(got!.n).toBe(sent!.n);
    // bit-exact: same length, same first/last, same f64 sum (no precision loss).
    expect(got!.first).toBe(sent!.first);
    expect(got!.last).toBe(sent!.last);
    expect(got!.sum).toBe(sent!.sum);

    await ctx.close();
  });
});
