import { test, expect } from "@playwright/test";

// Lever 2 ("talk faster") — actually MEASURE the per-round latency I keep
// asserting (~200 ms broker). Broker RTT = publish to a topic we're subscribed
// to (loopback client→broker→client). Local RTT = BroadcastChannel ping-pong
// across two same-origin pages. Median over N pings. No heavy compute.

test.describe("Transport RTT — broker vs local", () => {
  test("MQTT broker loopback RTT", async ({ page }) => {
    test.setTimeout(90 * 1000);
    page.on("console", (m) => { if (m.text().startsWith("[rtt]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    const r = await page.evaluate(async () => {
      const log = (s: string): void => { console.log(`[rtt] ${s}`); };
      await new Promise<void>((res, rej) => { const s = document.createElement("script");
        s.src = "https://unpkg.com/mqtt@5.10.1/dist/mqtt.min.js"; s.onload = () => res(); s.onerror = () => rej(new Error("cdn")); document.head.appendChild(s); });
      const mqtt = (window as unknown as { mqtt: { connect(u: string, o?: unknown): MqttC } }).mqtt;
      interface MqttC { on(e: string, cb: (...a: unknown[]) => void): void; subscribe(t: string, cb?: (e?: Error) => void): void; publish(t: string, m: string): void; end(): void; }
      const topic = `wgq-rtt/${Date.now()}`;
      const c = mqtt.connect("wss://broker.emqx.io:8084/mqtt", { connectTimeout: 20000 });
      const ok = await new Promise<boolean>((res) => { c.on("connect", () => res(true)); setTimeout(() => res(false), 22000); });
      if (!ok) return { ok: false };
      const rtts: number[] = [];
      await new Promise<void>((resolve) => {
        let sent = 0; const pending = new Map<number, number>();
        c.on("message", (_t: unknown, p: unknown) => {
          const id = Number(new TextDecoder().decode(p as Uint8Array));
          const t0 = pending.get(id); if (t0 !== undefined) rtts.push(performance.now() - t0);
          if (rtts.length >= 20) { resolve(); return; }
          if (sent < 25) { const t = performance.now(); pending.set(sent, t); c.publish(topic, String(sent++)); }
        });
        c.subscribe(topic, () => { const t = performance.now(); pending.set(sent, t); c.publish(topic, String(sent++)); });
        setTimeout(resolve, 60000);
      });
      c.end();
      rtts.sort((a, b) => a - b);
      const med = rtts[Math.floor(rtts.length / 2)] ?? NaN;
      log(`broker RTT (n=${rtts.length}): median ${med.toFixed(0)}ms, min ${(rtts[0] ?? NaN).toFixed(0)}ms, max ${(rtts[rtts.length - 1] ?? NaN).toFixed(0)}ms`);
      return { ok: true, median: med, n: rtts.length };
    });
    console.log(`\n[comm-rtt broker] ${JSON.stringify(r)}\n`);
    test.skip(!r.ok, "broker unreachable");
    expect(r.n).toBeGreaterThan(5);
    expect(r.median).toBeGreaterThan(0);
  });

  test("BroadcastChannel (local) RTT", async ({ browser }) => {
    test.setTimeout(60 * 1000);
    const ctx = await browser.newContext();
    const a = await ctx.newPage(); const b = await ctx.newPage();
    a.on("console", (m) => { if (m.text().startsWith("[rtt]")) console.log(m.text()); });
    await a.goto("/swarm.html", { waitUntil: "domcontentloaded" });
    await b.goto("/swarm.html", { waitUntil: "domcontentloaded" });
    // B echoes.
    await b.evaluate(() => { const ch = new BroadcastChannel("rtt-local"); ch.onmessage = (e) => { if (typeof e.data === "number") ch.postMessage(-e.data); }; });
    const r = await a.evaluate(async () => {
      const log = (s: string): void => { console.log(`[rtt] ${s}`); };
      const ch = new BroadcastChannel("rtt-local");
      const rtts: number[] = []; const pending = new Map<number, number>();
      await new Promise<void>((resolve) => {
        let sent = 0;
        ch.onmessage = (e) => { if (typeof e.data !== "number" || e.data > 0) return; const id = -e.data; const t0 = pending.get(id);
          if (t0 !== undefined) rtts.push(performance.now() - t0);
          if (rtts.length >= 50) { resolve(); return; }
          const t = performance.now(); pending.set(sent, t); ch.postMessage(sent++); };
        const t = performance.now(); pending.set(0, t); ch.postMessage(sent++);
        setTimeout(resolve, 10000);
      });
      rtts.sort((x, y) => x - y);
      const med = rtts[Math.floor(rtts.length / 2)] ?? NaN;
      log(`local RTT (n=${rtts.length}): median ${med.toFixed(2)}ms, min ${(rtts[0] ?? NaN).toFixed(2)}ms`);
      return { median: med, n: rtts.length };
    });
    console.log(`\n[comm-rtt local] ${JSON.stringify(r)}\n`);
    await ctx.close();
    expect(r.n).toBeGreaterThan(5);
  });
});
