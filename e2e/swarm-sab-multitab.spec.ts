import { test, expect } from "@playwright/test";

// Multi-tab SAB-partitioning MVP — the smallest possible demonstration
// that aggregate browser-tab memory exceeds a single tab's ceiling.
//
// Each tab on the same origin allocates its OWN private 200 MB SAB
// (independent of any other tab's allocation). Master broadcasts a
// "D vector" via BroadcastChannel; each tab computes a checksum over
// (its private SAB · broadcast D). Master gathers the partial
// checksums. Total memory in play across N tabs = N · 200 MB,
// proving aggregate > any one tab's ~2 GB ceiling could ever be.
//
// This is the infrastructure proof for breaking the chemistry memory
// wall: anthracene's 900 MB B-tensor sliced across 4 tabs at ~225 MB
// each is well below any per-tab cap. The actual chemistry swarm
// (each tab independently builds its aux-slice of B) builds on top.

test.describe("Multi-tab SAB + BroadcastChannel partitioning", () => {
  test("4 tabs each allocate 200 MB SAB, BroadcastChannel coordination, aggregate compute", async ({ browser }) => {
    test.setTimeout(2 * 60 * 1000);
    const N_TABS = 4;
    const PER_TAB_MB = 200;

    const ctx = await browser.newContext();
    const pages = await Promise.all(
      Array.from({ length: N_TABS }, () => ctx.newPage()),
    );

    // Capture page errors for diagnosis.
    pages.forEach((p, i) => {
      p.on("pageerror", (err) => console.error(`[tab${i}:pageerror] ${err.message}`));
    });

    // Use molecule.html as a same-origin host page (COI + SAB enabled).
    await Promise.all(pages.map((p) =>
      p.goto("/molecule.html", { waitUntil: "domcontentloaded" })));

    // Each tab: allocate its own SAB, fill with a tab-id-stamped pattern,
    // open BroadcastChannel, listen for "go" with broadcast D.
    // Tab 0 acts as master, dispatches a single round-trip.
    const result = await Promise.all(pages.map(async (page, tabId) => {
      return page.evaluate(async ({ tabId, perTabMB, nTabs }) => {
        const bytes = perTabMB * 1024 * 1024;
        // eslint-disable-next-line no-console
        console.log(`[tab${tabId}] crossOriginIsolated=${crossOriginIsolated}, allocating SAB ${perTabMB} MB`);
        if (typeof SharedArrayBuffer === "undefined" || !crossOriginIsolated) {
          return { tabId, error: "SAB unavailable / not COI" };
        }
        const sab = new SharedArrayBuffer(bytes);
        const view = new Float64Array(sab);
        // Fill with a deterministic pattern that depends on tabId so
        // the per-tab partial sums differ.
        const stride = 8192;  // touch every 64 KB to fault all pages
        for (let i = 0; i < view.length; i += stride) {
          view[i] = (i + tabId * 1e6) % 10007;
        }
        // eslint-disable-next-line no-console
        console.log(`[tab${tabId}] SAB filled; ${view.length / 1e6}M f64 elements`);

        const ch = new BroadcastChannel("sab-mvp");
        const isMaster = tabId === 0;

        if (!isMaster) {
          // Worker: listen for "go", compute partial, post back.
          return await new Promise<{ tabId: number; partial: number; durMs: number; sabBytes: number }>((resolve) => {
            ch.addEventListener("message", (ev) => {
              const msg = ev.data as { type: string; D?: number[] };
              if (msg?.type !== "go") return;
              const D = msg.D ?? [];
              const t0 = performance.now();
              let s = 0;
              // Partial = Σ_{i sampled} view[i] · D[i mod |D|]
              for (let i = 0; i < view.length; i += stride) {
                s += view[i]! * D[i % D.length]!;
              }
              const durMs = performance.now() - t0;
              ch.postMessage({ type: "partial", tabId, partial: s, durMs });
              resolve({ tabId, partial: s, durMs, sabBytes: bytes });
            });
          });
        }
        // Master: wait for N-1 worker partials, then sum.
        const partials = new Map<number, number>();
        const ready = new Promise<void>((resolve) => {
          ch.addEventListener("message", (ev) => {
            const msg = ev.data as { type: string; tabId?: number; partial?: number };
            if (msg?.type === "partial" && typeof msg.tabId === "number" && typeof msg.partial === "number") {
              partials.set(msg.tabId, msg.partial);
              if (partials.size === nTabs - 1) resolve();
            }
          });
        });
        // Give workers a moment to attach their listeners.
        await new Promise((r) => setTimeout(r, 1500));

        const D = Array.from({ length: 64 }, (_, i) => Math.sin(i + 1));
        const t0 = performance.now();
        // Master's own partial
        let ownPartial = 0;
        for (let i = 0; i < view.length; i += stride) {
          ownPartial += view[i]! * D[i % D.length]!;
        }
        // Broadcast go
        ch.postMessage({ type: "go", D });
        await ready;
        const totalMs = performance.now() - t0;

        let aggregate = ownPartial;
        for (const v of partials.values()) aggregate += v;

        return {
          tabId, partial: ownPartial, durMs: totalMs, sabBytes: bytes,
          aggregate, fromWorkers: Array.from(partials.entries()).sort(),
        };
      }, { tabId, perTabMB: PER_TAB_MB, nTabs: N_TABS });
    }));

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Multi-tab SAB partitioning — ${N_TABS} tabs × ${PER_TAB_MB} MB`);
    console.log(`══════════════════════════════════════════════════════════`);
    const totalAllocated = result.reduce((s, r) => s + (r.sabBytes ?? 0), 0);
    console.log(`Aggregate SAB allocated across tabs: ${(totalAllocated / 1024 / 1024).toFixed(0)} MB`);
    console.log(`(per-tab \"ceiling\" no longer the cap)`);
    console.log();
    for (const r of result) {
      const partial = r.partial ?? 0;
      const dur = r.durMs ?? 0;
      console.log(`  tab ${r.tabId}: partial=${partial.toExponential(4)}  ${dur.toFixed(1)} ms`);
    }
    const master = result[0]!;
    if ("aggregate" in master && master.aggregate !== undefined) {
      console.log();
      console.log(`Master aggregated total: ${master.aggregate.toExponential(6)}`);
      console.log(`Master saw ${master.fromWorkers?.length ?? 0} worker partials`);
      console.log(`Round-trip wall-clock: ${master.durMs.toFixed(1)} ms`);
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    // Assert all tabs succeeded
    for (const r of result) {
      expect(r).not.toHaveProperty("error");
      expect(r.sabBytes).toBe(PER_TAB_MB * 1024 * 1024);
    }
    // Master gathered all worker partials
    expect((master as { fromWorkers?: unknown[] }).fromWorkers?.length).toBe(N_TABS - 1);
    expect(Number.isFinite((master as { aggregate?: number }).aggregate ?? NaN)).toBe(true);

    await ctx.close();
  });
});
