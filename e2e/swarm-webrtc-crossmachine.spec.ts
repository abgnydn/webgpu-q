import { test, expect } from "@playwright/test";

// Cross-MACHINE WebRTC connectivity de-risk.
//
// Unlike e2e/swarm-webrtc.spec.ts (two browser CONTEXTS on one machine —
// trivial host-candidate connection, no NAT), this runs as TWO SEPARATE
// GitHub Actions jobs (two VMs, two NATs, public internet). The make-or-break
// question: can two cloud VMs establish a WebRTC DataChannel at all?
//
// Rendezvous without out-of-band ID exchange: both jobs derive deterministic
// peer IDs from a shared ROOM (the GitHub run id). Role "a" listens; role "b"
// dials a and round-trips a Float64Array. Signaling via the PeerJS public
// broker; ICE via Google STUN + a free public TURN relay (cloud NAT is usually
// symmetric, so STUN-only hole-punching fails — TURN is the fallback path).
//
// Hard pass/fail (not skip): we want an unambiguous yes/no on cross-machine
// WebRTC for the swarm.

const ROLE = process.env.ROLE ?? "a";          // "a" (listener) | "b" (dialer)
const ROOM = process.env.ROOM ?? "local-dev";  // shared rendezvous (github.run_id in CI)
const SELF = `wgq-${ROOM}-${ROLE}`;
const PEER_A = `wgq-${ROOM}-a`;

// Google STUN + metered.ca Open Relay free public TURN (no key, best-effort).
const ICE = [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "stun:stun.relay.metered.ca:80" },
  { urls: "turn:openrelay.metered.ca:80", username: "openrelayproject", credential: "openrelayproject" },
  { urls: "turn:openrelay.metered.ca:443", username: "openrelayproject", credential: "openrelayproject" },
  { urls: "turn:openrelay.metered.ca:443?transport=tcp", username: "openrelayproject", credential: "openrelayproject" },
];

test.describe("Cross-machine WebRTC connectivity (2 separate VMs)", () => {
  test(`role ${ROLE} — DataChannel round-trip over WebRTC across machines`, async ({ page }) => {
    test.setTimeout(4 * 60 * 1000);
    page.on("pageerror", (e) => console.error(`[${ROLE}:pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[rtc]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const result = await page.evaluate(async ({ role, self, peerA, ice }) => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[rtc] ${s}`); };
      const { WebRTCTransport } = await import("/src/parallel/swarm/webrtc-transport.ts" as string);

      const t = new WebRTCTransport({ id: self, iceServers: ice });
      const t0 = Date.now();
      try {
        await Promise.race([
          t.openAsync(),
          new Promise((_, rej) => setTimeout(() => rej(new Error("broker open timeout")), 30000)),
        ]);
      } catch (e) {
        return { ok: false, stage: "open", error: String(e) };
      }
      log(`peer open as ${t.self} (+${Date.now() - t0}ms)`);

      if (role === "a") {
        // Listener: reply to a "ping" with the array element-wise doubled.
        return await new Promise<{ ok: boolean; stage: string; error?: string }>((resolve) => {
          t.onMessage((msg: { from: string; type: string; payload?: unknown }) => {
            if (msg.type === "ping") {
              const arr = (msg.payload as { arr: number[] }).arr;
              log(`got ping from ${msg.from}, replying pong`);
              t.send({ to: msg.from, type: "pong", payload: { arr: arr.map((x) => x * 2) } });
              resolve({ ok: true, stage: "served-ping" });
            }
          });
          setTimeout(() => resolve({ ok: false, stage: "wait-ping", error: "no ping in 180s" }), 180000);
        });
      }

      // Dialer: retry connectTo until the channel opens, then ping/await pong.
      return await new Promise<{ ok: boolean; stage: string; error?: string; rtt?: number }>((resolve) => {
        const sent = { arr: [1.5, -2.25, 3.125, 4.0] };
        let connected = false;
        t.onMessage((msg: { from: string; type: string; payload?: unknown }) => {
          if (msg.type === "pong") {
            const got = (msg.payload as { arr: number[] }).arr;
            const ok = got.length === 4 && got.every((x, i) => Math.abs(x - sent.arr[i]! * 2) < 1e-12);
            log(`got pong, roundtrip ${ok ? "OK" : "MISMATCH"}`);
            resolve({ ok, stage: "roundtrip", rtt: Date.now() - t0 });
          }
        });
        const dial = setInterval(() => {
          const live = t.peers().length > 0;
          if (live && !connected) {
            connected = true;
            log(`connected to ${peerA}, sending ping`);
            t.send({ to: peerA, type: "ping", payload: sent });
          } else if (!live) {
            try { t.connectTo(peerA); } catch { /* peer maybe not registered yet */ }
          }
        }, 3000);
        setTimeout(() => { clearInterval(dial); resolve({ ok: false, stage: "connect", error: "no DataChannel in 180s" }); }, 180000);
      });
    }, { role: ROLE, self: SELF, peerA: PEER_A, ice: ICE });

    console.log(`\n[cross-machine ${ROLE}] result:`, JSON.stringify(result), "\n");
    expect(result.ok, `stage=${result.stage} ${result.error ?? ""}`).toBe(true);
  });
});
