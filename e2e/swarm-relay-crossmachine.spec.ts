import { test, expect } from "@playwright/test";

// Cross-MACHINE coordination via a free public relay hub (zero signup).
//
// The WebRTC de-risk (swarm-webrtc-crossmachine.spec.ts) showed PeerJS signaling
// works between two Actions VMs but the P2P DataChannel can't form across
// symmetric cloud NAT without a TURN relay (free public TURN is dead; a TURN key
// needs signup). The pragmatic free path: both VMs connect OUTBOUND to a free
// public MQTT broker (no inbound NAT problem, no account) and coordinate through
// it. Still genuinely cross-machine — two VMs, two NATs, public internet — just
// hub-relayed instead of pure P2P, which is how most distributed systems work.
//
// Role "a" echoes a ping back doubled; role "b" pings and checks the round-trip.

const ROLE = process.env.ROLE ?? "a";
const ROOM = process.env.ROOM ?? "local-dev";
const TOPIC = `wgq/${ROOM}`;
const BROKER = process.env.MQTT_BROKER ?? "wss://broker.emqx.io:8084/mqtt"; // free public, no auth

test.describe("Cross-machine swarm coordination via free relay (2 separate VMs)", () => {
  test(`role ${ROLE} — Float64Array round-trip through a public broker`, async ({ page }) => {
    test.setTimeout(4 * 60 * 1000);
    page.on("pageerror", (e) => console.error(`[${ROLE}:pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[relay]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const result = await page.evaluate(async ({ role, topic, broker }) => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[relay] ${s}`); };
      // Load mqtt.js (browser build) from CDN.
      await new Promise<void>((resolve, reject) => {
        const s = document.createElement("script");
        s.src = "https://unpkg.com/mqtt@5.10.1/dist/mqtt.min.js";
        s.onload = () => resolve();
        s.onerror = () => reject(new Error("mqtt.js CDN load failed"));
        document.head.appendChild(s);
      });
      const mqtt = (window as unknown as { mqtt: { connect(url: string, o?: unknown): MqttClient } }).mqtt;
      interface MqttClient {
        on(ev: "connect" | "error", cb: (a?: unknown) => void): void;
        on(ev: "message", cb: (t: string, p: Uint8Array) => void): void;
        subscribe(t: string, cb?: (e?: Error) => void): void;
        publish(t: string, m: string): void;
        end(): void;
      }
      const pingT = `${topic}/ping`, pongT = `${topic}/pong`;
      const t0 = Date.now();
      const client = mqtt.connect(broker, { connectTimeout: 20000, reconnectPeriod: 2000 });

      const connected = await new Promise<boolean>((resolve) => {
        client.on("connect", () => resolve(true));
        client.on("error", (e?: unknown) => log(`mqtt error: ${String(e)}`));
        setTimeout(() => resolve(false), 25000);
      });
      if (!connected) return { ok: false, stage: "broker-connect", error: "could not reach broker" };
      log(`connected to broker (+${Date.now() - t0}ms)`);

      if (role === "a") {
        return await new Promise<{ ok: boolean; stage: string; error?: string }>((resolve) => {
          client.on("message", (t: string, payload: Uint8Array) => {
            if (t !== pingT) return;
            const arr = JSON.parse(new TextDecoder().decode(payload)).arr as number[];
            log(`got ping, replying pong`);
            client.publish(pongT, JSON.stringify({ arr: arr.map((x) => x * 2) }));
            resolve({ ok: true, stage: "served-ping" });
          });
          client.subscribe(pingT);
          setTimeout(() => resolve({ ok: false, stage: "wait-ping", error: "no ping in 180s" }), 180000);
        });
      }

      return await new Promise<{ ok: boolean; stage: string; error?: string; rtt?: number }>((resolve) => {
        const sent = [1.5, -2.25, 3.125, 4.0];
        client.on("message", (t: string, payload: Uint8Array) => {
          if (t !== pongT) return;
          const got = JSON.parse(new TextDecoder().decode(payload)).arr as number[];
          const ok = got.length === 4 && got.every((x, i) => Math.abs(x - sent[i]! * 2) < 1e-12);
          log(`got pong, roundtrip ${ok ? "OK" : "MISMATCH"}`);
          resolve({ ok, stage: "roundtrip", rtt: Date.now() - t0 });
        });
        client.subscribe(pongT, () => {
          // Re-publish the ping periodically until a pong comes back (covers the
          // case where role a subscribed after our first publish).
          const send = (): void => client.publish(pingT, JSON.stringify({ arr: sent }));
          send();
          const iv = setInterval(send, 4000);
          setTimeout(() => { clearInterval(iv); resolve({ ok: false, stage: "roundtrip", error: "no pong in 180s" }); }, 180000);
        });
      });
    }, { role: ROLE, topic: TOPIC, broker: BROKER });

    console.log(`\n[relay-crossmachine ${ROLE}] result:`, JSON.stringify(result), "\n");
    expect(result.ok, `stage=${result.stage} ${result.error ?? ""}`).toBe(true);
  });
});
