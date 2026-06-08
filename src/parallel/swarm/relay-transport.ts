// Relay-hub SwarmTransport — cross-machine swarm over a free public broker.
//
// WebRTC P2P (webrtc-transport.ts) can't form a DataChannel between two cloud
// VMs behind symmetric NAT without a TURN key (free public TURN is dead). This
// transport sidesteps NAT entirely: every peer connects OUTBOUND to a shared
// MQTT broker and the broker fans messages out — a broadcast bus, but
// cross-machine. Proven between two GitHub Actions VMs (2026-06-08).
//
// All peers in a `room` subscribe + publish one topic; each ignores its own
// echoes and messages addressed to other peers. Payloads must be JSON-safe
// (Float64Array → number[] at the call site), per the SwarmMessage contract.
//
// Trade-off vs WebRTC: not peer-to-peer (a hub sees the traffic), and bounded
// by the broker's payload/throughput limits — fine for control + modest tensors,
// keep large slices local. The broker is best-effort free infra.

import {
  type SwarmTransport, type SwarmMessage, type SwarmListener, type PeerId,
} from "./transport.js";

interface MqttClient {
  on(ev: "connect", cb: () => void): void;
  on(ev: "error" | "close", cb: (e?: unknown) => void): void;
  on(ev: "message", cb: (topic: string, payload: Uint8Array) => void): void;
  subscribe(topic: string, cb?: (e?: Error) => void): void;
  publish(topic: string, msg: string): void;
  end(force?: boolean): void;
}
interface MqttGlobal { connect(url: string, opts?: Record<string, unknown>): MqttClient }

const MQTT_CDN = "https://unpkg.com/mqtt@5.10.1/dist/mqtt.min.js";
let mqttPromise: Promise<MqttGlobal> | null = null;

declare global {
  interface Window { mqtt?: MqttGlobal }
}

async function loadMqtt(): Promise<MqttGlobal> {
  if (mqttPromise) return mqttPromise;
  mqttPromise = (async () => {
    if (window.mqtt) return window.mqtt;
    await new Promise<void>((resolve, reject) => {
      const s = document.createElement("script");
      s.src = MQTT_CDN;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("mqtt.js CDN load failed"));
      document.head.appendChild(s);
    });
    if (!window.mqtt) throw new Error("mqtt.js loaded but window.mqtt missing");
    return window.mqtt;
  })();
  return mqttPromise;
}

export interface RelayTransportOpts {
  /** Rendezvous: all peers sharing a room reach each other through one topic. */
  readonly room: string;
  /** Self peer id (default: random 12-char). */
  readonly id?: string;
  /** Broker wss URL. Default: a free public broker (no auth). */
  readonly broker?: string;
  readonly onReady?: (id: PeerId) => void;
  readonly onError?: (err: Error) => void;
  readonly connectTimeoutMs?: number;
}

export class RelayTransport implements SwarmTransport {
  self: PeerId;
  private client: MqttClient | null = null;
  private readonly topic: string;
  private readonly broker: string;
  private readonly listeners = new Set<SwarmListener>();
  private readonly opts: RelayTransportOpts;

  constructor(opts: RelayTransportOpts) {
    this.opts = opts;
    this.self = opts.id ?? `wgq-${Math.abs(hashStr(opts.room + ":" + (opts.id ?? ""))).toString(36)}-${shortRand(opts.room)}`;
    this.topic = `wgq-relay/${opts.room}`;
    this.broker = opts.broker ?? "wss://broker.emqx.io:8084/mqtt";
  }

  /** Connect to the broker and subscribe to the room topic. */
  async openAsync(): Promise<void> {
    if (this.client) return;
    const mqtt = await loadMqtt();
    const client = mqtt.connect(this.broker, {
      connectTimeout: this.opts.connectTimeoutMs ?? 20000,
      reconnectPeriod: 2000,
    });
    this.client = client;
    client.on("error", (e) => this.opts.onError?.(e instanceof Error ? e : new Error(String(e))));
    client.on("message", (topic, payload) => {
      if (topic !== this.topic) return;
      let msg: SwarmMessage;
      try { msg = JSON.parse(new TextDecoder().decode(payload)) as SwarmMessage; }
      catch { return; }
      if (!msg || typeof msg !== "object" || !msg.from || !msg.type) return;
      if (msg.from === this.self) return;               // ignore our own echo
      if (msg.to && msg.to !== this.self) return;        // not addressed to us
      for (const fn of this.listeners) fn(msg);
    });
    await new Promise<void>((resolve, reject) => {
      const to = setTimeout(() => reject(new Error("broker connect timeout")), this.opts.connectTimeoutMs ?? 20000);
      client.on("connect", () => {
        client.subscribe(this.topic, (e) => {
          clearTimeout(to);
          if (e) { reject(e); return; }
          this.opts.onReady?.(this.self);
          resolve();
        });
      });
    });
  }

  open(): void {
    void this.openAsync().catch((e) => this.opts.onError?.(e instanceof Error ? e : new Error(String(e))));
  }

  close(): void {
    if (this.client) { try { this.client.end(true); } catch { /* ignore */ } this.client = null; }
    this.listeners.clear();
  }

  send(msg: Omit<SwarmMessage, "from">): void {
    if (!this.client) throw new Error("RelayTransport.send before open()");
    const full: SwarmMessage = { ...msg, from: this.self };
    this.client.publish(this.topic, JSON.stringify(full));
  }

  onMessage(listener: SwarmListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h | 0;
}
function shortRand(seed: string): string {
  // Deterministic-ish short suffix from the seed length + chars (no Math.random
  // in the swarm path; uniqueness within a room is the caller's job via `id`).
  return hashStr(seed + ":suffix").toString(36).slice(-4);
}
