// WebRTC-backed SwarmTransport.
//
// Step 2 of Phase D — cross-machine peer-to-peer swarm. Uses the
// PeerJS library (lazy-loaded from CDN) which encapsulates:
//   - signaling (SDP offer/answer + ICE candidates) via peerjs.com's
//     free public broker (no infra to deploy on our side)
//   - PeerConnection setup with Google STUN
//   - DataChannel (SCTP) for reliable in-order message passing
//
// API surface beyond SwarmTransport:
//   - `connectTo(otherId)`: dial an out-of-band peer ID. Required
//     because there's no broadcast discovery on WebRTC like there is
//     on BroadcastChannel — peers must exchange IDs out-of-band
//     (URL, QR code, copy/paste).
//   - `peers()`: list connected peer IDs.
//   - `onPeerConnected(cb)` / `onPeerDisconnected(cb)`: lifecycle.
//
// Honest limitations (carry-forward):
//   - peerjs.com broker is best-effort. If their broker is down, peers
//     can't discover each other. Self-hosted PeerServer is a follow-up.
//   - Symmetric-NAT corporate networks may fail without TURN (paid).
//   - PeerJS adds ~70 KB minified to the bundle (CDN, lazy-loaded so
//     it doesn't tax the default page weight).

import {
  type SwarmTransport, type SwarmMessage, type SwarmListener,
  type PeerId,
} from "./transport.js";

// Minimal PeerJS shape — we only use a tiny corner of the library.
interface PeerJSDataConnection {
  readonly peer: string;
  readonly open: boolean;
  send(data: unknown): void;
  close(): void;
  on(event: "open" | "close" | "error", cb: () => void): void;
  on(event: "data", cb: (data: unknown) => void): void;
  on(event: string, cb: (...args: unknown[]) => void): void;
}
interface PeerJSPeer {
  readonly id: string;
  connect(other: string, opts?: { reliable: boolean; serialization?: string }): PeerJSDataConnection;
  on(event: "open", cb: (id: string) => void): void;
  on(event: "connection", cb: (conn: PeerJSDataConnection) => void): void;
  on(event: "error", cb: (err: Error) => void): void;
  on(event: "close" | "disconnected", cb: () => void): void;
  destroy(): void;
}
interface PeerJSGlobal {
  Peer: new (id?: string, opts?: { debug?: number }) => PeerJSPeer;
}

const PEERJS_CDN = "https://unpkg.com/peerjs@1.5.4/dist/peerjs.min.js";
let peerJSPromise: Promise<PeerJSGlobal> | null = null;

declare global {
  interface Window { Peer?: PeerJSGlobal["Peer"] }
}

async function loadPeerJS(): Promise<PeerJSGlobal> {
  if (peerJSPromise) return peerJSPromise;
  peerJSPromise = (async () => {
    if (window.Peer) return { Peer: window.Peer };
    await new Promise<void>((resolve, reject) => {
      const s = document.createElement("script");
      s.src = PEERJS_CDN;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("PeerJS CDN load failed"));
      document.head.appendChild(s);
    });
    if (!window.Peer) throw new Error("PeerJS loaded but window.Peer missing");
    return { Peer: window.Peer };
  })();
  return peerJSPromise;
}

export interface WebRTCTransportOpts {
  /** Optional preferred peer ID. PeerJS will auto-generate one if
   *  omitted. Caller is responsible for ID uniqueness across the
   *  PeerJS public broker — a 12-character random ID is plenty. */
  readonly id?: string;
  /** Called when the underlying peer has opened on the broker (we
   *  have an ID and can be connected to). */
  readonly onReady?: (id: string) => void;
  /** Lifecycle hooks. */
  readonly onPeerConnected?: (id: PeerId) => void;
  readonly onPeerDisconnected?: (id: PeerId) => void;
  /** Called when the broker / signaling errors. */
  readonly onError?: (err: Error) => void;
}

export class WebRTCTransport implements SwarmTransport {
  self: PeerId;          // mutable: assigned once PeerJS opens
  private peer: PeerJSPeer | null = null;
  private readonly connections = new Map<PeerId, PeerJSDataConnection>();
  private readonly listeners = new Set<SwarmListener>();
  private readonly opts: WebRTCTransportOpts;

  constructor(opts: WebRTCTransportOpts = {}) {
    this.opts = opts;
    this.self = opts.id ?? "";
  }

  /** Open the peer connection. Resolves once PeerJS gives us an ID. */
  async openAsync(): Promise<void> {
    if (this.peer) return;
    const PJS = await loadPeerJS();
    const peer = new PJS.Peer(this.opts.id, { debug: 0 });
    this.peer = peer;
    return new Promise<void>((resolve, reject) => {
      peer.on("open", (id: string) => {
        this.self = id;
        this.opts.onReady?.(id);
        resolve();
      });
      peer.on("connection", (conn: PeerJSDataConnection) => {
        this.adoptConnection(conn);
      });
      peer.on("error", (err: Error) => {
        this.opts.onError?.(err);
        // Only reject the open() promise if we never got an "open" event.
        if (!this.self) reject(err);
      });
    });
  }

  /** SwarmTransport requires `open(): void`. The async version is the
   *  one callers should await; this no-op satisfies the interface for
   *  the synchronous code path. */
  open(): void {
    // Idempotent. Real init via openAsync().
    void this.openAsync().catch(() => { /* errors surface via onError */ });
  }

  /** Dial an out-of-band peer ID (shared by URL, QR, or copy/paste). */
  connectTo(otherId: PeerId): void {
    if (!this.peer) throw new Error("WebRTCTransport.connectTo before open()");
    if (otherId === this.self) return;
    if (this.connections.has(otherId)) return;
    const conn = this.peer.connect(otherId, { reliable: true, serialization: "json" });
    this.adoptConnection(conn);
  }

  /** Already-connected peer IDs (with open data channels). */
  peers(): readonly PeerId[] {
    const out: PeerId[] = [];
    for (const [id, c] of this.connections) if (c.open) out.push(id);
    return out;
  }

  close(): void {
    for (const c of this.connections.values()) {
      try { c.close(); } catch { /* ignore */ }
    }
    this.connections.clear();
    if (this.peer) {
      try { this.peer.destroy(); } catch { /* ignore */ }
      this.peer = null;
    }
    this.listeners.clear();
  }

  send(msg: Omit<SwarmMessage, "from">): void {
    if (!this.peer) {
      throw new Error("WebRTCTransport.send before open()");
    }
    const full: SwarmMessage = { ...msg, from: this.self };
    if (msg.to) {
      const conn = this.connections.get(msg.to);
      if (conn && conn.open) conn.send(full);
      return;
    }
    // Broadcast across all open connections.
    for (const conn of this.connections.values()) {
      if (conn.open) conn.send(full);
    }
  }

  onMessage(listener: SwarmListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private adoptConnection(conn: PeerJSDataConnection): void {
    const otherId = conn.peer;
    this.connections.set(otherId, conn);
    conn.on("open", () => {
      this.opts.onPeerConnected?.(otherId);
    });
    conn.on("data", (data: unknown) => {
      const msg = data as SwarmMessage;
      if (!msg || typeof msg !== "object" || !msg.from || !msg.type) return;
      if (msg.from === this.self) return;
      if (msg.to && msg.to !== this.self) return;
      for (const fn of this.listeners) fn(msg);
    });
    const cleanup = (): void => {
      this.connections.delete(otherId);
      this.opts.onPeerDisconnected?.(otherId);
    };
    conn.on("close", cleanup);
    conn.on("error", cleanup);
  }
}
