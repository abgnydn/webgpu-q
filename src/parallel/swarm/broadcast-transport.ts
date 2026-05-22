// BroadcastChannel-backed SwarmTransport.
//
// Works across SAME-ORIGIN tabs on a single browser. Zero infrastructure:
// no signaling server, no STUN/TURN, no relay. Browser's built-in
// BroadcastChannel API does same-origin pub/sub for us.
//
// Limitations vs full WebRTC:
//   - Same-origin only (all tabs must be on the same `webgpu-q` URL).
//   - Same-machine only — doesn't cross device boundaries.
//   - Messages are JSON-only (no transferable ArrayBuffer); large
//     Float64Array payloads are copied via structured clone, which is
//     fine for tile-sized work but bandwidth-limited at multi-MB.
//
// These constraints are fine for the typical "split a calc across 2-4
// tabs of my browser" use case. WebRTCTransport will lift them.

import {
  type SwarmTransport, type SwarmMessage, type SwarmListener,
  type PeerId, makePeerId,
} from "./transport.js";

export class BroadcastChannelTransport implements SwarmTransport {
  readonly self: PeerId;
  private channel: BroadcastChannel | null = null;
  private readonly channelName: string;
  private readonly listeners = new Set<SwarmListener>();

  constructor(channelName = "webgpu-q-swarm") {
    this.channelName = channelName;
    this.self = makePeerId();
  }

  open(): void {
    if (this.channel) return;
    this.channel = new BroadcastChannel(this.channelName);
    this.channel.addEventListener("message", (ev) => {
      const msg = ev.data as SwarmMessage;
      if (!msg || typeof msg !== "object" || !msg.from || !msg.type) return;
      if (msg.from === this.self) return;          // ignore own echoes
      if (msg.to && msg.to !== this.self) return;  // not for us
      for (const fn of this.listeners) fn(msg);
    });
  }

  close(): void {
    if (this.channel) {
      this.channel.close();
      this.channel = null;
    }
    this.listeners.clear();
  }

  send(msg: Omit<SwarmMessage, "from">): void {
    if (!this.channel) {
      throw new Error("BroadcastChannelTransport.send before open()");
    }
    const full: SwarmMessage = { ...msg, from: this.self };
    this.channel.postMessage(full);
  }

  onMessage(listener: SwarmListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}
