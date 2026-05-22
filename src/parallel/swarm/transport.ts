// Abstract swarm transport — the underlying message bus that connects
// peers in a webgpu-q distributed-compute swarm.
//
// We keep this interface narrow so the same `swarmMap` / kernel layer
// works over different transports:
//   - `BroadcastChannelTransport` for same-origin multi-tab on one
//     machine (ships in v0.7.x; no signaling, no STUN/TURN).
//   - `WebRTCTransport` (future) for cross-machine peer-to-peer via
//     a tiny signaling server. Same protocol; different wire.
//
// Messages are JSON-serializable; binary payloads (Float64Arrays etc.)
// get transferred as ArrayBuffer where possible.

export type PeerId = string;

export interface SwarmMessage {
  /** Sender peer id. */
  readonly from: PeerId;
  /** Optional target peer id; absent = broadcast. */
  readonly to?: PeerId;
  /** Protocol message type (see `protocol.ts`). */
  readonly type: string;
  /** Free-form payload — JSON-safe. */
  readonly payload?: unknown;
}

export type SwarmListener = (msg: SwarmMessage) => void;

export interface SwarmTransport {
  /** Stable id assigned at construction. */
  readonly self: PeerId;
  /** Connect to the bus (idempotent). */
  open(): void;
  /** Tear down. */
  close(): void;
  /** Send a message; to-id absent = broadcast to all peers. */
  send(msg: Omit<SwarmMessage, "from">): void;
  /** Listen for incoming messages addressed to us OR broadcast. */
  onMessage(listener: SwarmListener): () => void;
}

/** Generate a short stable peer id for the lifetime of a tab. */
export function makePeerId(): PeerId {
  return `peer-${Math.random().toString(36).slice(2, 8)}-${Date.now().toString(36).slice(-4)}`;
}
