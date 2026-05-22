// Swarm protocol — messages exchanged between peers.
//
// Roles are dynamic: any peer can act as master for a job it initiates,
// and any peer can opt in as a worker for someone else's job. A single
// tab can be master + worker simultaneously (it can claim its own tiles
// when no one else volunteers).

import type { PeerId } from "./transport.js";

/** Sent periodically by every peer announcing presence. */
export interface HelloMsg {
  readonly type: "hello";
  readonly payload: { capabilities: string[]; ts: number };
}

/** Master announces a new job. */
export interface JobAnnounceMsg {
  readonly type: "job-announce";
  readonly payload: {
    readonly jobId: string;
    readonly kind: string;          // e.g. "primes-tile" — what kernel to run
    readonly nTiles: number;
    readonly tileSpec?: unknown;    // shared per-tile context (broadcast)
  };
}

/** Worker claims a specific tile. */
export interface TileClaimMsg {
  readonly type: "tile-claim";
  readonly payload: { jobId: string; tileIndex: number };
}

/** Master accepts a claim (or rejects if already taken). */
export interface TileAssignMsg {
  readonly type: "tile-assign";
  readonly payload: {
    readonly jobId: string;
    readonly tileIndex: number;
    readonly worker: PeerId;
    readonly accepted: boolean;
    readonly tile?: unknown;        // tile-specific input
  };
}

/** Worker reports a tile's result. */
export interface TileResultMsg {
  readonly type: "tile-result";
  readonly payload: { jobId: string; tileIndex: number; result: unknown };
}

/** Worker reports a tile failed (master may reassign). */
export interface TileFailMsg {
  readonly type: "tile-fail";
  readonly payload: { jobId: string; tileIndex: number; error: string };
}

export type SwarmProtoMsg =
  | HelloMsg | JobAnnounceMsg | TileClaimMsg
  | TileAssignMsg | TileResultMsg | TileFailMsg;
