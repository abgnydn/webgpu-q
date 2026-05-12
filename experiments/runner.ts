// ─────────────────────────────────────────────────────────────
// runner.ts — entry point for experiments/index.html.
//
// Wires the per-level "run all" buttons AND exposes a typed
// `window.__webgpuq` hook so headless drivers (Playwright e2e)
// can run any level by name and grab the artifacts back as
// JSON without going through the download dialog.
// ─────────────────────────────────────────────────────────────

import { runLevel1, wireRunAllButton } from "./level-1-statevector/run-all.js";
import { runLevel2, wireRunLevel2Button } from "./level-2-mps/run-all.js";
import { runLevel3, wireRunLevel3Button } from "./level-3-fusion/run-all.js";
import { runLevel6, wireRunLevel6Button } from "./level-6-chemistry/run-all.js";
import { runE32 } from "./level-6-chemistry/E32-ccsdt-gpu.js";
import { runE33 } from "./level-6-chemistry/E33-h2o-uvvis.js";
import { runE34 } from "./level-6-chemistry/E34-wallclock-vs-pyscf.js";

wireRunAllButton();
wireRunLevel2Button();
wireRunLevel3Button();
wireRunLevel6Button();

declare global {
  interface Window {
    __webgpuq?: {
      runLevel1: typeof runLevel1;
      runLevel2: typeof runLevel2;
      runLevel3: typeof runLevel3;
      runLevel6: typeof runLevel6;
      runE32: typeof runE32;
      runE33: typeof runE33;
      runE34: typeof runE34;
      ready: true;
    };
  }
}

window.__webgpuq = {
  runLevel1,
  runLevel2,
  runLevel3,
  runLevel6,
  runE32,
  runE33,
  runE34,
  ready: true,
};
