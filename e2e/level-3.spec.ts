import { test } from "@playwright/test";
import { assertLevelStatus, runLevelE2E } from "./lib/run-level.js";

test.describe("Level 3 — Kernel fusion", () => {
  test("E8–E13 run end-to-end and produce schema-conformant artifacts", async ({ page }) => {
    test.setTimeout(10 * 60 * 1000);
    const result = await runLevelE2E(page, "runLevel3", 3);
    assertLevelStatus(result);
  });
});
