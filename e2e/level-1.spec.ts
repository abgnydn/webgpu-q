import { test } from "@playwright/test";
import { assertLevelStatus, runLevelE2E } from "./lib/run-level.js";

test.describe("Level 1 — Statevector", () => {
  test("E1–E4 run end-to-end and produce pass artifacts", async ({ page }) => {
    test.setTimeout(8 * 60 * 1000);
    const result = await runLevelE2E(page, "runLevel1", 1);
    assertLevelStatus(result);
  });
});
