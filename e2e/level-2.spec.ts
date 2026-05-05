import { test } from "@playwright/test";
import { assertLevelStatus, runLevelE2E } from "./lib/run-level.js";

test.describe("Level 2 — MPS", () => {
  test("E5–E7 + E18/E19 (Phase B) run end-to-end and produce artifacts", async ({ page }) => {
    test.setTimeout(15 * 60 * 1000);
    const result = await runLevelE2E(page, "runLevel2", 2);
    assertLevelStatus(result);
  });
});
