import { test } from "@playwright/test";
import { assertLevelStatus, runLevelE2E } from "./lib/run-level.js";

test.describe("Level 2 — MPS", () => {
  test("E5–E7 run end-to-end and produce pass artifacts", async ({ page }) => {
    test.setTimeout(8 * 60 * 1000);
    const result = await runLevelE2E(page, "runLevel2", 2);
    assertLevelStatus(result);
  });
});
