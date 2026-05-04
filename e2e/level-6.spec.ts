import { test } from "@playwright/test";
import { assertLevelStatus, runLevelE2E } from "./lib/run-level.js";

test.describe("Level 6 — Chemistry", () => {
  test("E16 runs end-to-end and produces an artifact", async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    const result = await runLevelE2E(page, "runLevel6", 6);
    assertLevelStatus(result);
  });
});
