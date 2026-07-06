import { test, expect, type Page } from "@playwright/test";

// ─────────────────────────────────────────────────────────────
// learn.spec.ts — e2e coverage for /learn.html ("Why is water
// special?"). A real browser bug motivated this spec: the atom
// label <text> sat on top of the draggable circle and swallowed
// pointerdown at the atom's center, so drags silently did
// nothing. DOM asserts can't catch that class of bug — only a
// REAL pointer gesture at the atom center can (test below).
//
// Robustness contract: the HF recompute may become async
// (worker, latest-wins) at any point. Every numeric assertion
// therefore POLLS the side panel until the value settles —
// nothing reads a number synchronously right after an action.
// ─────────────────────────────────────────────────────────────

const SIDE = "#side";
const MU_RE = /Dipole μ\s*(-?\d+(?:\.\d+)?)\s*D/;
const ANGLE_RE = /H–O–H angle\s*(-?\d+(?:\.\d+)?)°/;
const ETOT_RE = /Total energy\s*(-?\d+(?:\.\d+)?)\s*Ha/;
const SCF_RE = /SCF energy\s*(-?\d+(?:\.\d+)?)\s*Ha/;

/** Extract a number from the side panel via regex; NaN while absent (poll-friendly). */
async function readSideNumber(page: Page, re: RegExp): Promise<number> {
  const text = (await page.locator(SIDE).textContent()) ?? "";
  const m = text.match(re);
  return m ? Number(m[1]) : NaN;
}

/** Poll until the side panel value exists AND satisfies the matcher; return it. */
async function pollSideNumber(
  page: Page,
  re: RegExp,
  assert: (p: ReturnType<typeof expect.poll>) => Promise<void>,
  timeout = 60_000,
): Promise<number> {
  await assert(expect.poll(() => readSideNumber(page, re), { timeout }));
  return readSideNumber(page, re);
}

test.describe("Learn page — why is water special?", () => {
  // ── 8. no console errors anywhere in the spec ──────────────
  let errors: string[] = [];
  test.beforeEach(({ page }) => {
    errors = [];
    page.on("pageerror", (err) => errors.push(`[pageerror] ${err.message}`));
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(`[console.error] ${msg.text()}`);
    });
  });
  test.afterEach(() => {
    expect(errors, "no pageerror / console.error during the test").toEqual([]);
  });

  // ── 1. Feel: qualitative only — zero numbers, tutor + hint ─
  test("Feel: no numeric content, tutor chips + hint present", async ({ page }) => {
    await page.goto("/learn.html", { waitUntil: "domcontentloaded" });

    await expect(page.locator('.depth button[data-depth="feel"]')).toHaveAttribute("aria-selected", "true");
    await expect(page.locator("#hint")).toContainText("drag a white atom");
    await expect(page.locator(`${SIDE} .chip`)).toHaveCount(2);
    await expect(page.locator(SIDE)).toContainText("Why it matters");

    // Feel must stay qualitative even after the report lands: poll a moment,
    // then assert the panel never grew numbers (no "Dipole μ", no decimals).
    await expect.poll(async () => (await page.locator(SIDE).textContent()) ?? "").toContain("comfy shape");
    const text = (await page.locator(SIDE).textContent()) ?? "";
    expect(text).not.toContain("Dipole μ");
    expect(text).not.toMatch(/\d+\.\d+/);
  });

  // ── 2. Know: the real HF numbers for bent water ────────────
  test("Know: bent default shows μ≈1.7 D, angle≈104.5°, E≈−74.96 Ha", async ({ page }) => {
    await page.goto("/learn.html?d=know", { waitUntil: "domcontentloaded" });

    const mu = await pollSideNumber(page, MU_RE, (p) => p.toBeGreaterThan(1.5));
    expect(mu).toBeLessThan(2.0);

    const angle = await pollSideNumber(page, ANGLE_RE, (p) => p.toBeGreaterThan(104.0));
    expect(angle).toBeLessThan(105.0);

    const e = await pollSideNumber(page, ETOT_RE, (p) => p.toBeLessThan(-74.9));
    expect(Math.abs(e - -74.96)).toBeLessThan(0.02);
  });

  // ── 3. THE regression test: real pointer drag AT the atom center ──
  // The atom label <text> used to swallow pointerdown at the circle's
  // center (fixed with `svg text { pointer-events: none }`). A locator
  // .click() or DOM-event dispatch would not regress-test that — only a
  // real mouse gesture aimed at the exact center does.
  test("drag at the atom center recomputes μ and updates the URL g= param", async ({ page }) => {
    await page.goto("/learn.html?d=know", { waitUntil: "domcontentloaded" });

    // Wait for the first report + the boot syncURL to settle.
    const mu0 = await pollSideNumber(page, MU_RE, (p) => p.toBeGreaterThan(0.5));
    await expect.poll(() => new URL(page.url()).searchParams.get("g")).not.toBeNull();
    const g0 = new URL(page.url()).searchParams.get("g");

    // Real mouse drag, starting EXACTLY at the H atom's center (where the
    // label <text> sits) — ~60px toward lower-right.
    const h = page.locator("circle.draggable").last();
    const box = await h.boundingBox();
    expect(box).not.toBeNull();
    const cx = box!.x + box!.width / 2;
    const cy = box!.y + box!.height / 2;
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    await page.mouse.move(cx + 50, cy + 35, { steps: 8 });
    await page.mouse.up();

    // The gesture must have (a) changed the dipole, (b) written the new
    // geometry into the URL. Both polled — recompute may become async.
    await expect
      .poll(async () => Math.abs((await readSideNumber(page, MU_RE)) - mu0), { timeout: 60_000 })
      .toBeGreaterThan(0.01);
    await expect.poll(() => new URL(page.url()).searchParams.get("g")).not.toBe(g0);
  });

  // ── 4. Straighten: symmetry kills the dipole ───────────────
  test("Straighten at See → Know shows μ<0.05 D and angle≈180°", async ({ page }) => {
    await page.goto("/learn.html?d=see", { waitUntil: "domcontentloaded" });

    await page.locator("#straighten").click();
    // Wait out the ~650ms ease animation: the scene's live angle label
    // (visible at See) must reach ~180° before we read anything.
    await expect
      .poll(async () => {
        const t = (await page.locator("#scene").textContent()) ?? "";
        const m = t.match(/(-?\d+(?:\.\d+)?)°/);
        return m ? Number(m[1]) : NaN;
      }, { timeout: 15_000 })
      .toBeGreaterThan(179.5);

    // Re-skin to Know and assert the real numbers agree with symmetry.
    await page.locator('.depth button[data-depth="know"]').click();
    await pollSideNumber(page, MU_RE, (p) => p.toBeLessThan(0.05));
    await pollSideNumber(page, ANGLE_RE, (p) => p.toBeGreaterThan(179.5));
  });

  // ── 5. Depth switch = instant re-skin, never a recompute ───
  test("depth round-trip Know→Feel→Prove leaves Prove values bit-identical", async ({ page }) => {
    await page.goto("/learn.html?d=prove", { waitUntil: "domcontentloaded" });

    // Capture the fully-settled Prove numbers (exact rendered strings).
    await pollSideNumber(page, SCF_RE, (p) => p.toBeLessThan(-70));
    const grab = async (): Promise<string> => {
      const text = (await page.locator(SIDE).textContent()) ?? "";
      return [SCF_RE, MU_RE].map((re) => text.match(re)?.[0] ?? "").join(" | ");
    };
    const before = await grab();
    expect(before).toMatch(SCF_RE);

    // Round-trip through the other depths and back.
    for (const d of ["know", "feel", "prove"] as const) {
      await page.locator(`.depth button[data-depth="${d}"]`).click();
    }

    // Back at Prove: the strings must be IDENTICAL — a re-skin renders the
    // same report object; any recompute (or worker round-trip) that changed
    // the values would show up here.
    await expect.poll(grab).toBe(before);
  });

  // ── 6. Prove: basis switch to cc-pVDZ ──────────────────────
  test("Prove: switching basis to cc-pVDZ converges to SCF ≈ −76.027 Ha", async ({ page }) => {
    await page.goto("/learn.html?d=prove", { waitUntil: "domcontentloaded" });

    // STO-3G baseline must be on screen first.
    await pollSideNumber(page, SCF_RE, (p) => p.toBeLessThan(-74.9));

    await page.locator("#basisSel").selectOption("cc-pvdz");

    // Generous poll: the cc-pVDZ SCF is heavier and may move to a worker.
    const e = await pollSideNumber(page, SCF_RE, (p) => p.toBeLessThan(-75.5), 180_000);
    expect(Math.abs(e - -76.027)).toBeLessThan(0.005);
  });

  // ── 7. Reproducibility-as-a-URL ────────────────────────────
  test("loading a straightened ?g= URL reproduces μ<0.05 D directly", async ({ page }) => {
    await page.goto("/learn.html?d=know&g=-0.958,0.000;0.958,0.000", { waitUntil: "domcontentloaded" });

    await pollSideNumber(page, MU_RE, (p) => p.toBeLessThan(0.05));
    await pollSideNumber(page, ANGLE_RE, (p) => p.toBeGreaterThan(179.5));
  });

  test("Feel idle-wiggles the molecule (real normal modes), and stops on See", async ({ page }) => {
    await page.goto("/learn.html?d=feel", { waitUntil: "domcontentloaded" });
    const cy = () => page.locator("#scene circle.draggable").first().getAttribute("cy").then(Number);
    const samples: number[] = [];
    for (let i = 0; i < 8; i++) { samples.push(await cy()); await page.waitForTimeout(55); }
    const range = Math.max(...samples) - Math.min(...samples);
    expect(range, "Feel should animate the atoms").toBeGreaterThan(0.5);

    // Deeper depths are still (lab mode): See must not wiggle.
    await page.locator('.depth button[data-depth="see"]').click();
    await page.waitForTimeout(120);
    const s2: number[] = [];
    for (let i = 0; i < 6; i++) { s2.push(await cy()); await page.waitForTimeout(55); }
    expect(Math.max(...s2) - Math.min(...s2), "See should be still").toBeLessThan(0.5);
  });

  test("Know: orbital toggle renders a real MO field; μ-vs-angle plot is a computed curve", async ({ page }) => {
    await page.goto("/learn.html?d=know", { waitUntil: "domcontentloaded" });
    await pollSideNumber(page, MU_RE, (p) => p.toBeGreaterThan(0)); // wait for first report

    // μ(angle) sweep: a real polyline with the full 23-point ladder appears.
    await expect.poll(async () =>
      (await page.locator("#side polyline").first().getAttribute("points").catch(() => ""))?.split(" ").length ?? 0,
      { timeout: 30_000 },
    ).toBeGreaterThan(20);

    // Orbital overlay: off by default, an <image> field appears after toggling on.
    await expect(page.locator("#scene image")).toHaveCount(0);
    await page.locator("#orbToggle").click();
    await expect(page.locator("#scene image")).toHaveCount(1, { timeout: 30_000 });
    await expect(page.locator("#scene text").filter({ hasText: "HOMO−1" })).toBeVisible();
    await expect(page).toHaveURL(/orb=1/);
  });

  test("depth tablist is keyboard-operable (ArrowRight moves + activates)", async ({ page }) => {
    await page.goto("/learn.html?d=feel", { waitUntil: "domcontentloaded" });
    await page.locator('.depth button[data-depth="feel"]').focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.locator('.depth button[data-depth="see"]')).toHaveAttribute("aria-selected", "true");
    await expect(page).toHaveURL(/d=see/);
  });
});
