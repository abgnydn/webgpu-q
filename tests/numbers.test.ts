// ─────────────────────────────────────────────────────────────
// numbers.test.ts — empirical reproducibility check on every
// numerical claim the project makes.
//
// For each entry in `benchmarks/numbers.ts`, we:
//   1. Run the verify() function (re-derives the value from
//      running source code at HEAD).
//   2. Assert the empirical value is within the declared
//      tolerance of the claimed value.
//   3. Assert the claim is also within tolerance of the
//      experimental / literature reference where one is
//      provided (these double as documentation tests).
//
// CI fails if any number drifts past tolerance. Drift means
// either:
//   • A genuine regression (caught early), or
//   • An intentional improvement that should bump the claim
//     value in `numbers.ts` (one-line edit) so the docs
//     stay in sync.
// ─────────────────────────────────────────────────────────────
import { describe, expect, test } from "vitest";
import { CLAIMS, type Claim } from "../benchmarks/numbers.js";

describe("benchmarks/numbers.ts — empirical reproducibility", () => {
  for (const claim of CLAIMS) {
    test(`${claim.id}: ${claim.label}`, { timeout: 120_000 }, () => {
      const got = claim.verify();
      const diff = Math.abs(got - claim.value);
      if (diff > claim.tolerance) {
        throw new Error(
          `claim "${claim.id}" drifted from numbers.ts:\n` +
          `  claim:      ${claim.value} ${claim.units}\n` +
          `  empirical:  ${got} ${claim.units}\n` +
          `  diff:       ${diff} (tolerance ${claim.tolerance})\n` +
          `If the new value is an intentional improvement, edit\n` +
          `benchmarks/numbers.ts to update "${claim.id}".`,
        );
      }
      expect(Number.isFinite(got)).toBe(true);
    });
  }

  // Reference cross-checks — the project's documented agreement
  // with experiment / literature SHOULD hold within the reference's
  // own tolerance. These are surfaced separately so a drift in
  // experimental match (e.g. a basis change) doesn't get hidden.
  describe("reference agreement", () => {
    const claimsWithRef: readonly Claim[] = CLAIMS.filter(
      (c) => c.reference !== undefined && c.reference.tolerance !== undefined,
    );
    for (const claim of claimsWithRef) {
      const ref = claim.reference!;
      test(`${claim.id} matches ${ref.source}`, () => {
        const diff = Math.abs(claim.value - ref.value);
        expect(diff).toBeLessThanOrEqual(ref.tolerance!);
      });
    }
  });
});
