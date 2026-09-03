import { describe, expect, test } from "vitest";
import { bell, ghz, qft } from "../src/circuits.js";
import { CpuCircuit } from "../src/cpu-reference.js";

describe("circuit builders on CpuCircuit", () => {
  test("bell(2) produces a normalized Bell pair", () => {
    const c = new CpuCircuit(2);
    bell(c);
    expect(c.norm()).toBeCloseTo(1, 12);
  });

  test("ghz(3) has two equal-amplitude nonzero components", () => {
    const c = new CpuCircuit(3);
    ghz(c);
    const p = c.probabilities();
    const nonzero = Array.from(p).filter((x) => x > 1e-15);
    expect(nonzero.length).toBe(2);
    expect(nonzero[0]).toBeCloseTo(0.5, 12);
    expect(nonzero[1]).toBeCloseTo(0.5, 12);
  });

  test("qft(2) runs without throwing", () => {
    const c = new CpuCircuit(2);
    expect(() => qft(c)).not.toThrow();
    expect(c.norm()).toBeCloseTo(1, 12);
  });
});
