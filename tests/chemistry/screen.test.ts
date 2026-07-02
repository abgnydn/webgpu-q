// screenMolecules — the batch-screen primitive that a headless driver
// (Claude Science / Playwright / Modal / swarm) calls via window.__webgpuq.screen.

import { describe, expect, test } from "vitest";
import { screenMolecules } from "../../src/chemistry/screen.js";
import { molecules } from "../../src/chemistry/molecules.js";

describe("screenMolecules", () => {
  test("screens a library and ranks by HOMO–LUMO gap (descending)", () => {
    const art = screenMolecules(
      [
        { name: "h2", atoms: molecules.h2 },
        { name: "h2o", atoms: molecules.h2o },
        { name: "ch4", atoms: molecules.ch4 },
        { name: "lih", atoms: molecules.liH },
      ],
      { basis: "sto-3g", rankBy: "gap" },
    );

    expect(art.meta.kind).toBe("screen");
    expect(art.meta.count).toBe(4);
    expect(art.rows).toHaveLength(4);
    expect(art.status).toBe("pass");
    expect(art.rows.every((r) => r.status === "ok")).toBe(true);

    // Every ok row carries the standardized descriptors.
    for (const r of art.rows) {
      expect(Number.isFinite(r.totalEnergy!)).toBe(true);
      expect(Number.isFinite(r.homoLumoGapEv!)).toBe(true);
      expect(Number.isFinite(r.dipoleDebye!)).toBe(true);
    }

    // Ranked descending by gap.
    const gaps = art.rows.map((r) => r.homoLumoGapEv!);
    for (let i = 1; i < gaps.length; i++) expect(gaps[i - 1]!).toBeGreaterThanOrEqual(gaps[i]!);
  });

  test("ascending sort by energy flips the order", () => {
    const desc = screenMolecules([{ name: "h2", atoms: molecules.h2 }, { name: "h2o", atoms: molecules.h2o }], { rankBy: "energy", descending: false });
    const e = desc.rows.map((r) => r.totalEnergy!);
    for (let i = 1; i < e.length; i++) expect(e[i - 1]!).toBeLessThanOrEqual(e[i]!);
  });

  test("a bad molecule becomes an error row, not a thrown batch; status → partial", () => {
    const art = screenMolecules(
      [
        { name: "good", atoms: molecules.h2 },
        { name: "empty" }, // no atoms and no xyz
      ],
      {},
    );
    expect(art.status).toBe("partial");
    expect(art.diagnosis).toMatch(/1\/2/);
    const bad = art.rows.find((r) => r.name === "empty");
    expect(bad?.status).toBe("error");
    expect(bad?.error).toBeTruthy();
    // The good one still screened, and errored rows sink to the bottom.
    expect(art.rows[0]!.name).toBe("good");
    expect(art.rows[art.rows.length - 1]!.name).toBe("empty");
  });

  test("accepts raw XYZ text as input", () => {
    const xyz = "3\nwater\nO 0 0 0\nH 0 0 0.958\nH 0.926 0 -0.24\n";
    const art = screenMolecules([{ name: "water-xyz", xyz }], { basis: "sto-3g" });
    expect(art.status).toBe("pass");
    expect(art.rows[0]!.status).toBe("ok");
    expect(art.rows[0]!.nAtoms).toBe(3);
  });
});
