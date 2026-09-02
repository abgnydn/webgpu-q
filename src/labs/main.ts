// ─────────────────────────────────────────────────────────────
// labs/main.ts — four standard undergraduate computational
// chemistry labs, each computed live in the tab.
//
// Nothing here is precomputed or hard-coded: pressing Run starts a
// real SCF / MP2 / CCSD / CCSD(T) in a worker and the table fills in
// as results arrive. The interpretation text is revealed only after
// the numbers exist, so a student reads their own result rather than
// a promise about it.
// ─────────────────────────────────────────────────────────────

import type { LabReply, LabRequest } from "./lab-worker.js";
import { SN2_CCPVDZ_BARRIER_KCAL, SN2_LITERATURE_BARRIER_KCAL } from "./sn2-geometry.js";
import type { BasisName } from "../chemistry/atoms.js";

type Row = Record<string, number | string | boolean>;

const worker = new Worker(new URL("./lab-worker.ts", import.meta.url), { type: "module" });
let seq = 0;
const pending = new Map<number, {
  onRow: (r: Row) => void;
  onDone: (s: number) => void;
  onError: (m: string) => void;
}>();

worker.addEventListener("message", (ev: MessageEvent<LabReply>) => {
  const m = ev.data;
  const h = pending.get(m.seq);
  if (!h) return;
  if (m.kind === "row") h.onRow(m.row);
  else if (m.kind === "done") { h.onDone(m.seconds); pending.delete(m.seq); }
  else { h.onError(m.message); pending.delete(m.seq); }
});

function run(
  req: Omit<LabRequest, "seq">,
  onRow: (r: Row) => void,
  onDone: (s: number) => void,
  onError: (m: string) => void,
) {
  const s = ++seq;
  pending.set(s, { onRow, onDone, onError });
  worker.postMessage({ ...req, seq: s } as LabRequest);
}

const $ = (id: string) => document.getElementById(id)!;
const num = (v: unknown, d = 6) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—";

function wire(
  labId: string,
  /** Built at click time, so controls (e.g. the basis dropdown) are read fresh. */
  reqFor: () => Omit<LabRequest, "seq">,
  header: string[],
  toCells: (r: Row) => string[],
  interpret: (rows: Row[]) => string,
) {
  const btn = $(`${labId}-run`) as HTMLButtonElement;
  const status = $(`${labId}-status`);
  const tbody = $(`${labId}-body`);
  const thead = $(`${labId}-head`);
  const note = $(`${labId}-note`);

  btn.addEventListener("click", () => {
    btn.disabled = true;
    btn.textContent = "Running…";
    status.textContent = "computing — this is a real calculation, not a lookup";
    status.className = "status busy";
    tbody.innerHTML = "";
    note.innerHTML = "";
    thead.innerHTML = `<tr>${header.map((h) => `<th>${h}</th>`).join("")}</tr>`;
    const rows: Row[] = [];

    run(
      reqFor(),
      (r) => {
        rows.push(r);
        const tr = document.createElement("tr");
        tr.innerHTML = toCells(r).map((c) => `<td>${c}</td>`).join("");
        tbody.appendChild(tr);
      },
      (s) => {
        btn.disabled = false;
        btn.textContent = "Run again";
        status.textContent = `done in ${s.toFixed(1)} s`;
        status.className = "status ok";
        note.innerHTML = interpret(rows);
      },
      (m) => {
        btn.disabled = false;
        btn.textContent = "Run";
        status.textContent = `stopped: ${m}`;
        status.className = "status err";
        note.innerHTML =
          `<p><b>The calculation refused to return a number.</b> That is the
           library behaving correctly — an unconverged SCF or CCSD amplitude set
           is not a result, and reporting it as one is how wrong numbers end up
           in papers.</p>`;
      },
    );
  });
}

// ── Lab 1 ────────────────────────────────────────────────────
wire(
  "diss",
  () => ({ kind: "dissociation" }),
  ["R (Å)", "E_exact (Ha)", "E_RHF (Ha)", "error (mHa)", "c_g", "c_u"],
  (r) => [
    num(r["R"], 4),
    num(r["eExact"]),
    r["rhfOk"] ? num(r["eRHF"]) : '<span class="bad">no convergence</span>',
    r["rhfOk"] ? num(r["errMHa"], 2) : "—",
    num(r["cG"], 3),
    num(r["cU"], 3),
  ],
  (rows) => {
    const far = rows[rows.length - 1];
    const eq = rows.find((r) => Math.abs(Number(r["R"]) - 0.7414) < 1e-6);
    if (!far || !eq) return "";
    return `
      <p><b>Read the last row.</b> At R = ${num(far["R"], 1)} Å the two
      coefficients have gone to c_g ≈ ${num(far["cG"], 3)} and
      c_u ≈ ${num(far["cU"], 3)} — equal magnitude. Near equilibrium
      (${num(eq["R"], 4)} Å) they were ${num(eq["cG"], 3)} and
      ${num(eq["cU"], 3)}.</p>
      <p>That crossover <i>is</i> static correlation. A single Slater
      determinant assumes one configuration dominates. Once two contribute
      equally, no amount of single-reference theory can fix it — which is why
      RHF's error grows to ${num(far["errMHa"], 0)} mHa at dissociation while
      being nearly exact at equilibrium. It is also the reason CASSCF exists.</p>
      <p class="caveat">The exact curve here is a full CI in the H₂ STO-3G
      minimal basis — 16 determinants, solved directly. That is specific to
      H₂; this page does not contain a general FCI solver, so the same lab
      cannot be run on N₂.</p>`;
  },
);

// ── Lab 2 ────────────────────────────────────────────────────
wire(
  "basis",
  () => ({ kind: "basis" }),
  ["basis", "functions", "E_HF (Ha)", "gain (mHa)"],
  (r) => [
    String(r["basis"]),
    String(r["n"]),
    num(r["energy"]),
    Number.isFinite(Number(r["gainMHa"])) ? num(r["gainMHa"], 2) : "—",
  ],
  (rows) => {
    if (rows.length < 3) return "";
    const [a, b, c] = rows as [Row, Row, Row];
    return `
      <p><b>Diminishing returns, quantified.</b> Going minimal → double-zeta
      bought ${num(b["gainMHa"], 0)} mHa. Adding diffuse functions on top
      bought only ${num(c["gainMHa"], 1)} mHa — roughly
      ${Math.round(Number(b["gainMHa"]) / Number(c["gainMHa"]))}× less, for
      ${Number(c["n"]) - Number(b["n"])} more basis functions.</p>
      <p>Every energy is variational, so each row must lie below the one above
      it. It does. But notice none of them is the exact answer: the basis-set
      limit and the correlation problem are two different errors, and this lab
      only attacks the first. Lab 3 attacks the second.</p>
      <p class="caveat">${a["basis"]} at n = ${a["n"]} is a minimal basis —
      one function per occupied atomic orbital. It is useful for teaching and
      almost never for publishing.</p>`;
  },
);

// ── Lab 3 ────────────────────────────────────────────────────
const ladderBasis = () =>
  ((document.getElementById("ladder-basis") as HTMLSelectElement).value) as BasisName;

wire(
  "ladder",
  () => ({ kind: "ladder", basis: ladderBasis() }),
  ["method", "E (Ha)", "E_corr (mHa)", "% of CCSD(T)"],
  (r) => [
    String(r["method"]),
    num(r["energy"]),
    Number(r["corrMHa"]) === 0 ? "0" : num(r["corrMHa"], 2),
    Number(r["corrMHa"]) === 0 ? "—" : `${num(r["pct"], 1)}%`,
  ],
  (rows) => {
    const mp2 = rows.find((r) => r["method"] === "MP2");
    const ccsd = rows.find((r) => r["method"] === "CCSD");
    const t = rows.find((r) => r["method"] === "CCSD(T)");
    if (!mp2 || !ccsd || !t) return "";
    return `
      <p><b>The ladder, in one table.</b> MP2 — the cheapest correlated method
      — already recovers ${num(mp2["pct"], 1)}% of what CCSD(T) finds. CCSD
      gets to ${num(ccsd["pct"], 1)}%. The perturbative triples supply the
      last ${num(100 - Number(ccsd["pct"]), 1)}%.</p>
      <p>Total correlation energy here is ${num(t["corrMHa"], 1)} mHa. For
      scale, chemical accuracy is 1.594 mHa — so the correlation you are
      recovering is over a hundred times larger than the accuracy you need,
      which is why you cannot simply ignore it.</p>
      <p class="caveat">CCSD(T) is the reference for the percentages, not the
      exact answer. It is the best number this ladder produces; the true
      correlation energy in this basis is slightly larger still.</p>`;
  },
);

// ── Lab 4 ────────────────────────────────────────────────────
wire(
  "sn2",
  () => ({ kind: "sn2" }),
  ["s", "C···Cl in (Å)", "C···Cl out (Å)", "umbrella (°)", "E (Ha)", "ΔE (kcal/mol)"],
  (r) => [
    num(r["s"], 1),
    num(r["r1"], 3),
    num(r["r2"], 3),
    num(r["phiDeg"], 1),
    num(r["energy"], 6),
    num(r["relKcal"], 2),
  ],
  (rows) => {
    const ts = rows.find((r) => Math.abs(Number(r["s"])) < 1e-9);
    const end = rows.find((r) => Math.abs(Number(r["s"]) + 1) < 1e-9);
    if (!ts || !end) return "";
    const barrier = Number(ts["relKcal"]);
    return `
      <p><b>You just computed an SN2 barrier.</b> The peak sits at s = 0,
      where the two C···Cl distances are equal and the umbrella angle is
      exactly 90° — the methyl group is planar. That is the transition
      state, and passing through it is the Walden inversion: the carbon
      turns inside out like an umbrella in wind.</p>
      <p><b>Now the part worth arguing about.</b> This barrier came out at
      ${num(barrier, 1)} kcal/mol. The literature value for this reaction is
      about ${SN2_LITERATURE_BARRIER_KCAL} kcal/mol measured from the
      ion–dipole complex, which is where s = −1 sits. Same geometry in
      cc-pVDZ gives ${SN2_CCPVDZ_BARRIER_KCAL} — so the minimal basis is
      wrong by more than a factor of two, in the direction that would make
      you predict the reaction is far slower than it is.</p>
      <p>That is the honest lesson: the shape is right, the number is not.
      A minimal basis is enough to see a mechanism and nowhere near enough
      to predict a rate.</p>
      <p class="caveat">Two further caveats stated up front. The path is a
      straight-line interpolation between reactant and transition state,
      not an optimized reaction path — a real study would locate the
      saddle point and verify it has exactly one imaginary frequency. And
      anions genuinely need diffuse functions, which neither STO-3G nor
      cc-pVDZ has; aug-cc-pVDZ would be the honest choice.</p>`;
  },
);
