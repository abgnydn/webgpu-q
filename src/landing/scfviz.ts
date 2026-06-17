// Hero centerpiece: a live readout of a real Hartree–Fock SCF.
//   • convergence plot — the actual per-iteration energy trace (hf.history)
//     animated as it drops to the ground state;
//   • MO ladder — the real orbital energies around the HOMO–LUMO gap.
// Every pixel is real data from the calculation that just ran in this tab.

const GREEN = "#34d399", BLUE = "#6ea8ff", WARM = "#ffb56e", MUTED = "#7a8294", TEXT = "#e6e9f0";
const HARTREE_EV = 27.211386;

export interface ScfViz { show(history: readonly number[], eps: Float64Array, nOcc: number): void }

export function initScfViz(plot: HTMLCanvasElement, ladder: HTMLCanvasElement): ScfViz {
  const p = plot.getContext("2d");
  const l = ladder.getContext("2d");
  if (!p || !l) return { show() {} };
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const DPR = Math.min(window.devicePixelRatio || 1, 2);
  // Match the backing store to the CSS-driven display box ONLY (never write the
  // display size back, or a ResizeObserver loop ensues). The observer re-runs
  // this once layout settles, so the canvas ends at the correct full width.
  const sizeCanvas = (c: HTMLCanvasElement, ctx: CanvasRenderingContext2D, hh: number): { w: number; h: number } => {
    const w = c.clientWidth || c.parentElement?.clientWidth || 300;
    const h = c.clientHeight || hh;
    c.width = Math.max(1, Math.round(w * DPR)); c.height = Math.max(1, Math.round(h * DPR));
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    return { w, h };
  };

  function drawPlot(hist: readonly number[], upTo: number, total: number, done: boolean): void {
    const { w, h } = sizeCanvas(plot, p!, 150);
    p!.clearRect(0, 0, w, h);
    if (total < 2) return;
    const padL = 8, padR = 10, padT = 12, padB = 18;
    let emin = Infinity, emax = -Infinity;
    for (const e of hist) { if (e < emin) emin = e; if (e > emax) emax = e; }
    if (emax - emin < 1e-9) { emax += 0.5; emin -= 0.5; }
    const xAt = (i: number): number => padL + (i / (total - 1)) * (w - padL - padR);
    const yAt = (e: number): number => padT + (1 - (e - emin) / (emax - emin)) * (h - padT - padB);
    // baseline grid
    p!.strokeStyle = "rgba(122,130,148,0.16)"; p!.lineWidth = 1;
    for (let g = 0; g <= 2; g++) { const y = padT + (g / 2) * (h - padT - padB); p!.beginPath(); p!.moveTo(padL, y); p!.lineTo(w - padR, y); p!.stroke(); }
    // area + line up to `upTo`
    const lim = Math.max(1, Math.min(upTo, total));
    p!.beginPath(); p!.moveTo(xAt(0), yAt(hist[0]!));
    for (let i = 1; i < lim; i++) p!.lineTo(xAt(i), yAt(hist[i]!));
    const grad = p!.createLinearGradient(0, padT, 0, h - padB);
    grad.addColorStop(0, "rgba(52,211,153,0.28)"); grad.addColorStop(1, "rgba(52,211,153,0)");
    p!.lineTo(xAt(lim - 1), h - padB); p!.lineTo(xAt(0), h - padB); p!.closePath();
    p!.fillStyle = grad; p!.fill();
    p!.beginPath(); p!.moveTo(xAt(0), yAt(hist[0]!));
    for (let i = 1; i < lim; i++) p!.lineTo(xAt(i), yAt(hist[i]!));
    p!.strokeStyle = GREEN; p!.lineWidth = 2; p!.shadowColor = GREEN; p!.shadowBlur = 8; p!.stroke(); p!.shadowBlur = 0;
    for (let i = 0; i < lim; i++) { p!.beginPath(); p!.arc(xAt(i), yAt(hist[i]!), 2.1, 0, Math.PI * 2); p!.fillStyle = BLUE; p!.fill(); }
    // converged marker
    const last = lim - 1;
    p!.beginPath(); p!.arc(xAt(last), yAt(hist[last]!), done ? 4 : 3, 0, Math.PI * 2);
    p!.fillStyle = done ? GREEN : WARM; p!.fill();
    p!.fillStyle = MUTED; p!.font = "10px ui-monospace, monospace";
    p!.fillText("energy / Ha", padL, 9);
    p!.fillText(`${total - 1} iters`, w - padR - 42, h - 5);
    if (done) { p!.fillStyle = TEXT; p!.fillText("✓ ground state", xAt(last) - 78, yAt(hist[last]!) - 7); }
  }

  function drawLadder(eps: Float64Array, nOcc: number): void {
    const { w, h } = sizeCanvas(ladder, l!, 104);
    l!.clearRect(0, 0, w, h);
    const n = eps.length;
    if (n === 0) return;
    const homo = Math.max(0, nOcc - 1), lumo = Math.min(n - 1, nOcc);
    const lo = Math.max(0, homo - 4), hi = Math.min(n - 1, lumo + 4);
    let emin = Infinity, emax = -Infinity;
    for (let i = lo; i <= hi; i++) { const e = eps[i]!; if (e < emin) emin = e; if (e > emax) emax = e; }
    if (emax - emin < 1e-9) { emax += 0.1; emin -= 0.1; }
    const padT = 10, padB = 10, x1 = 14, x2 = w - 14;
    const yAt = (e: number): number => padT + (1 - (e - emin) / (emax - emin)) * (h - padT - padB);
    for (let i = lo; i <= hi; i++) {
      const occ = i < nOcc;
      const y = yAt(eps[i]!);
      l!.strokeStyle = occ ? BLUE : "rgba(122,130,148,0.5)";
      l!.lineWidth = (i === homo || i === lumo) ? 2.4 : 1.4;
      l!.beginPath(); l!.moveTo(x1, y); l!.lineTo(x2, y); l!.stroke();
      if (occ) { // two electrons
        l!.fillStyle = i === homo ? GREEN : BLUE;
        l!.beginPath(); l!.arc(x1 + (x2 - x1) * 0.36, y, 1.9, 0, Math.PI * 2); l!.fill();
        l!.beginPath(); l!.arc(x1 + (x2 - x1) * 0.64, y, 1.9, 0, Math.PI * 2); l!.fill();
      }
    }
    // gap annotation
    const gapEv = (eps[lumo]! - eps[homo]!) * HARTREE_EV;
    const yH = yAt(eps[homo]!), yL = yAt(eps[lumo]!), xa = x1 + 2;
    l!.strokeStyle = WARM; l!.lineWidth = 1; l!.beginPath(); l!.moveTo(xa, yH); l!.lineTo(xa, yL); l!.stroke();
    l!.fillStyle = WARM; l!.font = "9px ui-monospace, monospace";
    l!.fillText(`${gapEv.toFixed(1)} eV`, x1, (yH + yL) / 2 + 3);
    l!.fillStyle = GREEN; l!.fillText("HOMO", x2 - 30, yH + (yH < h - 14 ? 11 : -4));
    l!.fillStyle = MUTED; l!.fillText("LUMO", x2 - 30, yL - 4);
  }

  let raf = 0;
  let last: { h: readonly number[]; eps: Float64Array; nOcc: number } | null = null;
  function render(animate: boolean): void {
    if (!last) return;
    cancelAnimationFrame(raf);
    drawLadder(last.eps, last.nOcc);
    const total = last.h.length;
    if (total < 2) { drawPlot(last.h, total, total, true); return; }
    const dur = (animate && !reduce) ? Math.min(1500, 130 * total + 300) : 0;
    const t0 = performance.now();
    const step = (now: number): void => {
      const k = dur ? Math.min(1, (now - t0) / dur) : 1;
      drawPlot(last!.h, Math.max(1, Math.ceil(k * total)), total, k >= 1);
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  }
  let rT: number | undefined;
  const reflow = (): void => { window.clearTimeout(rT); rT = window.setTimeout(() => render(false), 60); };
  window.addEventListener("resize", reflow);
  if ("ResizeObserver" in window) {
    const ro = new ResizeObserver(reflow);
    ro.observe(plot);
    ro.observe(ladder);
  }
  return {
    show(history: readonly number[], eps: Float64Array, nOcc: number): void {
      last = { h: history, eps, nOcc };
      render(true);
    },
  };
}
