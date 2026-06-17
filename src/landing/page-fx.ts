// Shared page effects for sub-pages (demo, viz, swarm, …): the fixed
// wavefunction field, the spectral scroll rail, and generic scroll-reveal.
// Mirrors the landing's feel via the shared q.css. Reduced-motion-safe; if JS
// never runs, the html.fx gate keeps all content fully visible.

import { initField } from "./field.js";
import { initScrollRail } from "./reveal.js";

export function initPageFx(): void {
  document.documentElement.classList.add("fx");

  const f = document.getElementById("field");
  if (f instanceof HTMLCanvasElement) initField(f);
  initScrollRail();

  const items = Array.from(document.querySelectorAll<HTMLElement>(
    ".section, table, .glass, [data-reveal]",
  ));
  items.forEach((el) => el.classList.add("reveal"));

  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce || !("IntersectionObserver" in window)) {
    items.forEach((el) => el.classList.add("in"));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      e.target.classList.add("in");
      io.unobserve(e.target);
    }
  }, { threshold: 0.08, rootMargin: "0px 0px -5% 0px" });
  items.forEach((el) => io.observe(el));
}
