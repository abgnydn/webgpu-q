// Scroll-reveal: fade + rise elements as they enter the viewport, lightly
// staggered within each row. Pure IntersectionObserver; if unsupported or the
// visitor prefers reduced motion, everything is shown immediately.

export function initReveal(): void {
  const sel = ".stats .stat, .demos .demo-card, .ladder .lvl, .section > .eyebrow, .section > h2, .section > .sub, .twocol > *, #proof .hero-live";
  const items = Array.from(document.querySelectorAll<HTMLElement>(sel));
  items.forEach((el) => el.classList.add("reveal"));

  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce || !("IntersectionObserver" in window)) {
    items.forEach((el) => el.classList.add("in"));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const el = e.target as HTMLElement;
      const sibs = Array.from(el.parentElement?.children ?? []).filter((c) => c.classList.contains("reveal"));
      el.style.transitionDelay = `${Math.min(sibs.indexOf(el), 6) * 55}ms`;
      el.classList.add("in");
      io.unobserve(el);
    }
  }, { threshold: 0.1, rootMargin: "0px 0px -6% 0px" });
  items.forEach((el) => io.observe(el));
}

// Spectral scroll-progress rail along the top of the viewport.
export function initScrollRail(): void {
  const rail = document.getElementById("scrollRail");
  if (!rail) return;
  const update = (): void => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    rail.style.width = `${max > 0 ? Math.min(100, (window.scrollY / max) * 100) : 0}%`;
  };
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update, { passive: true });
  update();
}
