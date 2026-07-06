// webgpu-q service worker.
// Strategy:
//   - HTML documents: network-first, fall back to cache (so deploys update).
//   - Static assets (JS, CSS, SVG, PNG, JSON): cache-first (immutable bundle hashes).
//   - Cross-origin: bypass (don't cache 3rd-party).
// Cache key bumps on every release so old shells get evicted.

const CACHE_NAME = "webgpu-q-v0.6.1";
const SHELL = [
  "/",
  "/index.html",
  "/molecule.html",
  "/viz.html",
  "/learn.html",
  "/site.webmanifest",
  "/favicon-32.png",
  "/icon-192.png",
  "/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      // Best-effort precache of the shell. Don't fail install if any miss.
      Promise.allSettled(SHELL.map((u) => cache.add(u))),
    ).then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)),
      ),
    ).then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  // Only GET, only same-origin.
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  const isHtml = req.mode === "navigate" ||
                 req.headers.get("accept")?.includes("text/html");
  if (isHtml) {
    // Network-first: try the network, on failure fall back to cache.
    event.respondWith(
      fetch(req).then((res) => {
        const copy = res.clone();
        caches.open(CACHE_NAME).then((c) => c.put(req, copy)).catch(() => {});
        return res;
      }).catch(() => caches.match(req).then((r) => r ?? caches.match("/index.html"))),
    );
  } else {
    // Cache-first for static assets.
    event.respondWith(
      caches.match(req).then((cached) => {
        if (cached) return cached;
        return fetch(req).then((res) => {
          if (res.ok && res.status === 200) {
            const copy = res.clone();
            caches.open(CACHE_NAME).then((c) => c.put(req, copy)).catch(() => {});
          }
          return res;
        });
      }),
    );
  }
});
