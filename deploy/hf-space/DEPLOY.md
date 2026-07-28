# Hugging Face Space deploy

Mirror of the static build as a HF Space: <https://huggingface.co/spaces/abgunaydin/webgpu-q>

`README.md` in this directory is the **Space card**, kept byte-identical to what is
live. It is uploaded to the Space repo root, where its YAML frontmatter configures
the Space. Do not put deploy notes in it — everything in that file is public.

Vercel (`webgpu-q.vercel.app`) stays canonical. The Space is a pinned snapshot that
links home; it is not a second identity to keep in sync.

## Update

```sh
STAGE=$(mktemp -d)
npm run build
cp -R dist/. "$STAGE"/
cp deploy/hf-space/README.md "$STAGE"/README.md      # dist/ has no README; the card must be added
hf upload abgunaydin/webgpu-q "$STAGE" . --repo-type=space \
  --commit-message="webgpu-q $(git describe --tags --abbrev=0) ($(git rev-parse --short=12 HEAD))"
```

Then bump the snapshot line at the bottom of the card so provenance stays truthful,
and re-upload just that file:

```sh
hf upload abgunaydin/webgpu-q deploy/hf-space/README.md README.md --repo-type=space
```

CLI is `hf` (the old `huggingface-cli` name is retired): `uv tool install huggingface_hub`.
There is no `[cli]` extra — asking for one warns and installs the base package anyway.

## Things that cost time once

- **The served host is `abgunaydin-webgpu-q.static.hf.space`** — static Spaces carry a
  `.static.` segment. Without it every path 404s with an identical 3020-byte HF error
  page, which looks exactly like a broken deploy. The bare `*.hf.space` form is wrong
  for static Spaces, and the Space API reports `host: null` either way, so it is no help.
- **`short_description` is capped at 60 characters.** Server-side validation, and the
  upload is rejected outright with the whole commit failing — not truncated.
- **`custom_headers` accepts only COEP, COOP and CORP**, keys *and* values lowercase.
  This is the load-bearing part: without COOP/COEP there is no `crossOriginIsolated`,
  so no `SharedArrayBuffer`, and `runRHFSCFAsync` silently degrades to single-threaded
  instead of failing — the same trap `vite.config.ts` documents for the dev server.
- **The wasm is served via HF's Xet CDN behind a 302.** It ends up correct
  (`application/wasm`, `access-control-allow-origin: *`, which satisfies COEP
  `require-corp`), but a `curl -I` without `-L` only shows the redirect. Follow it.
- **Static Spaces have no logs and cannot be restarted** — `hf spaces logs` returns
  500 and `hf spaces restart` refuses. Diagnose from response headers instead.
- **`app_build_command` is deliberately unused.** Letting HF run the build would stamp
  `__GIT_SHA__` from the *Space* repo's commit rather than this repo's, quietly
  mislabelling provenance in the UI. Build locally, upload `dist/`.
- The swarm needs same-origin sibling tabs, so it only works from the direct host, not
  from the iframe on the Space page.

## Verify after any deploy

```sh
B=https://abgunaydin-webgpu-q.static.hf.space
curl -sSI "$B/learn.html" | grep -i cross-origin        # expect COOP same-origin + COEP require-corp
curl -sSIL "$B/assets/wasm_eri_bg-*.wasm" | grep -i content-type   # expect application/wasm
```

In the page console, `crossOriginIsolated` must be `true`. Check it on the direct host
*and* on the Space page, since the embed depends on HF passing
`allow="cross-origin-isolated"` on its iframe.
