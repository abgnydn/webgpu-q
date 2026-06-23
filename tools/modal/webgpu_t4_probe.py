"""
webgpu_t4_probe.py — run webgpu-q's GPU kernels on a real NVIDIA T4 via Modal,
and HARD-GATE on a non-fallback adapter (reject SwiftShader / llvmpipe / lavapipe).

Why: every webgpu-q benchmark is on a single M2 Pro (n=1). This gets ONE honest
second-GPU artifact for ~$0 (Modal's free $30/mo credits), and is the de-risk
step before building any CI lane. See docs/webgpu-ci-providers.md.

It points Playwright at the LIVE deployed site — the WGSL/WASM runs client-side
in the headless Chrome on Modal's T4 regardless of where the page is served, so
no clone/npm-build is needed in the container.

────────────────────────────────────────────────────────────────────────────
STATUS: written from the recipe (jasonmayes / Chrome team + tigerabrodi Feb
2026), SYNTAX-CHECKED ONLY — never run from the dev container (no GPU there).
Expect to iterate once on the FRAGILE points, all flagged `# VERIFY:` below:
  1. NVIDIA Vulkan *userspace* / ICD visibility inside Modal's container
     (the _diag() step prints nvidia-smi + vulkaninfo so the FIRST run tells you
     immediately whether you're on the NVIDIA ICD or fell back to lavapipe).
  2. Dawn's driver-version blocklist cutoff (the 570+ number drifts).
The GATE is deliberately strict: if it lands on software it RAISES rather than
emitting fake "GPU" numbers.
────────────────────────────────────────────────────────────────────────────

Usage:
    pip install modal && modal setup
    modal run tools/modal/webgpu_t4_probe.py                 # adapter probe only (cheap, ~30s)
    modal run tools/modal/webgpu_t4_probe.py --kernels       # probe + L1 & L3 kernels
    modal run tools/modal/webgpu_t4_probe.py --kernels --levels 1,3,E34   # + the (T) CPU/GPU bench (heavy)
"""

import json
import subprocess

import modal

# The deployed experiments page exposes window.__webgpuq.{runLevel1,runLevel3,
# runLevel6,runE34,ready}. Override with --url to hit a preview/local tunnel.
DEFAULT_URL = "https://webgpu-q.vercel.app/experiments/"

# Verified headless-Chrome-on-NVIDIA-Vulkan flag set (compute path).
# --disable-vulkan-surface disables canvas but leaves WGSL compute intact, which
# is exactly the L1/L3/(T) case. The Dawn blocklist flag is the one most likely
# to matter on a 2026 driver.
CHROME_ARGS = [
    "--no-sandbox",
    "--headless=new",
    "--use-angle=vulkan",
    "--enable-features=Vulkan",
    "--disable-vulkan-surface",
    "--enable-unsafe-webgpu",
    "--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist",  # VERIFY: 570+ cutoff drifts
]

# requestAdapter + return its info. Tries high-performance, then default. Handles
# both the modern (adapter.info) and legacy (requestAdapterInfo) shapes.
GATE_JS = r"""
async () => {
  if (!('gpu' in navigator) || !navigator.gpu) return { ok: false, reason: 'no navigator.gpu' };
  let a = null;
  try { a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }); } catch (e) {}
  if (!a) { try { a = await navigator.gpu.requestAdapter(); } catch (e) {} }
  if (!a) return { ok: false, reason: 'requestAdapter returned null' };
  const info = a.info ? a.info : (a.requestAdapterInfo ? await a.requestAdapterInfo() : {});
  const fb = (a.info && 'isFallbackAdapter' in a.info) ? a.info.isFallbackAdapter
           : ('isFallbackAdapter' in a) ? a.isFallbackAdapter : null;
  return {
    ok: true,
    vendor: info.vendor || '', architecture: info.architecture || '',
    device: info.device || '', description: info.description || '',
    isFallbackAdapter: fb,
  };
}
"""

_SOFTWARE = ("swiftshader", "llvmpipe", "lavapipe", "software", "google inc.")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # vulkan-tools (vulkaninfo) for the diag; dbus is required before Chrome launches.
    .apt_install("vulkan-tools", "libvulkan1", "dbus", "ca-certificates")
    .pip_install("playwright==1.49.0")
    .run_commands("playwright install-deps chromium", "playwright install chromium")
    # VERIFY: ask the NVIDIA Container runtime to mount the GL/Vulkan userspace +
    # ICD (not just compute). If vulkaninfo still shows only llvmpipe, install a
    # libnvidia-gl-<branch> matching `nvidia-smi`'s driver, or set VK_ICD_FILENAMES.
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "NVIDIA_VISIBLE_DEVICES": "all"})
)

app = modal.App("webgpu-q-t4-probe", image=image)


def _diag() -> dict:
    """Self-diagnosis: prove (or disprove) the NVIDIA ICD is visible before Chrome."""
    out = {}
    for name, cmd in (
        ("nvidia_smi", ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        ("vulkaninfo", ["bash", "-lc", "vulkaninfo 2>/dev/null | grep -iE 'deviceName|driverName' | head"]),
    ):
        try:
            out[name] = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:  # noqa: BLE001
            out[name] = f"<{name} failed: {e}>"
    return out


def _run(levels: list[str], url: str) -> dict:
    from playwright.sync_api import sync_playwright

    diag = _diag()
    # Chrome needs a running dbus session bus.
    subprocess.run(["bash", "-lc", "service dbus start || /etc/init.d/dbus start || true"], check=False)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,  # we pass --headless=new ourselves; Playwright's --headless suppresses GPU
            args=CHROME_ARGS,
            ignore_default_args=["--use-angle=swiftshader-webgl"],  # trap #1: the injected SwiftShader arg
        )
        page = browser.new_page()
        page.set_default_timeout(0)  # heavy kernels (CCSD(T)) can run minutes; gate Modal-side instead
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)

        info = page.evaluate(GATE_JS)

        # ── HARD GATE ── reject software / missing adapter outright.
        blob = " ".join(
            str(info.get(k, "")) for k in ("vendor", "architecture", "device", "description")
        ).lower()
        is_software = info.get("isFallbackAdapter") is True or any(s in blob for s in _SOFTWARE)
        if not info.get("ok") or is_software:
            browser.close()
            raise AssertionError(
                "REJECTED — software or no GPU adapter (this is the whole point of the gate).\n"
                f"  adapter = {json.dumps(info)}\n"
                f"  diag    = {json.dumps(diag)}\n"
                "  → fix the NVIDIA Vulkan ICD / Dawn-blocklist flag and re-run. "
                "See docs/webgpu-ci-providers.md."
            )

        results: dict[str, object] = {}
        if levels:
            page.wait_for_function("window.__webgpuq && window.__webgpuq.ready === true", timeout=120_000)
            for lv in levels:
                fn = "runE34" if lv.upper() == "E34" else f"runLevel{lv}"
                results[fn] = page.evaluate(
                    f"async () => await window.__webgpuq.{fn}()"
                )
        browser.close()

    return {"adapter": info, "diag": diag, "results": results}


@app.function(gpu="T4", timeout=300)
def probe(url: str = DEFAULT_URL) -> dict:
    """Cheap (~30s): just prove the T4 yields a real, non-fallback adapter."""
    return _run([], url)


@app.function(gpu="T4", timeout=1800)
def run_kernels(levels: list[str], url: str = DEFAULT_URL) -> dict:
    """Probe + run the named kernels (default L1 + L3). 'E34' = the (T) CPU/GPU bench (heavy)."""
    return _run(levels, url)


@app.local_entrypoint()
def main(kernels: bool = False, levels: str = "1,3", url: str = DEFAULT_URL):
    lv = [s.strip() for s in levels.split(",") if s.strip()]
    result = run_kernels.remote(lv, url) if kernels else probe.remote(url)
    info = result["adapter"]
    print("\n=== adapter ===")
    print(json.dumps(info, indent=2))
    print("=== diag (nvidia-smi / vulkaninfo) ===")
    print(json.dumps(result["diag"], indent=2))
    real = info.get("ok") and info.get("isFallbackAdapter") is not True
    print(f"\n{'✅ REAL GPU' if real else '❌ software/none'} — vendor={info.get('vendor')!r} "
          f"arch={info.get('architecture')!r} desc={info.get('description')!r}")
    if result["results"]:
        print(f"\n=== ran {list(result['results'])} — full artifacts below ===")
        print(json.dumps(result["results"], indent=2)[:4000])
