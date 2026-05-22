// Pyodide-backed Python REPL embedded in /molecule.html.
//
// Lazy-loaded — the ~10 MB Pyodide bundle is only fetched when the user
// clicks "Open Python REPL". Once loaded, the REPL exposes the current
// calculation context (`ctx.D`, `ctx.C_MO`, `ctx.orbitalEnergies`, etc.)
// as JS-side values that Python can read via `pyodide.toJs()` /
// `pyodide.globals.get("ctx")`.
//
// Use case: click any value in the SI report, drop into Python, poke at
// intermediates with familiar numpy syntax. Cross-check our chemistry
// against PySCF / libxc when those are loaded as Pyodide packages.

const PYODIDE_VERSION = "0.26.4";
const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full`;

interface PyodideInterface {
  runPython(code: string): unknown;
  runPythonAsync(code: string): Promise<unknown>;
  loadPackage(pkg: string | string[]): Promise<void>;
  globals: {
    set(name: string, value: unknown): void;
    get(name: string): unknown;
  };
}

declare global {
  interface Window {
    loadPyodide?: (opts: { indexURL?: string }) => Promise<PyodideInterface>;
  }
}

let pyodidePromise: Promise<PyodideInterface> | null = null;

async function loadPyodideOnce(): Promise<PyodideInterface> {
  if (pyodidePromise) return pyodidePromise;
  pyodidePromise = (async () => {
    // Inject the Pyodide loader script.
    if (!window.loadPyodide) {
      await new Promise<void>((resolve, reject) => {
        const s = document.createElement("script");
        s.src = `${PYODIDE_CDN}/pyodide.js`;
        s.onload = () => resolve();
        s.onerror = () => reject(new Error("Pyodide CDN load failed"));
        document.head.appendChild(s);
      });
    }
    if (!window.loadPyodide) throw new Error("Pyodide loader unavailable");
    return window.loadPyodide({ indexURL: PYODIDE_CDN });
  })();
  return pyodidePromise;
}

/**
 * Bring up a REPL panel inside `host`. Initially empty; on first run,
 * loads Pyodide and exposes `ctxProvider()` as the Python `ctx`.
 */
export function attachPythonREPL(
  host: HTMLElement,
  ctxProvider: () => Record<string, unknown>,
): void {
  host.innerHTML = `
    <div class="repl">
      <div class="repl-toolbar">
        <span class="repl-status">Click "Run" to lazy-load Pyodide (~10 MB).</span>
        <span class="repl-hint">Globals: <code>ctx</code> — read-only snapshot of the last calculation.</span>
      </div>
      <textarea class="repl-input" rows="6" spellcheck="false" placeholder="# Try:
# import numpy as np
# D = np.asarray(ctx['D']).reshape(ctx['n'], ctx['n'])
# print('|D| Frobenius =', np.linalg.norm(D))
# print('eigenvalues:', np.linalg.eigvalsh(D))
"></textarea>
      <div class="repl-buttons">
        <button class="btn btn-secondary repl-run">Run (Cmd/Ctrl+Enter)</button>
        <button class="btn btn-secondary repl-clear">Clear output</button>
      </div>
      <pre class="repl-output" aria-live="polite"></pre>
    </div>`;

  const status = host.querySelector(".repl-status") as HTMLSpanElement;
  const input  = host.querySelector(".repl-input") as HTMLTextAreaElement;
  const output = host.querySelector(".repl-output") as HTMLPreElement;
  const runBtn = host.querySelector(".repl-run") as HTMLButtonElement;
  const clearBtn = host.querySelector(".repl-clear") as HTMLButtonElement;

  let pyodide: PyodideInterface | null = null;

  async function ensureLoaded(): Promise<PyodideInterface> {
    if (pyodide) return pyodide;
    status.textContent = "Loading Pyodide (~10 MB)…";
    runBtn.disabled = true;
    try {
      pyodide = await loadPyodideOnce();
      // Capture stdout/stderr so prints land in the REPL output.
      pyodide.runPython(`
import sys, io
_stdout = io.StringIO()
_stderr = io.StringIO()
sys.stdout = _stdout
sys.stderr = _stderr
`);
      status.textContent = "Pyodide ready.";
      return pyodide;
    } finally {
      runBtn.disabled = false;
    }
  }

  async function run(): Promise<void> {
    const code = input.value.trim();
    if (!code) return;
    output.textContent += `>>> ${code}\n`;
    output.scrollTop = output.scrollHeight;
    try {
      const py = await ensureLoaded();
      // Refresh the ctx global each run so the user sees the LATEST
      // calculation, even if they re-ran Run on the main page.
      py.globals.set("ctx", py.runPython("None"));
      py.globals.set("ctx", ctxProvider());
      const result = await py.runPythonAsync(code);
      // Drain captured stdout/stderr.
      const sout = py.runPython("_stdout.getvalue(); _stdout.truncate(0); _stdout.seek(0); _") as unknown;
      const serr = py.runPython("_stderr.getvalue(); _stderr.truncate(0); _stderr.seek(0); _") as unknown;
      if (typeof sout === "string" && sout) output.textContent += sout;
      if (typeof serr === "string" && serr) output.textContent += `[stderr] ${serr}`;
      if (result !== undefined && result !== null) output.textContent += `${String(result)}\n`;
    } catch (e) {
      output.textContent += `[error] ${e instanceof Error ? e.message : String(e)}\n`;
    }
    output.scrollTop = output.scrollHeight;
  }

  runBtn.addEventListener("click", () => void run());
  clearBtn.addEventListener("click", () => { output.textContent = ""; });
  input.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      void run();
    }
  });
}
