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
        <button class="btn btn-secondary repl-sanity" title="Run numpy invariant checks against ctx (Hermitian D, orthonormal C, etc.)">Sanity-check</button>
        <button class="btn btn-secondary repl-pyscf" title="Best-effort: install PySCF via micropip and diff HF energy against ours">Compare to PySCF</button>
        <button class="btn btn-secondary repl-clear">Clear output</button>
      </div>
      <pre class="repl-output" aria-live="polite"></pre>
    </div>`;

  const status = host.querySelector(".repl-status") as HTMLSpanElement;
  const input  = host.querySelector(".repl-input") as HTMLTextAreaElement;
  const output = host.querySelector(".repl-output") as HTMLPreElement;
  const runBtn = host.querySelector(".repl-run") as HTMLButtonElement;
  const clearBtn = host.querySelector(".repl-clear") as HTMLButtonElement;
  const sanityBtn = host.querySelector(".repl-sanity") as HTMLButtonElement;
  const pyscfBtn = host.querySelector(".repl-pyscf") as HTMLButtonElement;

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

  sanityBtn.addEventListener("click", () => {
    input.value = SANITY_SCRIPT;
    void run();
  });

  pyscfBtn.addEventListener("click", () => {
    input.value = PYSCF_COMPARE_SCRIPT;
    void run();
  });
}

const SANITY_SCRIPT = `import numpy as np
n = ctx['n']
D = np.asarray(ctx['D'], dtype=float).reshape(n, n)
C = np.asarray(ctx['C_MO'], dtype=float).reshape(n, n)
S = np.asarray(ctx['S_AO'], dtype=float).reshape(n, n)
h = np.asarray(ctx['h_AO'], dtype=float).reshape(n, n)

print(f"Molecule: {ctx['molecule']} / {ctx['method']} (n={n}, E={ctx['energy']:.10f} Ha)")
print()
print("Density matrix sanity:")
print(f"  Hermitian? max|D - D.T|     = {np.abs(D - D.T).max():.2e}")
print(f"  Idempotent? max|D.S.D - 2D| = {np.abs(D @ S @ D - 2*D).max():.2e}  (factor 2 = closed-shell RHF)")
print(f"  Trace(D.S) = {np.trace(D @ S):.4f}  (= 2*nOccupied = {2*ctx['nOccupied']})")
print()
print("MO coefficient sanity:")
print(f"  C.T S C orthonormal?  max|C.T S C - I| = {np.abs(C.T @ S @ C - np.eye(n)).max():.2e}")
print()
print("Core integral sanity:")
print(f"  h_AO Hermitian? max|h - h.T| = {np.abs(h - h.T).max():.2e}")
print(f"  S_AO Hermitian? max|S - S.T| = {np.abs(S - S.T).max():.2e}")
print(f"  S min eigenvalue = {np.linalg.eigvalsh(S).min():.4e}  (must be > 0)")
print()
print("Orbital energies:")
print(f"  occupied:  {np.array(ctx['orbitalEnergies'][:ctx['nOccupied']])}")
print(f"  virtual:   {np.array(ctx['orbitalEnergies'][ctx['nOccupied']:])}")
print(f"  HOMO-LUMO gap = {ctx['orbitalEnergies'][ctx['nOccupied']] - ctx['orbitalEnergies'][ctx['nOccupied']-1]:.4f} Ha")
`;

const PYSCF_COMPARE_SCRIPT = `# Best-effort: install PySCF via micropip and diff HF energy.
# PySCF on Pyodide is experimental — depends on libcint compileability.
# If install fails the script reports the error and you can still use
# the Sanity-check button.
import sys, micropip
try:
    print("Installing pyscf via micropip… (this can take a few minutes on first run)")
    await micropip.install("pyscf")
    from pyscf import gto, scf
    print(f"  pyscf {sys.modules['pyscf'].__version__} loaded.")
except Exception as e:
    print(f"[unavailable] {e}")
    print()
    print("PySCF doesn't currently compile cleanly to WASM (libcint C extensions).")
    print("Workarounds:")
    print("  - Click 'Sanity-check' for numpy-based invariant checks.")
    print("  - Download our QCSchema export and run pyscf locally:")
    print("      pyscf.tools.qcschema.recipe(open('webgpu-q-result.qcjson').read())")
    raise SystemExit
# If we reach here, pyscf loaded. Build the same molecule and compare.
# (Hand-coded for the built-in molecule library; for imported molecules
# the user can adapt the atom list below.)
known = {
    "h2o":  ("O 0 0 0; H 0.7572 0.5864 0; H -0.7572 0.5864 0", 0),
    "h2":   ("H 0 0 0; H 0.7414 0 0", 0),
    "lih":  ("Li 0 0 0; H 1.595 0 0", 0),
    "beh2": ("Be 0 0 0; H 1.330 0 0; H -1.330 0 0", 0),
}
if ctx['molecule'] not in known:
    print(f"No built-in geometry for '{ctx['molecule']}' — adapt the atom list manually.")
    raise SystemExit
atoms, charge = known[ctx['molecule']]
mol = gto.M(atom=atoms, basis="sto-3g", charge=charge, unit="Angstrom", verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-10)
print(f"  PySCF HF (STO-3G): {mf.e_tot:.10f} Ha")
print(f"  Ours            : {ctx['energy']:.10f} Ha")
print(f"  Δ              : {abs(mf.e_tot - ctx['energy']):.2e} Ha  (chemical accuracy = 1.6e-3 Ha)")
`;
