// IndexedDB-backed history of past calculations on /molecule.html.
// One object store, one record per (molecule, method, timestamp).

const DB_NAME = "webgpu-q";
const DB_VERSION = 1;
const STORE = "calculations";

export interface HistoryRecord {
  id?: number;
  molecule: string;
  method: string;
  timestamp: number;
  durationMs: number;
  scfEnergy: number;
  // Lightweight summary blob — not the full Float64Arrays.
  summary: Record<string, unknown>;
}

let dbPromise: Promise<IDBDatabase> | null = null;

function open(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
        store.createIndex("by_timestamp", "timestamp", { unique: false });
        store.createIndex("by_molecule", "molecule", { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

export async function saveCalculation(rec: HistoryRecord): Promise<number> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const store = tx.objectStore(STORE);
    const req = store.add(rec);
    req.onsuccess = () => resolve(req.result as number);
    req.onerror = () => reject(req.error);
  });
}

export async function listCalculations(limit = 25): Promise<HistoryRecord[]> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const idx = tx.objectStore(STORE).index("by_timestamp");
    const results: HistoryRecord[] = [];
    const req = idx.openCursor(null, "prev");
    req.onsuccess = () => {
      const cursor = req.result;
      if (cursor && results.length < limit) {
        results.push(cursor.value as HistoryRecord);
        cursor.continue();
      } else {
        resolve(results);
      }
    };
    req.onerror = () => reject(req.error);
  });
}

export async function clearCalculations(): Promise<void> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const req = tx.objectStore(STORE).clear();
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

/** Register the service worker. No-op outside http(s) (file://, dev iframes). */
export function registerServiceWorker(): void {
  if (!("serviceWorker" in navigator)) return;
  if (window.location.protocol !== "http:" && window.location.protocol !== "https:") return;
  window.addEventListener("load", () => {
    void navigator.serviceWorker.register("/sw.js").catch(() => {
      // Best-effort; offline support is nice-to-have.
    });
  });
}
