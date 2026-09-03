#!/usr/bin/env node
import { readFileSync } from 'node:fs';

const args = process.argv.slice(2);
const gate = args.includes('--gate');
const file = args.find(a => !a.startsWith('--'));

if (!file) {
  console.error('Usage: node bench/check-perf.mjs <vitest-json-report> [--gate]');
  process.exit(1);
}

const report = JSON.parse(readFileSync(file, 'utf8'));
if (!report || !Array.isArray(report.testResults)) {
  console.error('Unexpected schema: expected top-level testResults[]');
  process.exit(1);
}

const baseline = JSON.parse(readFileSync(new URL('baseline.json', import.meta.url), 'utf8'));

const actuals = new Map();
for (const r of report.testResults) {
  const name = r.name || '';
  let ms = 0;
  if (typeof r.duration === 'number') {
    ms = r.duration;
  } else if (Array.isArray(r.assertionResults)) {
    for (const a of r.assertionResults) {
      if (typeof a.duration === 'number') ms += a.duration;
    }
  } else {
    console.error(`Unexpected schema for ${name}: no duration or assertionResults`);
    process.exit(1);
  }
  actuals.set(name, ms / 1000);
}

let warnings = 0;
for (const [key, entry] of Object.entries(baseline.files)) {
  const base = entry.seconds;
  const match = [...actuals.keys()].find(n => n === key || n.endsWith('/' + key));
  if (!match) {
    console.log(`NOTE baseline file not in report: ${key}`);
    continue;
  }
  const actual = actuals.get(match);
  if (actual > base * 2) {
    warnings++;
    const gated = entry.noise && entry.noise.stdOverMedian <= 0.1 && entry.noise.samples >= 20;
    console.log(`WARN perf: ${match} ${actual.toFixed(1)}s vs baseline ${base}s${gated ? ' (gated)' : ''}`);
  }
}

if (gate) {
  const gated = Object.entries(baseline.files).filter(([, e]) =>
    e.noise && e.noise.stdOverMedian <= 0.1 && e.noise.samples >= 20
  );
  if (gated.length === 0) {
    console.log('no gated cells yet');
    process.exit(0);
  }
  process.exit(warnings > 0 ? 1 : 0);
}

console.log(warnings ? `OK with ${warnings} warning(s)` : 'OK all within 2x baseline');
process.exit(0);
