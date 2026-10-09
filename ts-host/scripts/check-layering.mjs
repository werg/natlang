#!/usr/bin/env node
// Freezes the directory-level import cycles of ts-host/src: the set of directory edges that sit on a cycle may
// shrink but not grow. Usage: node scripts/check-layering.mjs [--update]
import { readdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const srcRoot = resolve(here, '../src');
const baselinePath = join(here, 'layering-baseline.json');

const files = [];
(function walk(dir) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) walk(full);
    else if (/\.ts$/.test(name) && !/\.d\.ts$/.test(name)) files.push(full);
  }
})(srcRoot);

const groupOf = file => {
  const parts = relative(srcRoot, file).split(sep);
  return parts.length === 1 ? '(root)' : parts[0];
};
const specifier = /(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+|\brequire\s*\(\s*)['"](\.{1,2}\/[^'"]*|\.{1,2})['"]/g;
const resolveTarget = (from, spec) => {
  const base = resolve(dirname(from), spec);
  const stem = base.replace(/\.(js|mjs|ts)$/, '');
  for (const cand of [stem + '.ts', base + '.ts', join(base, 'index.ts'), base]) {
    if (existsSync(cand) && statSync(cand).isFile()) return cand;
  }
  return undefined;
};

// edge "a -> b" => importing files (relative to src)
const edges = new Map();
for (const file of files) {
  const text = readFileSync(file, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
  for (const m of text.matchAll(specifier)) {
    const target = resolveTarget(file, m[1]);
    if (!target || !target.startsWith(srcRoot)) continue;
    const a = groupOf(file), b = groupOf(target);
    if (a === b) continue;
    const key = `${a} -> ${b}`;
    if (!edges.has(key)) edges.set(key, new Set());
    edges.get(key).add(`${relative(srcRoot, file)} -> ${relative(srcRoot, target)}`);
  }
}

// Strongly connected components (Tarjan); an edge is cyclic when both ends share a component of size > 1.
const nodes = new Set();
const adj = new Map();
for (const key of edges.keys()) {
  const [a, b] = key.split(' -> ');
  nodes.add(a); nodes.add(b);
  if (!adj.has(a)) adj.set(a, []);
  adj.get(a).push(b);
}
const index = new Map(), low = new Map(), onStack = new Set(), stack = [], comp = new Map();
let counter = 0, compId = 0;
const connect = v => {
  index.set(v, counter); low.set(v, counter); counter++; stack.push(v); onStack.add(v);
  for (const w of adj.get(v) ?? []) {
    if (!index.has(w)) { connect(w); low.set(v, Math.min(low.get(v), low.get(w))); }
    else if (onStack.has(w)) low.set(v, Math.min(low.get(v), index.get(w)));
  }
  if (low.get(v) === index.get(v)) {
    let w; do { w = stack.pop(); onStack.delete(w); comp.set(w, compId); } while (w !== v);
    compId++;
  }
};
for (const v of [...nodes].sort()) if (!index.has(v)) connect(v);

const current = [...edges.keys()].filter(key => {
  const [a, b] = key.split(' -> ');
  return comp.get(a) === comp.get(b);
}).sort();

if (process.argv.includes('--update')) {
  writeFileSync(baselinePath, JSON.stringify({
    note: 'Directory-level import edges (under ts-host/src) that participate in cycles. May shrink, must not grow. Regenerate with: node scripts/check-layering.mjs --update',
    cyclicEdges: current,
  }, null, 2) + '\n');
  console.log(`layering baseline written: ${current.length} cyclic edges`);
  process.exit(0);
}

const baseline = new Set(JSON.parse(readFileSync(baselinePath, 'utf8')).cyclicEdges);
const added = current.filter(e => !baseline.has(e));
const removed = [...baseline].filter(e => !current.includes(e)).sort();
if (removed.length) {
  console.log(`layering: ${removed.length} baseline cyclic edge(s) no longer cyclic; shrink the baseline with --update:`);
  for (const e of removed) console.log(`  - ${e}`);
}
if (added.length) {
  console.error(`layering: ${added.length} new directory-level cyclic edge(s) not in layering-baseline.json:`);
  for (const e of added) {
    console.error(`  ${e}`);
    for (const f of [...edges.get(e)].sort()) console.error(`    imported by ${f}`);
  }
  console.error('Move the shared code to a lower layer instead of adding a back-edge.');
  process.exit(1);
}
console.log(`layering ok: ${current.length} cyclic edges (baseline ${baseline.size})`);
