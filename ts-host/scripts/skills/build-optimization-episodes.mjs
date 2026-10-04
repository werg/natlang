#!/usr/bin/env node
/** Deterministic, model-free optimization skill fixtures. Bounds are host-only episode labels. */
import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';

const digest = value => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');
const hashInt = s => Number.parseInt(digest(s).slice(0, 8), 16);
const targetText = {
  knapsack: 'Given an instance JSON string, return JSON with selectedIds: a list of item IDs. Maximize total value without exceeding capacity. Verify every chosen ID and sum the weights.',
  'bin-packing': 'Given an instance JSON string, return JSON with bins: a list of objects, each having itemIds. Place every item exactly once; each bin load must fit capacity. Minimize bin count.',
  'weighted-tardiness': 'Given an instance JSON string, return JSON with order: a permutation of job IDs. Minimize the sum of weight × max(0, completion time − due time).',
};
function makeInstance(kind, seed) {
  const r = (i, salt, max = 9) => 1 + hashInt(`${kind}/${seed}/${i}/${salt}`) % max;
  if (kind === 'knapsack') return { capacity: 34 + r(0, 'cap', 17), items: Array.from({ length: 14 }, (_, i) => ({ id: `i${i}`, weight: r(i, 'w', 14), value: 5 + r(i, 'v', 41) })) };
  if (kind === 'bin-packing') {
    const hard = [4, 4, 8, 2, 5, 9, 2, 2, 3];
    const sizes = seed === 0 ? hard : Array.from({ length: 9 }, (_, i) => 1 + hashInt(`${kind}/${seed}/${i}/s`) % 9);
    return { capacity: 10, items: sizes.map((size, i) => ({ id: `i${i}`, size })) };
  }
  return { jobs: Array.from({ length: 12 }, (_, i) => ({ id: `j${i}`, processing: r(i, 'p'), due: 4 + r(i, 'd') * 3, weight: r(i, 'w') })) };
}
function bounds(kind, x) {
  if (kind === 'knapsack') {
    let best = 0;
    for (let mask = 0; mask < 2 ** x.items.length; mask++) { let w = 0, v = 0; x.items.forEach((i, n) => { if (mask & (2 ** n)) { w += i.weight; v += i.value; } }); if (w <= x.capacity) best = Math.max(best, v); }
    return { kind: 'objective-bound', worst: 0, best };
  }
  if (kind === 'bin-packing') {
    let best = x.items.length; const bins = [];
    const items = [...x.items].sort((a, b) => b.size - a.size);
    const visit = i => { if (i === items.length) { best = Math.min(best, bins.length); return; } if (bins.length >= best) return;
      const seen = new Set(); for (let b = 0; b < bins.length; b++) if (bins[b] + items[i].size <= x.capacity && !seen.has(bins[b])) { seen.add(bins[b]); bins[b] += items[i].size; visit(i + 1); bins[b] -= items[i].size; }
      bins.push(items[i].size); visit(i + 1); bins.pop(); };
    visit(0); return { kind: 'objective-bound', worst: x.items.length, best };
  }
  const jobs = x.jobs, count = 2 ** jobs.length, min = new Float64Array(count).fill(Infinity), max = new Float64Array(count).fill(-Infinity);
  min[0] = max[0] = 0;
  for (let mask = 0; mask < count; mask++) {
    let time = 0; jobs.forEach((job, i) => { if (mask & (2 ** i)) time += job.processing; });
    for (let i = 0; i < jobs.length; i++) if (!(mask & (2 ** i))) {
      const next = mask + 2 ** i, job = jobs[i], cost = job.weight * Math.max(0, time + job.processing - job.due);
      min[next] = Math.min(min[next], min[mask] + cost); max[next] = Math.max(max[next], max[mask] + cost);
    }
  }
  return { kind: 'objective-bound', worst: max[count - 1], best: min[count - 1] };
}
function baseline(kind, x) {
  if (kind === 'knapsack') {
    let weight = 0, value = 0;
    for (const item of [...x.items].sort((a, b) => b.value / b.weight - a.value / a.weight))
      if (weight + item.weight <= x.capacity) { weight += item.weight; value += item.value; }
    return value;
  }
  if (kind === 'bin-packing') {
    const bins = [];
    for (const item of [...x.items].sort((a, b) => b.size - a.size)) {
      const bin = bins.find(row => row.load + item.size <= x.capacity);
      if (bin) bin.load += item.size; else bins.push({ load: item.size });
    }
    return bins.length;
  }
  let time = 0, total = 0;
  for (const job of [...x.jobs].sort((a, b) => a.processing / a.weight - b.processing / b.weight)) {
    time += job.processing; total += job.weight * Math.max(0, time - job.due);
  }
  return total;
}
function quality(kind, value, bound) {
  if (bound.best === bound.worst) return 1;
  return kind === 'knapsack' ? (value - bound.worst) / (bound.best - bound.worst)
    : (bound.worst - value) / (bound.worst - bound.best);
}
function buildKind(kind, index) {
  const seeds = {
    knapsack: [66, 31, 57, 44, 78, 63, 3, 38],
    'bin-packing': [0, 3, 7, 11, 18, 27, 39, 52],
    'weighted-tardiness': [30, 34, 33, 13, 80, 38, 93, 87],
  }[kind];
  const all = Array.from({ length: 8 }, (_, n) => {
    const instance = makeInstance(kind, seeds[n]), source = `${kind}:instance-${String(n).padStart(2, '0')}`;
    const bound = bounds(kind, instance);
    return { id: `case-${digest(source).slice(0, 20)}`, group: `g-${digest(source).slice(20, 40)}`, args: [JSON.stringify(instance)], expected: bound,
      baseline_quality: quality(kind, baseline(kind, instance), bound) };
  });
  const support = all.slice(0, 4).map(({ baseline_quality, ...row }) => row), query = all.slice(4, 8).map(({ baseline_quality, ...row }) => row);
  const other = ['knapsack', 'bin-packing', 'weighted-tardiness'][(index + 1) % 3];
  const transfer = Array.from({ length: 4 }, (_, n) => {
    const instance = makeInstance(other, 100 + index * 10 + n);
    const key = `${other}:transfer:${index}:${n}`;
    return { id: `case-${digest(key).slice(0, 20)}`, group: `g-${digest(key).slice(20, 40)}`, args: [JSON.stringify(instance)], expected: bounds(other, instance) };
  });
  const allGroups = [...support, ...query, ...transfer].map(c => c.group).sort();
  const commit = digest(allGroups);
  const target = family => ({ kind: 'improvement-case', entry: 'solve.nl', exportName: 'default',
    source: { schema: 'natlang.skill-objective-target/1', id: `objective-${family}-v1` },
    files: { 'solve.nl': `---\nargs: { instance: string }\nreturns: string\n---\n${targetText[family]} Return only the requested JSON object. The host independently checks feasibility and computes the objective.\n` } });
  const episode = { version: 'natlang.skill-episode/1', id: `skill-objective-${kind}-v1`, family: `optimization-${kind}`, split: 'train',
    source_groups: [`group-commitment:sha256:${commit}`], license: 'project-generated', target: target(kind),
    library: { kind: 'empty', skills: {} }, support: { cases: support }, query: { cases: query },
    transfer: { family: `optimization-${other}`, target: target(other), cases: transfer }, operations: ['create', 'revise', 'select', 'test'],
    limits: { maxSteps: 6 }, provenance: { generator: 'natlang.skill-optimization-episodes/1',
      metric: { schema: 'natlang.skill-objective/1', kind },
      transfer_metric: { schema: 'natlang.skill-objective/1', kind: other } } };
  return { episode, baseline_quality: all.reduce((s, row) => s + row.baseline_quality, 0) / all.length };
}

const outArg = process.argv.indexOf('--out');
if (outArg < 0 || !process.argv[outArg + 1]) throw Error('Usage: node build-optimization-episodes.mjs --out DIR');
const out = resolve(process.argv[outArg + 1]);
const built = ['knapsack', 'bin-packing', 'weighted-tardiness'].map(buildKind);
const rows = built.map(row => row.episode);
await mkdir(out, { recursive: true });
const body = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
await writeFile(join(out, 'optimization-episodes.jsonl'), body, { flag: 'wx' });
const manifest = { schema: 'natlang.skill-objective-episodes/1', episodes: rows.length, instances_per_kind: 8,
  support: 12, query: 12, transfer: 12, license: 'project-generated', model_calls: 0,
  groups_disjoint_across_roles: true,
  heuristic_baseline_mean_quality: Object.fromEntries(built.map(({ episode, baseline_quality }) => [episode.provenance.metric.kind, baseline_quality])),
  sha256: digest(body), source: 'deterministic-generated-small-instances' };
await writeFile(join(out, 'optimization-episodes.manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify(manifest, null, 2));
