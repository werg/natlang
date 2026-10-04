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
const skill = (name, description, body) => ({ 'SKILL.md': `---\nname: ${name}\ndescription: ${JSON.stringify(description)}\n---\n${body}\n` });
/** Starting skills. Bodies are general procedures, never instance answers. */
const GOOD = {
  knapsack: skill('knapsack-exact', 'Use for 0/1 knapsack selection under one capacity: choosing item IDs to maximize value within a weight limit.',
    'With at most about twenty items, enumerate or use dynamic programming over capacity to find the best feasible subset.\nTrack chosen IDs, re-sum weights and values, and confirm the total weight does not exceed capacity before returning.'),
  'bin-packing': skill('bin-packing-search', 'Use for packing items into the fewest equal-capacity bins, each item exactly once.',
    'Sort items by decreasing size and place each into the first bin it fits (first-fit decreasing).\nThen try to empty the least-loaded bin by moving or swapping items; a lower bound is ceil(total size / capacity).\nVerify every ID appears once and every bin load fits capacity.'),
  'weighted-tardiness': skill('tardiness-ordering', 'Use for ordering jobs on one machine to minimize total weighted tardiness given processing times, due dates and weights.',
    'Start from earliest-due-date and weighted-shortest-processing-time orders and keep the better one.\nImprove with adjacent and pairwise swaps while the computed total weighted tardiness decreases; recompute completion times after each swap.'),
};
/** Correct bodies with uninformative or misleading descriptions: metadata-only repair cases. */
const MISDESCRIBED = Object.fromEntries(Object.entries(GOOD).map(([kind, files]) => [kind,
  { 'SKILL.md': files['SKILL.md'].replace(/description: .*/, 'description: "General notes; not needed for typical tasks."') }]));
/** Plausible but flawed procedures: revision cases. */
const INCORRECT = {
  knapsack: skill('knapsack-exact', 'Use for 0/1 knapsack selection under one capacity.', 'Pick items in order of highest value first until the next item does not fit, then stop.'),
  'bin-packing': skill('bin-packing-search', 'Use for packing items into the fewest bins.', 'Open a new bin for every item larger than half the capacity, and put all remaining items into one shared bin.'),
  'weighted-tardiness': skill('tardiness-ordering', 'Use for ordering jobs to minimize weighted tardiness.', 'Order jobs by decreasing weight; ignore due dates, which do not affect tardiness.'),
};
const VARIANTS = ['empty', 'distractor', 'misdescribed', 'incorrect', 'redundant'];
const OPERATIONS = {
  empty: ['create', 'revise', 'select', 'test'],
  distractor: ['create', 'revise', 'select', 'repair-irrelevant', 'test'],
  misdescribed: ['revise', 'select', 'test'],
  incorrect: ['revise', 'repair-incorrect', 'select', 'test'],
  redundant: ['revise', 'select', 'retire', 'test'],
};
function library(kind, variant) {
  const others = Object.keys(GOOD).filter(k => k !== kind);
  const name = files => /name: (.*)/.exec(files['SKILL.md'])[1];
  const of = list => Object.fromEntries(list.map(files => [name(files), files]));
  if (variant === 'empty') return { kind: 'empty', skills: {} };
  if (variant === 'distractor') return { kind: 'existing', skills: of(others.map(k => GOOD[k])) };
  if (variant === 'misdescribed') return { kind: 'existing', skills: of([MISDESCRIBED[kind], GOOD[others[0]]]) };
  if (variant === 'incorrect') return { kind: 'corrupted', skills: of([INCORRECT[kind], GOOD[others[1]]]),
    defect: { kind: 'incorrect', skill: name(INCORRECT[kind]), detail: 'plausible but flawed procedure', verified: false } };
  // Redundant: a correct skill plus a duplicate-purpose copy under another name; selection and retirement case.
  const copy = { 'SKILL.md': GOOD[kind]['SKILL.md'].replace(/name: (.*)/, 'name: $1-notes') };
  return { kind: 'existing', skills: of([GOOD[kind], copy, GOOD[others[0]]]) };
}
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
function buildKind(kind, index, replica = 0, variant = 'empty') {
  const pilot = {
    knapsack: [66, 31, 57, 44, 78, 63, 3, 38],
    'bin-packing': [0, 3, 7, 11, 18, 27, 39, 52],
    'weighted-tardiness': [30, 34, 33, 13, 80, 38, 93, 87],
  }[kind];
  // Replica 0 keeps the pilot instances; later replicas draw disjoint seeds.
  const seeds = replica === 0 ? pilot : Array.from({ length: 8 }, (_, n) => 1000 * replica + n);
  const all = Array.from({ length: 8 }, (_, n) => {
    const instance = makeInstance(kind, seeds[n]), source = replica === 0 ? `${kind}:instance-${String(n).padStart(2, '0')}` : `${kind}:r${replica}:instance-${String(n).padStart(2, '0')}`;
    const bound = bounds(kind, instance);
    return { id: `case-${digest(source).slice(0, 20)}`, group: `g-${digest(source).slice(20, 40)}`, args: [JSON.stringify(instance)], expected: bound,
      baseline_quality: quality(kind, baseline(kind, instance), bound) };
  });
  const support = all.slice(0, 4).map(({ baseline_quality, ...row }) => row), query = all.slice(4, 8).map(({ baseline_quality, ...row }) => row);
  const other = ['knapsack', 'bin-packing', 'weighted-tardiness'][(index + 1) % 3];
  const transfer = Array.from({ length: 4 }, (_, n) => {
    const instance = makeInstance(other, replica === 0 ? 100 + index * 10 + n : 500000 + 1000 * replica + 10 * index + n);
    const key = replica === 0 ? `${other}:transfer:${index}:${n}` : `${other}:transfer:r${replica}:${index}:${n}`;
    return { id: `case-${digest(key).slice(0, 20)}`, group: `g-${digest(key).slice(20, 40)}`, args: [JSON.stringify(instance)], expected: bounds(other, instance) };
  });
  const allGroups = [...support, ...query, ...transfer].map(c => c.group).sort();
  const commit = digest(allGroups);
  const target = family => ({ kind: 'improvement-case', entry: 'solve.nl', exportName: 'default',
    source: { schema: 'natlang.skill-objective-target/1', id: `objective-${family}-v1` },
    files: { 'solve.nl': `---\nargs: { instance: string }\nreturns: string\n---\n${targetText[family]} Return only the requested JSON object. The host independently checks feasibility and computes the objective.\n` } });
  const suffix = replica === 0 && variant === 'empty' ? '' : `-r${replica}-${variant}`;
  const episode = { version: 'natlang.skill-episode/1', id: `skill-objective-${kind}-v1${suffix}`, family: `optimization-${kind}`, split: 'train',
    source_groups: [`group-commitment:sha256:${commit}`], license: 'project-generated', target: target(kind),
    library: library(kind, variant), support: { cases: support }, query: { cases: query },
    transfer: { family: `optimization-${other}`, target: target(other), cases: transfer }, operations: OPERATIONS[variant],
    limits: { maxSteps: 6 }, provenance: { generator: 'natlang.skill-optimization-episodes/2', library_variant: variant, replica,
      metric: { schema: 'natlang.skill-objective/1', kind },
      transfer_metric: { schema: 'natlang.skill-objective/1', kind: other } } };
  return { episode, baseline_quality: all.reduce((s, row) => s + row.baseline_quality, 0) / all.length };
}

const arg = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
if (!arg('--out')) throw Error('Usage: node build-optimization-episodes.mjs --out DIR [--replicas N] [--variants empty,distractor,...]');
const out = resolve(arg('--out'));
const replicas = Number(arg('--replicas') ?? 1), variants = (arg('--variants') ?? 'empty').split(',');
if (!Number.isSafeInteger(replicas) || replicas < 1 || variants.some(v => !VARIANTS.includes(v))) throw Error('invalid --replicas or --variants');
const KINDS = ['knapsack', 'bin-packing', 'weighted-tardiness'];
// Each (replica, variant) pair gets its own instances, so no case is shared between episodes.
const built = [];
for (let r = 0; r < replicas; r++) variants.forEach((variant, v) => KINDS.forEach((kind, index) =>
  built.push(buildKind(kind, index, r * variants.length + v, variant))));
const rows = built.map(row => row.episode);
await mkdir(out, { recursive: true });
const body = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
await writeFile(join(out, 'optimization-episodes.jsonl'), body, { flag: 'wx' });
const manifest = { schema: 'natlang.skill-objective-episodes/1', episodes: rows.length, replicas, variants, instances_per_episode: 8,
  support_per_episode: 4, query_per_episode: 4, transfer_per_episode: 4, license: 'project-generated', model_calls: 0,
  groups_disjoint_across_roles: true,
  heuristic_baseline_mean_quality: Object.fromEntries(built.map(({ episode, baseline_quality }) => [episode.id, baseline_quality])),
  sha256: digest(body), source: 'deterministic-generated-small-instances' };
await writeFile(join(out, 'optimization-episodes.manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify(manifest, null, 2));
