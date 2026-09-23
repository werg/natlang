#!/usr/bin/env node
// Select a balanced shard from verified case pools.
// node scripts/inline-curriculum/select.mjs POOL.jsonl [...] --out OUT.jsonl [--size N] [--seed S] [--split train|test] [--track interpreter|authoring] [--no-balance]
// Counterfactual groups stay whole. Domains and slices are filled up to the plan's shares of the shard size
// (by default the largest shard whose shares all stay within three points of target), preferring the categories
// with the most room left and spreading picks across families.
import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { DOMAIN_TARGETS, SLICE_TARGETS } from '../../dist/teacher/curriculum.js';
import { Random } from './lib.mjs';

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  out: { type: 'string' }, size: { type: 'string' }, seed: { type: 'string', default: '1' }, split: { type: 'string' }, track: { type: 'string', default: 'interpreter' }, 'no-balance': { type: 'boolean', default: false } } });
if (!positionals.length || !values.out) throw new Error('usage: select.mjs POOL.jsonl [...] --out OUT.jsonl [--size N] [--seed S]');

const records = positionals.flatMap(path => readFileSync(path, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line)));
const seen = new Set();
// Authoring rows (curriculum.track "authoring") are a separate track: select them with --track authoring.
const unique = records.filter(record => (record.curriculum.track ?? 'interpreter') === values.track &&
  (!values.split || record.split === values.split) && !seen.has(record.id) && seen.add(record.id));
const units = new Map();
for (const record of unique) {
  const key = record.curriculum.pair_group ?? record.id;
  units.set(key, [...(units.get(key) ?? []), record]);
}
const count = (list, axis) => { const out = {}; for (const r of list) out[r.curriculum[axis]] = (out[r.curriculum[axis]] ?? 0) + 1; return out; };
const available = { domain: count(unique, 'domain'), slice: count(unique, 'slice') };
const targets = { domain: DOMAIN_TARGETS, slice: SLICE_TARGETS };
const bound = Math.floor(Math.min(...['domain', 'slice'].flatMap(axis =>
  Object.entries(targets[axis]).map(([key, share]) => (available[axis][key] ?? 0) / share))));
/** Greedy fill of a shard of this size with units that keep every share under its cap. */
function fill(size) {
  const caps = Object.fromEntries(['domain', 'slice'].map(axis =>
    [axis, Object.fromEntries(Object.entries(targets[axis]).map(([key, share]) => [key, Math.ceil(share * size)]))]));
  const rng = new Random(values.seed, 'select');
  const byFamily = new Map();
  for (const unit of rng.shuffle([...units.values()])) {
    const family = unit[0].curriculum.family;
    byFamily.set(family, [...(byFamily.get(family) ?? []), unit]);
  }
  const queues = rng.shuffle([...byFamily.values()]);
  const used = { domain: {}, slice: {} };
  const chosen = [];
  const fits = unit => ['domain', 'slice'].every(axis => Object.entries(count(unit, axis))
    .every(([key, n]) => (used[axis][key] ?? 0) + n <= (caps[axis][key] ?? 0)));
  // Each step takes, from the family used least so far among the best candidates, the next unit that fits and
  // goes furthest toward the domains and slices with the most room left.
  const taken = new Map();
  const room = unit => ['domain', 'slice'].reduce((sum, axis) => sum + Object.entries(count(unit, axis))
    .reduce((part, [key, n]) => part + n * (1 - (used[axis][key] ?? 0) / Math.max(1, caps[axis][key] ?? 0)), 0), 0) / unit.length;
  while (chosen.length < size) {
    let best;
    for (const queue of queues) {
      const at = queue.findIndex(fits);
      if (at === -1) continue;
      const unit = queue[at], family = unit[0].curriculum.family;
      const score = room(unit) - 0.02 * (taken.get(family) ?? 0);
      if (!best || score > best.score) best = { queue, at, unit, family, score };
    }
    if (!best) break;
    best.queue.splice(best.at, 1);
    taken.set(best.family, (taken.get(best.family) ?? 0) + 1);
    for (const axis of ['domain', 'slice']) for (const [key, n] of Object.entries(count(best.unit, axis))) used[axis][key] = (used[axis][key] ?? 0) + n;
    chosen.push(...best.unit);
  }
  return chosen;
}
// Without --size, the largest shard whose domain and slice shares are all within three points of target.
const near = list => list.length > 0 && ['domain', 'slice'].every(axis => Object.entries(targets[axis])
  .every(([key, share]) => Math.abs((count(list, axis)[key] ?? 0) / list.length - share) <= 0.03));
let size = values.size ? Number(values.size) : bound;
// --no-balance keeps every case (small held-out or authoring pools, whose shares cannot follow the targets).
let chosen = values['no-balance'] ? unique : fill(size);
while (!values['no-balance'] && !values.size && !near(chosen) && size > 10) chosen = fill(size = Math.floor(size * 0.98));
writeFileSync(values.out, chosen.map(record => JSON.stringify(record)).join('\n') + '\n');
const shares = axis => Object.fromEntries(Object.entries(count(chosen, axis)).map(([key, n]) => [key, `${(n / chosen.length * 100).toFixed(1)}% (target ${(targets[axis][key] * 100).toFixed(0)}%)`]));
console.log(JSON.stringify({ pool: unique.length, bound, selected: chosen.length, families: new Set(chosen.map(r => r.curriculum.family)).size,
  domain: shares('domain'), slice: shares('slice') }, null, 1));
