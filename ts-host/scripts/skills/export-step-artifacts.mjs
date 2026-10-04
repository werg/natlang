#!/usr/bin/env node
/** Save the Neuralese artifacts that improvement steps name (soft skills, adapters, soft values) from the server that
 * holds them into one `.nz` file, so the records stay resolvable after the server's in-memory store is gone.
 *
 * Usage: export-step-artifacts.mjs --steps improvement-steps.jsonl --endpoint URL --out artifacts.nz
 *
 * Every `before`, `after` and proposal artifact of kind soft-skill, soft-value, adapter or adapter-code with a block
 * ID is exported under its ID (`<kind>_<id>`); missing blocks are reported, not fatal. Crisp artifacts are named by
 * source digest and are not blocks.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { HttpNeuraleseStore } from '../../dist/model/neuralese-server.js';
import { saveNz } from '../../dist/native/nz-file.js';
import { neuraleseRef } from '../../dist/native/neuralese.js';

const arg = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
const [stepsPath, endpoint, out] = [arg('--steps'), arg('--endpoint'), arg('--out')];
if (!stepsPath || !endpoint || !out) throw Error('Usage: export-step-artifacts.mjs --steps FILE --endpoint URL --out FILE.nz');
const KINDS = new Set(['soft-skill', 'soft-value', 'adapter', 'adapter-code']);
const wanted = new Map();
for (const line of readFileSync(stepsPath, 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const step = JSON.parse(line);
  for (const ref of [...step.before, ...step.after, ...step.proposal.deltas.map(item => item.artifact)])
    if (KINDS.has(ref.kind) && typeof ref.id === 'string' && ref.id.startsWith('nz1_')) wanted.set(ref.id, ref.kind);
}
const store = new HttpNeuraleseStore(endpoint);
const exports = {}, missing = [];
for (const [id, kind] of wanted) {
  const meta = await store.meta(id);
  if (!meta) { missing.push(id); continue; }
  const type = kind === 'adapter' ? 'Adapter' : meta.type && meta.type !== 'Adapter' ? meta.type : 'Neuralese<string>';
  exports[`${kind.replace('-', '_')}_${id}`] = { type, value: neuraleseRef(type, id) };
}
if (!Object.keys(exports).length) throw Error(`none of the ${wanted.size} artifacts are on ${endpoint}`);
writeFileSync(out, await saveNz(exports, { store, dialect: 'nd:natlang@1', provenance: { steps: stepsPath, endpoint } }), { flag: 'wx' });
console.log(JSON.stringify({ exported: Object.keys(exports).length, missing: missing.length, out }));
