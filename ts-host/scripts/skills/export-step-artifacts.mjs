#!/usr/bin/env node
/** Archive native state and coefficient artifacts named by improvement-step records.
 *
 * Usage: export-step-artifacts.mjs --steps improvement-steps.jsonl --endpoint URL --out artifacts.nz
 *
 * Current soft-skill and soft-value artifacts use nzs1 state IDs; adapters use nnt1 coefficient IDs. Historical nz1
 * references are reported as unavailable and are never interpreted as native state.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { HttpNeuraleseStore } from '../../dist/model/neuralese-server.js';
import { saveNz } from '../../dist/native/nz-file.js';
import { neuraleseRef, nativeAdapterRef } from '../../dist/native/neuralese.js';

const arg = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
const [stepsPath, endpoint, out] = [arg('--steps'), arg('--endpoint'), arg('--out')];
if (!stepsPath || !endpoint || !out) throw Error('Usage: export-step-artifacts.mjs --steps FILE --endpoint URL --out FILE.nz');
const STATE_KINDS = new Set(['soft-skill', 'soft-value']);
const TENSOR_KINDS = new Set(['adapter', 'adapter-code']);
const wanted = new Map(), unavailableLegacy = new Set();
for (const line of readFileSync(stepsPath, 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const step = JSON.parse(line);
  for (const ref of [...step.before, ...step.after, ...step.proposal.deltas.map(item => item.artifact)]) {
    if (typeof ref.id !== 'string') continue;
    if (ref.id.startsWith('nz1_')) { unavailableLegacy.add(ref.id); continue; }
    if (STATE_KINDS.has(ref.kind) && ref.id.startsWith('nzs1_') || TENSOR_KINDS.has(ref.kind) && ref.id.startsWith('nnt1_'))
      wanted.set(ref.id, ref.kind);
  }
}
const store = new HttpNeuraleseStore(endpoint);
const exports = {}, missing = [], dialects = new Set();
for (const [id, kind] of wanted) {
  const meta = await store.meta(id);
  if (!meta) { missing.push(id); continue; }
  dialects.add(meta.dialect);
  exports[`${kind.replace('-', '_')}_${id}`] = id.startsWith('nnt1_') ? nativeAdapterRef(id) :
    neuraleseRef(`Neuralese<unknown, ${JSON.stringify(meta.dialect)}>`, id);
}
if (!Object.keys(exports).length) throw Error(`none of the ${wanted.size} native artifacts are on ${endpoint}`);
const dialect = dialects.values().next().value ?? 'native-artifact-bundle/1';
writeFileSync(out, await saveNz(exports, { store, dialect, provenance: { steps: stepsPath, endpoint } }), { flag: 'wx' });
console.log(JSON.stringify({ exported: Object.keys(exports).length, missing: missing.length,
  unavailable_legacy_refs: unavailableLegacy.size, out }));
