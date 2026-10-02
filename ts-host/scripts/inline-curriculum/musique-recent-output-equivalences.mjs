/** Exact, source-pinned answer equivalences; primary gold and version 1 stay intact. */
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {sourceConversionDigest} from '../../dist/teacher/source-conversion.js';

const REGISTRY = 'musique-reviewed-output-equivalence/2026-10-02-v2';
const REGISTRY_SHA = 'd28ae9c50a7cb802b6e3b5b2ed9c8bc2b493c28402497e2e3439898d2946d06a';
const SUFFIX = ':reviewed-output-equivalence-v2';
const bytes = readFileSync(new URL('./musique-reviewed-output-equivalences-v2.json', import.meta.url));
const sha = value => createHash('sha256').update(value).digest('hex');
if (sha(bytes) !== REGISTRY_SHA) throw Error('recent_musique_equivalence_registry_hash_mismatch');
const registry = JSON.parse(bytes);
if (registry.version !== REGISTRY || registry.entries.length !== 2) throw Error('recent_musique_equivalence_registry_shape_mismatch');
const entries = new Map(registry.entries.map(entry => [entry.source_id, entry]));
const same = (a, b) => sourceConversionDigest(a) === sourceConversionDigest(b);

export function applyRecentMusiqueOutputEquivalences(record) {
  if (record?.source !== 'musique') return record;
  const entry = entries.get(record.external_source?.source_id ?? record.source_ids?.[0]);
  if (!entry) return record;
  const review = {registry: REGISTRY, registry_sha256: REGISTRY_SHA,
    base_ir_id: entry.base_ir_id, base_ir_sha256: entry.base_ir_sha256,
    exact_accepted_outputs: entry.accepted, review: registry.review};
  const oracle = {level: 'normalized', alternates: [...new Set([
    ...(entry.base_oracle.alternates ?? []), ...entry.accepted])]};
  const base = structuredClone(record);
  if (record.id === entry.base_ir_id + SUFFIX) {
    if (!same(base.generation?.reviewed_answer_equivalence, review) || !same(base.semantics.oracle, oracle))
      throw Error('recent_musique_equivalence_variant_drift');
    base.id = entry.base_ir_id;
    base.semantics.oracle = structuredClone(entry.base_oracle);
    delete base.generation.reviewed_answer_equivalence;
  }
  // This digest pins all instructions, inputs, evidence, identity, split, license and reference.
  if (base.id !== entry.base_ir_id || sourceConversionDigest(base) !== entry.base_ir_sha256 ||
      base.semantics.expected !== entry.primary)
    throw Error('recent_musique_equivalence_source_proof_mismatch:' + entry.source_id);
  const result = structuredClone(base);
  result.id += SUFFIX;
  result.semantics.oracle = oracle;
  result.generation.reviewed_answer_equivalence = review;
  return result;
}
