import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { isDeepStrictEqual } from 'node:util';
import { curriculumCase, evalCall, returnCall } from '../inline-curriculum/lib.mjs';

export const VERSION = 'natlang.recovered_source_adapter/1';
export const sha = value => createHash('sha256').update(typeof value === 'string' || Buffer.isBuffer(value) ? value : JSON.stringify(value)).digest('hex');
export const equal = isDeepStrictEqual;
export function requireValue(condition, reason) { if (!condition) throw new Error(reason); }
export async function* jsonLines(path) {
  const lines = createInterface({input:createReadStream(path), crlfDelay:Infinity});
  try { for await (const line of lines) if (line.trim()) yield JSON.parse(line); }
  finally { lines.close(); }
}
export async function fileHash(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}
export function typeOf(value) {
  if (value === null) return 'null';
  if (typeof value === 'number') { requireValue(Number.isFinite(value), 'nonfinite_value'); return 'number'; }
  if (typeof value === 'boolean' || typeof value === 'string') return typeof value;
  if (Array.isArray(value)) {
    requireValue(value.length > 0, 'untyped_empty_array');
    return `(${[...new Set(value.map(typeOf))].join(' | ')})[]`;
  }
  if (value && typeof value === 'object') return `{ ${Object.entries(value).map(([key, child]) => `${JSON.stringify(key)}: ${typeOf(child)}`).join(', ')} }`;
  throw new Error('unsupported_value_type');
}

/** Gold, scripted references and quality metadata remain outside visible inputs/files. */
export function adaptedCase(original, info, {family, suffix='program', root, files={}, inputs={}, expected,
  actions, children=[], quality='eligible', checks=[], scope='source_task_reference', sourceGroups, sourceIds}) {
  requireValue(typeof original.id==='string' && original.id.length>0, 'missing_original_id');
  requireValue(['train','dev','val','validation','test'].includes(original.split),'missing_original_split');
  requireValue(original.license || (info.license && info.license!=='source-record-required'),'missing_source_license');
  const group = original.group_id ?? original.source_groups?.[0] ?? original.id;
  const record = curriculumCase({family:`recovered_${family}`, shape:sha([original.id, suffix, VERSION]).slice(0,24),
    variant:suffix, splitGroup:`${info.source}:${group}`, slice:children.length ? 'nested_scoped' : 'inline_placement',
    domain:'other', mode:'single_call', reference:{root:actions, ...(children.length ? {children} : {})},
    root, files, inputs, expected, split:original.split === 'train' ? 'train' : 'test'});
  record.source = info.source;
  record.source_ids = sourceIds ?? original.source_ids ?? [original.id];
  record.source_groups = [...new Set(sourceGroups ?? (original.source_groups?.map(g=>`${info.source}:${g}`) ?? [`${info.source}:${group}`]))];
  requireValue(record.source_groups.length>0 && record.source_groups.every(g=>typeof g==='string' && g && !g.endsWith(':undefined')), 'invalid_source_groups');
  record.source_revisions = original.source_revisions ?? [original.source_revision ?? info.sha256];
  record.license = original.license ?? info.license;
  requireValue(typeof record.license === 'string' && record.license.length > 0, 'missing_source_license');
  record.gold_sources = original.gold_sources ?? [original.gold_source ?? 'source-annotation'];
  record.external_source = {source:info.source, original_split:original.split, snapshot_sha256:info.sha256,
    original_row_sha256:sha(original), original_row:original, original_id:original.id, source_path:info.path,
    quality:{version:'natlang.source_quality/1', status:quality, checks}, conversion_scope:scope};
  record.generation = {generator:VERSION};
  return record;
}
export {evalCall, returnCall};
