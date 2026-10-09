/**
 * The label and writer of advisory files, shared by the advisory command-line tools (explain-advisory.mjs,
 * curriculum-advisory.mjs). An advisory file sits beside the crisp result it comments on, says it is advisory, names the
 * explainer by hash and records the hash of the inputs it read. Nothing the pipeline acts on reads it:
 * `acted_on_by_pipeline` is the constant false.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

export const ADVISORY_SCHEMA = 'natlang.advisory_explanation/1';

const sha256 = text => createHash('sha256').update(text).digest('hex');
export const canonical = value => JSON.stringify(value, (_key, item) => item && typeof item === 'object' && !Array.isArray(item) ?
  Object.fromEntries(Object.entries(item).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) : item);

/** Who explained: the hash of the function sources (name to text) and the executor's name. */
export function explainerIdentity(sources, executor) {
  const sources_sha256 = sha256(canonical(sources));
  return { explainer: `natlang@${sha256(`${sources_sha256}\n${executor}`).slice(0, 16)}`, executor, sources_sha256 };
}

/** The file's content: labelled advisory, with the explainer's hash and the hash of the inputs it read. */
export function advisoryFile(name, identity, inputs, items) {
  return { schema: ADVISORY_SCHEMA, advisory: true, function: name, explainer: identity.explainer, executor: identity.executor,
    inputs_sha256: sha256(canonical(inputs)), acted_on_by_pipeline: false, items };
}

/** Fails before any model call when the file already exists. */
export function refuseExisting(path) {
  if (existsSync(path)) throw new Error(`${path} exists; advisory files are written once, next to the result they explain`);
}

/** Writes beside the crisp result and never over anything: an existing file is an error. */
export function writeAdvisory(path, file) {
  if (existsSync(path)) throw new Error(`${path} exists; advisory files are written once, next to the result they explain`);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(file, null, 2)}\n`, { flag: 'wx' });
}
