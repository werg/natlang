import { ADAPTATION_SCHEMA, type AdaptationArtifact, type ComponentValue } from './types.js';
import { fingerprint, immutable, cloneData } from './identity.js';
export class AdaptationError extends Error { constructor(message: string) { super(message); this.name = 'AdaptationError'; } }
const fail = (message: string): never => { throw new AdaptationError(message); };
/** JSON recursive descent with duplicate member detection, including escaped spellings. */
export function parseStrictJSON(text: string): unknown {
  let at = 0;
  const ws = () => { while (/[ \t\r\n]/.test(text[at] ?? '') && at < text.length) at++; };
  const string = (): string => {
    const start = at++; let escaped = false;
    while (at < text.length) { const c = text[at++]!;
      if (!escaped && c === '"') { try { return JSON.parse(text.slice(start, at)); } catch { fail('invalid JSON string'); } }
      if (!escaped && c === '\\') escaped = true; else escaped = false;
    }
    return fail('unterminated JSON string');
  };
  const value = (): unknown => {
    ws(); const c = text[at];
    if (c === '"') return string();
    if (c === '{') { at++; ws(); const object: Record<string, unknown> = Object.create(null); const seen = new Set<string>();
      if (text[at] === '}') { at++; return object; }
      while (true) { ws(); if (text[at] !== '"') fail('expected JSON member'); const key = string();
        if (seen.has(key)) fail('duplicate JSON key: ' + key); seen.add(key); ws(); if (text[at++] !== ':') fail('expected colon');
        object[key] = value(); ws(); const delimiter = text[at++]; if (delimiter === '}') return object;
        if (delimiter !== ',') fail('expected object delimiter'); }
    }
    if (c === '[') { at++; ws(); const array: unknown[] = []; if (text[at] === ']') { at++; return array; }
      while (true) { array.push(value()); ws(); const delimiter = text[at++]; if (delimiter === ']') return array;
        if (delimiter !== ',') fail('expected array delimiter'); }
    }
    const token = /^(?:true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)/.exec(text.slice(at));
    if (!token) return fail('invalid JSON value at ' + at); at += token[0].length;
    const parsed: unknown = JSON.parse(token[0]); if (typeof parsed === 'number' && !Number.isFinite(parsed)) fail('nonfinite JSON number'); return parsed;
  };
  const result = value(); ws(); if (at !== text.length) fail('trailing JSON input'); return result;
}
export function object(value: unknown, keys: readonly string[], path: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return fail(path + ' must be an object');
  const record = value as Record<string, unknown>;
  if (Object.keys(record).some(key => !keys.includes(key)) || keys.some(key => !Object.hasOwn(record, key))) fail(path + ' has missing or unknown fields');
  return record;
}
export function nonempty(value: unknown, path: string): string { if (typeof value !== 'string' || !value.length) return fail(path + ' must be a nonempty string'); return value; }
function hash(value: unknown, path: string): string {
  const text = nonempty(value, path); if (!/^[a-f0-9]{64}$/.test(text)) fail(path + ' must be a SHA-256 digest'); return text;
}
export function strings(value: unknown, path: string): readonly string[] {
  if (!Array.isArray(value) || value.some(part => typeof part !== 'string')) return fail(path + ' must contain strings'); return value;
}
export function validateValue(value: unknown): ComponentValue {
  if (!value || typeof value !== 'object') return fail('invalid component value');
  const kind = (value as Record<string, unknown>).kind;
  if (kind === 'program.guidance') { const v = object(value, ['kind', 'text'], 'guidance'); if (typeof v.text !== 'string') fail('guidance text must be a string'); }
  else if (kind === 'lambda.instructions') { const v = object(value, ['kind', 'template'], 'instructions');
    const template = object(v.template, ['segments', 'slotIds'], 'template'); const segments = strings(template.segments, 'segments');
    const slots = strings(template.slotIds, 'slotIds'); if (segments.length !== slots.length + 1) fail('template segment count does not match slots');
    if (slots.some(slot => !slot.length)) fail('slotIds must be nonempty');
  } else fail('unsupported component kind');
  return value as ComponentValue;
}
export function artifactDigest(artifact: Omit<AdaptationArtifact, 'digest'> | AdaptationArtifact): string {
  const { digest: _digest, ...data } = artifact as AdaptationArtifact; return fingerprint(data);
}
export function validateAdaptation(value: unknown): AdaptationArtifact {
  const a = object(value, ['schema', 'digest', 'program', 'executor', 'policy', 'components', 'provenance'], 'artifact');
  if (a.schema !== ADAPTATION_SCHEMA) fail('unsupported adaptation schema'); nonempty(a.digest, 'digest');
  const p = object(a.program, ['id', 'buildHash', 'protocol', 'guidanceScope'], 'program'); nonempty(p.id, 'program.id'); hash(p.buildHash, 'buildHash');
  if (!Number.isInteger(p.protocol) || Number(p.protocol) < 1) fail('invalid protocol');
  const scope = object(p.guidanceScope, ['importedPrograms'], 'guidanceScope'); strings(scope.importedPrograms, 'importedPrograms');
  const imports = scope.importedPrograms as string[];
  if (imports.some(name => !name.length) || new Set(imports).size !== imports.length) fail('guidance imports must be unique nonempty identities');
  const e = object(a.executor, ['id', 'configuration'], 'executor'); nonempty(e.id, 'executor.id');
  if (!e.configuration || typeof e.configuration !== 'object' || Array.isArray(e.configuration)) fail('invalid executor configuration');
  const policyKeys = ['codeEdits', 'settings', ...((a.policy as Record<string, unknown>)?.limits !== undefined ? ['limits'] : []),
    ...((a.policy as Record<string, unknown>)?.systemPromptHash !== undefined ? ['systemPromptHash'] : [])];
  const policy = object(a.policy, policyKeys, 'policy');
  if (policy.codeEdits !== 'allow' && policy.codeEdits !== 'deny') fail('invalid code edit policy');
  if (!policy.settings || typeof policy.settings !== 'object' || Array.isArray(policy.settings)) fail('invalid policy settings');
  if (policy.systemPromptHash !== undefined) hash(policy.systemPromptHash, 'systemPromptHash');
  if (policy.limits !== undefined) {
    if (!policy.limits || typeof policy.limits !== 'object' || Array.isArray(policy.limits)) fail('invalid inference limits');
    for (const [key, limit] of Object.entries(policy.limits as Record<string, unknown>))
      if (!['maxEpisodes', 'maxDepth', 'maxActions', 'maxToolCalls', 'timeoutMs'].includes(key) || !Number.isSafeInteger(limit) || Number(limit) < 0) fail('invalid inference limit: ' + key);
  }
  if (!Array.isArray(a.components)) fail('components must be an array'); const seen = new Set<string>();
  for (const raw of a.components as unknown[]) { const c = object(raw, ['key', 'baselineHash', 'contractHash', 'value'], 'component');
    const key = nonempty(c.key, 'key'); if (seen.has(key)) fail('duplicate component: ' + key); seen.add(key);
    hash(c.baselineHash, 'baselineHash'); hash(c.contractHash, 'contractHash'); validateValue(c.value); }
  const provenance = object(a.provenance, ['runId', 'strategy', 'engine', 'suiteHash', 'seed', 'promotion', 'evidence'], 'provenance');
  for (const key of ['runId', 'strategy', 'engine', 'suiteHash']) nonempty(provenance[key], key);
  hash(provenance.suiteHash, 'suiteHash');
  if (!Number.isSafeInteger(provenance.seed)) fail('invalid seed');
  if (!['selected', 'revalidated', 'incumbent'].includes(provenance.promotion as string)) fail('invalid promotion');
  if (!provenance.evidence || typeof provenance.evidence !== 'object' || Array.isArray(provenance.evidence)) fail('invalid evidence');
  // Canonical serialization rejects nested non-JSON/nonfinite evidence/configuration as well.
  if (artifactDigest(value as AdaptationArtifact) !== a.digest) fail('artifact digest mismatch');
  return immutable(cloneData(value as AdaptationArtifact));
}
export function parseAdaptation(text: string): AdaptationArtifact { return validateAdaptation(parseStrictJSON(text)); }
