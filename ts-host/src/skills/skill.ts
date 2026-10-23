/**
 * Skills as natlang context items (S2 §2).
 *
 * A skill is a standard agent-harness skill: a folder with a `SKILL.md` whose YAML frontmatter has `name` and
 * `description`, followed by markdown instructions, plus optional supporting files (references, examples, scripts,
 * assets). Natlang reads such folders unchanged and adds an optional `natlang` block (scope injection, typed exports,
 * requirements, tests, provenance). The block may be written as a top-level `natlang:` key or, for validators that only
 * accept the standard keys, as `metadata.natlang` (an object or a YAML/JSON string). Native state references appear
 * directly in typed metadata values; `.nz` archives carry artifact custody, not skill manifests.
 */
import YAML from 'yaml';
import { isNeuraleseRef, neuraleseRef, type NeuraleseRef } from '../native/neuralese.js';
import { hexDigest } from '../native/hash.js';
import { parseType, TypeSyntaxError } from '../native/types.js';

export const SKILL_FILE = 'SKILL.md';

/** Keys of the standard skill frontmatter; anything else is kept but reported. */
export const STANDARD_KEYS = new Set(['name', 'description', 'summary', 'license', 'allowed-tools', 'metadata', 'compatibility', 'natlang']);

/** A scope binding comes from a literal value or a JSON file in the skill (with an optional JSON pointer). */
export type ScopeSource = { type: string; value?: unknown; file?: string; pointer?: string; description?: string };
export type SkillTest = { name: string; input?: unknown; expected?: unknown; check?: string };
export type NatlangSkillMeta = {
  scope: Record<string, ScopeSource>;
  exports: Record<string, string>;
  requires: { skills: string[]; services: string[] };
  tests: SkillTest[];
  provenance?: Record<string, unknown>;
};
export type SkillDiagnostic = { path: string; code: string; message: string; severity: 'error' | 'warning' };

/** Semantic discovery metadata: crisp text or an actual soft value, never an implicit gloss. */
export type SkillDescription = string | NeuraleseRef;

export type Skill = {
  readonly name: string;
  readonly description: SkillDescription;
  readonly summary?: SkillDescription;
  readonly format: 'markdown';
  /** Skill folder, relative to the context root. */
  readonly root: string;
  /** Markdown instructions after the frontmatter. */
  readonly body: string;
  readonly frontmatter: Readonly<Record<string, unknown>>;
  readonly natlang: NatlangSkillMeta;
  /** Supporting files, relative to the skill root, read on demand. */
  readonly files: readonly string[];
  /** Content digest over the skill's files: the skill's revision. */
  readonly revision: string;
};

const NAME = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const IDENT = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

export function splitFrontmatter(text: string): { data: unknown; body: string } | undefined {
  const normalized = text.replace(/^﻿/, '').replace(/\r\n/g, '\n');
  if (!normalized.startsWith('---\n')) return undefined;
  const end = normalized.indexOf('\n---', 3);
  if (end < 0) return undefined;
  const after = normalized.indexOf('\n', end + 4);
  const body = after < 0 ? '' : normalized.slice(after + 1);
  return { data: YAML.parse(normalized.slice(4, end + 1)) ?? {}, body: body.replace(/^\n+/, '') };
}

const plain = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

function safeRelative(path: string): boolean {
  return !!path && !path.startsWith('/') && !path.split('/').some(part => part === '..' || part === '');
}

/** Read and validate the natlang block; unknown fields are warnings, malformed ones errors. */
export function readNatlangMeta(frontmatter: Record<string, unknown>, path: string, files?: readonly string[],
  diagnostics: SkillDiagnostic[] = []): NatlangSkillMeta {
  const error = (code: string, message: string) => diagnostics.push({ path, code, message, severity: 'error' });
  let raw: unknown = frontmatter.natlang;
  const metadata = frontmatter.metadata;
  if (raw === undefined && plain(metadata) && metadata.natlang !== undefined) {
    raw = metadata.natlang;
    if (typeof raw === 'string') {
      try { raw = YAML.parse(raw); } catch (cause) { error('skill-natlang-syntax', `metadata.natlang is not YAML or JSON: ${(cause as Error).message}`); raw = {}; }
    }
  }
  const meta: NatlangSkillMeta = { scope: {}, exports: {}, requires: { skills: [], services: [] }, tests: [] };
  if (raw === undefined || raw === null) return meta;
  if (!plain(raw)) { error('skill-natlang-shape', 'the natlang block must be a mapping'); return meta; }
  for (const key of Object.keys(raw)) if (!['scope', 'exports', 'requires', 'tests', 'provenance'].includes(key))
    diagnostics.push({ path, code: 'skill-natlang-unknown-field', message: `unknown natlang field "${key}"`, severity: 'warning' });
  const parses = (where: string, text: unknown): text is string => {
    if (typeof text !== 'string') { error('skill-type', `${where}: a type must be a string in natlang type syntax`); return false; }
    try { parseType(text); return true; } catch (cause) {
      error('skill-type', `${where}: ${cause instanceof TypeSyntaxError ? cause.message : String(cause)}`); return false;
    }
  };
  if (raw.scope !== undefined) {
    if (!plain(raw.scope)) error('skill-scope-shape', 'natlang.scope must map binding names to sources');
    else for (const [name, source] of Object.entries(raw.scope)) {
      const where = `natlang.scope.${name}`;
      if (!IDENT.test(name)) { error('skill-scope-name', `${where}: "${name}" is not an identifier`); continue; }
      if (!plain(source)) { error('skill-scope-shape', `${where}: expected { type, value | file }`); continue; }
      if (!parses(where, source.type)) continue;
      const given = ['value', 'file'].filter(key => source[key] !== undefined);
      if (given.length !== 1) { error('skill-scope-source', `${where}: give exactly one of value or file`); continue; }
      const target = typeof source.file === 'string' ? source.file : undefined;
      if (given[0] !== 'value') {
        if (typeof target !== 'string' || !safeRelative(target)) { error('skill-scope-path', `${where}: path must be relative and inside the skill`); continue; }
        if (files !== undefined && !files.includes(target)) { error('skill-scope-missing-file', `${where}: ${target} is not a file of this skill`); continue; }
      }
      meta.scope[name] = { type: source.type, ...(source.value !== undefined ? { value: source.value } : {}),
        ...(typeof source.file === 'string' ? { file: source.file } : {}), ...(typeof source.pointer === 'string' ? { pointer: source.pointer } : {}),
        ...(typeof source.description === 'string' ? { description: source.description } : {}) };
    }
  }
  if (raw.exports !== undefined) {
    if (!plain(raw.exports)) error('skill-exports-shape', 'natlang.exports must map names to types');
    else for (const [name, type] of Object.entries(raw.exports)) if (parses(`natlang.exports.${name}`, type)) meta.exports[name] = type;
  }
  if (raw.requires !== undefined) {
    const requires = raw.requires;
    const list = (value: unknown, where: string): string[] => {
      if (value === undefined) return [];
      if (Array.isArray(value) && value.every(item => typeof item === 'string')) return value;
      error('skill-requires-shape', `${where} must be a list of names`); return [];
    };
    if (!plain(requires)) error('skill-requires-shape', 'natlang.requires must be { skills?, services? }');
    else meta.requires = { skills: list(requires.skills, 'natlang.requires.skills'), services: list(requires.services, 'natlang.requires.services') };
  }
  if (raw.tests !== undefined) {
    if (!Array.isArray(raw.tests)) error('skill-tests-shape', 'natlang.tests must be a list');
    else raw.tests.forEach((test, index) => {
      if (!plain(test) || typeof test.name !== 'string') { error('skill-tests-shape', `natlang.tests[${index}] needs a name`); return; }
      if (test.expected === undefined && typeof test.check !== 'string')
        { error('skill-tests-shape', `natlang.tests[${index}] needs expected or check`); return; }
      meta.tests.push({ name: test.name, input: test.input, expected: test.expected, ...(typeof test.check === 'string' ? { check: test.check } : {}) });
    });
  }
  if (raw.provenance !== undefined) {
    if (plain(raw.provenance)) meta.provenance = raw.provenance;
    else error('skill-provenance-shape', 'natlang.provenance must be a mapping');
  }
  return meta;
}

/** Validate semantic metadata without converting a soft value to text. IDs and routing fields remain crisp. */
export function readSkillDescription(value: unknown, field: 'description' | 'summary', path: string,
  diagnostics: SkillDiagnostic[]): SkillDescription | undefined {
  const error = (code: string, message: string) => diagnostics.push({ path, code, message, severity: 'error' });
  if (typeof value === 'string') {
    const text = value.trim();
    if (!text) { error(`skill-${field}-missing`, `${field} must not be empty`); return undefined; }
    if (text.length > 1024) { error(`skill-${field}-length`, `${field} exceeds 1024 characters`); return undefined; }
    return text;
  }
  if (isNeuraleseRef(value)) {
    try {
      const type = parseType(value.$neuralese.type);
      if (type.kind !== 'neuralese' || type.element.kind !== 'prim' || type.element.name !== 'string')
        throw new Error('expected Neuralese<string>');
      return neuraleseRef(value.$neuralese.type, value.$neuralese.id,value.$neuralese.stop_gradient,value.$neuralese.source_occurrence);
    } catch { error(`skill-${field}-type`, `${field} must have type Neuralese<string> (optionally dialect-qualified)`); return undefined; }
  }
  error(`skill-${field}-${value === undefined ? 'missing' : 'type'}`, `${field} must be nonempty text or a Neuralese<string> reference`);
  return undefined;
}

/** Validate standard skill frontmatter fields. */
export function readStandardFields(frontmatter: Record<string, unknown>, path: string, expectedName: string | undefined,
  diagnostics: SkillDiagnostic[], extraKeys: readonly string[] = []): { name: string; description: SkillDescription; summary?: SkillDescription } {
  const error = (code: string, message: string) => diagnostics.push({ path, code, message, severity: 'error' });
  const name = typeof frontmatter.name === 'string' ? frontmatter.name : '';
  const description = readSkillDescription(frontmatter.description, 'description', path, diagnostics) ?? '';
  const summary = frontmatter.summary === undefined ? undefined : readSkillDescription(frontmatter.summary, 'summary', path, diagnostics);
  if (!name) error('skill-name-missing', 'frontmatter needs a name');
  else if (!NAME.test(name) || name.length > 64) error('skill-name-format', `name "${name}" must be lowercase letters, digits and hyphens, at most 64 characters`);
  else if (expectedName !== undefined && name !== expectedName) error('skill-name-folder', `name "${name}" must match its folder "${expectedName}"`);

  for (const key of Object.keys(frontmatter)) if (!STANDARD_KEYS.has(key) && !extraKeys.includes(key))
    diagnostics.push({ path, code: 'skill-frontmatter-unknown', message: `non-standard frontmatter key "${key}" is kept but other harnesses may reject it`, severity: 'warning' });
  return { name, description, ...(summary !== undefined ? { summary } : {}) };
}

export function skillRevision(contents: Readonly<Record<string, string | Uint8Array>>): string {
  const parts = Object.keys(contents).sort().map(path => {
    const body = contents[path]!;
    return `${path}\u0000${typeof body === 'string' ? body : hexDigest(body)}`;
  });
  return 'sk1_' + hexDigest(parts.join('\u0001')).slice(0, 32);
}

/** Parse one markdown skill. `files` maps paths relative to the skill folder to their contents (SKILL.md included). */
export function parseMarkdownSkill(root: string, files: Readonly<Record<string, string | Uint8Array>>,
  diagnostics: SkillDiagnostic[] = []): Skill | undefined {
  const path = `${root}/${SKILL_FILE}`;
  const source = files[SKILL_FILE];
  if (typeof source !== 'string') { diagnostics.push({ path, code: 'skill-file-missing', message: 'SKILL.md is missing', severity: 'error' }); return undefined; }
  const parsed = (() => { try { return splitFrontmatter(source); } catch (cause) { return { error: (cause as Error).message }; } })();
  if (!parsed || 'error' in parsed) {
    diagnostics.push({ path, code: 'skill-frontmatter', message: parsed ? `frontmatter is not YAML: ${parsed.error}` : 'SKILL.md must begin with YAML frontmatter between --- lines', severity: 'error' });
    return undefined;
  }
  if (!plain(parsed.data)) { diagnostics.push({ path, code: 'skill-frontmatter', message: 'frontmatter must be a mapping', severity: 'error' }); return undefined; }
  const before = diagnostics.filter(d => d.severity === 'error').length;
  const folder = root.split('/').pop();
  const { name, description, summary } = readStandardFields(parsed.data, path, folder, diagnostics);
  const supporting = Object.keys(files).filter(file => file !== SKILL_FILE).sort();
  const natlang = readNatlangMeta(parsed.data, path, supporting, diagnostics);
  if (!parsed.body.trim()) diagnostics.push({ path, code: 'skill-body-empty', message: 'SKILL.md has no instructions after the frontmatter', severity: 'warning' });
  if (diagnostics.filter(d => d.severity === 'error').length > before) return undefined;
  return { name, description, ...(summary !== undefined ? { summary } : {}), format: 'markdown', root, body: parsed.body, frontmatter: parsed.data, natlang,
    files: supporting, revision: skillRevision(files) };
}
