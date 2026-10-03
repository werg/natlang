/**
 * Skills as natlang context items (S2 §2).
 *
 * A skill is a standard agent-harness skill: a folder with a `SKILL.md` whose YAML frontmatter has `name` and
 * `description`, followed by markdown instructions, plus optional supporting files (references, examples, scripts,
 * assets). Natlang reads such folders unchanged and adds an optional `natlang` block (scope injection, typed exports,
 * requirements, tests, provenance). The block may be written as a top-level `natlang:` key or, for validators that only
 * accept the standard keys, as `metadata.natlang` (an object or a YAML/JSON string). A soft skill keeps the same fields
 * in the metadata of a `.nz` file instead of YAML frontmatter.
 */
import YAML from 'yaml';
import { hexDigest } from '../native/hash.js';
import { parseType, TypeSyntaxError } from '../native/types.js';

export const SKILL_FILE = 'SKILL.md';

/** Keys of the standard skill frontmatter; anything else is kept but reported. */
export const STANDARD_KEYS = new Set(['name', 'description', 'license', 'allowed-tools', 'metadata', 'compatibility', 'natlang']);

/** Where a scope binding's value comes from: a literal, a JSON file in the skill (with an optional JSON pointer), or a
 * `.nz` export (`file.nz#export`). Exactly one source is given. */
export type ScopeSource = { type: string; value?: unknown; file?: string; pointer?: string; nz?: string; description?: string };
export type SkillTest = { name: string; input?: unknown; expected?: unknown; check?: string };
export type NatlangSkillMeta = {
  scope: Record<string, ScopeSource>;
  exports: Record<string, string>;
  requires: { skills: string[]; services: string[] };
  tests: SkillTest[];
  provenance?: Record<string, unknown>;
};
export type SkillDiagnostic = { path: string; code: string; message: string; severity: 'error' | 'warning' };

export type Skill = {
  readonly name: string;
  readonly description: string;
  readonly format: 'markdown' | 'nz';
  /** Folder (markdown skill) or file (`.nz` skill), relative to the context root. */
  readonly root: string;
  /** Markdown instructions after the frontmatter; for a `.nz` skill, the name of its Neuralese body export. */
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
      if (!plain(source)) { error('skill-scope-shape', `${where}: expected { type, value | file | nz }`); continue; }
      if (!parses(where, source.type)) continue;
      const given = ['value', 'file', 'nz'].filter(key => source[key] !== undefined);
      if (given.length !== 1) { error('skill-scope-source', `${where}: give exactly one of value, file or nz`); continue; }
      const target = typeof source.file === 'string' ? source.file : typeof source.nz === 'string' ? source.nz.split('#')[0]! : undefined;
      if (given[0] !== 'value') {
        if (typeof target !== 'string' || !safeRelative(target)) { error('skill-scope-path', `${where}: path must be relative and inside the skill`); continue; }
        if (files !== undefined && !files.includes(target)) { error('skill-scope-missing-file', `${where}: ${target} is not a file of this skill`); continue; }
        if (given[0] === 'nz' && !/^[^#]+\.nz#[A-Za-z_$][A-Za-z0-9_$]*$/.test(String(source.nz)))
          { error('skill-scope-nz', `${where}: nz must be "file.nz#export"`); continue; }
      }
      meta.scope[name] = { type: source.type, ...(source.value !== undefined ? { value: source.value } : {}),
        ...(typeof source.file === 'string' ? { file: source.file } : {}), ...(typeof source.pointer === 'string' ? { pointer: source.pointer } : {}),
        ...(typeof source.nz === 'string' ? { nz: source.nz } : {}), ...(typeof source.description === 'string' ? { description: source.description } : {}) };
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

/** Validate the standard fields shared by markdown and `.nz` skills. */
export function readStandardFields(frontmatter: Record<string, unknown>, path: string, expectedName: string | undefined,
  diagnostics: SkillDiagnostic[], extraKeys: readonly string[] = []): { name: string; description: string } {
  const error = (code: string, message: string) => diagnostics.push({ path, code, message, severity: 'error' });
  const name = typeof frontmatter.name === 'string' ? frontmatter.name : '';
  const description = typeof frontmatter.description === 'string' ? frontmatter.description.trim() : '';
  if (!name) error('skill-name-missing', 'frontmatter needs a name');
  else if (!NAME.test(name) || name.length > 64) error('skill-name-format', `name "${name}" must be lowercase letters, digits and hyphens, at most 64 characters`);
  else if (expectedName !== undefined && name !== expectedName) error('skill-name-folder', `name "${name}" must match its folder "${expectedName}"`);
  if (!description) error('skill-description-missing', 'frontmatter needs a description saying what the skill does and when to use it');
  else if (description.length > 1024) error('skill-description-length', 'description exceeds 1024 characters');
  for (const key of Object.keys(frontmatter)) if (!STANDARD_KEYS.has(key) && !extraKeys.includes(key))
    diagnostics.push({ path, code: 'skill-frontmatter-unknown', message: `non-standard frontmatter key "${key}" is kept but other harnesses may reject it`, severity: 'warning' });
  return { name, description };
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
  const { name, description } = readStandardFields(parsed.data, path, folder, diagnostics);
  const supporting = Object.keys(files).filter(file => file !== SKILL_FILE).sort();
  const natlang = readNatlangMeta(parsed.data, path, supporting, diagnostics);
  if (!parsed.body.trim()) diagnostics.push({ path, code: 'skill-body-empty', message: 'SKILL.md has no instructions after the frontmatter', severity: 'warning' });
  if (diagnostics.filter(d => d.severity === 'error').length > before) return undefined;
  return { name, description, format: 'markdown', root, body: parsed.body, frontmatter: parsed.data, natlang,
    files: supporting, revision: skillRevision(files) };
}

/**
 * Hook for `.nz` skills: the `.nz` reader (S4) supplies the safetensors metadata of a file. The skill fields live under
 * the metadata key `skill` (the same fields as SKILL.md frontmatter, plus `body`: the name of the Neuralese export that
 * holds the instructions).
 */
export type NzSkillHook = { readMetadata(path: string, bytes: Uint8Array): Promise<Record<string, unknown>> };

export const unsupportedNzSkills: NzSkillHook = {
  async readMetadata(path) { throw new Error(`reading .nz skill metadata is not available yet (${path})`); },
};

export async function parseNzSkill(path: string, bytes: Uint8Array, hook: NzSkillHook, siblings: readonly string[] = [],
  diagnostics: SkillDiagnostic[] = []): Promise<Skill | undefined> {
  let metadata: Record<string, unknown>;
  try { metadata = await hook.readMetadata(path, bytes); }
  catch (cause) { diagnostics.push({ path, code: 'nz-skill-unsupported', message: (cause as Error).message, severity: 'error' }); return undefined; }
  const fields = metadata.skill;
  if (!plain(fields)) { diagnostics.push({ path, code: 'nz-skill-metadata', message: 'the .nz metadata has no "skill" mapping', severity: 'error' }); return undefined; }
  const before = diagnostics.filter(d => d.severity === 'error').length;
  const base = path.split('/').pop()!.replace(/\.nz$/, '');
  const folder = path.split('/').slice(-2, -1)[0];
  const expected = path.endsWith(`/${base}.nz`) && folder && folder !== 'skills' ? folder : base;
  const { name, description } = readStandardFields(fields, path, expected, diagnostics, ['body']);
  const natlang = readNatlangMeta(fields, path, siblings, diagnostics);
  const body = typeof fields.body === 'string' ? fields.body : '';
  if (!body) diagnostics.push({ path, code: 'nz-skill-body', message: 'skill.body must name the Neuralese export holding the instructions', severity: 'error' });
  if (diagnostics.filter(d => d.severity === 'error').length > before) return undefined;
  return { name, description, format: 'nz', root: path, body, frontmatter: fields, natlang, files: [...siblings].sort(),
    revision: skillRevision({ [path]: bytes }) };
}
