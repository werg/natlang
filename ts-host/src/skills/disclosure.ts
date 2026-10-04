/**
 * Progressive disclosure of bound skills (S2 §2.2), and scope injection.
 *
 * 1. A call's opening lists each bound skill by name and description.
 * 2. The model loads a skill's instructions when it judges them relevant (`read_code("skills.<name>")`), and reads
 *    supporting files on demand (`read_code("skills.<name>/references/x.md")`).
 * 3. Each bound skill's declared scope bindings are injected into the eval scope automatically and declared in the
 *    opening, so the model sees them without loading anything.
 *
 * The runtime decides where these strings go; this module only produces them.
 */
import { neuraleseRef, neuraleseSentinel, type NeuraleseRef } from '../native/neuralese.js';
import { formatType, parseType, TypeEnv, type Type } from '../native/types.js';
import { coerce, Reject, type Value } from '../native/values.js';
import { readSkillFile, type SkillSet } from './registry.js';
import type { Skill, SkillDiagnostic } from './skill.js';

export const SKILL_READ_PREFIX = 'skills.';

/** The opening's skill listing; empty when no skills are bound. */
export function renderSkillListing(set: SkillSet, diagnostics: readonly SkillDiagnostic[] = []): string {
  if (!set.size && !diagnostics.length) return '';
  const semantic = (value: Skill['description']): string => typeof value === 'string'
    ? value.replace(/\s+/g, ' ').trim() : neuraleseSentinel(value.$neuralese.id);
  const lines = set.list().map(skill => `- ${skill.name}: ${semantic(skill.description)}` +
    (skill.summary !== undefined ? `\n  Summary: ${semantic(skill.summary)}` : '') +
    `\n  Read instructions: read_code("skills.${skill.name}")`);
  const notes = diagnostics.length ? ['', 'Skill diagnostics (invalid skills or scope bindings were omitted):',
    ...diagnostics.map(item => `- ${item.path} [${item.code}]`)] : [];
  return ['Skills bound to this call. Choose the skills whose descriptions fit the current task; you do not need to read every skill.',
    'Read selected instructions with read_code("skills.<name>") before relying on them, then apply the relevant steps to the current inputs. Reading instructions does not execute a procedure.',
    'Read supporting files only when needed with read_code("skills.<name>/<path>"). Callable helpers, when provided, are shown in the eval scope. If no skill fits, solve using the task instructions and available tools.',
    ...lines, ...notes].join('\n');
}

export type SkillDocument =
  | { kind: 'text'; skill: string; path: string; text: string }
  | { kind: 'neuralese'; skill: string; path: string; export: string; note: string };

/** Whether a `read_code` target names a skill (so the runtime can route it here). */
export const isSkillTarget = (target: string): boolean => target.startsWith(SKILL_READ_PREFIX);

/**
 * Resolve `skills.<name>` (instructions plus a list of the skill's files) or `skills.<name>/<path>` (one supporting
 * file). A `.nz` skill's instructions are a Neuralese export: the caller renders it as a literal through the read port.
 */
export async function readSkillDocument(set: SkillSet, target: string): Promise<SkillDocument> {
  if (!isSkillTarget(target)) throw new Error(`not a skill target: ${target}`);
  const rest = target.slice(SKILL_READ_PREFIX.length);
  const slash = rest.indexOf('/');
  const name = slash < 0 ? rest : rest.slice(0, slash);
  const skill = set.get(name);
  if (!skill) throw new Error(`no skill named "${name}"; bound skills: ${set.names.join(', ') || 'none'}`);
  if (slash >= 0) {
    const file = rest.slice(slash + 1);
    return { kind: 'text', skill: name, path: `${skill.root}/${file}`, text: await readSkillFile(set, name, file) };
  }
  if (skill.format === 'nz') return { kind: 'neuralese', skill: name, path: skill.root, export: skill.body,
    note: `Instructions of skill "${name}" (soft):` };
  const files = skill.files.length ? `\n\nFiles of this skill (read with read_code("skills.${name}/<path>")):\n` +
    skill.files.map(file => `- ${file}: read_code("skills.${name}/${file}")`).join('\n') : '';
  return { kind: 'text', skill: name, path: `${skill.root}/SKILL.md`, text: skill.body.trimEnd() + files };
}

/** Resolves a `.nz` scope source (`file.nz#export`) to a stored block ID; supplied by the `.nz` reader (S4). */
export type NzExportResolver = (skill: Skill, file: string, exportName: string) => Promise<string>;

export type ScopeBinding = {
  name: string;
  type: Type;
  typeText: string;
  value: Value | NeuraleseRef;
  skill: string;
  description?: string;
};

function pointerGet(value: unknown, pointer: string): unknown {
  if (!pointer || pointer === '/') return value;
  if (!pointer.startsWith('/')) throw new Error(`JSON pointer must start with "/": ${pointer}`);
  let current: unknown = value;
  for (const raw of pointer.slice(1).split('/')) {
    const key = raw.replace(/~1/g, '/').replace(/~0/g, '~');
    if (current === null || typeof current !== 'object' || !(key in (current as Record<string, unknown>)))
      throw new Error(`JSON pointer ${pointer} does not resolve`);
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

/**
 * Compute the scope bindings that bound skills inject. Values are type-checked against their declared types; a name
 * declared by two skills, or a binding that fails to load or check, is reported and left out.
 */
export async function scopeBindings(set: SkillSet, options: { env?: TypeEnv; nz?: NzExportResolver; reserved?: readonly string[] } = {}):
  Promise<{ bindings: ScopeBinding[]; diagnostics: SkillDiagnostic[] }> {
  const env = options.env ?? new TypeEnv();
  const diagnostics: SkillDiagnostic[] = [];
  const bindings: ScopeBinding[] = [];
  const owner = new Map<string, string>();
  const reserved = new Set(options.reserved ?? []);
  for (const skill of set.list()) for (const [name, source] of Object.entries(skill.natlang.scope)) {
    const path = `${skill.root}#natlang.scope.${name}`;
    const report = (code: string, message: string) => diagnostics.push({ path, code, message, severity: 'error' });
    if (reserved.has(name)) { report('skill-scope-reserved', `"${name}" is already a name in this call's scope`); continue; }
    if (owner.has(name)) { report('skill-scope-conflict', `"${name}" is also injected by skill "${owner.get(name)}"`); continue; }
    let type: Type;
    try { type = parseType(source.type); } catch (cause) { report('skill-type', (cause as Error).message); continue; }
    try {
      let value: Value | NeuraleseRef;
      if (source.nz !== undefined) {
        if (type.kind !== 'neuralese') { report('skill-scope-nz-type', `a .nz source needs a Neuralese type, not ${source.type}`); continue; }
        if (!options.nz) { report('nz-skill-unsupported', `cannot load ${source.nz}: .nz reading is not available yet`); continue; }
        const [file, exportName] = source.nz.split('#') as [string, string];
        value = neuraleseRef(formatType(type), await options.nz(skill, file, exportName));
      } else {
        const raw = source.file !== undefined
          ? pointerGet(JSON.parse(await readSkillFile(set, skill.name, source.file)), source.pointer ?? '')
          : source.value;
        value = coerce(raw, type, env, name);
      }
      owner.set(name, skill.name);
      bindings.push({ name, type, typeText: formatType(type), value, skill: skill.name,
        ...(source.description ? { description: source.description } : {}) });
    } catch (cause) {
      report(cause instanceof Reject ? 'skill-scope-type' : 'skill-scope-load', (cause as Error).message);
    }
  }
  return { bindings, diagnostics };
}

/** Declarations for the opening, in the style of the runtime's own `declare` lines. */
export function renderScopeDeclarations(bindings: readonly ScopeBinding[]): string {
  return bindings.map(binding => `declare const ${binding.name}: ${binding.typeText};  // from skill ${binding.skill}` +
    (binding.description ? `: ${binding.description}` : '')).join('\n');
}
