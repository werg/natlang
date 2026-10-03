/**
 * Loading the skills of a context folder, immutable skill sets, and the global skill pool (S2 §2.3).
 *
 * Skills are part of the context a function is bound to. A program's own `skills/` folder is bound implicitly; a host
 * may also keep a global, mutable pool. Changing the pool makes a new pool revision and never changes a skill set that
 * was already bound: a program sees pool changes only when it is rebound.
 */
import { hexDigest } from '../native/hash.js';
import { parseMarkdownSkill, parseNzSkill, SKILL_FILE, unsupportedNzSkills, type NzSkillHook, type Skill,
  type SkillDiagnostic } from './skill.js';

/** Read access to a context folder. Paths are relative to the context root and use `/`. */
export type SkillSource = {
  list(): Promise<readonly string[]>;
  read(path: string): Promise<string | Uint8Array>;
};

export function memorySkillSource(files: Readonly<Record<string, string | Uint8Array>>): SkillSource {
  return { async list() { return Object.keys(files).sort(); }, async read(path) {
    const body = files[path];
    if (body === undefined) throw new Error(`no such file: ${path}`);
    return body;
  } };
}

const text = (body: string | Uint8Array) => typeof body === 'string' ? body : new TextDecoder().decode(body);
const bytes = (body: string | Uint8Array) => typeof body === 'string' ? new TextEncoder().encode(body) : body;
/** Supporting files are kept as text when they decode cleanly, so revisions are stable across sources. */
const asStored = (body: string | Uint8Array): string | Uint8Array => {
  if (typeof body === 'string') return body;
  try { return new TextDecoder('utf-8', { fatal: true }).decode(body); } catch { return body; }
};

export type LoadedSkills = { set: SkillSet; diagnostics: SkillDiagnostic[] };

/**
 * Load every skill under `root` (default `skills`): `skills/<name>/SKILL.md` folders, and `.nz` skills either as
 * `skills/<name>.nz` or as a folder `skills/<name>/` whose only skill file is `<name>.nz`. Invalid skills are left out
 * and reported.
 */
export async function loadSkills(source: SkillSource, options: { root?: string; nz?: NzSkillHook } = {}): Promise<LoadedSkills> {
  const root = (options.root ?? 'skills').replace(/\/+$/, '');
  const hook = options.nz ?? unsupportedNzSkills;
  const diagnostics: SkillDiagnostic[] = [];
  const paths = (await source.list()).filter(path => path.startsWith(root + '/'));
  const folders = new Map<string, string[]>();
  const looseNz: string[] = [];
  for (const path of paths) {
    const rest = path.slice(root.length + 1);
    const slash = rest.indexOf('/');
    if (slash < 0) { if (rest.endsWith('.nz')) looseNz.push(path); continue; }
    const folder = rest.slice(0, slash);
    folders.set(folder, [...(folders.get(folder) ?? []), rest.slice(slash + 1)]);
  }
  const skills: Skill[] = [];
  for (const [folder, inner] of [...folders].sort(([a], [b]) => a.localeCompare(b))) {
    const base = `${root}/${folder}`;
    if (inner.includes(SKILL_FILE)) {
      const files: Record<string, string | Uint8Array> = {};
      for (const file of inner) files[file] = asStored(await source.read(`${base}/${file}`));
      const skill = parseMarkdownSkill(base, files, diagnostics);
      if (skill) skills.push(skill);
    } else if (inner.includes(`${folder}.nz`)) {
      const skill = await parseNzSkill(`${base}/${folder}.nz`, bytes(await source.read(`${base}/${folder}.nz`)), hook,
        inner.filter(file => file !== `${folder}.nz`), diagnostics);
      if (skill) skills.push(skill);
    } else diagnostics.push({ path: base, code: 'skill-file-missing', message: `${base} has no SKILL.md or ${folder}.nz`, severity: 'error' });
  }
  for (const path of looseNz.sort()) {
    const skill = await parseNzSkill(path, bytes(await source.read(path)), hook, [], diagnostics);
    if (skill) skills.push(skill);
  }
  const seen = new Map<string, Skill>();
  const unique: Skill[] = [];
  for (const skill of skills) {
    const earlier = seen.get(skill.name);
    if (earlier) { diagnostics.push({ path: skill.root, code: 'skill-duplicate', message: `skill "${skill.name}" is also defined at ${earlier.root}`, severity: 'error' }); continue; }
    seen.set(skill.name, skill); unique.push(skill);
  }
  return { set: new SkillSet(unique, source), diagnostics };
}

/** Read a supporting file of a skill through the source it was loaded from. */
export async function readSkillFile(set: SkillSet, name: string, file: string): Promise<string> {
  const skill = set.get(name);
  if (!skill) throw new Error(`no skill named "${name}"`);
  if (skill.format !== 'markdown' || !skill.files.includes(file)) throw new Error(`skill "${name}" has no file ${file}`);
  const source = set.sourceOf(name);
  if (!source) throw new Error(`skill "${name}" has no readable source`);
  return text(await source.read(`${skill.root}/${file}`));
}

/** An immutable set of skills, as bound into a context. */
export class SkillSet {
  readonly #skills: ReadonlyMap<string, Skill>;
  readonly #sources: ReadonlyMap<string, SkillSource>;
  readonly revision: string;

  constructor(skills: readonly Skill[] = [], source?: SkillSource, sources?: ReadonlyMap<string, SkillSource>) {
    const map = new Map<string, Skill>();
    const owners = new Map<string, SkillSource>(sources ?? []);
    for (const skill of skills) {
      if (map.has(skill.name) && map.get(skill.name)!.revision !== skill.revision)
        throw new Error(`conflicting revisions of skill "${skill.name}"`);
      map.set(skill.name, skill);
      if (source && !owners.has(skill.name)) owners.set(skill.name, source);
    }
    this.#skills = new Map([...map].sort(([a], [b]) => a.localeCompare(b)));
    this.#sources = owners;
    this.revision = 'ss1_' + hexDigest([...this.#skills.values()].map(skill => `${skill.name}@${skill.revision}`).join('\n')).slice(0, 32);
  }

  get names(): string[] { return [...this.#skills.keys()]; }
  get size(): number { return this.#skills.size; }
  get(name: string): Skill | undefined { return this.#skills.get(name); }
  has(name: string): boolean { return this.#skills.has(name); }
  list(): Skill[] { return [...this.#skills.values()]; }
  sourceOf(name: string): SkillSource | undefined { return this.#sources.get(name); }

  /** Union with another set; the same name at different revisions is a conflict. */
  union(other: SkillSet): SkillSet {
    const sources = new Map(this.#sources);
    for (const name of other.names) if (!sources.has(name) && other.sourceOf(name)) sources.set(name, other.sourceOf(name)!);
    return new SkillSet([...this.list(), ...other.list()], undefined, sources);
  }

  /** The subset with the given names (rebinding selects skills this way). Unknown names are an error. */
  select(names: readonly string[]): SkillSet {
    const missing = names.filter(name => !this.#skills.has(name));
    if (missing.length) throw new Error(`unknown skills: ${missing.join(', ')}`);
    return new SkillSet(names.map(name => this.#skills.get(name)!), undefined,
      new Map(names.flatMap(name => this.#sources.has(name) ? [[name, this.#sources.get(name)!] as const] : [])));
  }

  without(names: readonly string[]): SkillSet { return this.select(this.names.filter(name => !names.includes(name))); }

  /** Skills whose `requires.skills` name a skill outside this set. */
  unmetRequirements(): { skill: string; missing: string[] }[] {
    return this.list().flatMap(skill => {
      const missing = skill.natlang.requires.skills.filter(name => !this.has(name));
      return missing.length ? [{ skill: skill.name, missing }] : [];
    });
  }
}

/**
 * The host's global, mutable skill pool. Every change makes a new revision; snapshots are immutable `SkillSet`s, so
 * functions bound earlier keep the skills they were bound with.
 */
export class SkillPool {
  #current = new SkillSet();
  readonly #history: { revision: string; change: string }[] = [];

  get revision(): string { return this.#current.revision; }
  get history(): readonly { revision: string; change: string }[] { return this.#history; }
  snapshot(): SkillSet { return this.#current; }

  /** Add or replace skills (publishing an improved skill replaces the earlier revision). */
  publish(skills: SkillSet | readonly Skill[]): string {
    const incoming = skills instanceof SkillSet ? skills : new SkillSet(skills);
    const kept = this.#current.without(incoming.names.filter(name => this.#current.has(name)));
    this.#current = kept.union(incoming);
    this.#history.push({ revision: this.#current.revision, change: `publish ${incoming.names.join(', ')}` });
    return this.#current.revision;
  }

  retire(names: readonly string[]): string {
    this.#current = this.#current.without(names.filter(name => this.#current.has(name)));
    this.#history.push({ revision: this.#current.revision, change: `retire ${names.join(', ')}` });
    return this.#current.revision;
  }
}

/** A skill's executable nodes (S0 §7.3): `.nl` and TypeScript files under `helpers/`. Everything else is data. */
export function skillExecutables(skill: Skill): string[] {
  return skill.format === 'markdown'
    ? skill.files.filter(file => file.startsWith('helpers/') && /\.(nl|ts|mts)$/.test(file)).map(file => `${skill.root}/${file}`)
    : [];
}

/**
 * Executable nodes present in `after` that `before` did not have. Rebinding may edit nodes and add data, but new
 * executable nodes must come from files written to a staged tree and compiled, so a non-empty result means the new
 * set needs that path rather than a plain rebinding.
 */
export function addedExecutables(before: SkillSet, after: SkillSet): string[] {
  const known = new Set(before.list().flatMap(skillExecutables));
  return after.list().flatMap(skillExecutables).filter(path => !known.has(path));
}

/**
 * The skills a program's functions are bound with by default: the program's own skills, plus an explicit selection
 * from a snapshot of the pool. Requirements must be met inside the result.
 */
export function bindSkills(program: SkillSet, pool?: SkillSet, selection: readonly string[] = []): SkillSet {
  const chosen = selection.length ? (pool ?? new SkillSet()).select(selection) : new SkillSet();
  const bound = program.union(chosen);
  const unmet = bound.unmetRequirements();
  if (unmet.length) throw new Error('unmet skill requirements: ' + unmet.map(item => `${item.skill} needs ${item.missing.join(', ')}`).join('; '));
  return bound;
}
