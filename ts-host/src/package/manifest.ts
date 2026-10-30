import { isAbsolute, posix } from 'node:path';
import { parseTargetArguments, type TargetArgument } from '../runtime/target-arguments.js';

export const PACKAGE_SCHEMA = 'natlang.package/v2' as const;

/** A launchable application target: a TypeScript entry module and the function it exports. */
export type NatlangTarget = {
  entry: string;
  /** Relative path to a portable adaptation artifact included in this package. */
  adaptation?: string;
  /** Exported function receiving the target context; default `main`. */
  export?: string;
  description?: string;
  authority?: string[];
  commands?: string[];
  /** Ordered positional arguments with natlang types (runtime/target-arguments.ts): `natlang run` checks them into
   * `context.arguments`, and the dashboard renders its run form from them. */
  arguments?: TargetArgument[];
};
export type NatlangPackageManifest = {
  schema: typeof PACKAGE_SCHEMA;
  name: string;
  version: string;
  description?: string;
  include: string[];
  targets?: Record<string, NatlangTarget>;
  exports?: Record<string, string>;
  dependencies?: Record<string, string>;
  engines?: { natlang?: string; node?: string };
  /** What the runtime records of this program's calls (plans/TRACE_SPECIALIZATION.md §3.2). */
  recording?: { exclude?: string[] };
  /** How far stored compilations may serve this program's calls; the machine setting still bounds it. */
  specialization?: 'off' | 'shadow' | 'on';
  /** How `Is<T, "predicate">` slots are checked: threshold, uncertainty band, policy, mode (plans/REFINEMENT_TYPES.md). */
  refinements?: Record<string, unknown>;
  /** Fused hand-offs between functions: mode, planner, certificate (plans/FUSED_PIPELINES.md); validated by `parseFusionSettings`. */
  fusion?: Record<string, unknown>;
  /** The model the program is built, warmed and run with unless the command selects another (`--profile`, `--provider`,
   * `--model`, NATLANG_PROFILE/SERVER/PROVIDER): a machine profile by name, a stock model, or profile fields. */
  model?: ProgramModel;
  /** The calls `natlang build --warm` runs ahead of time: their traces are recorded and their prompt prefixes prefilled. */
  warmup?: WarmupCase[];
};
export type ProgramModel = { profile?: string; stock?: string; provider?: string; model?: string; endpoint?: string;
  request?: Record<string, unknown>; contextTokens?: number; parallel?: number; escalate?: string };
/** One warm-up call: a named `.nl` function of the package and its inputs by parameter name. */
export type WarmupCase = { name?: string; function: string; inputs: Record<string, unknown> };

const PACKAGE_NAME = /^(?:@[a-z0-9][a-z0-9._-]*\/)?[a-z0-9][a-z0-9._-]*$/;
const VERSION = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/;

export function packagePath(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value || value.includes('\\') || value.includes('\0') || isAbsolute(value))
    throw new TypeError(`${label} must be a relative portable path`);
  const normalized = posix.normalize(value);
  if (normalized === '..' || normalized.startsWith('../') || normalized === '.' || normalized.startsWith('/'))
    throw new TypeError(`${label} escapes the package`);
  return normalized.replace(/^\.\//, '');
}

function stringMap(value: unknown, label: string): Record<string, string> | undefined {
  if (value === undefined) return undefined;
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`${label} must be an object`);
  const out: Record<string, string> = {};
  for (const [key, item] of Object.entries(value)) {
    if (!key || typeof item !== 'string' || !item) throw new TypeError(`${label} entries must be nonempty strings`);
    out[key] = item;
  }
  return out;
}

export function parsePackageManifest(value: unknown): NatlangPackageManifest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('package manifest must be an object');
  const raw = value as Record<string, unknown>;
  const known = new Set(['schema', 'name', 'version', 'description', 'include', 'targets', 'exports', 'dependencies', 'engines',
    'recording', 'specialization', 'refinements', 'fusion', 'model', 'warmup']);
  for (const key of Object.keys(raw)) if (!known.has(key)) throw new TypeError(`unknown package manifest field: ${key}`);
  if (raw.schema !== PACKAGE_SCHEMA) throw new TypeError(`package schema must be ${PACKAGE_SCHEMA}`);
  if (typeof raw.name !== 'string' || !PACKAGE_NAME.test(raw.name)) throw new TypeError('invalid package name');
  if (typeof raw.version !== 'string' || !VERSION.test(raw.version)) throw new TypeError('version must be semantic x.y.z');
  if (!Array.isArray(raw.include) || !raw.include.length) throw new TypeError('include must name at least one file or directory');
  const include = [...new Set(raw.include.map((item, index) => packagePath(item, `include[${index}]`)))].sort();
  const targets: Record<string, NatlangTarget> = {};
  if (raw.targets !== undefined) {
    if (!raw.targets || typeof raw.targets !== 'object' || Array.isArray(raw.targets)) throw new TypeError('targets must be an object');
    for (const [name, item] of Object.entries(raw.targets)) {
      if (!/^[a-z0-9][a-z0-9._-]*$/.test(name) || !item || typeof item !== 'object' || Array.isArray(item))
        throw new TypeError(`invalid target ${name}`);
      const target = item as Record<string, unknown>;
      const targetKnown = new Set(['entry', 'adaptation', 'export', 'description', 'authority', 'commands', 'arguments']);
      for (const key of Object.keys(target)) if (!targetKnown.has(key)) throw new TypeError(`unknown field in target ${name}: ${key}`);
      const authority = target.authority === undefined ? undefined : requireStrings(target.authority, `target ${name} authority`);
      const commands = target.commands === undefined ? undefined : requireStrings(target.commands, `target ${name} commands`);
      const args = target.arguments === undefined ? undefined : parseTargetArguments(target.arguments, name);
      targets[name] = { entry: packagePath(target.entry, `target ${name} entry`),
        ...(target.adaptation === undefined ? {} : { adaptation: packagePath(target.adaptation, `target ${name} adaptation`) }),
        ...(target.export === undefined ? {} : { export: requireString(target.export, `target ${name} export`) }),
        ...(target.description === undefined ? {} : { description: requireString(target.description, `target ${name} description`) }),
        ...(authority ? { authority } : {}), ...(commands ? { commands } : {}), ...(args ? { arguments: args } : {}) };
    }
  }
  const exports = stringMap(raw.exports, 'exports');
  if (exports) for (const [name, path] of Object.entries(exports)) exports[name] = packagePath(path, `export ${name}`);
  const dependencies = stringMap(raw.dependencies, 'dependencies');
  let engines: NatlangPackageManifest['engines'];
  if (raw.engines !== undefined) {
    const values = stringMap(raw.engines, 'engines')!;
    for (const key of Object.keys(values)) if (key !== 'natlang' && key !== 'node') throw new TypeError(`unknown engine ${key}`);
    engines = values;
  }
  let recording: NatlangPackageManifest['recording'];
  if (raw.recording !== undefined) {
    if (!raw.recording || typeof raw.recording !== 'object' || Array.isArray(raw.recording)) throw new TypeError('recording must be an object');
    const fields = raw.recording as Record<string, unknown>;
    for (const key of Object.keys(fields)) if (key !== 'exclude') throw new TypeError(`unknown recording field: ${key}`);
    recording = fields.exclude === undefined ? {} : { exclude: requireStrings(fields.exclude, 'recording.exclude') };
  }
  if (raw.specialization !== undefined && !['off', 'shadow', 'on'].includes(raw.specialization as string))
    throw new TypeError('specialization must be "off", "shadow" or "on"');
  if (raw.refinements !== undefined && (!raw.refinements || typeof raw.refinements !== 'object' || Array.isArray(raw.refinements)))
    throw new TypeError('refinements must be an object');
  // The fields are validated by parseRefinementSettings where the runtime is configured (native/refinement-settings.ts).
  const refinements = raw.refinements as Record<string, unknown> | undefined;
  if (raw.fusion !== undefined && (!raw.fusion || typeof raw.fusion !== 'object' || Array.isArray(raw.fusion))) throw new TypeError('fusion must be an object');
  const fusion = raw.fusion as Record<string, unknown> | undefined;
  const model = raw.model === undefined ? undefined : programModel(raw.model);
  let warmup: WarmupCase[] | undefined;
  if (raw.warmup !== undefined) {
    if (!Array.isArray(raw.warmup)) throw new TypeError('warmup must be an array of { function, inputs } calls');
    warmup = raw.warmup.map((item, index) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) throw new TypeError(`warmup[${index}] must be an object`);
      const fields = item as Record<string, unknown>;
      for (const key of Object.keys(fields)) if (!['name', 'function', 'inputs'].includes(key)) throw new TypeError(`unknown field in warmup[${index}]: ${key}`);
      const fn = packagePath(fields.function, `warmup[${index}].function`);
      if (!fn.endsWith('.nl')) throw new TypeError(`warmup[${index}].function must name a .nl file of the package`);
      if (!fields.inputs || typeof fields.inputs !== 'object' || Array.isArray(fields.inputs))
        throw new TypeError(`warmup[${index}].inputs must be an object keyed by parameter name`);
      return { ...(fields.name === undefined ? {} : { name: requireString(fields.name, `warmup[${index}].name`) }), function: fn,
        inputs: fields.inputs as Record<string, unknown> };
    });
  }
  return { ...(model ? { model } : {}), ...(warmup ? { warmup } : {}), schema: PACKAGE_SCHEMA, name: raw.name, version: raw.version, include,
    ...(recording ? { recording } : {}),
    ...(refinements ? { refinements } : {}), ...(fusion ? { fusion } : {}),
    ...(raw.specialization !== undefined ? { specialization: raw.specialization as 'off' | 'shadow' | 'on' } : {}),
    ...(raw.description === undefined ? {} : { description: requireString(raw.description, 'description') }),
    ...(Object.keys(targets).length ? { targets } : {}), ...(exports ? { exports } : {}),
    ...(dependencies ? { dependencies } : {}), ...(engines ? { engines } : {}) };
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value) throw new TypeError(`${label} must be a nonempty string`);
  return value;
}
function requireStrings(value: unknown, label: string): string[] {
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string' || !item))
    throw new TypeError(`${label} must be an array of nonempty strings`);
  return [...new Set(value)].sort();
}

function programModel(value: unknown): ProgramModel {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('model must be an object');
  const raw = value as Record<string, unknown>, out: ProgramModel = {};
  for (const [key, item] of Object.entries(raw)) {
    if (['profile', 'stock', 'provider', 'model', 'endpoint', 'escalate'].includes(key)) (out as Record<string, unknown>)[key] = requireString(item, `model.${key}`);
    else if (key === 'request') {
      if (!item || typeof item !== 'object' || Array.isArray(item)) throw new TypeError('model.request must be an object');
      out.request = item as Record<string, unknown>;
    } else if (key === 'contextTokens' || key === 'parallel') {
      if (!Number.isSafeInteger(item) || (item as number) < 1) throw new TypeError(`model.${key} must be a positive integer`);
      out[key] = item as number;
    } else throw new TypeError(`unknown model field: ${key}`);
  }
  if (out.provider && out.endpoint) throw new TypeError('model cannot set both provider and endpoint');
  if (out.stock && (out.provider || out.endpoint)) throw new TypeError('model.stock is a local model; it cannot be combined with provider or endpoint');
  if (out.provider && !out.model) throw new TypeError('model.provider needs model.model');
  return out;
}
