import { isAbsolute, posix } from 'node:path';

export const PACKAGE_SCHEMA = 'natlang.package/v2' as const;

/** A launchable application target: a TypeScript entry module and the function it exports. */
export type NatlangTarget = {
  entry: string;
  /** Exported function receiving the target context; default `main`. */
  export?: string;
  description?: string;
  authority?: string[];
  commands?: string[];
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
};

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
  const known = new Set(['schema', 'name', 'version', 'description', 'include', 'targets', 'exports', 'dependencies', 'engines']);
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
      const targetKnown = new Set(['entry', 'export', 'description', 'authority', 'commands']);
      for (const key of Object.keys(target)) if (!targetKnown.has(key)) throw new TypeError(`unknown field in target ${name}: ${key}`);
      const authority = target.authority === undefined ? undefined : requireStrings(target.authority, `target ${name} authority`);
      const commands = target.commands === undefined ? undefined : requireStrings(target.commands, `target ${name} commands`);
      targets[name] = { entry: packagePath(target.entry, `target ${name} entry`),
        ...(target.export === undefined ? {} : { export: requireString(target.export, `target ${name} export`) }),
        ...(target.description === undefined ? {} : { description: requireString(target.description, `target ${name} description`) }),
        ...(authority ? { authority } : {}), ...(commands ? { commands } : {}) };
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
  return { schema: PACKAGE_SCHEMA, name: raw.name, version: raw.version, include,
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
