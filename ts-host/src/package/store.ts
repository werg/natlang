import { chmodSync, existsSync, linkSync, lstatSync, mkdirSync, readFileSync, readdirSync,
  renameSync, rmSync, unlinkSync, writeFileSync } from 'node:fs';
import { homedir, platform } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { canonicalJson, parsePackageArchive, readPackageArchive, sha256, type NatlangPackageArchive } from './archive.js';

export type InstalledPackageIdentity = { name: string; version: string; digest: string; root: string };
export type InstalledPackage = InstalledPackageIdentity & { dependencies: Record<string, InstalledPackageIdentity> };
type DependencyPin = { version: string; digest: string };
type PackageReference = { name: string; version: string; digest: string; dependencies?: Record<string, DependencyPin> };

type Version = [number, number, number, string];
function version(value: string): Version {
  const match = /^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$/.exec(value);
  if (!match) throw new TypeError(`invalid semantic version: ${value}`);
  return [Number(match[1]), Number(match[2]), Number(match[3]), match[4] ?? ''];
}
function compare(left: Version, right: Version): number {
  for (let index = 0; index < 3; index++) if (left[index] !== right[index]) return Number(left[index]) - Number(right[index]);
  if (left[3] === right[3]) return 0;
  if (!left[3]) return 1; if (!right[3]) return -1;
  return left[3].localeCompare(right[3]);
}
export function compareVersions(left: string, right: string): number { return compare(version(left), version(right)); }
export function satisfiesVersion(actual: string, range: string): boolean {
  if (range === '*' || range === 'latest') return true;
  const candidate = version(actual);
  if (range.startsWith('^') || range.startsWith('~')) {
    const base = version(range.slice(1));
    if (compare(candidate, base) < 0) return false;
    const ceiling: Version = range[0] === '~' ? [base[0], base[1] + 1, 0, ''] :
      base[0] > 0 ? [base[0] + 1, 0, 0, ''] : base[1] > 0 ? [0, base[1] + 1, 0, ''] : [0, 0, base[2] + 1, ''];
    return compare(candidate, ceiling) < 0;
  }
  if (range.startsWith('>=')) return compare(candidate, version(range.slice(2))) >= 0;
  return compare(candidate, version(range)) === 0;
}

export function defaultNatlangDataDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  if (environment.NATLANG_HOME) return resolve(environment.NATLANG_HOME);
  if (platform() === 'win32') return join(environment.LOCALAPPDATA ?? homedir(), 'natlang');
  if (platform() === 'darwin') return join(homedir(), 'Library', 'Application Support', 'natlang');
  return join(environment.XDG_DATA_HOME ?? join(homedir(), '.local', 'share'), 'natlang');
}
export function defaultNatlangConfigDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  if (environment.NATLANG_CONFIG_HOME) return resolve(environment.NATLANG_CONFIG_HOME);
  if (platform() === 'win32') return join(environment.APPDATA ?? homedir(), 'natlang');
  if (platform() === 'darwin') return join(homedir(), 'Library', 'Application Support', 'natlang');
  return join(environment.XDG_CONFIG_HOME ?? join(homedir(), '.config'), 'natlang');
}
export function defaultNatlangStateDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  if (environment.NATLANG_STATE_HOME) return resolve(environment.NATLANG_STATE_HOME);
  if (platform() === 'win32') return join(environment.LOCALAPPDATA ?? homedir(), 'natlang', 'state');
  if (platform() === 'darwin') return join(homedir(), 'Library', 'Application Support', 'natlang', 'state');
  return join(environment.XDG_STATE_HOME ?? join(homedir(), '.local', 'state'), 'natlang');
}
export function defaultNatlangCacheDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  if (environment.NATLANG_CACHE_HOME) return resolve(environment.NATLANG_CACHE_HOME);
  if (platform() === 'win32') return join(environment.LOCALAPPDATA ?? homedir(), 'natlang', 'cache');
  if (platform() === 'darwin') return join(homedir(), 'Library', 'Caches', 'natlang');
  return join(environment.XDG_CACHE_HOME ?? join(homedir(), '.cache'), 'natlang');
}

function encodedName(name: string): string { return encodeURIComponent(name); }
export class NatlangPackageStore {
  readonly root: string;
  constructor(root = defaultNatlangDataDirectory()) { this.root = resolve(root); }
  private refPath(name: string, version: string): string { return join(this.root, 'refs', encodedName(name), `${version}.json`); }
  private objectRoot(digest: string): string { return join(this.root, 'objects', digest); }

  install(value: NatlangPackageArchive | string): InstalledPackage {
    return this.installMany([value])[0]!;
  }

  installMany(values: Array<NatlangPackageArchive | string>): InstalledPackage[] {
    const archives = values.map(value => typeof value === 'string' ? readPackageArchive(value) : parsePackageArchive(value));
    const identities = new Set<string>(), byIdentity = new Map<string, NatlangPackageArchive>();
    for (const archive of archives) {
      const identity = `${archive.manifest.name}@${archive.manifest.version}`;
      if (identities.has(identity)) throw new Error(`duplicate package candidate: ${identity}`);
      identities.add(identity); byIdentity.set(identity, archive);
      const ref = this.refPath(archive.manifest.name, archive.manifest.version);
      if (existsSync(ref) && this.resolve(identity).digest !== archive.digest)
        throw new Error(`${identity} is already bound to another digest`);
    }
    const installed = this.list();
    const pins = new Map<string, Record<string, DependencyPin>>();
    for (const archive of archives) {
      const identity = `${archive.manifest.name}@${archive.manifest.version}`;
      const existingRef = this.refPath(archive.manifest.name, archive.manifest.version);
      if (existsSync(existingRef)) {
        const locked = JSON.parse(readFileSync(existingRef, 'utf8')) as PackageReference;
        for (const [name, range] of Object.entries(archive.manifest.dependencies ?? {})) {
          const pin = locked.dependencies?.[name];
          if (!pin || !satisfiesVersion(pin.version, range)) throw new Error(`${identity} has an invalid dependency lock for ${name}`);
        }
        pins.set(identity, locked.dependencies ?? {}); continue;
      }
      const selected: Record<string, DependencyPin> = {};
      for (const [name, range] of Object.entries(archive.manifest.dependencies ?? {})) {
        const choices = [...installed.filter(item => item.name === name).map(item =>
          ({ name, version: item.version, digest: item.digest })),
          ...archives.filter(item => item.manifest.name === name).map(item =>
            ({ name, version: item.manifest.version, digest: item.digest }))]
          .filter(item => satisfiesVersion(item.version, range))
          .sort((left, right) => compare(version(right.version), version(left.version)) || left.digest.localeCompare(right.digest));
        const chosen = choices[0];
        if (!chosen) throw new Error(`${archive.manifest.name}@${archive.manifest.version} needs ${name}@${range}`);
        selected[name] = { version: chosen.version, digest: chosen.digest };
      }
      pins.set(identity, selected);
    }
    const visiting = new Set<string>(), visited = new Set<string>(), order: NatlangPackageArchive[] = [];
    const visit = (identity: string): void => {
      if (visiting.has(identity)) throw new Error(`package dependency cycle includes ${identity}`);
      if (visited.has(identity)) return;
      visiting.add(identity);
      for (const [name, pin] of Object.entries(pins.get(identity) ?? {})) {
        const dependencyIdentity = `${name}@${pin.version}`;
        if (byIdentity.has(dependencyIdentity)) visit(dependencyIdentity);
      }
      visiting.delete(identity); visited.add(identity); order.push(byIdentity.get(identity)!);
    };
    for (const identity of identities) visit(identity);
    for (const archive of order) this.installChecked(archive,
      pins.get(`${archive.manifest.name}@${archive.manifest.version}`) ?? {});
    return archives.map(archive => this.resolve(`${archive.manifest.name}@${archive.manifest.version}`));
  }

  private installChecked(archive: NatlangPackageArchive, dependencies: Record<string, DependencyPin>): void {
    const ref = this.refPath(archive.manifest.name, archive.manifest.version);
    if (existsSync(ref)) {
      const current = JSON.parse(readFileSync(ref, 'utf8')) as PackageReference;
      if (current.digest !== archive.digest) throw new Error(`${archive.manifest.name}@${archive.manifest.version} is already bound to another digest`);
      if (canonicalJson(current.dependencies ?? {}) !== canonicalJson(dependencies))
        throw new Error(`${archive.manifest.name}@${archive.manifest.version} already has a different dependency lock`);
      return;
    }
    mkdirSync(join(this.root, 'tmp'), { recursive: true });
    const destination = this.objectRoot(archive.digest);
    if (!existsSync(destination)) {
      mkdirSync(dirname(destination), { recursive: true });
      const temporary = join(this.root, 'tmp', `${archive.digest}-${process.pid}-${Date.now()}`);
      mkdirSync(join(temporary, 'files'), { recursive: true });
      try {
        for (const file of archive.files) {
          const path = join(temporary, 'files', ...file.path.split('/'));
          mkdirSync(dirname(path), { recursive: true });
          writeFileSync(path, Buffer.from(file.content, 'base64'), { mode: 0o444 });
        }
        writeFileSync(join(temporary, 'manifest.json'), canonicalJson(archive.manifest) + '\n', { mode: 0o444 });
        writeFileSync(join(temporary, 'archive.json'), canonicalJson(archive) + '\n', { mode: 0o444 });
        try { renameSync(temporary, destination); }
        catch (error) { if (!existsSync(destination)) throw error; rmSync(temporary, { recursive: true, force: true }); }
      } catch (error) { rmSync(temporary, { recursive: true, force: true }); throw error; }
      chmodSync(join(destination, 'files'), 0o555);
    } else {
      const installed = readPackageArchive(join(destination, 'archive.json'));
      if (installed.digest !== archive.digest) throw new Error(`corrupt package object: ${archive.digest}`);
    }
    mkdirSync(dirname(ref), { recursive: true });
    const temporaryRef = `${ref}.${process.pid}.tmp`;
    writeFileSync(temporaryRef, canonicalJson({ name: archive.manifest.name, version: archive.manifest.version,
      digest: archive.digest, dependencies }) + '\n');
    try { linkSync(temporaryRef, ref); }
    catch (error) {
      if (!existsSync(ref)) throw error;
      const current = JSON.parse(readFileSync(ref, 'utf8')) as PackageReference;
      if (current.digest !== archive.digest || canonicalJson(current.dependencies ?? {}) !== canonicalJson(dependencies))
        throw new Error(`${archive.manifest.name}@${archive.manifest.version} was concurrently bound to another package lock`);
    } finally { unlinkSync(temporaryRef); }
  }

  resolve(specifier: string): InstalledPackage {
    const at = specifier.lastIndexOf('@');
    if (at <= 0 || at === specifier.length - 1) throw new TypeError('package specifier must be name@version');
    const name = specifier.slice(0, at), version = specifier.slice(at + 1);
    const ref = this.refPath(name, version);
    if (!existsSync(ref)) throw new Error(`package is not installed: ${specifier}`);
    const value = JSON.parse(readFileSync(ref, 'utf8')) as PackageReference;
    if (typeof value.digest !== 'string') throw new Error(`invalid package reference: ${specifier}`);
    const installed = this.verifyInstalled(name, version, value.digest);
    const dependencies: Record<string, InstalledPackageIdentity> = {};
    for (const [dependencyName, pin] of Object.entries(value.dependencies ?? {})) {
      const dependencyRef = this.refPath(dependencyName, pin.version);
      if (!existsSync(dependencyRef)) throw new Error(`locked dependency is missing: ${dependencyName}@${pin.version}`);
      const dependency = JSON.parse(readFileSync(dependencyRef, 'utf8')) as PackageReference;
      if (dependency.digest !== pin.digest) throw new Error(`locked dependency changed: ${dependencyName}@${pin.version}`);
      dependencies[dependencyName] = this.verifyInstalled(dependencyName, pin.version, pin.digest);
    }
    return { ...installed, dependencies };
  }

  private verifyInstalled(name: string, packageVersion: string, digest: string): InstalledPackageIdentity {
    const root = this.objectRoot(digest);
    const archive = readPackageArchive(join(root, 'archive.json'));
    if (archive.manifest.name !== name || archive.manifest.version !== packageVersion || archive.digest !== digest)
      throw new Error(`corrupt package object: ${name}@${packageVersion}`);
    for (const file of archive.files) {
      const path = join(root, 'files', ...file.path.split('/'));
      const status = lstatSync(path);
      if (!status.isFile() || status.isSymbolicLink() || status.size !== file.size || sha256(readFileSync(path)) !== file.sha256)
        throw new Error(`installed package content changed: ${file.path}`);
    }
    return { name, version: packageVersion, digest, root: join(root, 'files') };
  }

  list(): InstalledPackage[] {
    const refs = join(this.root, 'refs');
    if (!existsSync(refs)) return [];
    return readdirSync(refs).flatMap(encoded => readdirSync(join(refs, encoded)).filter(file => file.endsWith('.json')).map(file =>
      this.resolve(`${decodeURIComponent(encoded)}@${file.slice(0, -5)}`))).sort((a, b) =>
        a.name.localeCompare(b.name) || a.version.localeCompare(b.version));
  }

  manifest(specifier: string) {
    const installed = this.resolve(specifier);
    return readPackageArchive(join(dirname(installed.root), 'archive.json')).manifest;
  }

  resolveDependencies(dependencies: Record<string, string> = {}): Record<string, InstalledPackageIdentity> {
    const installed = this.list(), resolved: Record<string, InstalledPackageIdentity> = {};
    for (const [name, range] of Object.entries(dependencies)) {
      const matches = installed.filter(item => item.name === name && satisfiesVersion(item.version, range))
        .sort((left, right) => compareVersions(right.version, left.version));
      const selected = matches[0];
      if (!selected) throw new Error(`no installed package satisfies ${name}@${range}`);
      resolved[name] = { name: selected.name, version: selected.version,
        digest: selected.digest, root: selected.root };
    }
    return resolved;
  }
}
