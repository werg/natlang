import { chmodSync, existsSync, linkSync, lstatSync, mkdirSync, readFileSync, readdirSync,
  renameSync, rmSync, unlinkSync, writeFileSync } from 'node:fs';
import { homedir, platform } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { canonicalJson, parsePackageArchive, readPackageArchive, sha256, type NatlangPackageArchive } from './archive.js';

export type InstalledPackage = { name: string; version: string; digest: string; root: string };

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
    const archive = typeof value === 'string' ? readPackageArchive(value) : parsePackageArchive(value);
    this.checkDependencies(archive, []);
    return this.installChecked(archive);
  }

  installMany(values: Array<NatlangPackageArchive | string>): InstalledPackage[] {
    const archives = values.map(value => typeof value === 'string' ? readPackageArchive(value) : parsePackageArchive(value));
    const identities = new Set<string>();
    for (const archive of archives) {
      const identity = `${archive.manifest.name}@${archive.manifest.version}`;
      if (identities.has(identity)) throw new Error(`duplicate package candidate: ${identity}`);
      identities.add(identity);
      const ref = this.refPath(archive.manifest.name, archive.manifest.version);
      if (existsSync(ref) && this.resolve(identity).digest !== archive.digest)
        throw new Error(`${identity} is already bound to another digest`);
      this.checkDependencies(archive, archives);
    }
    return archives.map(archive => this.installChecked(archive));
  }

  private checkDependencies(archive: NatlangPackageArchive, candidates: NatlangPackageArchive[]): void {
    for (const [name, range] of Object.entries(archive.manifest.dependencies ?? {})) {
      const available = [...this.list().filter(item => item.name === name).map(item => item.version),
        ...candidates.filter(item => item.manifest.name === name).map(item => item.manifest.version)];
      if (!available.some(item => satisfiesVersion(item, range)))
        throw new Error(`${archive.manifest.name}@${archive.manifest.version} needs ${name}@${range}`);
    }
  }

  private installChecked(archive: NatlangPackageArchive): InstalledPackage {
    const ref = this.refPath(archive.manifest.name, archive.manifest.version);
    if (existsSync(ref)) {
      const current = JSON.parse(readFileSync(ref, 'utf8')) as { digest?: string };
      if (current.digest !== archive.digest) throw new Error(`${archive.manifest.name}@${archive.manifest.version} is already bound to another digest`);
      return this.resolve(`${archive.manifest.name}@${archive.manifest.version}`);
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
      digest: archive.digest }) + '\n');
    try { linkSync(temporaryRef, ref); }
    catch (error) {
      if (!existsSync(ref)) throw error;
      const current = JSON.parse(readFileSync(ref, 'utf8')) as { digest?: string };
      if (current.digest !== archive.digest) throw new Error(`${archive.manifest.name}@${archive.manifest.version} was concurrently bound to another digest`);
    } finally { unlinkSync(temporaryRef); }
    return this.resolve(`${archive.manifest.name}@${archive.manifest.version}`);
  }

  resolve(specifier: string): InstalledPackage {
    const at = specifier.lastIndexOf('@');
    if (at <= 0 || at === specifier.length - 1) throw new TypeError('package specifier must be name@version');
    const name = specifier.slice(0, at), version = specifier.slice(at + 1);
    const ref = this.refPath(name, version);
    if (!existsSync(ref)) throw new Error(`package is not installed: ${specifier}`);
    const value = JSON.parse(readFileSync(ref, 'utf8')) as { digest?: string };
    if (typeof value.digest !== 'string') throw new Error(`invalid package reference: ${specifier}`);
    const root = this.objectRoot(value.digest);
    const archive = readPackageArchive(join(root, 'archive.json'));
    if (archive.manifest.name !== name || archive.manifest.version !== version || archive.digest !== value.digest)
      throw new Error(`corrupt package reference: ${specifier}`);
    for (const file of archive.files) {
      const path = join(root, 'files', ...file.path.split('/'));
      const status = lstatSync(path);
      if (!status.isFile() || status.isSymbolicLink() || status.size !== file.size || sha256(readFileSync(path)) !== file.sha256)
        throw new Error(`installed package content changed: ${file.path}`);
    }
    return { name, version, digest: value.digest, root: join(root, 'files') };
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
}
