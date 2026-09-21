import { chmodSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { homedir, platform } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { canonicalJson, parsePackageArchive, readPackageArchive, type NatlangPackageArchive } from './archive.js';

export type InstalledPackage = { name: string; version: string; digest: string; root: string };

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

function encodedName(name: string): string { return encodeURIComponent(name); }
export class NatlangPackageStore {
  readonly root: string;
  constructor(root = defaultNatlangDataDirectory()) { this.root = resolve(root); }
  private refPath(name: string, version: string): string { return join(this.root, 'refs', encodedName(name), `${version}.json`); }
  private objectRoot(digest: string): string { return join(this.root, 'objects', digest); }

  install(value: NatlangPackageArchive | string): InstalledPackage {
    const archive = typeof value === 'string' ? readPackageArchive(value) : parsePackageArchive(value);
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
    renameSync(temporaryRef, ref);
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
