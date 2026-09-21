import { createHash } from 'node:crypto';
import { lstatSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { packagePath, parsePackageManifest, type NatlangPackageManifest } from './manifest.js';

export const ARCHIVE_SCHEMA = 'natlang.package-archive/v1' as const;
export type PackageFile = { path: string; encoding: 'base64'; size: number; sha256: string; content: string };
export type NatlangPackageArchive = { schema: typeof ARCHIVE_SCHEMA; manifest: NatlangPackageManifest;
  files: PackageFile[]; digest: string };

export function canonicalJson(value: unknown): string {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError('canonical JSON cannot contain non-finite numbers');
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value as object).sort().map(key =>
    `${JSON.stringify(key)}:${canonicalJson((value as Record<string, unknown>)[key])}`).join(',')}}`;
  throw new TypeError('canonical JSON contains an unsupported value');
}
export const sha256 = (value: string | Uint8Array): string => createHash('sha256').update(value).digest('hex');

function filesBelow(root: string, candidate: string): string[] {
  const status = lstatSync(candidate);
  if (status.isSymbolicLink()) throw new TypeError(`package input cannot be a symbolic link: ${relative(root, candidate)}`);
  if (status.isFile()) return [candidate];
  if (!status.isDirectory()) throw new TypeError(`package input is not a regular file or directory: ${relative(root, candidate)}`);
  return readdirSync(candidate).sort().flatMap(name => filesBelow(root, join(candidate, name)));
}

export function createPackageArchive(manifestValue: unknown, rootDirectory: string): NatlangPackageArchive {
  const manifest = parsePackageManifest(manifestValue);
  const root = resolve(rootDirectory);
  const paths = new Set<string>();
  for (const include of manifest.include) {
    const absolute = resolve(root, include);
    if (absolute !== root && !absolute.startsWith(root + sep)) throw new TypeError(`include escapes package: ${include}`);
    for (const file of filesBelow(root, absolute)) paths.add(packagePath(relative(root, file).split(sep).join('/'), 'file path'));
  }
  for (const [name, target] of Object.entries(manifest.targets ?? {})) {
    for (const path of [target.entry, target.reducer, target.view].filter(Boolean) as string[])
      if (!paths.has(path)) throw new TypeError(`target ${name} references a file outside include: ${path}`);
  }
  for (const [name, path] of Object.entries(manifest.exports ?? {}))
    if (!paths.has(path)) throw new TypeError(`export ${name} references a file outside include: ${path}`);
  const files = [...paths].sort().map(path => {
    const bytes = readFileSync(join(root, ...path.split('/')));
    return { path, encoding: 'base64' as const, size: bytes.byteLength, sha256: sha256(bytes), content: bytes.toString('base64') };
  });
  const digest = sha256(canonicalJson({ schema: ARCHIVE_SCHEMA, manifest, files }));
  return { schema: ARCHIVE_SCHEMA, manifest, files, digest };
}

export function parsePackageArchive(value: unknown): NatlangPackageArchive {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('package archive must be an object');
  const raw = value as Record<string, unknown>;
  if (raw.schema !== ARCHIVE_SCHEMA || typeof raw.digest !== 'string' || !/^[0-9a-f]{64}$/.test(raw.digest))
    throw new TypeError('invalid package archive header');
  const manifest = parsePackageManifest(raw.manifest);
  if (!Array.isArray(raw.files)) throw new TypeError('package files must be an array');
  const seen = new Set<string>();
  const files = raw.files.map((item, index): PackageFile => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) throw new TypeError(`invalid package file ${index}`);
    const file = item as Record<string, unknown>, path = packagePath(file.path, `file ${index} path`);
    if (seen.has(path)) throw new TypeError(`duplicate package file: ${path}`); seen.add(path);
    if (file.encoding !== 'base64' || typeof file.content !== 'string' || typeof file.sha256 !== 'string' ||
        typeof file.size !== 'number' || !Number.isSafeInteger(file.size) || file.size < 0) throw new TypeError(`invalid package file: ${path}`);
    const bytes = Buffer.from(file.content, 'base64');
    if (bytes.byteLength !== file.size || sha256(bytes) !== file.sha256) throw new TypeError(`package file checksum mismatch: ${path}`);
    return { path, encoding: 'base64', content: file.content, sha256: file.sha256, size: file.size };
  });
  const sorted = [...files].sort((a, b) => a.path.localeCompare(b.path));
  if (files.some((file, index) => file.path !== sorted[index]!.path)) throw new TypeError('package files are not canonically ordered');
  const digest = sha256(canonicalJson({ schema: ARCHIVE_SCHEMA, manifest, files }));
  if (digest !== raw.digest) throw new TypeError('package archive digest mismatch');
  return { schema: ARCHIVE_SCHEMA, manifest, files, digest };
}

export function readPackageArchive(path: string): NatlangPackageArchive {
  return parsePackageArchive(JSON.parse(readFileSync(path, 'utf8')));
}

export function writePackageArchive(path: string, archive: NatlangPackageArchive): void {
  const checked = parsePackageArchive(archive);
  mkdirSync(dirname(resolve(path)), { recursive: true });
  // A trailing newline keeps archives pleasant to inspect without affecting their content digest.
  writeFileSync(path, canonicalJson(checked) + '\n');
}
