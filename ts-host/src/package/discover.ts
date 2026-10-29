/** Applications on disk: directories holding a natlang.package/v2 natlang.json manifest (`natlang apps`, the dashboard). */
import { existsSync, lstatSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';

export type LocalApplication = { path: string; name: string; version: string; targets: string[]; description: string };

export function discoverLocalApplications(value = '.'): LocalApplication[] {
  const root = resolve(value);
  if (!existsSync(root)) throw new Error(`application search path does not exist: ${value}`);
  const manifests: string[] = [], ignored = new Set(['.git', '.natlang', 'node_modules', '.venv', 'dist']);
  const visit = (path: string, explicitRoot = false) => {
    const status = explicitRoot ? statSync(path) : lstatSync(path);
    if (status.isSymbolicLink()) return;
    if (status.isFile()) { if ((basename(path) === 'natlang.json' || path.endsWith('.natlang.json')) && isApplicationManifest(path)) manifests.push(path); return; }
    if (!status.isDirectory() || ignored.has(basename(path))) return;
    for (const name of readdirSync(path).sort()) visit(join(path, name));
  };
  visit(root, true);
  return manifests.map(path => {
    const manifest = JSON.parse(readFileSync(path, 'utf8')) as { name: string; version: string;
      description?: string; targets?: Record<string, unknown> };
    return { path, name: manifest.name, version: manifest.version,
      targets: Object.keys(manifest.targets ?? {}), description: manifest.description ?? '' };
  }).sort((left, right) => left.path.localeCompare(right.path));
}

export function isApplicationManifest(path: string): boolean {
  if (!existsSync(path) || !lstatSync(path).isFile()) return false;
  try {
    const value = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>;
    return value.schema === 'natlang.package/v2' && Boolean(value.targets && typeof value.targets === 'object');
  } catch { return false; }
}
