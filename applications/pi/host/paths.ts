/**
 * POSIX path arithmetic for the platform-neutral core, which runs where `node:path` does not (the browser host's
 * virtual folder, host/folder-env.ts). Paths are `/`-separated; the Node host's own paths are POSIX too (Linux).
 */

/** `path` with `.` and `..` segments resolved and repeated slashes collapsed; `..` never climbs above `/`. */
export function normalizePath(path: string): string {
  const absolute = path.startsWith('/');
  const parts: string[] = [];
  for (const part of path.split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') { if (parts.length && parts.at(-1) !== '..') parts.pop(); else if (!absolute) parts.push('..'); }
    else parts.push(part);
  }
  const joined = parts.join('/');
  return absolute ? `/${joined}` : joined || '.';
}

export const isAbsolutePath = (path: string) => path.startsWith('/');

/** The parts joined and normalized, as `path.join`. */
export const joinPath = (...parts: string[]) => normalizePath(parts.filter(Boolean).join('/'));

/** `path` made absolute against `base` and normalized, as `path.resolve(base, path)`. */
export const resolvePath = (base: string, path: string) => normalizePath(isAbsolutePath(path) ? path : `${base}/${path}`);

/** The path from `from` to `to` (both absolute), as `path.relative`. */
export function relativePath(from: string, to: string): string {
  const a = normalizePath(from).split('/').filter(Boolean), b = normalizePath(to).split('/').filter(Boolean);
  let shared = 0;
  while (shared < a.length && shared < b.length && a[shared] === b[shared]) shared++;
  return [...a.slice(shared).map(() => '..'), ...b.slice(shared)].join('/');
}

/** The last segment of `path`. */
export const basenamePath = (path: string) => normalizePath(path).split('/').at(-1) ?? '';

/** `path` without its last segment (`/` for a child of the root). */
export function dirnamePath(path: string): string {
  const normalized = normalizePath(path), at = normalized.lastIndexOf('/');
  return at > 0 ? normalized.slice(0, at) : at === 0 ? '/' : '.';
}
