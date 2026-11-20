/** Canonical POSIX coordinates for compiler and in-memory runtime sources. */
export function normalizeSourcePath(path: string): string {
  const absolute = path.startsWith('/');
  const segments: string[] = [];
  for (const segment of path.split('/')) {
    if (!segment || segment === '.') continue;
    if (segment === '..') segments.pop();
    else segments.push(segment);
  }
  return (absolute ? '/' : '') + segments.join('/');
}

export function joinSourcePath(...parts: string[]): string {
  return normalizeSourcePath(parts.filter(Boolean).join('/'));
}
