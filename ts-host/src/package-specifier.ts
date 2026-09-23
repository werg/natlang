/** Root package name for an npm dependency import, including exported subpaths. */
export function packageNameFromSpecifier(specifier: string): string {
  if (typeof specifier !== 'string' || !specifier || specifier.startsWith('.') ||
      specifier.startsWith('/') || specifier.startsWith('#') || specifier.includes('\\') ||
      specifier.includes(':') || specifier.includes('?'))
    throw new Error(`Only declared package imports are supported: ${JSON.stringify(specifier)}`);
  const parts = specifier.split('/');
  const name = specifier.startsWith('@') ? parts.slice(0, 2).join('/') : parts[0]!;
  if (!/^(@[A-Za-z0-9._-]+\/)?[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name) ||
      parts.some(part => part === '' || part === '.' || part === '..'))
    throw new Error(`Invalid package import: ${JSON.stringify(specifier)}`);
  return name;
}
