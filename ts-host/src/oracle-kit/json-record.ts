/** Generic canonical JSON helpers shared by the answer comparators (and by benchmark-specific comparators). */
export function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as Record<string, unknown>)
    .sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(',')}}`;
  return JSON.stringify(value);
}
/** Formatting is irrelevant for JSON string records; duplicate keys and other shapes are invalid. */
export function jsonStringRecordCanonical(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const string = String.raw`"(?:[^"\\\u0000-\u001f]|\\(?:["\\/bfnrt]|u[\da-fA-F]{4}))*"`;
  const pair = `${string}\\s*:\\s*${string}`;
  if (!new RegExp(`^\\s*\\{\\s*(?:${pair}(?:\\s*,\\s*${pair})*)?\\s*\\}\\s*$`).test(value)) return null;
  try {
    const names = new Set<string>();
    for (const match of value.matchAll(new RegExp(`(${string})\\s*:\\s*${string}`, 'g'))) {
      const key = JSON.parse(match[1]!) as string;
      if (names.has(key)) return null;
      names.add(key);
    }
    return canonical(JSON.parse(value));
  } catch { return null; }
}
