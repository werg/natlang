/** A value-only TypeScript view of an undeclared host service; never evaluate accessors or show field values. */
export function undeclaredServiceType(value: unknown, depth = 0): string {
  if (typeof value === 'function') return '(...args: unknown[]) => unknown';
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'readonly unknown[]';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ||
      typeof value === 'bigint' || typeof value === 'symbol' || typeof value === 'undefined') return typeof value;
  if (typeof value !== 'object' || depth >= 2) return 'unknown';

  // Read descriptors rather than properties so getters are not run while rendering an opening or an inspection.
  // If an exotic service refuses descriptor inspection, expose a read-only unknown record and let its declaration
  // provide detail.
  let descriptors: PropertyDescriptorMap;
  try { descriptors = Object.getOwnPropertyDescriptors(value); }
  catch { return 'Readonly<Record<string, unknown>>'; }
  const entries = Object.entries(descriptors).filter(([, descriptor]) => descriptor.enumerable);
  if (!entries.length || entries.length > 64) return 'Readonly<Record<string, unknown>>';
  const fields = entries.map(([key, descriptor]) => {
    const type = Object.hasOwn(descriptor, 'value') ? undeclaredServiceType(descriptor.value, depth + 1) : 'unknown';
    return `readonly ${JSON.stringify(key)}: ${type}`;
  });
  return `Readonly<{ ${fields.join('; ')} }>`;
}
