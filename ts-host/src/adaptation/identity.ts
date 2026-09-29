import { hexDigest } from '../native/hash.js';
/** Hash domains are versioned; object order is irrelevant, array order remains significant. */
export function canonical(value: unknown): string {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return '[' + Array.from(value, canonical).join(',') + ']';
  if (value && typeof value === 'object' && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null))
    return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + canonical((value as Record<string, unknown>)[key])).join(',') + '}';
  throw new Error('adaptation data must contain only finite JSON values');
}
export function fingerprint(value: unknown, domain = 'natlang.adaptation/v1'): string { return hexDigest(domain + '\0' + canonical(value)); }
export function componentKey(program: string, path: string, site = 'instructions'): string {
  return [program, path.replace(/\\/g, '/'), site].map(part => encodeURIComponent(part).replace(/%2F/g, '/')).join('::');
}
export function immutable<T>(value: T): T {
  if (value && typeof value === 'object') { Object.values(value).forEach(immutable); Object.freeze(value); }
  return value;
}
export function cloneData<T>(value: T): T { return JSON.parse(canonical(value)) as T; }
