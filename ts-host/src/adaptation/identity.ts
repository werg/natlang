import { hexDigest } from '../native/hash.js';
/** Hash domains are versioned; object order is irrelevant, array order remains significant. */
export { canonical } from '../native/canonical-data.js';
import { canonical } from '../native/canonical-data.js';
export function fingerprint(value: unknown, domain = 'natlang.adaptation/v1'): string { return hexDigest(domain + '\0' + canonical(value)); }
export function componentKey(program: string, path: string, site = 'instructions'): string {
  return [program, path.replace(/\\/g, '/'), site].map(part => encodeURIComponent(part).replace(/%2F/g, '/')).join('::');
}
export function immutable<T>(value: T): T {
  if (value && typeof value === 'object') { Object.values(value).forEach(immutable); Object.freeze(value); }
  return value;
}
export function cloneData<T>(value: T): T { return JSON.parse(canonical(value)) as T; }
