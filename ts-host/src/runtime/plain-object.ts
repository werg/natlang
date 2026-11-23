/**
 * A plain object (`{...}` or `Object.create(null)`) from any realm: callable-folder modules, the eval sandbox, workers
 * and browser compartments each bring their own `Object.prototype`, so an identity test against this realm's misreads
 * their records as opaque values. Arrays, class instances, dates and other built-ins are not plain.
 */
export function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  if (Object.prototype.toString.call(value) !== '[object Object]') return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === null || Object.getPrototypeOf(prototype) === null;
}
