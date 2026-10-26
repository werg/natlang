/** Exact native resource surfaces, registered by their constructors. No proxy discovery. */
type MethodKind = 'promise' | 'synchronous';
const resources = new WeakMap<object, { surface: object; kind: string; cleanup?: () => void }>();
const originals = new WeakMap<object, object>();

export function declareSelectedSurface(value: object, kind: string,
    methods: Record<string, MethodKind>, properties: readonly string[] = [], cleanup?: () => void,
    writableProperties: readonly string[] = []): void {
  const surface = Object.create(null);
  for (const [name, callKind] of Object.entries(methods)) {
    const invoke = (...args: unknown[]) => Reflect.apply(Reflect.get(value, name), value, args);
    Object.defineProperty(invoke, Symbol.for('natlang.native-call-kind'), { value: callKind });
    Object.defineProperty(surface, name, { value: invoke, enumerable: true });
  }
  for (const name of properties) Object.defineProperty(surface, name,
    { get: () => Reflect.get(value, name), ...(writableProperties.includes(name) ?
      { set: (next: unknown) => { Reflect.set(value, name, next); } } : {}), enumerable: true });
  Object.freeze(surface); resources.set(value, { surface, kind, cleanup }); originals.set(surface, value);
}

export function selectedNativeSurface(value: unknown): unknown {
  if (!value || (typeof value !== 'object' && typeof value !== 'function')) return value;
  // ArrayBuffer.isView does not run Proxy traps. Only the exact native byte view
  // gets this surface; class instances and arbitrary objects are never guessed.
  if (!resources.has(value) && ArrayBuffer.isView(value) && Object.getPrototypeOf(value) === Uint8Array.prototype) {
    const bytes = value as Uint8Array;
    declareSelectedSurface(bytes, 'Uint8Array', { at: 'synchronous', slice: 'synchronous', subarray: 'synchronous',
      set: 'synchronous', fill: 'synchronous' }, ['length', 'byteLength']);
  }
  return resources.get(value)?.surface ?? value;
}
export function selectedNativeIdentity(value: unknown): object | null {
  return value && (typeof value === 'object' || typeof value === 'function') ? originals.get(value) ??
    (resources.has(value) ? value : null) : null;
}
export function selectedNativeCleanup(value: object): (() => void) | null { return resources.get(value)?.cleanup ?? null; }
