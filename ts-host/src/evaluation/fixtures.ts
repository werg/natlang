import { cloneData, immutable } from '../adaptation/identity.js';
/** Explicit full-payload capture, independent of truncated interpreter previews. */
export class FixtureObservationStore {
  private readonly values: Record<string, unknown> = Object.create(null);
  put(key: string, value: unknown): void {
    if (!key || Object.hasOwn(this.values, key)) throw new Error('observation needs a unique nonempty key: ' + key);
    this.values[key] = immutable(cloneData(value));
  }
  get(key: string): unknown { return this.values[key]; }
  snapshot(): Readonly<Record<string, unknown>> { return immutable(cloneData(this.values)); }
}
