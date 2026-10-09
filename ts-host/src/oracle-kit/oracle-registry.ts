/**
 * Registry of benchmark-specific record comparators. The core oracle module knows only generic comparators;
 * a benchmark package (src/benchmarks/<name>/) registers additional ones by name, and `checkOracle`,
 * `checkFiles` and `checkFileReturn` look them up here. This module imports nothing, so benchmark packages
 * may depend on evaluation/ without creating a cycle.
 */
export type RecordComparator = {
  /** Workspace file holding the answer record for `compare` (files oracles). */
  answerFile: string;
  /** True when the actual answer equals the expected one under this comparator. */
  equal(actual: unknown, expected: unknown): boolean;
};

const comparators = new Map<string, RecordComparator>();

export function registerRecordComparator(name: string, comparator: RecordComparator): void {
  const existing = comparators.get(name);
  if (existing && existing !== comparator) throw new Error(`record comparator already registered: ${name}`);
  comparators.set(name, comparator);
}
export function recordComparator(name: unknown): RecordComparator | undefined {
  return typeof name === 'string' ? comparators.get(name) : undefined;
}
export function registeredRecordComparators(): string[] { return [...comparators.keys()].sort(); }

/** Names a spec may use without a registered comparator. Anything else must be registered, or the spec is rejected. */
export function assertKnownComparator(kind: 'compare' | 'normalization', name: unknown, builtin: readonly string[]): void {
  if (name === undefined || (typeof name === 'string' && (builtin.includes(name) || comparators.has(name)))) return;
  throw new RangeError(`unknown ${kind} "${String(name)}": not built in and no benchmark registered it ` +
    `(import benchmarks/builtin.js, or register it with registerRecordComparator)`);
}
