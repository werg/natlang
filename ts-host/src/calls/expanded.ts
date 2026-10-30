import type { CallStore } from './store.js';
import type { CallRecord } from './types.js';

/** A record with its values in place of their references. */
export function expandedRecord(store: CallStore, record: CallRecord): Record<string, unknown> {
  const value = (ref: Parameters<CallStore['value']>[0]) => ref?.complete ? store.value(ref) : ref;
  return { ...record, inputs: Object.fromEntries(Object.entries(record.inputs).map(([name, ref]) => [name, value(ref)])),
    captures: Object.fromEntries(Object.entries(record.captures).map(([name, ref]) => [name, value(ref)])),
    output: value(record.output), effects: record.effects.map(effect => ({ ...effect, args: value(effect.args), result: value(effect.result) })),
    capture_writes: record.capture_writes.map(write => ({ ...write, after: value(write.after) })),
    definition: { ...record.definition, instructions: value(record.definition.instructions) } };
}
