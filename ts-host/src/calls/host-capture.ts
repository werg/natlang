/**
 * Exact host capture as a filter over the call store (plans/TRACE_SPECIALIZATION.md §4): the teacher collector's
 * `host_capture` records (exact portable inputs and outputs of a call, bounded, explicitly incomplete otherwise),
 * rebuilt from a stored call instead of a separate recording path. Field for field what the native runtime emits inline
 * (native/runtime.ts captureExactValue), without the trace envelope (seq, observed_at, ...).
 */
import type { CallRecord, ValueRef } from './types.js';

export type HostCapture = { capture_kind: 'invocation_input' | 'invocation_output'; call_id: string; parent_call_id: string | null;
  definition_source: string | null; name: string; complete: boolean; value?: unknown; bytes?: number; reason?: string;
  value_sha256?: string; result_type?: string | null; terminal_action_seq?: number | null; origin?: string };

/** What a store must answer for the view. */
export interface CaptureSource { call(callId: string): CallRecord | undefined; value(ref: ValueRef | null | undefined): unknown;
  events?(callId: string): Record<string, unknown>[] | undefined }

const OUTPUT_ORIGIN = 'observed-host-result; not a model-generated writer target';

/**
 * The exact captures of a stored call: the named inputs (`inputArguments`; a name the call did not get is recorded as
 * missing) and, when `output` is set and the call succeeded, its result. `maxBytes` bounds a value as the inline
 * capture's budget does; the store's own bound (`maxValueBytes`) applies first.
 */
export function hostCaptures(store: CaptureSource, callId: string, spec: { inputArguments?: readonly string[]; output?: boolean;
  maxBytes?: number } = {}): HostCapture[] | undefined {
  const record = store.call(callId);
  if (!record) return undefined;
  const base = { call_id: record.call_id, parent_call_id: record.parent_call_id, definition_source: record.definition.source };
  const exact = (ref: ValueRef | null | undefined): Pick<HostCapture, 'complete' | 'value' | 'bytes' | 'reason' | 'value_sha256'> => {
    if (!ref) return { complete: false, reason: 'missing' };
    if (!ref.complete) return { complete: false, reason: ref.reason === 'oversize' ? 'oversize' : 'nonportable' };
    if (spec.maxBytes !== undefined && ref.bytes > spec.maxBytes) return { complete: false, reason: 'oversize' };
    const value = store.value(ref);
    if (value === undefined) return { complete: false, reason: 'nonportable' };
    return { complete: true, value, bytes: ref.bytes, value_sha256: ref.hash };
  };
  const captures: HostCapture[] = (spec.inputArguments ?? []).map(name => ({ capture_kind: 'invocation_input', ...base, name,
    ...exact(record.inputs[name]) }));
  if (spec.output && record.outcome === 'done') {
    const actions = store.events?.(callId)?.filter(event => event.kind === 'action');
    captures.push({ capture_kind: 'invocation_output', ...base, name: 'return', ...exact(record.output),
      result_type: record.definition.returns ?? null, terminal_action_seq: (actions?.at(-1)?.seq as number | undefined) ?? null,
      origin: OUTPUT_ORIGIN });
  }
  return captures;
}
