import { hexDigest } from '../native/hash.js';
import type { EffectRecord } from './types.js';

/** Exact per-call binding for child inventory and observed task-wide service requests. */
export function graphCustodyDigest(taskId: string, callId: string, parentCallId: string | null,
  children: readonly string[], effects: readonly EffectRecord[]): string {
  const value = { schema:'natlang.call-graph-chronology/1', task_id:taskId, call_id:callId,
    parent_call_id:parentCallId, child_call_ids:children, effects };
  const text=JSON.stringify(value, (_key,item) => item && typeof item === 'object' && !Array.isArray(item) ?
    Object.fromEntries(Object.entries(item).sort(([a],[b]) => a.localeCompare(b))) : item);
  return hexDigest(text);
}
