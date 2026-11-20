import { hexDigest } from '../native/hash.js';
import type { CallRecord, EffectRecord } from './types.js';

/** Exact per-call binding for child inventory and observed task-wide service requests. */
export function graphCustodyDigest(taskId: string, callId: string, parentCallId: string | null,
  children: readonly string[], effects: readonly EffectRecord[]): string {
  const value = { schema:'natlang.call-graph-chronology/1', task_id:taskId, call_id:callId,
    parent_call_id:parentCallId, child_call_ids:children, effects };
  const text=JSON.stringify(value, (_key,item) => item && typeof item === 'object' && !Array.isArray(item) ?
    Object.fromEntries(Object.entries(item).sort(([a],[b]) => a.localeCompare(b))) : item);
  return hexDigest(text);
}

export function graphCustodyMatches(
  call: Pick<CallRecord, 'task_id' | 'call_id' | 'parent_call_id' | 'effects' | 'graph_custody'>,
  children: readonly string[],
): boolean {
  const custody = call.graph_custody;
  if (!custody || custody.schema !== 'natlang.call-graph-chronology/1' || custody.task_id !== call.task_id ||
      !Array.isArray(custody.child_call_ids) || custody.child_call_ids.length !== children.length)
    return false;
  const recorded = new Set(custody.child_call_ids);
  return recorded.size === children.length && new Set(children).size === children.length &&
    children.every(id => recorded.has(id)) &&
    custody.sha256 === graphCustodyDigest(call.task_id, call.call_id, call.parent_call_id, custody.child_call_ids, call.effects);
}
