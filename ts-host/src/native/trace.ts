import { createHash } from 'node:crypto';

export const TRACE_VERSION = 'reduction-trace/1';
export type TraceEvent = { version: typeof TRACE_VERSION; seq: number; kind: string; [key: string]: unknown };

export function deriveSeed(root: number, path: string, attempt: number, purpose: string, ordinal = 0): number {
  const payload = { attempt, ordinal, path, purpose, root, version: 'sha256-json-v1' };
  const digest = createHash('sha256').update(JSON.stringify(payload)).digest();
  return Number(digest.readBigUInt64BE(0) % (2n ** 31n));
}

function valueType(value: unknown): string {
  if (value === null) return 'Null';
  if (typeof value === 'boolean') return 'Bool';
  if (typeof value === 'number') return 'Num';
  if (typeof value === 'string') return 'Text';
  if (Array.isArray(value)) return 'List';
  return 'Record';
}

export function changes(before: unknown, after: unknown, path: (string | number)[] = []): Record<string, unknown>[] {
  if (before && after && typeof before === 'object' && typeof after === 'object' &&
      !Array.isArray(before) && !Array.isArray(after)) {
    const a = before as Record<string, unknown>, b = after as Record<string, unknown>;
    return [...new Set([...Object.keys(a), ...Object.keys(b)])].sort().flatMap(key => {
      if (!(key in a) || !(key in b)) return [{ path: [...path, key], before_present: key in a,
        after_present: key in b, before: a[key] ?? null, after: b[key] ?? null,
        type: key in b ? valueType(b[key]) : null }];
      return changes(a[key], b[key], [...path, key]);
    });
  }
  if (JSON.stringify(before) === JSON.stringify(after)) return [];
  return [{ path, before_present: true, after_present: true, before, after, type: valueType(after) }];
}

export class NativeTraceRecorder {
  readonly events: TraceEvent[] = [];
  constructor(manifest: Record<string, unknown>) { this.emit('manifest', manifest); }
  emit(kind: string, data: Record<string, unknown> = {}): TraceEvent {
    const event = JSON.parse(JSON.stringify({ version: TRACE_VERSION, seq: this.events.length, kind, ...data })) as TraceEvent;
    this.events.push(event);
    return event;
  }
  finalState(): unknown {
    const states = this.events.filter(event => event.kind === 'state');
    if (!states.length) throw new Error('trace has no state');
    return states.at(-1)!.value;
  }
  reconstruct(): unknown {
    const first = this.events.find(event => event.kind === 'state' && event.phase === 'initial');
    if (!first) throw new Error('trace has no initial state');
    let current = structuredClone(first.value);
    for (const event of this.events.filter(event => event.kind === 'reduction')) {
      for (const change of event.changes as Record<string, unknown>[]) {
        const path = change.path as (string | number)[];
        if (!path.length) { current = structuredClone(change.after); continue; }
        let target = current as Record<string, unknown>;
        for (const part of path.slice(0, -1)) target = target[String(part)] as Record<string, unknown>;
        if (change.after_present) target[String(path.at(-1))] = structuredClone(change.after);
        else delete target[String(path.at(-1))];
      }
    }
    if (JSON.stringify(current) !== JSON.stringify(this.finalState())) throw new Error('trace reconstruction mismatch');
    return current;
  }
}
