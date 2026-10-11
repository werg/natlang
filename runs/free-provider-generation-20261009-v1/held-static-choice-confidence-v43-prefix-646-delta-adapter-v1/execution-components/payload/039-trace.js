import { digest } from './hash.js';
export const TRACE_VERSION = 'reduction-trace/1';
export function deriveSeed(root, path, attempt, purpose, ordinal = 0) {
    const payload = { attempt, ordinal, path, purpose, root, version: 'sha256-json-v1' };
    const bytes = digest(JSON.stringify(payload));
    let first = 0n;
    for (const byte of bytes.slice(0, 8))
        first = (first << 8n) | BigInt(byte);
    return Number(first % (2n ** 31n));
}
function valueType(value) {
    if (value === null)
        return 'null';
    if (typeof value === 'boolean')
        return 'boolean';
    if (typeof value === 'number')
        return 'number';
    if (typeof value === 'string')
        return 'string';
    if (Array.isArray(value))
        return 'List';
    if (value && typeof value === 'object')
        for (const [key, body] of Object.entries(value)) {
            if (key.startsWith('$') && body && typeof body === 'object' && 'type' in body)
                return String(body.type);
        }
    return 'Record';
}
function canonical(value) {
    if (Array.isArray(value))
        return value.map(canonical);
    if (value && typeof value === 'object')
        return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
            .map(([key, item]) => [key, canonical(item)]));
    return value;
}
const sameValue = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
export function changes(before, after, path = []) {
    if (before && after && typeof before === 'object' && typeof after === 'object' &&
        !Array.isArray(before) && !Array.isArray(after)) {
        const a = before, b = after;
        return [...new Set([...Object.keys(a), ...Object.keys(b)])].sort().flatMap(key => {
            if (!(key in a) || !(key in b))
                return [{ path: [...path, key], before_present: key in a,
                        after_present: key in b, before: a[key] ?? null, after: b[key] ?? null,
                        type: key in b ? valueType(b[key]) : null }];
            return changes(a[key], b[key], [...path, key]);
        });
    }
    if (sameValue(before, after))
        return [];
    return [{ path, before_present: true, after_present: true, before, after, type: valueType(after) }];
}
export class NativeTraceRecorder {
    events = [];
    started = performance.now();
    constructor(manifest) { this.emit('manifest', manifest); }
    /** Open a captured trace for read-only reconstruction and admission checks. */
    static fromEvents(events) {
        const trace = new NativeTraceRecorder({});
        trace.events.splice(0, trace.events.length, ...structuredClone(events));
        trace.reconstruct();
        return trace;
    }
    emit(kind, data = {}) {
        const event = JSON.parse(JSON.stringify({ version: TRACE_VERSION, seq: this.events.length, kind,
            observed_at: new Date().toISOString(), elapsed_ms: Math.round(performance.now() - this.started),
            ...data }));
        this.events.push(event);
        return event;
    }
    finalState() {
        const states = this.events.filter(event => event.kind === 'state');
        if (!states.length)
            throw new Error('trace has no state');
        return states.at(-1).value;
    }
    coverage() {
        const states = this.events.filter(event => event.kind === 'state');
        const incomplete = states.some(event => /"\$stream"|"\$opaque"/.test(JSON.stringify(event.value)));
        return { state_reconstructable: states.length > 0, live_source_reconstructable: !incomplete,
            native_effects_replayable: false, effect_count: this.events.filter(event => event.kind === 'effect').length,
            mode: 'recorded-observations-only' };
    }
    replayObservations() {
        const states = this.events.filter(event => event.kind === 'state');
        if (states[0]?.phase !== 'initial' || states.at(-1)?.phase !== 'final')
            throw new Error('trace lacks complete initial/final observations');
        return { initial: states[0].value, final: states.at(-1).value, outcome: states.at(-1).outcome,
            actions: this.events.filter(event => event.kind === 'action'),
            effects: this.events.filter(event => event.kind === 'effect'), coverage: this.coverage() };
    }
    reconstruct() {
        if (this.events[0]?.kind !== 'manifest' || this.events.some((event, index) => event.version !== TRACE_VERSION || event.seq !== index))
            throw new Error('unsupported or discontinuous trace');
        const first = this.events.find(event => event.kind === 'state' && event.phase === 'initial');
        if (!first)
            throw new Error('trace has no initial state');
        let current = structuredClone(first.value);
        for (const event of this.events.filter(event => event.kind === 'reduction')) {
            for (const change of event.changes) {
                const path = change.path;
                if (!path.length) {
                    current = structuredClone(change.after);
                    continue;
                }
                let target = current;
                for (const part of path.slice(0, -1))
                    target = target[String(part)];
                if (change.after_present)
                    target[String(path.at(-1))] = structuredClone(change.after);
                else
                    delete target[String(path.at(-1))];
            }
        }
        if (!sameValue(current, this.finalState()))
            throw new Error('trace reconstruction mismatch');
        return current;
    }
}
