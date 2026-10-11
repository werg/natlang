function canonical(value) {
    if (Array.isArray(value))
        return value.map(canonical);
    if (value && typeof value === 'object')
        return Object.fromEntries(Object.entries(value)
            .sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => [key, canonical(child)]));
    return value;
}
const same = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
/** Admit recorded observations without rerunning a model, eval, or host effect. */
export function admitNativeTrace(trace, contract) {
    const replay = trace.replayObservations();
    if (replay.outcome !== contract.outcome)
        throw new Error(`outcome changed: ${replay.outcome} != ${contract.outcome}`);
    // The final state is the invocation's lambda; its captured value is the typed return.
    const final = replay.final?.$lambda?.return ?? null;
    if (contract.outcome === 'done' && !same(final, contract.value))
        throw new Error('final captured value does not match the contract');
    const effects = replay.effects;
    const requested = effects.filter(event => event.phase === 'requested');
    if (contract.effects) {
        const observed = requested.map(event => [event.capability, event.args]);
        if (!same(observed, contract.effects))
            throw new Error('ordered effect sequence does not match the contract');
        const completed = new Set(effects.filter(event => event.phase === 'completed')
            .map(event => JSON.stringify([event.call_id ?? null, event.capability, event.sequence])));
        if (requested.some(event => !completed.has(JSON.stringify([event.call_id ?? null, event.capability, event.sequence]))))
            throw new Error('required effect was requested but did not complete');
    }
    const actions = replay.actions;
    let cursor = 0;
    for (const required of contract.requiredActions ?? []) {
        while (cursor < actions.length && (actions[cursor].name !== required.name ||
            Object.entries(required.arguments ?? {}).some(([key, value]) => !same((actions[cursor].arguments ?? {})[key], value))))
            cursor++;
        if (cursor === actions.length)
            throw new Error(`required ordered action absent: ${JSON.stringify(required)}`);
        cursor++;
    }
    return { admitted: true, trace_version: trace.events[0].version,
        source_sha256: trace.events[0].source_sha256 ?? trace.events[0].source_revision ?? null,
        run_id: trace.events[0].run_id, coverage: replay.coverage };
}
