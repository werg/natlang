import type { NativeTraceRecorder } from './trace.js';

export type NativeScenarioContract = { outcome: string; value?: unknown;
  effects?: [string, unknown[]][];
  requiredActions?: { name: string; arguments?: Record<string, unknown> }[];
  constrainedCalls?: { function: string; to: string; inputs: Record<string, string> }[] };

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value)
    .sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => [key, canonical(child)]));
  return value;
}
const same = (a: unknown, b: unknown) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));

/** Admit recorded observations without rerunning a model, eval, or host effect. */
export function admitNativeTrace(trace: NativeTraceRecorder, contract: NativeScenarioContract): Record<string, unknown> {
  const replay = trace.replayObservations();
  if (replay.outcome !== contract.outcome)
    throw new Error(`outcome changed: ${replay.outcome} != ${contract.outcome}`);
  if (contract.outcome === 'done' && !same(replay.final, contract.value))
    throw new Error('final captured value does not match the contract');
  const effects = replay.effects as Record<string, unknown>[];
  const requested = effects.filter(event => event.phase === 'requested');
  if (contract.effects) {
    const observed = requested.map(event => [event.capability, event.args]);
    if (!same(observed, contract.effects)) throw new Error('ordered effect sequence does not match the contract');
    const completed = new Set(effects.filter(event => event.phase === 'completed')
      .map(event => JSON.stringify([event.call_id ?? null, event.capability, event.sequence])));
    if (requested.some(event => !completed.has(JSON.stringify([event.call_id ?? null, event.capability, event.sequence]))))
      throw new Error('required effect was requested but did not complete');
  }
  const actions = replay.actions as Record<string, unknown>[];
  for (const rule of contract.constrainedCalls ?? []) for (const action of actions) {
    const args = action.arguments as Record<string, unknown> ?? {};
    if (action.name === 'call' && args.function === rule.function &&
        (args.to !== rule.to || !same(args.inputs ?? {}, rule.inputs)))
      throw new Error('required call destination or inputs changed');
  }
  let cursor = 0;
  for (const required of contract.requiredActions ?? []) {
    while (cursor < actions.length && (actions[cursor]!.name !== required.name ||
      Object.entries(required.arguments ?? {}).some(([key, value]) =>
        !same((actions[cursor]!.arguments as Record<string, unknown> ?? {})[key], value)))) cursor++;
    if (cursor === actions.length) throw new Error(`required ordered action absent: ${JSON.stringify(required)}`);
    cursor++;
  }
  return { admitted: true, trace_version: trace.events[0]!.version,
    source_sha256: trace.events[0]!.source_sha256 ?? trace.events[0]!.source_revision ?? null,
    run_id: trace.events[0]!.run_id, coverage: replay.coverage };
}
