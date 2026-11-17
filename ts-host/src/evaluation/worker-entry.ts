import { parentPort, workerData } from 'node:worker_threads';
import { createNatlangRuntime, NatlangCallError } from '../index.js';
import { bindAdaptation } from '../adaptation/compatibility.js';
import { validateMetrics } from './metrics.js';
import { cloneData } from '../adaptation/identity.js';
import { FixtureObservationStore } from './fixtures.js';
import type { EvaluationSuite, EvaluationResult } from './types.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { remoteDriver, type DriverCall } from '../runtime/driver.js';
import { currentFrame } from '../runtime/context.js';
import type { WorkerInput } from './worker.js';
const input = workerData as WorkerInput;
const abort = new AbortController();
let sequence = 0;
const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
parentPort!.on('message', message => {
  if (message.type === 'cancel') { abort.abort(new Error('evaluation cancelled')); for (const p of pending.values()) p.reject(abort.signal.reason); pending.clear(); }
  if (message.type === 'response') { const p = pending.get(message.id); pending.delete(message.id); if (message.error) p?.reject(new Error(message.error)); else p?.resolve(message.value); }
});
async function run(): Promise<void> {
  const suite = (await import(input.moduleURL)).default as EvaluationSuite;
  const traces: EvaluationResult['traces'][number][] = [];
  const modelTurns: NonNullable<EvaluationResult['modelTurns']>[number][] = [];
  const counters = new Map<string, number>();
  let captureError: Error | undefined;
  const requestModel = async (role: 'executor' | 'judge', call: DriverCall): Promise<unknown> => {
    if (input.capture && role === 'executor' && call.kind !== 'turn') {
      captureError = new Error('teacher capture needs generated executor turns; use a teacher driver without decision scoring');
      abort.abort(captureError);
      throw captureError;
    }
    const frame = currentFrame(), invocation = frame?.parentCallId ?? null;
    const capture = input.capture && role === 'executor';
    const request = capture ? structuredClone(call.request) as ModelTurnRequest : undefined;
    const turn = (counters.get(invocation ?? '') ?? 0) + 1;
    if (capture) counters.set(invocation ?? '', turn);
    const reply = await new Promise<unknown>((resolve, reject) => {
      const id = ++sequence; pending.set(id, { resolve, reject }); parentPort!.postMessage({ type: 'request', id, role, call });
    });
    if (capture) modelTurns.push({ invocation_id: invocation, parent_call_id: null, turn,
      request: request!, reply: structuredClone(reply) as ModelTurn });
    return reply;
  };
  const executorIdentity = input.executorIdentity ?? suite.executorIdentity;
  const runtime = createNatlangRuntime({ program: input.program, adaptation: input.artifact ? bindAdaptation(input.artifact, input.program, suite.executorIdentity) : null,
    limits: suite.policy?.limits, systemPrompt: suite.systemPrompt,
    evaluation: { id: input.id, caseId: input.testCase.id, replicate: input.replicate },
    executorIdentity, serviceDeclarations: suite.services?.declarations,
    serviceScopes: suite.services?.scopes as Record<string, string[]> | undefined,
    signal: abort.signal, isolateModules: true, codeEdits: suite.policy?.codeEdits ?? 'deny', trace: trace => traces.push(trace),
    seed: { mode: 'derived', root: input.seed }, model: { ...suite.policy?.settings, driver: remoteDriver(call => requestModel('executor', call), input.decisions === true, input.continuesReplies === true) } });
  let fixture: Awaited<ReturnType<EvaluationSuite['fixture']['create']>> | undefined;
  let outcome: EvaluationResult['outcome'] = { kind: 'returned' };
  let result: unknown; const start = Date.now();
  const observations = new FixtureObservationStore();
  try {
    fixture = await suite.fixture.create(input.testCase, { runtime, program: input.program, seed: input.seed, signal: abort.signal, observations,
      judge: request => requestModel('judge', { kind: 'turn', request }) as Promise<ModelTurn>,
      loadFreshProgram: () => import(input.compiledEntry) });
    if (input.preflight) {
      if (typeof fixture?.execute !== 'function') throw new Error('fixture must provide execute()');
      parentPort!.postMessage({ type: 'result', result: { id: input.id, caseId: input.testCase.id, split: input.testCase.split,
        status: 'scored', quality: 0, gates: {}, metrics: {}, feedback: '', outcome: { kind: 'returned' },
        coverage: [], latencyMs: Date.now() - start, traces } });
      return;
    }
    try { result = await fixture.execute(); outcome = { kind: 'returned', value: result }; }
    catch (error) {
      abort.signal.throwIfAborted();
      outcome = { kind: 'threw', error: { name: error instanceof Error ? error.name : 'Error',
        message: error instanceof Error ? error.message : String(error), ...(error instanceof NatlangCallError ? { outcome: error.outcome } : {}) } };
    }
    abort.signal.throwIfAborted();
    if (captureError) throw captureError;
    const observation = fixture.observe ? await fixture.observe(result, outcome) : { ...(result === undefined ? {} : { result }), outcome };
    const metric = validateMetrics(await suite.score(input.testCase, observation, outcome), suite.requiredGates);
    // The score is independent evidence about the calls that produced it (plans/TRACE_SPECIALIZATION.md §3.6).
    try {
      const store = runtime.callStore();
      for (const trace of traces.filter(item => !item.parentCallId))
        store?.annotate?.(trace.callId, 'evaluation', { evaluation: input.id, case: input.testCase.id, quality: metric.quality,
          gates: metric.gates ?? {}, feedback: metric.feedback ?? '' }, 'evaluation', false);
    } catch { /* recording never fails an evaluation */ }
    let captured: unknown;
    try { captured = cloneData(observation); }
    catch { throw new Error('fixture observation must be finite JSON; supply observe() to project live handles and captures'); }
    // Scoring has access to live values. Only its explicit observation crosses
    // the worker boundary; handles and closures are never serialized.
    let portableOutcome = outcome;
    try { portableOutcome = cloneData(outcome); }
    catch { portableOutcome = { kind: outcome.kind, ...(outcome.error ? { error: outcome.error } : {}) }; }
    const coverage = [...new Set(traces.flatMap(trace => [trace.adaptation?.component, trace.adaptation?.guidanceComponent]).filter((key): key is string => typeof key === 'string'))].sort();
    if (input.capture) for (const turn of modelTurns)
      turn.parent_call_id = traces.find(trace => trace.callId === turn.invocation_id)?.parentCallId ?? null;
    parentPort!.postMessage({ type: 'result', result: { id: input.id, caseId: input.testCase.id, split: input.testCase.split,
      status: 'scored', quality: metric.quality, gates: { ...(metric.gates ?? {}), ...(metric.judge ? { judge: metric.judge === 'accepted' } : {}) },
      metrics: metric.metrics ?? {}, feedback: metric.feedback ?? '', outcome: portableOutcome, observation: captured,
      observations: observations.snapshot(), coverage, latencyMs: Date.now() - start, traces,
      ...(input.capture ? { modelTurns } : {}) } });
  } finally { await fixture?.dispose?.(); runtime.close(); }
}
run().then(() => parentPort!.postMessage({ type: 'disposed' }), error => parentPort!.postMessage({ type: 'error', error: error instanceof Error ? error.message : String(error), cancelled: abort.signal.aborted }));
