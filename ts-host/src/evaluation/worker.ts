import { Worker } from 'node:worker_threads';
import type { AdaptationArtifact, ProgramDescriptor } from '../adaptation/types.js';
import type { ModelDriver } from '../runtime/runtime.js';
import type { EvaluationCase, EvaluationResult } from './types.js';
import { UsageGateway, type ModelRole } from './usage.js';
export type WorkerInput = { moduleURL: string; compiledEntry: string; testCase: EvaluationCase; program: ProgramDescriptor;
  artifact: AdaptationArtifact | null; seed: number; replicate: number; id: string; preflight?: boolean };
export function runWorker(input: WorkerInput, driver: ModelDriver, gateway: UsageGateway,
  signal?: AbortSignal, timeoutMs = 120000, judge?: ModelDriver): Promise<EvaluationResult> {
  return new Promise((resolve, reject) => {
    const lifetime = new AbortController();
    const requestSignal = signal ? AbortSignal.any([signal, lifetime.signal]) : lifetime.signal;
    const worker = new Worker(new URL('./worker-entry.js', import.meta.url), { workerData: input });
    let result: EvaluationResult | undefined, finished = false, transportError: Error | undefined;
    const usage = { modelCalls: 0, inputTokens: 0 as number | null, outputTokens: 0 as number | null, cost: gateway.limits.pricing ? 0 as number | null : null };
    const fail = (error: Error) => { if (finished) return; finished = true; lifetime.abort(error); cleanup(); void worker.terminate(); reject(error); };
    const onAbort = () => { worker.postMessage({ type: 'cancel' }); fail(signal?.reason instanceof Error ? signal.reason : new Error('evaluation cancelled')); };
    const timer = setTimeout(() => fail(new Error('evaluation worker timed out')), timeoutMs);
    const cleanup = () => { clearTimeout(timer); signal?.removeEventListener('abort', onAbort); };
    signal?.addEventListener('abort', onAbort, { once: true }); if (signal?.aborted) { onAbort(); return; }
    worker.on('message', async message => {
      if (message.type === 'request') {
        try {
          const role: ModelRole = message.role === 'judge' ? 'judge' : 'executor';
          const selectedDriver = role === 'judge' ? judge : driver;
          if (!selectedDriver) throw new Error('fixture requested a judge but no judge driver is configured');
          const before = gateway.ledger.usage.modelCalls;
          const response = gateway.request(selectedDriver, message.request, requestSignal, role);
          if (gateway.ledger.usage.modelCalls > before) usage.modelCalls++;
          const turn = await response;
          if (turn.prompt_tokens === undefined) usage.inputTokens = null; else if (usage.inputTokens !== null) usage.inputTokens += turn.prompt_tokens;
          if (turn.completion_tokens === undefined) usage.outputTokens = null; else if (usage.outputTokens !== null) usage.outputTokens += turn.completion_tokens;
          const pricing = gateway.limits.pricing;
          if (!pricing || turn.prompt_tokens === undefined || turn.completion_tokens === undefined) usage.cost = null;
          else if (usage.cost !== null) usage.cost += (turn.prompt_tokens * pricing.inputPerMillion + turn.completion_tokens * pricing.outputPerMillion) / 1_000_000;
          if (!finished) worker.postMessage({ type: 'turn', id: message.id, turn });
        } catch (error) { transportError = error instanceof Error ? error : new Error(String(error)); if (!finished) worker.postMessage({ type: 'turn', id: message.id, error: transportError.message }); }
      } else if (message.type === 'result') result = message.result;
      else if (message.type === 'error') fail(transportError ?? new Error(message.error));
      else if (message.type === 'disposed') {
        if (transportError) { fail(transportError); return; }
        if (!result) { fail(new Error('worker disposed without evaluation result')); return; }
        if (finished) return; finished = true; cleanup(); void worker.terminate();
        result.usage = usage; resolve(result);
      }
    });
    worker.on('error', fail); worker.on('exit', code => { if (!finished) fail(new Error('evaluation worker exited before disposal: ' + code)); });
  });
}
