/**
 * Runs pi as a natlang program: pi.nl is the agent on the big model, and its tools and System One judgments are the
 * natural-language functions of its folder, on the small model. The host supplies the outside world (services.ts),
 * writes the session log from the runtime's traces, and counts what the run did.
 */
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { NatlangRuntime, type InvocationTrace, type ModelConfig, type ModelDriver } from '@natlang/node';
import pi from './pi.nl';
import { PI_SERVICE_DECLARATIONS, piServices } from './services.js';
import type { AgentResult, Intervention } from './agent.js';

const TOOLS = new Set(['read', 'bash', 'edit', 'write']);
const JUDGMENTS = new Set(['risk', 'digest', 'review', 'progress', 'done', 'scout']);

export type PureOptions = {
  task: string,
  cwd: string,
  /** The launcher's runtime options: its model runs the tools and judgments. */
  base: NatlangRuntime['options'],
  /** Runs pi.nl itself. */
  big: ModelDriver | ModelConfig,
  skillDirs?: string[],
  confirm?: (question: string) => Promise<boolean>,
  maxTurns?: number,
  /** Session log: one JSON line per finished call, with its full trace. */
  session?: string,
  onTrace?: (trace: InvocationTrace) => void,
  signal?: AbortSignal,
};

export async function runPure(options: PureOptions): Promise<AgentResult> {
  const started = performance.now();
  const traces: InvocationTrace[] = [];
  if (options.session) {
    mkdirSync(dirname(options.session), { recursive: true });
    writeFileSync(options.session, JSON.stringify({ type: 'session', version: 'natlang', cwd: options.cwd, task: options.task, timestamp: new Date().toISOString() }) + '\n');
  }
  const big = typeof options.big === 'function' ? { driver: options.big } : options.big;
  const runtime = new NatlangRuntime({ ...options.base, models: { ...options.base.models, big: { ...big, maxTurns: options.maxTurns ?? big.maxTurns } } });
  let answer = '', stopped: AgentResult['stopped'] = 'answered', failure: string | undefined;
  try {
    answer = await runtime.run(() => pi(options.task, options.cwd, options.skillDirs ?? []), {
      services: piServices(options.cwd, options.confirm), serviceDeclarations: PI_SERVICE_DECLARATIONS, signal: options.signal,
      trace: trace => {
        traces.push(trace);
        if (options.session) appendFileSync(options.session, JSON.stringify({ type: 'call', ...trace }) + '\n');
        options.onTrace?.(trace);
      } });
  } catch (error) {
    // Stopped from outside (a time limit), or the agent's call failed: what the run did so far is its result.
    stopped = options.signal?.aborted ? 'aborted' : 'failed';
    if (stopped === 'failed') failure = String((error as Error)?.message ?? error).slice(0, 2000);
  }
  return { answer, stopped, ...tally(traces), ms: Math.round(performance.now() - started), ...failure ? { error: failure } : {} };
}

/** What the run did, from its traces: the agent's turns, tool calls, judgments (with their distributions) and tokens. */
function tally(traces: InvocationTrace[]): Omit<AgentResult, 'answer' | 'stopped' | 'ms'> {
  const root = traces.find(trace => trace.parentCallId === null);
  const requests = (trace: InvocationTrace) => trace.events.filter(event => event.kind === 'model_request' && event.phase === 'end');
  const tokens = (key: string) => traces.reduce((sum, trace) => sum + requests(trace).reduce((n, event) => n + Number(event[key] ?? 0), 0), 0);
  const interventions: Intervention[] = traces.filter(trace => JUDGMENTS.has(trace.name)).map(trace => {
    const scored = trace.events.find(event => event.kind === 'decision_readout' && event.phase === 'scored');
    const probabilities = scored ? scored.probabilities as number[] : [];
    const best = probabilities.indexOf(Math.max(...probabilities));
    return { kind: trace.name as Intervention['kind'], turn: 0, action: trace.outcome,
      ...scored ? { value: JSON.parse((scored.options as string[])[best]!), confidence: probabilities[best] } : {},
      ms: 0, ...trace.outcome === 'done' ? {} : { error: trace.detail.slice(0, 300) } };
  });
  return { turns: root ? requests(root).length : 0, smallTurns: traces.filter(trace => trace !== root).reduce((n, trace) => n + requests(trace).length, 0),
    toolCalls: traces.filter(trace => TOOLS.has(trace.name)).length, interventions,
    promptTokens: tokens('prompt_tokens'), completionTokens: tokens('completion_tokens') };
}
