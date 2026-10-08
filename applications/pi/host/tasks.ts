/**
 * The three built-in task kinds as natural-language tasks. Each kind keeps pi-durable's name, version and initial
 * checkpoint; every phase handler, and the abort handler, calls the kind's natural-language entry (`generation.nl`,
 * `tool.nl`, `compaction.nl`) with the phase facts and the services bound to this invocation. The entry returns once
 * it has committed the task's next state; the scheduler's step rules then judge the phase as they judge pi's.
 *
 * Executor failures (the call failed, was blocked, or returned a bad value): when the task's committed state changed,
 * the phase made progress and stands; otherwise the phase runs once more from its checkpoint, and a second failure
 * faults the task, as pi faults a phase that throws (owner decision 16).
 */
import type { Context } from '@earendil-works/chord';
import type { NatlangRuntime } from '@natlang/node';
import { defineTask } from '../vendor/durable/src/tasks.ts';
import type { RunningTask, TaskRuntime } from '../vendor/durable/src/types.ts';
import type { Agent as PiAgent, AnyTask, Settings as PiSettings } from '../vendor/durable/src/harness/types.ts';
import { ensureProviderSessionId } from '../vendor/durable/src/harness/provider.ts';
import { LiveDoc } from '../vendor/durable/src/harness/live.ts';
import type { Agent, PhaseFacts, Settings } from '../types.ts';
import { AI_DECLARATION, aiService } from './ai.ts';
import { DURABLE_DECLARATION, durableService, type PhaseState, plain, type DurableHost } from './durable.ts';
import { TOOLS_DECLARATION, toolsService } from './tools.ts';

type Runtime = TaskRuntime<unknown, unknown, unknown, Record<string, unknown>>;
/** A natural-language task entry: phase facts in, a one-line account of what it committed out. */
export type Entry = (facts: PhaseFacts) => Promise<string>;

export type TaskHost = DurableHost & {
  natlang: NatlangRuntime;
  /** Runs per phase; at least 1. Default 2: one rerun after an executor failure. */
  attempts?: number;
  /** Called with each phase's outcome, for logs. */
  onPhase?: (event: { kind: string; taskId: number; phase: string; mode: 'run' | 'abort'; attempt: number; error?: string; summary?: string }) => void;
};

/** The agent as JSON: tool declarations with their replay, mode and limits; section keys with their tag. */
export function agentFacts(agent: PiAgent): Agent {
  return plain({
    ...(agent.model ? { model: agent.model } : {}),
    thinkingLevel: agent.thinkingLevel,
    tools: agent.tools.map(tool => ({ name: tool.name, description: tool.description, parameters: tool.parameters,
      ...(tool.constrainedSampling === undefined ? {} : { constrainedSampling: tool.constrainedSampling }),
      ...(tool.replay ? { replay: tool.replay } : {}), ...(tool.executionMode ? { executionMode: tool.executionMode } : {}),
      ...(tool.outputLimits ? { outputLimits: tool.outputLimits } : {}) })),
    sections: agent.sections.map(section => ({ key: section.key, tag: section.tag !== false })),
    ...(agent.instructions !== undefined ? { instructions: agent.instructions } : {}),
    ...(agent.cwd !== undefined ? { cwd: agent.cwd } : {}),
  }) as Agent;
}

export function settingsFacts(settings: PiSettings): Settings {
  return plain({ stream: settings.stream, retry: settings.retry, compaction: settings.compaction, toolExecution: settings.toolExecution,
    steeringMode: settings.steeringMode, followUpMode: settings.followUpMode }) as Settings;
}

const stateKey = (record: { state: unknown } | undefined) => JSON.stringify(record?.state ?? null);

/**
 * Aborted tool tasks of one round write their results in call order, as pi-durable's (synchronous) abort handlers do:
 * an abort waits, a bounded time, until every earlier slot of the round is done. Mechanism only; the abort itself is
 * the natural-language phase.
 */
async function awaitEarlierSlots(task: RunningTask<unknown, unknown, unknown>, runtime: Runtime, context: Context): Promise<void> {
  const deadline = Date.now() + 600_000;
  while (Date.now() < deadline && !runtime.signal.aborted) {
    const slots = ((await runtime.snapshot(LiveDoc, task.conversationId, context)) as { tools?: { taskId?: number; status: string }[] } | undefined)?.tools ?? [];
    const mine = slots.findIndex(slot => slot.taskId === task.id);
    if (mine <= 0 || slots.slice(0, mine).every(slot => slot.status === 'done' || slot.taskId === undefined)) return;
    await new Promise(resolve => setTimeout(resolve, 250));
  }
}

async function invoke(host: TaskHost, entry: Entry, mode: 'run' | 'abort', task: RunningTask<unknown, unknown, unknown>, runtime: Runtime, context: Context) {
  const agent = await runtime.agent(context);
  const checkpoint = task.state.checkpoint as { phase?: string };
  const facts: PhaseFacts = {
    task: plain({ id: task.id, kind: task.kind, conversationId: task.conversationId, input: task.input, checkpoint }) as PhaseFacts['task'],
    mode, agent: agentFacts(agent), settings: settingsFacts(runtime.settings),
    sessionId: await ensureProviderSessionId(runtime, context), now: runtime.now(),
  };
  // One phase state per attempt: a failure the ai service sees ends what durable may still commit.
  const phaseState: PhaseState = {};
  const services: Record<string, object> = { durable: durableService(runtime, context, host, agent, phaseState), ai: aiService(runtime, context,
    task.kind === 'pi.generation' ? () => (checkpoint as { attempt?: number }).attempt ?? 1 : undefined, phaseState) };
  const serviceDeclarations: Record<string, string> = { durable: DURABLE_DECLARATION, ai: AI_DECLARATION };
  if (task.kind === 'pi.tool') {
    services.tools = toolsService(runtime, context, agent, (task.input as { callId: string }).callId);
    serviceDeclarations.tools = TOOLS_DECLARATION;
  }
  if (task.kind === 'pi.tool' && mode === 'abort') await awaitEarlierSlots(task, runtime, context);
  const before = stateKey(task);
  const attempts = Math.max(1, host.attempts ?? 2);
  const phase = mode === 'abort' ? 'abort' : String(checkpoint.phase);
  let previousAttempt = '';
  for (let attempt = 1; ; attempt++) {
    try {
      phaseState.failed = undefined;
      const summary = await host.natlang.run(() => entry(attempt === 1 ? facts : { ...facts, previousAttempt }), { services, serviceDeclarations, signal: runtime.signal,
        name: `${task.kind}#${task.id}:${phase}` });
      // An entry must commit the task's next state. One that says it did, while the state is unchanged, failed.
      if (stateKey(await runtime.getTask(task.id, context)) === before)
        throw new Error(`the ${phase} phase returned "${String(summary ?? '').slice(0, 200)}" without committing the task's next state`);
      host.onPhase?.({ kind: task.kind, taskId: task.id, phase, mode, attempt, summary: String(summary ?? '') });
      return;
    } catch (error) {
      if (runtime.signal.aborted) throw error;
      const message = error instanceof Error ? error.message : String(error);
      host.onPhase?.({ kind: task.kind, taskId: task.id, phase, mode, attempt, error: message });
      // A phase that committed its next state before failing made progress; the step rules judge it.
      if (stateKey(await runtime.getTask(task.id, context)) !== before) return;
      if (attempt >= attempts) throw new Error(`${task.kind} ${phase}: the executor failed: ${message}`);
      previousAttempt = message.slice(0, 2000);
    }
  }
}

/** A natural-language task kind with pi-durable's name, version, initial checkpoint and phase names. */
export function naturalLanguageTask(host: TaskHost, name: string, version: number, initial: (input: never) => unknown,
  phases: readonly string[], entry: Entry): AnyTask {
  const handler = (task: RunningTask<unknown, unknown, unknown>, runtime: Runtime, context: Context) => invoke(host, entry, 'run', task, runtime, context);
  return defineTask({
    name, version, initial: initial as never,
    phases: Object.fromEntries(phases.map(phase => [phase, handler])) as never,
    abort: (task: RunningTask<unknown, unknown, unknown>, runtime: Runtime, context: Context) => invoke(host, entry, 'abort', task, runtime, context),
  } as never) as AnyTask;
}
