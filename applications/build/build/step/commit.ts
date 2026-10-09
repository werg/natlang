/**
 * The exact side of a build round. admit checks a decision against the state before any effect happens. settle is the
 * pure, bounded commit: it appends one round's outcome to the state and sets the status. It makes no judgments.
 */
import { crisp as readyIds } from './ready.js';
import type { BuildState, Decision, Diagnosis, TaskResult } from '../../types.js';

/** Whether a decision to run or reuse names a task that is ready now. problem says which tasks are, and what to do. */
export function admit(state: BuildState, decision: Decision): { ok: boolean, problem: string } {
  if (decision.kind !== 'run' && decision.kind !== 'reuse') return { ok: true, problem: '' };
  const ready = readyIds(state.graph, state.needed, state.order);
  if (ready.includes(decision.task)) return { ok: true, problem: '' };
  return { ok: false, problem: `chosen task is not ready: ${decision.task}; choose the id of one of the ready tasks: ${ready.join(', ') || 'none'}` };
}

/**
 * The next state after one round. A stop blocks the build on the unfinished closure. A reject invalidates it with the
 * decision's reason. A run or reuse appends the task's result: a task that finished joins order (and the judgments
 * with the decision's reason), a failed or unknown one ends the build with its diagnosis. The goal finishing is done.
 */
export function settle(state: BuildState, decision: Decision, result: TaskResult | null, diagnosis: Diagnosis | null): BuildState {
  if (decision.kind === 'stop')
    return { ...state, status: 'blocked', blocked: state.needed.filter(id => !state.order.includes(id)).sort(),
      detail: `${decision.why}; dependencies may be missing or cyclic.` };
  if (decision.kind === 'reject') return { ...state, status: 'invalid', detail: decision.why };
  if (!result || result.id !== decision.task) throw new Error(`settle needs the result of task ${decision.task}`);
  if (state.attempted.includes(decision.task)) throw new Error(`task ${decision.task} was already attempted`);
  const attempted = [...state.attempted, decision.task], results = [...state.results, result];
  if (result.status !== 'ok')
    return { ...state, attempted, results, status: result.status, detail: `${decision.task}: ${result.detail}`,
      ...(diagnosis ? { diagnosis } : {}) };
  return { ...state, attempted, results, order: [...state.order, decision.task],
    judgments: [...state.judgments, { task: decision.task, action: decision.kind === 'reuse' ? 'reused' : 'ran', reason: decision.why }],
    status: decision.task === state.goal ? 'done' : 'running', detail: '' };
}
