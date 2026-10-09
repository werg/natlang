/**
 * Revisioned local scheduling in natural language. `scheduler.nl` is the scheduler: it reads a request, extracts the
 * constraints, builds and repairs candidate schedules, ranks them by the user's wishes and explains the choice. The
 * workspace (workspace.ts) is the exact side: the verifier that checks everything the stages produce, the exact
 * enumeration, and the revisioned commit. `plan` takes a snapshot, runs the scheduler on it and commits conditionally.
 */
import planner from './scheduler.nl';
import { calendarDeclaration, calendarService, type Implementation } from './calendar.js';
import type { ScheduleResult, ScheduleWorkspace } from './workspace.js';

export * from './workspace.js';
export * from './calendar.js';
export type * from './types.js';

/** What a run of natural-language work needs: the services its stages call. */
export type RunServices = { services: Record<string, unknown>, serviceDeclarations: Record<string, string> };
/** Runs natural-language work in a task with these services (e.g. `(fn, options) => runtime.run(fn, options)`). */
export type Run = <T>(fn: () => Promise<T>, options: RunServices) => Promise<T>;
export type PlanOptions = {
  run: Run,
  /** How many complete schedules to build and compare at most (default 16). */
  candidates?: number,
  /** Which implementation of the enumeration hot path runs (default: the exact crisp search). */
  enumeration?: Implementation,
};

/**
 * Plan the day for a free-text request: the scheduler decides on a snapshot, the verifier re-checks what it chose, and
 * the workspace commits it only if the revision is unchanged.
 */
export async function plan(workspace: ScheduleWorkspace, request: string, options: PlanOptions): Promise<ScheduleResult> {
  const revision = workspace.revision, view = workspace.view(), problem = workspace.problem();
  const enumeration = options.enumeration ?? 'crisp';
  const proposal = await options.run(() => planner(request, view, { candidates: options.candidates ?? 16 }),
    { services: { calendar: calendarService(problem, { enumeration }) }, serviceDeclarations: { calendar: calendarDeclaration } });
  workspace.note({ operation: 'schedule.proposal', revision, status: proposal.status, considered: proposal.considered,
    truncated: proposal.truncated, enumeration });
  if (proposal.status !== 'chosen') {
    const snapshot = workspace.snapshot();
    return { status: proposal.status === 'unclear' ? 'unclear' : 'infeasible', revision: snapshot.revision, plan: snapshot.plan,
      detail: proposal.status === 'unclear' ? proposal.questions.join(' ') : proposal.explanation, explanation: proposal.explanation };
  }
  return { ...workspace.commitPlan(proposal.placements, revision, proposal.hard), explanation: proposal.explanation };
}
