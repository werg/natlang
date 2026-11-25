/**
 * What a program's own recorded runs say about each of its `.nl` functions (WS8 N4): the evidence `natlang improve DIR`
 * hands its natural-language improver. Crisp and exact: it reads the call store and reports what happened, nothing more.
 *
 * - failures: calls that ended without a result (`failed`, `blocked`, an error), with their inputs and the reason;
 * - escalations: decisions `escalate` sent to the stronger model, with the local answer and the fallback's answer
 *   (a disagreement is a labelled hard case);
 * - repairs: calls whose result broke an `ensures:` postcondition or an `Is<T, P>` refinement and had to be revised;
 * - feedback: calls a tester marked `bad` or fixed (`natlang traces bad|fix`), with the expected result when given.
 *
 * The same reading decides which recorded calls are trustworthy expected results for the case pack
 * (`trustedCallOutputs`): the local half of an escalated decision and a call a tester marked wrong are not.
 */
import { relative, resolve, sep } from 'node:path';
import { existsSync } from 'node:fs';
import { canonicalValue } from '../native/refinement.js';

type Ref = { complete: boolean; hash?: string } | null | undefined;
type Summary = { call_id: string; parent_call_id?: string | null; definition_id?: string; definition_source?: string | null;
  outcome?: string; started_at?: string };
type RecordLike = { call_id: string; parent_call_id?: string | null; outcome: string; detail?: string; definition: { id?: string; source: string | null };
  inputs: Record<string, Ref>; output: Ref; started_at?: string };

/** The call store reads the evidence needs (`CallStore` has all of them; the optional ones add evidence kinds). */
export type RunEvidenceStore = {
  calls(filter?: { program?: string; outcome?: string; definition?: string; limit?: number }): readonly Summary[];
  call(callId: string): RecordLike | undefined;
  value(ref: Ref): unknown;
  events?(callId: string): Record<string, unknown>[] | undefined;
  children?(callId: string): readonly Summary[];
  annotations?(callId: string): { kind: string; value: unknown }[];
};

export type RunFailure = { inputs: Record<string, unknown>; outcome: string; reason: string };
export type RunEscalation = { inputs: Record<string, unknown>; local: unknown; fallback: unknown; agreed: boolean; reason: string };
export type RunRepair = { inputs: Record<string, unknown>; problems: string[]; outcome: string };
export type RunFeedback = { inputs: Record<string, unknown>; output: unknown; verdict: string; expected?: unknown };

/** The evidence about one `.nl` function of the program, by its path relative to the program directory. */
export type FunctionEvidence = {
  function: string;
  /** Recorded calls of the function (finished, successful or not). */
  calls: number;
  failures: RunFailure[];
  escalations: RunEscalation[];
  repairs: RunRepair[];
  feedback: RunFeedback[];
};

/**
 * A value as the improver's stages receive it: whole. The stages (improveDiagnose, the `readout: decision`
 * improveShouldRevise) are natlang calls whose own listing shows a long value: cut with its variable named where the
 * call can read its scope, whole within the context budget where it cannot. A cut here left the rest nowhere.
 */
const shown = (value: unknown): unknown => value;

function inputsOf(store: RunEvidenceStore, record: RecordLike): Record<string, unknown> | undefined {
  const inputs: Record<string, unknown> = {};
  for (const [name, ref] of Object.entries(record.inputs)) {
    if (!ref?.complete) return undefined;
    const value = store.value(ref);
    if (value === undefined) return undefined;
    inputs[name] = value;
  }
  return inputs;
}

/** The program-relative path of a recorded definition source, when it is a `.nl` file inside the program. */
function programFunction(root: string, source: string | null | undefined): string | undefined {
  if (!source || !source.endsWith('.nl')) return undefined;
  const absolute = resolve(root, source);
  const rel = relative(root, absolute).split(sep).join('/');
  return rel && !rel.startsWith('..') && existsSync(absolute) ? rel : undefined;
}

/** The `escalation` note `escalate` left on a local call (std/policy.ts EscalationNote), when it escalated. */
type Escalated = { reason: string; fallbackCallId?: string; agreed?: boolean };

function escalationOf(store: RunEvidenceStore, callId: string): Escalated | undefined {
  const note = (store.annotations?.(callId) ?? []).filter(row => row.kind === 'escalation').at(-1)?.value as
    { escalated?: unknown; unavailable?: unknown; reason?: unknown; fallback_call?: unknown; agreed?: unknown } | undefined;
  if (!note || note.escalated !== true || note.unavailable === true) return undefined;
  return { reason: String(note.reason ?? 'policy'), ...(typeof note.fallback_call === 'string' ? { fallbackCallId: note.fallback_call } : {}),
    ...(typeof note.agreed === 'boolean' ? { agreed: note.agreed } : {}) };
}

function problemsOf(events: readonly Record<string, unknown>[]): string[] {
  const problems: string[] = [];
  for (const event of events) {
    if (event.kind === 'ensures_check' && event.outcome === 'fail')
      problems.push(`postcondition ${String(event.check)} failed${event.error ? ` (threw: ${String(event.error)})` : ''}`);
    else if (event.kind === 'refinement_check' && event.outcome === 'fail')
      problems.push(`result did not satisfy ${JSON.stringify(event.predicate ?? event.path ?? 'its refinement')}`);
  }
  return [...new Set(problems)];
}

function feedbackOf(store: RunEvidenceStore, callId: string): { verdict: string; expected?: unknown } | undefined {
  const rows = store.annotations?.(callId) ?? [];
  const last = rows.filter(row => row.kind === 'feedback').at(-1)?.value as { verdict?: unknown; expected?: unknown } | undefined;
  if (!last || typeof last.verdict !== 'string') return undefined;
  return { verdict: last.verdict, ...(Object.hasOwn(last, 'expected') ? { expected: last.expected } : {}) };
}

/**
 * Whether a recorded successful call's output can serve as an expected result, and which value: the local answer of an
 * escalated decision cannot (the fallback answered those inputs), a call a tester marked `bad` cannot, and a fixed call
 * gives the tester's expected value.
 */
export function trustedCallOutput(store: RunEvidenceStore, record: RecordLike):
    { trusted: true; expected?: unknown } | { trusted: false; why: string } {
  if (escalationOf(store, record.call_id)) return { trusted: false, why: 'the local answer of an escalated decision' };
  const feedback = feedbackOf(store, record.call_id);
  if (feedback?.verdict === 'bad' && !Object.hasOwn(feedback, 'expected')) return { trusted: false, why: 'marked bad by a tester' };
  if (feedback && Object.hasOwn(feedback, 'expected')) return { trusted: true, expected: feedback.expected };
  return { trusted: true };
}

/**
 * Read the program's recorded runs: per `.nl` function under `directory`, the failures, escalations, repairs and
 * feedback of its most recent `maxCalls` calls, at most `perKind` examples of each kind (newest first).
 */
export function gatherRunEvidence(store: RunEvidenceStore, directory: string, options: { maxCalls?: number; perKind?: number } = {}):
    Record<string, FunctionEvidence> {
  const root = resolve(directory);
  const perKind = options.perKind ?? 6;
  const byFunction: Record<string, FunctionEvidence> = {};
  for (const summary of store.calls({ program: root, limit: options.maxCalls ?? 1000 })) {
    if (summary.outcome === 'running') continue;
    const record = store.call(summary.call_id);
    if (!record) continue;
    const fn = programFunction(root, record.definition.source);
    if (!fn) continue;
    const evidence = byFunction[fn] ??= { function: fn, calls: 0, failures: [], escalations: [], repairs: [], feedback: [] };
    evidence.calls++;
    const inputs = inputsOf(store, record);
    if (!inputs) continue;
    const output = record.outcome === 'done' && record.output?.complete ? store.value(record.output) : undefined;
    if (record.outcome !== 'done' && evidence.failures.length < perKind)
      evidence.failures.push({ inputs: shown(inputs) as Record<string, unknown>, outcome: record.outcome, reason: String(shown(record.detail ?? '')) });
    const problems = problemsOf(store.events?.(record.call_id) ?? []);
    if (problems.length && evidence.repairs.length < perKind)
      evidence.repairs.push({ inputs: shown(inputs) as Record<string, unknown>, problems, outcome: record.outcome });
    const escalated = escalationOf(store, record.call_id);
    if (escalated?.fallbackCallId && evidence.escalations.length < perKind) {
      const fallback = store.call(escalated.fallbackCallId);
      const fallbackValue = fallback?.outcome === 'done' && fallback.output?.complete ? store.value(fallback.output) : undefined;
      if (fallbackValue !== undefined)
        evidence.escalations.push({ inputs: shown(inputs) as Record<string, unknown>, local: shown(output), fallback: shown(fallbackValue),
          agreed: escalated.agreed ?? canonicalValue(output) === canonicalValue(fallbackValue), reason: escalated.reason });
    }
    const feedback = feedbackOf(store, record.call_id);
    if (feedback && feedback.verdict !== 'good' && evidence.feedback.length < perKind)
      evidence.feedback.push({ inputs: shown(inputs) as Record<string, unknown>, output: shown(output), verdict: feedback.verdict,
        ...(Object.hasOwn(feedback, 'expected') ? { expected: shown(feedback.expected) } : {}) });
  }
  return byFunction;
}

/** Whether the runs show anything an improver could act on (a function whose every call went well has nothing to fix). */
export function hasActionableEvidence(evidence: FunctionEvidence): boolean {
  return evidence.failures.length > 0 || evidence.repairs.length > 0 || evidence.feedback.length > 0 ||
    evidence.escalations.some(item => !item.agreed);
}
