/**
 * In-run training guidance (applications/guidance-writer/README.md). A training run hands the agent verified facts
 * (`x_natlang_guidance`, `natlang.guidance-run/1`); for each consumer call, on its first model turn, the agent keeps the
 * facts the call may see, selects them with `selectGuidanceFacts` (the pluggable's crisp or natural-language side) and
 * runs `writeGuidance` in-process as a child call of the consumer, on the same runtime, model and recorder. Every turn of
 * the consumer then carries the `natlang.guidance-request/1` the engine renders (rl/privileged_context `guidance_plan`,
 * whose `_request_parts` is the exact contract), and every turn of the policy calls carries `x_natlang_guidance_policy`.
 *
 * The package's crisp parts and definitions are loaded by the host (teacher/guidance-writer.ts, Node); this module is the
 * platform-neutral mechanism around them.
 */
import type { CallableDefinition } from '../runtime/kernel.js';
import { runInFrame } from '../runtime/context.js';
import type { PluggableMode } from '../runtime/pluggable.js';
import { isNeuraleseRef } from './neuralese.js';
import type { LambdaNode } from './values.js';
import type { NativeSession } from './runtime.js';

export const GUIDANCE_RUN_SCHEMA = 'natlang.guidance-run/1';
export const GUIDANCE_REQUEST_SCHEMA = 'natlang.guidance-request/1';
export const GUIDANCE_FACT_KINDS = ['expected_result', 'expected_stop', 'expected_files', 'expected_effects',
  'reference_failure', 'reference_solution', 'child_value', 'minimum_sequence', 'decisive', 'teacher_trajectory',
  'attempt_feedback', 'fit_hindsight'] as const;
/** The token budget a selection is asked to fit when the run names none (the engine's declared rendering budget). */
export const DEFAULT_GUIDANCE_BUDGET_TOKENS = 512;
/** The recorded text-fallback reason of a run whose writer is switched off. */
export const WRITER_OFF = 'writer disabled by run setting';

export type GuidanceFact = { id: string; kind: typeof GUIDANCE_FACT_KINDS[number]; text: string;
  invocation?: string | null; match?: string[] };
export type GuidanceConsumer = { name: string; role: 'root' | 'child' | 'binding'; text: string };
export type GuidanceRun = { schema: typeof GUIDANCE_RUN_SCHEMA; role: 'teacher' | 'student';
  custody: Record<string, unknown>; facts: GuidanceFact[]; selection?: PluggableMode; budgetTokens?: number;
  /** `off` (opt-out, the measurement baseline): no writer; a teacher gets the selected facts as text, a student nothing. */
  writer?: 'native' | 'off' };
export type GuidancePolicyFunction = { function: 'selectGuidanceFacts' | 'writeGuidance'; sha256: string };

/** The guidance-writer package as loaded by the host: its two natural-language definitions and its crisp parts. */
export type GuidanceWriterPackage = {
  selection: { definition: CallableDefinition; sha256: string };
  writer: { definition: CallableDefinition; sha256: string };
  /** visibility.ts `visibleFacts`. */
  visibleFacts(consumer: GuidanceConsumer, facts: readonly GuidanceFact[]): GuidanceFact[];
  /** crispSelection.ts `crispSelection`. */
  crispSelection(consumer: GuidanceConsumer, facts: readonly GuidanceFact[], budgetTokens: number): string[];
  /** sha256 hex of crispSelection.ts `CRISP_SELECTION_IDENTITY`. */
  crispSha256: string;
};

const fail = (message: string): never => {
  throw Object.assign(new Error(`guidance-run-invalid: ${message}`), { code: 'guidance-run-invalid' });
};
const isRecord = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);

/** Validate a run-level `x_natlang_guidance` control (rl/jolt.py `guidance_controls` writes it). Returns it unchanged. */
export function guidanceRun(value: unknown): GuidanceRun {
  if (!isRecord(value) || value.schema !== GUIDANCE_RUN_SCHEMA) fail(`x_natlang_guidance is {schema: ${GUIDANCE_RUN_SCHEMA}, ...}`);
  const run = value as Record<string, unknown>;
  const extra = Object.keys(run).filter(key => !['schema', 'role', 'custody', 'facts', 'selection', 'budgetTokens', 'writer'].includes(key));
  if (extra.length) fail(`unsupported fields ${extra.join(', ')}`);
  if (run.role !== 'teacher' && run.role !== 'student') fail('role is teacher or student');
  if (!isRecord(run.custody)) fail('custody is the custody receipt object of the training world');
  if (!Array.isArray(run.facts) || !run.facts.length) fail('facts is a nonempty list');
  const ids = new Set<string>();
  for (const fact of run.facts as unknown[]) {
    if (!isRecord(fact) || typeof fact.id !== 'string' || !fact.id || typeof fact.text !== 'string' || !fact.text.trim() ||
        !(GUIDANCE_FACT_KINDS as readonly unknown[]).includes(fact.kind) ||
        Object.keys(fact).some(key => !['id', 'kind', 'text', 'invocation', 'match'].includes(key)))
      fail(`facts are {id, kind: ${GUIDANCE_FACT_KINDS.join('|')}, text, invocation?, match?}`);
    const item = fact as Record<string, unknown>;
    if (item.invocation !== undefined && item.invocation !== null && typeof item.invocation !== 'string')
      fail('a fact\'s invocation is a call id or null');
    if (item.match !== undefined && (!Array.isArray(item.match) || item.match.some(entry => typeof entry !== 'string')))
      fail('a fact\'s match is a list of prompt strings');
    if (ids.has(item.id as string)) fail(`fact id ${JSON.stringify(item.id)} repeats`);
    ids.add(item.id as string);
  }
  if (run.selection !== undefined && !['crisp', 'nl', 'shadow'].includes(run.selection as string))
    fail('selection is crisp, nl or shadow');
  if (run.writer !== undefined && run.writer !== 'native' && run.writer !== 'off') fail('writer is native or off');
  if (run.budgetTokens !== undefined && (!Number.isSafeInteger(run.budgetTokens) || (run.budgetTokens as number) < 1))
    fail('budgetTokens is a positive integer');
  return value as GuidanceRun;
}

/** The prompt text of one model request, as the engine computes it for visibility: string contents joined by newlines. */
export function requestText(messages: readonly unknown[]): string {
  return messages.flatMap(message => isRecord(message) && typeof message.content === 'string' ? [message.content] : []).join('\n');
}

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error)).split('\n')[0]!.slice(0, 300);

/**
 * One run's training guidance. The agent asks `policyOf` whether a call is the guidance policy's own (its turns carry
 * `x_natlang_guidance_policy` and get no guidance), and `requestFor` for a consumer call's guidance request.
 */
export class TrainingGuidance {
  /** Calls running the guidance policy: they and every call below them get no guidance. */
  private readonly policyCalls = new Set<string>();
  private readonly requests = new Map<string, Promise<Record<string, unknown> | undefined>>();
  constructor(readonly run: GuidanceRun, readonly writerPackage: GuidanceWriterPackage) { guidanceRun(run); }

  /** The policy function a call runs, by its exact definition (name and instructions), and remembers the call. */
  policyOf(lam: LambdaNode, callId: string | null): GuidancePolicyFunction | undefined {
    const body = lam.originalBody ?? lam.body;
    for (const [name, part] of [['selectGuidanceFacts', this.writerPackage.selection], ['writeGuidance', this.writerPackage.writer]] as const) {
      if (lam.functionName === part.definition.name && body === part.definition.body) {
        if (callId) this.policyCalls.add(callId);
        return { function: name, sha256: part.sha256 };
      }
    }
    return undefined;
  }

  /** Whether a call runs inside the guidance policy (its own calls and theirs). */
  insidePolicy(session: NativeSession): boolean {
    return (session.runtime.frame?.chain ?? []).some(id => this.policyCalls.has(id)) ||
      (!!session.runtime.currentCallId && this.policyCalls.has(session.runtime.currentCallId));
  }

  /**
   * The `natlang.guidance-request/1` of a consumer call, computed once on its first model turn from that turn's request
   * messages and reused by its later turns; undefined when the call gets no guidance.
   */
  requestFor(session: NativeSession, callId: string, messages: readonly unknown[]): Promise<Record<string, unknown> | undefined> {
    let request = this.requests.get(callId);
    if (!request) {
      request = this.compute(session, callId, requestText(messages));
      this.requests.set(callId, request);
    }
    return request;
  }

  private async compute(session: NativeSession, callId: string, text: string): Promise<Record<string, unknown> | undefined> {
    const trace = session.runtime.trace, pkg = this.writerPackage;
    const role = session.runtime.options.parentCallId ? 'child' as const : 'root' as const;
    const consumer: GuidanceConsumer = { name: callId, role, text };
    const visible = pkg.visibleFacts(consumer, this.run.facts);
    const emit = (phase: string, data: Record<string, unknown> = {}) =>
      trace.emit('training_guidance', { call_id: callId, phase, ...data });
    if (!visible.length) { emit('none', { reason: 'no visible facts' }); return undefined; }
    if (this.run.writer === 'off' && this.run.role === 'student') { emit('none', { reason: WRITER_OFF }); return undefined; }
    const budget = this.run.budgetTokens ?? DEFAULT_GUIDANCE_BUDGET_TOKENS;
    const frame = session.runtime.frame;
    const { invokeDefinition } = await import('../runtime/kernel.js');
    const { pluggable } = await import('../runtime/pluggable.js');
    const known = new Set(visible.map(fact => fact.id));
    // model-call-kind: runtime-authored-call (selectGuidanceFacts, guidance policy) -- training/model-call-kinds.json
    const nl = async (who: GuidanceConsumer, facts: GuidanceFact[], tokens: number): Promise<string[]> => {
      const chosen = await invokeDefinition(frame!, pkg.selection.definition, [who, facts, tokens]);
      if (!Array.isArray(chosen) || chosen.some(id => typeof id !== 'string' || !known.has(id)) || new Set(chosen).size !== chosen.length)
        throw new Error('selectGuidanceFacts must return distinct ids of the visible facts');
      return chosen as string[];
    };
    let implementation: 'nl' | 'crisp' = (this.run.selection ?? 'nl') === 'crisp' ? 'crisp' : 'nl';
    let ids: string[];
    try {
      ids = await runInFrame(frame, () => pluggable({ crisp: pkg.crispSelection, nl }, this.run.selection ?? 'nl',
        { name: 'selectGuidanceFacts' })(consumer, visible, budget));
    } catch (error) {
      implementation = 'crisp';
      emit('selection_fallback', { reason: errorText(error) });
      ids = pkg.crispSelection(consumer, visible, budget);
    }
    const byId = new Map(visible.map(fact => [fact.id, fact]));
    const facts = ids.map(id => byId.get(id)!);
    if (!facts.length) { emit('none', { reason: 'no facts selected', selection: implementation }); return undefined; }
    const selection = { implementation, sha256: implementation === 'nl' ? pkg.selection.sha256 : pkg.crispSha256 };
    const base = { schema: GUIDANCE_REQUEST_SCHEMA, invocation: { name: callId, role }, custody: this.run.custody,
      facts: facts.map(fact => structuredClone(fact)), selection };
    let state: { id: string; source_occurrence?: string } | undefined, failure: string | undefined;
    if (this.run.writer === 'off') failure = WRITER_OFF;
    else try {
      // model-call-kind: runtime-authored-call (writeGuidance, guidance policy) -- training/model-call-kinds.json
      const written = await runInFrame(frame, () => invokeDefinition(frame!, pkg.writer.definition, [consumer, facts]));
      if (isNeuraleseRef(written)) state = { id: written.$neuralese.id,
        ...(written.$neuralese.source_occurrence ? { source_occurrence: written.$neuralese.source_occurrence } : {}) };
      else failure = 'writeGuidance returned no native state';
    } catch (error) { failure = `writeGuidance failed: ${errorText(error)}`; }
    if (state) {
      const mode = this.run.role === 'teacher' ? 'import' : 'write-only';
      emit('written', { mode, state: state.id, facts: ids, selection: implementation });
      return { ...base, mode, writer: { function: 'writeGuidance', sha256: pkg.writer.sha256 }, state };
    }
    if (this.run.role === 'student') { emit('none', { reason: failure, selection: implementation }); return undefined; }
    emit('text_fallback', { reason: failure, facts: ids, selection: implementation });
    return { ...base, mode: 'text', fallback: failure };
  }
}
