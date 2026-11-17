/**
 * In-run training guidance (applications/guidance-writer/README.md). A training run hands the agent verified facts
 * (`x_natlang_guidance`, `natlang.guidance-run/1`); for each consumer call, on its first model turn, the agent keeps the
 * facts the call may see, selects them with `selectGuidanceFacts` (the pluggable's crisp or natural-language side) and
 * runs `writeGuidance` in-process as a child call of the consumer, on the same runtime, model and recorder. Every turn of
 * the consumer then carries the `natlang.guidance-request/2` whose guidance message the engine renders and marks (rl/privileged_context `guidance_plan`,
 * whose `_request_parts` is the exact contract), and every turn of the policy calls carries `x_natlang_guidance_policy`.
 *
 * The package's crisp parts and definitions are loaded by the host (runtime/node-guidance-writer.ts, Node); this module is the
 * platform-neutral mechanism around them.
 */
import type { CallableDefinition } from '../runtime/kernel.js';
import { runInFrame } from '../runtime/context.js';
import type { PluggableMode } from '../runtime/pluggable.js';
import { isNeuraleseRef } from './neuralese.js';
import { PAGE_CHARS_PER_TOKEN } from './cutoff.js';
import { parseType } from './types.js';
import type { LambdaNode } from './values.js';
import type { NativeSession } from './runtime.js';

export const GUIDANCE_RUN_SCHEMA = 'natlang.guidance-run/1';
export const GUIDANCE_REQUEST_SCHEMA = 'natlang.guidance-request/2';
/** The guidance turn's first line: a writer's state follows it; the facts as text follow `GUIDANCE_TEXT_HEADER`. */
export const GUIDANCE_HEADER = 'Guidance for this call (training only):';
export const GUIDANCE_TEXT_HEADER = 'Verified reference for this call (training only):';

type GuidanceMessage = { role: 'system'; content: string | ({ type: 'text'; text: string } |
  { type: 'native_state'; id: string; source_occurrence?: string })[] };

/**
 * The guidance turn as one ordinary system message, the only definition of its form (plans/neuralese/DECISIONS.md
 * 2026-11-18): every engine renders it with its chat template after the leading system message and marks its span;
 * the learner re-verifies the captured tokens (rl/privileged_context `verify_captured_block`). A writer's state follows
 * the header; text guidance is the header and the facts, packed whole into `budgetTokens` (estimated at
 * PAGE_CHARS_PER_TOKEN characters per token), the rest `omitted`.
 */
export function guidanceMessage(facts: readonly GuidanceFact[], state?: { id: string; source_occurrence?: string },
    budgetTokens = DEFAULT_GUIDANCE_BUDGET_TOKENS): { message: GuidanceMessage; included: GuidanceFact[];
      omitted: { id: string; reason: 'token_budget' }[] } {
  if (state) return { message: { role: 'system', content: [{ type: 'text', text: GUIDANCE_HEADER + '\n' },
    { type: 'native_state', ...state }] }, included: [...facts], omitted: [] };
  const included: GuidanceFact[] = [], omitted: { id: string; reason: 'token_budget' }[] = [];
  for (const fact of facts) {
    const body = [GUIDANCE_TEXT_HEADER, ...[...included, fact].map(item => item.text)].join('\n\n');
    if (body.length <= budgetTokens * PAGE_CHARS_PER_TOKEN) included.push(fact);
    else omitted.push({ id: fact.id, reason: 'token_budget' });
  }
  return { message: { role: 'system', content: [GUIDANCE_TEXT_HEADER, ...included.map(fact => fact.text)].join('\n\n') },
    included, omitted };
}
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
  /** `off` (opt-out, the measurement baseline): no writer; a teacher gets the selected facts as text, a student has the
   * same text turn recorded for its OPSD twin (`text-twin`) and sees nothing. */
  writer?: 'native' | 'off';
  /** Hindsight (teacher only): a verified successful run of the task as a `TrajectoryCall`
   * (applications/guidance-writer/types.ts); the root call's guidance is the hint `writeHint` writes from it. */
  trajectory?: Record<string, unknown> };
export type GuidancePolicyFunction = { function: 'selectGuidanceFacts' | 'writeGuidance' | 'writeHint'; sha256: string };

/** The guidance-writer package as loaded by the host: its two natural-language definitions and its crisp parts. */
export type GuidanceWriterPackage = {
  selection: { definition: CallableDefinition; sha256: string };
  writer: { definition: CallableDefinition; sha256: string };
  /** writeHint.nl (hindsight hints from a verified run). */
  hint: { definition: CallableDefinition; sha256: string };
  /** trajectoryOverview.ts `trajectoryOverview`: the crisp cold-start view of a run the hint writer opens with. */
  trajectoryOverview(call: unknown): unknown;
  /** hintCheck.ts `checkHint`: why a hint is rejected (one sentence), or undefined. */
  checkHint(hint: unknown, execution: unknown): string | undefined;
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
  const extra = Object.keys(run).filter(key => !['schema', 'role', 'custody', 'facts', 'selection', 'budgetTokens', 'writer', 'trajectory'].includes(key));
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
  if (run.trajectory !== undefined) {
    const call = run.trajectory as Record<string, unknown>;
    if (!isRecord(call) || typeof call.call !== 'string' || typeof call.prompt !== 'string' || !Array.isArray(call.steps) ||
        !Array.isArray(call.children) || !isRecord(call.result))
      fail('trajectory is a TrajectoryCall {call, function, role, prompt, steps, result, children}');
    if (run.role !== 'teacher') fail('a hindsight trajectory guides teacher attempts only');
  }
  if (run.budgetTokens !== undefined && (!Number.isSafeInteger(run.budgetTokens) || (run.budgetTokens as number) < 1))
    fail('budgetTokens is a positive integer');
  return value as GuidanceRun;
}

/** The prompt text of one model request, as the engine computes it for visibility: string contents joined by newlines. */
export function requestText(messages: readonly unknown[]): string {
  return messages.flatMap(message => isRecord(message) && typeof message.content === 'string' ? [message.content] : []).join('\n');
}

/**
 * What the writer reads about its consumer: the call's own request text without the shared system prompt. Visibility
 * still matches the whole request text; the writer only needs the call's opening, and repeating the ~5k-token system
 * prompt inside the writer's argument doubled its prompt and buried the facts (MEMETIC_META_LEARNING_REVIEW_20261103.md).
 */
export function consumerText(messages: readonly unknown[]): string {
  const own = messages.filter(message => !(isRecord(message) && message.role === 'system'));
  return requestText(own.length ? own : messages);
}

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error)).split('\n')[0]!.slice(0, 300);

/**
 * One run's training guidance. The agent asks `policyOf` whether a call is the guidance policy's own (its turns carry
 * `x_natlang_guidance_policy` and get no guidance), and `requestFor` for a consumer call's guidance request.
 */
export class TrainingGuidance {
  /** Calls running the guidance policy: they and every call below them get no guidance. */
  private readonly policyCalls = new Set<string>();
  /** The last model turn node of each writeGuidance call: the source of the state its text is written as. */
  private readonly writerTurns = new Map<string, { node: string; ancestors: readonly string[] }>();
  private readonly requests = new Map<string, Promise<Record<string, unknown> | undefined>>();
  constructor(readonly run: GuidanceRun, readonly writerPackage: GuidanceWriterPackage) { guidanceRun(run); }

  /** The policy function a call runs, by its exact definition (name and instructions), and remembers the call. */
  policyOf(lam: LambdaNode, callId: string | null): GuidancePolicyFunction | undefined {
    const body = lam.originalBody ?? lam.body;
    for (const [name, part] of [['selectGuidanceFacts', this.writerPackage.selection], ['writeGuidance', this.writerPackage.writer],
      ['writeHint', this.writerPackage.hint]] as const) {
      if (lam.functionName === part.definition.name && body === part.definition.body) {
        if (callId) this.policyCalls.add(callId);
        return { function: name, sha256: part.sha256 };
      }
    }
    return undefined;
  }

  /**
   * The exact check of a writeHint call's result (hintCheck.ts) against the run's verified trajectory, or undefined for
   * any other call. The agent applies it where a result is proposed; a failure is the writer's one-sentence tool error.
   */
  hintGuard(lam: LambdaNode): ((value: unknown) => string | undefined) | undefined {
    const trajectory = this.run.trajectory;
    if (!trajectory || this.policyOf(lam, null)?.function !== 'writeHint') return undefined;
    return value => this.writerPackage.checkHint(value, trajectory);
  }

  /** The agent records each writeGuidance model turn as it is taken (its graph node). */
  noteWriterTurn(callId: string, node: string, ancestors: readonly string[]): void { this.writerTurns.set(callId, { node, ancestors }); }

  /** Whether a call runs inside the guidance policy (its own calls and theirs). */
  insidePolicy(session: NativeSession): boolean {
    return (session.runtime.frame?.chain ?? []).some(id => this.policyCalls.has(id)) ||
      (!!session.runtime.currentCallId && this.policyCalls.has(session.runtime.currentCallId));
  }

  /**
   * The `natlang.guidance-request/2` of a consumer call, computed once on its first model turn from that turn's request
   * messages and reused by its later turns; undefined when the call gets no guidance.
   */
  requestFor(session: NativeSession, callId: string, messages: readonly unknown[]): Promise<Record<string, unknown> | undefined> {
    let request = this.requests.get(callId);
    if (!request) {
      request = this.compute(session, callId, requestText(messages), consumerText(messages));
      this.requests.set(callId, request);
    }
    return request;
  }

  private async compute(session: NativeSession, callId: string, text: string, ownText: string): Promise<Record<string, unknown> | undefined> {
    const trace = session.runtime.trace, pkg = this.writerPackage;
    const role = session.runtime.options.parentCallId ? 'child' as const : 'root' as const;
    const consumer: GuidanceConsumer = { name: callId, role, text };
    const visible = pkg.visibleFacts(consumer, this.run.facts);
    const emit = (phase: string, data: Record<string, unknown> = {}) =>
      trace.emit('training_guidance', { call_id: callId, phase, ...data });
    if (!visible.length) { emit('none', { reason: 'no visible facts' }); return undefined; }
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
    let writer: { function: 'writeGuidance' | 'writeHint'; sha256: string } = { function: 'writeGuidance', sha256: pkg.writer.sha256 };
    if (this.run.writer === 'off') failure = WRITER_OFF;
    else try {
      // model-call-kind: runtime-authored-call (writeGuidance, guidance policy) -- training/model-call-kinds.json
      const known = new Set(this.writerTurns.keys());
      // Hindsight: the root call of a teacher attempt with a verified run gets writeHint's hint over that run.
      const trajectory = role === 'root' ? this.run.trajectory : undefined;
      writer = trajectory ? { function: 'writeHint', sha256: pkg.hint.sha256 } : writer;
      const written = await runInFrame(frame, () => trajectory
        ? invokeDefinition(frame!, pkg.hint.definition, [{ ...consumer, text: ownText }, trajectory, pkg.trajectoryOverview(trajectory), facts])
        : invokeDefinition(frame!, pkg.writer.definition, [{ ...consumer, text: ownText }, facts]));
      // The hint's check also holds a result proposed past the agent's gate (the agent applies it to return_result and
      // to eval results; this is the last word).
      const rejected = trajectory && typeof written === 'string' ? pkg.checkHint(written, trajectory) : undefined;
      if (rejected) throw new Error(`hint-rejected: ${rejected}`);
      if (typeof written === 'string' && written.trim()) {
        // Span basis: the writer answers in its own plain text (an ordinary string result, measured to work on full67
        // where a native producer span and a forced template value did not); the configured writer materializes that
        // exact text as the native state, a typed-text-result producer span bound to the writer's final model turn.
        // The writer runs as a child call of this consumer; concurrent consumers' writers are told apart by call id.
        const own = [...this.writerTurns].filter(([id, entry]) => !known.has(id) && entry.ancestors.includes(callId));
        if (own.length !== 1) throw new Error(`${writer.function} returned text without exactly one recorded writer call`);
        const turn = own[0]![1].node;
        const ref = await session.runtime.writeNeuraleseText(written, parseType('Neuralese<string>'), {
          sourceKind: 'typed-text-result', source: 'return_result', markerContext: 'return-result',
          inputs: [{ node: turn, port: 'result-source' }] });
        if (isNeuraleseRef(ref)) state = { id: ref.$neuralese.id,
          ...(ref.$neuralese.source_occurrence ? { source_occurrence: ref.$neuralese.source_occurrence } : {}) };
        else failure = 'the configured writer returned no native state';
      } else if (isNeuraleseRef(written)) state = { id: written.$neuralese.id,
        ...(written.$neuralese.source_occurrence ? { source_occurrence: written.$neuralese.source_occurrence } : {}) };
      else failure = `${writer.function} returned no guidance text`;
    } catch (error) { failure = `${writer.function} failed: ${errorText(error)}`; }
    if (state) {
      const mode = this.run.role === 'teacher' ? 'import' : 'write-only';
      emit('written', { mode, state: state.id, facts: ids, selection: implementation, writer: writer.function });
      return { ...base, mode, writer, state, message: guidanceMessage(facts, state).message };
    }
    // Text fallback: a teacher gets the selected facts as a text turn; a student's engine renders the same turn and only
    // records it (`text-twin`: token ids and insertion point in the capture receipt) for the learner's OPSD twin.
    const mode = this.run.role === 'teacher' ? 'text' : 'text-twin';
    emit('text_fallback', { mode, reason: failure, facts: ids, selection: implementation });
    const packed = guidanceMessage(facts, undefined, budget);
    if (!packed.included.length) { emit('none', { reason: 'no fact fits the token budget', selection: implementation }); return undefined; }
    return { ...base, facts: packed.included.map(fact => structuredClone(fact)), mode, fallback: failure,
      message: packed.message, ...(packed.omitted.length ? { omitted: packed.omitted } : {}) };
  }
}
