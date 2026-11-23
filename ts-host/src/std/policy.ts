/**
 * Policy & higher-order combinators (plans/semantic-fp/03_SEMANTIC_STDLIB_AND_PATTERNS.md S3;
 * ts-host/std/DECOMPOSITION.md):
 * - `escalate`: confidence- and risk-gated escalation from a local `readout: decision` function to a stronger model;
 * - `withContext`: ergonomic sugar over `fn.in(Context.ofCallable(fn).with(entries))` preserving companion helpers;
 * - `compileRule`: turn a written rule into a typed semantic function, validated and repaired against examples.
 */
import { Context } from '../runtime/contexts.js';
import { decideInFrame, type Decision } from '../runtime/runtime.js';
import { callableMeta, isNatlangCallable } from '../runtime/callable.js';
import type { NatlangFunction } from '../runtime/surface.js';
import { pluggable, type PluggableSetting } from '../runtime/pluggable.js';
import { stableHash } from '../runtime/iterate.js';
import { semanticRepair, type Problems } from './converge.js';
import { compileRuleFn as shippedCompileRule, shouldEscalate as shippedShouldEscalate,
  type EscalationHistory, type TierCost } from './judgments.js';
import { StdSpan, stdError } from './trace.js';

type Maybe<T> = T | Promise<T>;

// --- escalate -------------------------------------------------------------------------------------------------------

export type EscalateOptions<R> = {
  /** Target model name in the runtime's `models` map (for example `'teacher'`). Alias: `fallback`. */
  to?: string;
  /** Target model name in the runtime's `models` map (alias for `to`). */
  fallback?: string;
  /** Confidence threshold below which the call escalates to `to`/`fallback`. Default `0.8`. */
  floor?: number;
  /**
   * Asymmetric risk floors keyed by JSON or string label (for example `{ block: 0.2 }` or `{ true: 0.25 }`):
   * escalate whenever any named value has probability at least its risk threshold, even if another value won argmax.
   */
  risk?: Record<string, number>;
  /**
   * Escalation policy mode (`pluggable`): `'crisp'` (default: asymmetric risk floors then confidence floor),
   * `'nl'` (the shipped `shouldEscalate.nl` decision policy), or `'shadow'` (runs both, serves crisp, records `pluggable_shadow`).
   */
  policy?: PluggableSetting;
  /** Optional custom escalation predicate over the local decision (overrides `policy` when supplied). */
  shouldEscalate?: (decision: Decision<R>) => Maybe<boolean>;
  /** Replacement for the shipped `shouldEscalate.nl` judgment when `policy` is `'nl'` or `'shadow'`. */
  nlPolicy?: (decision: Decision<R>, floor?: number, risk?: Record<string, number>,
    history?: EscalationHistory, cost?: TierCost) => Maybe<boolean>;
  /**
   * Prior escalation statistics for the `nl` policy. By default the policy reads them from the call store itself: this
   * function's earlier escalate runs (how often it escalated, how often the fallback disagreed with the local answer),
   * plus this wrapper's own runs since. Pass statistics to use yours, or `false` for none.
   */
  history?: EscalationHistory | false;
  /** Local vs fallback tier cost for the `nl` policy. By default, the mean milliseconds per call measured in those runs. */
  cost?: TierCost | false;
};

/** What the recorded escalate runs of one function say: the `nl` policy's history and measured tier cost. */
export type EscalationRecord = { history: EscalationHistory; cost?: TierCost };

/**
 * What one escalate run records on its local call (annotation kind `escalation`, not pinned): the history the `nl`
 * policy reads, `natlang profile` counts and `natlang improve DIR` reads as a labelled hard case.
 */
export type EscalationNote = { function: string; escalated: boolean; unavailable?: true; reason?: string; to?: string;
  agreed?: boolean; fallback_call?: string; local_ms: number; fallback_ms?: number };

type HistoryStore = { annotationsOf(kind: string, options?: { limit?: number }): readonly { call_id: string; value: unknown }[] };
const isHistoryStore = (store: unknown): store is HistoryStore =>
  !!store && typeof (store as HistoryStore).annotationsOf === 'function';

/** Running totals of escalate runs of one function. */
type Tally = { runs: number; escalated: number; disagreed: number; localMs: number; localTimed: number; fallbackMs: number; fallbackTimed: number };
const emptyTally = (): Tally => ({ runs: 0, escalated: 0, disagreed: 0, localMs: 0, localTimed: 0, fallbackMs: 0, fallbackTimed: 0 });

function tallyOf(note: EscalationNote): Tally {
  const tally = emptyTally();
  tally.runs = 1;
  if (typeof note.local_ms === 'number') { tally.localMs = note.local_ms; tally.localTimed = 1; }
  if (note.escalated) {
    tally.escalated = 1;
    if (note.agreed === false) tally.disagreed = 1;
    if (typeof note.fallback_ms === 'number') { tally.fallbackMs = note.fallback_ms; tally.fallbackTimed = 1; }
  }
  return tally;
}

const addTally = (into: Tally, more: Tally): Tally => {
  for (const key of Object.keys(into) as (keyof Tally)[]) into[key] += more[key];
  return into;
};

function recordOf(tally: Tally): EscalationRecord {
  const round = (value: number) => Math.round(value * 1e4) / 1e4;
  const history: EscalationHistory = { totalCalls: tally.runs, ...(tally.runs ? { escalationRate: round(tally.escalated / tally.runs) } : {}),
    ...(tally.escalated ? { disagreementRate: round(tally.disagreed / tally.escalated) } : {}) };
  const local = tally.localTimed ? tally.localMs / tally.localTimed : undefined;
  const fallback = tally.fallbackTimed ? tally.fallbackMs / tally.fallbackTimed : undefined;
  const cost: TierCost | undefined = local !== undefined || fallback !== undefined ? { ...(local !== undefined ? { local: Math.round(local) } : {}),
    ...(fallback !== undefined ? { fallback: Math.round(fallback) } : {}),
    ...(local && fallback !== undefined ? { ratio: round(fallback / local) } : {}) } : undefined;
  return { history, ...(cost ? { cost } : {}) };
}

/**
 * The escalation history of the function `functionId` (a definition id) in a call store, from the `escalation` notes
 * on its local calls: its most recent `limit` escalate runs, how many escalated, how many of those the fallback answered differently, and the mean milliseconds
 * of the local and the fallback call. Undefined when the store cannot be read this way.
 */
export function escalationHistoryFromStore(store: unknown, functionId: string, limit = 500): EscalationRecord | undefined {
  const tally = tallyFromStore(store, functionId, limit);
  return tally && recordOf(tally);
}

function tallyFromStore(store: unknown, functionId: string, limit = 500): Tally | undefined {
  if (!isHistoryStore(store)) return undefined;
  const tally = emptyTally();
  for (const row of store.annotationsOf('escalation', { limit })) {
    const note = row.value as EscalationNote | undefined;
    if (note && typeof note === 'object' && note.function === functionId) addTally(tally, tallyOf(note));
  }
  return tally;
}

export type Escalated<R> = Decision<R> & {
  /** Whether the stronger model settled this call. */
  escalated: boolean;
  /** Why the call escalated (`'confidence'`, `'risk:<label>'`, `'policy'`), when `escalated` is true; when it should have
   * but the runtime has no model `to`, `escalated` is false and this starts with `unavailable:`. */
  reason?: string;
  /** The local model's initial argmax before escalation, when `escalated` is true. */
  initialValue?: R;
};

/**
 * Wrap a `readout: decision` natlang function with confidence- and risk-gated escalation.
 * Most calls settle locally; only uncertain or high-risk calls invoke `fn.on({ model })` (keeping `fn`'s context).
 */
export function escalate<A extends unknown[], R>(fn: NatlangFunction<A, R>, options: EscalateOptions<R>): (...args: A) => Promise<Escalated<R>> {
  if (typeof fn !== 'function' || typeof fn.on !== 'function')
    throw stdError('std-arguments', 'escalate(fn, { to | fallback, floor?, risk?, policy? }) needs a natlang function with .on({ model }).');
  const rawTo = typeof options?.to === 'string' ? options.to.trim() : '';
  const rawFallback = typeof options?.fallback === 'string' ? options.fallback.trim() : '';
  if (rawTo && rawFallback && rawTo !== rawFallback)
    throw stdError('std-arguments', `escalate received conflicting to (${JSON.stringify(rawTo)}) and fallback (${JSON.stringify(rawFallback)}); pass one target model name.`);
  const to = rawTo || rawFallback;
  if (!to) throw stdError('std-arguments', 'escalate(fn, { to | fallback }) needs a nonempty target model name.');
  const floor = options.floor ?? 0.8;
  if (!(floor >= 0 && floor <= 1)) throw stdError('std-arguments', 'escalate floor is a probability between 0 and 1.');
  for (const [label, threshold] of Object.entries(options.risk ?? {}))
    if (!(threshold >= 0 && threshold <= 1))
      throw stdError('std-arguments', `escalate risk floor for ${JSON.stringify(label)} must be a probability between 0 and 1.`);

  const crispReason = (decision: Decision<R>): string | undefined => {
    // Check asymmetric risk floors before the general confidence floor so high-risk escalations keep their specific reason.
    if (options.risk) {
      for (const [label, threshold] of Object.entries(options.risk)) {
        const match = decision.probabilities.find(entry => String(entry.value) === label || JSON.stringify(entry.value) === label);
        if (match && match.probability >= threshold) return `risk:${label}`;
      }
    }
    if (decision.confidence < floor) return 'confidence';
    return undefined;
  };
  const nlJudge = options.nlPolicy ?? (shippedShouldEscalate as unknown as NonNullable<EscalateOptions<R>['nlPolicy']>);
  // The nl policy reads this function's escalation history from the call store once, then counts this wrapper's runs.
  const functionId = callableMeta(fn)?.definition.id;
  let stored: Tally | undefined;
  const live = emptyTally();
  const recorded = (frame: StdSpan['frame']): EscalationRecord | undefined => {
    if (!functionId) return undefined;
    stored ??= tallyFromStore(frame.task.runtime.callStore(), functionId) ?? emptyTally();
    return recordOf(addTally({ ...stored }, live));
  };
  const policyFacts = (frame: StdSpan['frame']): { history?: EscalationHistory; cost?: TierCost } => {
    const wantHistory = options.history === undefined, wantCost = options.cost === undefined;
    const measured = wantHistory || wantCost ? recorded(frame) : undefined;
    const history = options.history === false ? undefined : options.history ?? measured?.history;
    const cost = options.cost === false ? undefined : options.cost ?? measured?.cost;
    return { ...(history ? { history } : {}), ...(cost ? { cost } : {}) };
  };
  const decidePolicy = pluggable<[Decision<R>, StdSpan['frame']], string | undefined>(
    {
      crisp: decision => crispReason(decision),
      nl: async (decision, frame) => {
        const facts = policyFacts(frame);
        return (await nlJudge(decision, floor, options.risk, facts.history, facts.cost)) ? 'policy' : undefined;
      },
    },
    options.policy,
    { default: 'crisp', serve: 'crisp', name: 'std.escalate', same: (a, b) => Boolean(a) === Boolean(b) },
  );

  const fallback = fn.on({ model: to });
  return async (...args: A): Promise<Escalated<R>> => {
    const span = new StdSpan('escalate');
    return span.around(async () => {
      const localStarted = Date.now();
      const decision = await decideInFrame(span.frame, fn, args);
      const localMs = Date.now() - localStarted;
      const [localCall] = span.stage('local', { confidence: decision.confidence, scored: decision.scored, ms: localMs }).slice(-1);
      const reason = options.shouldEscalate
        ? ((await span.run(() => options.shouldEscalate!(decision))) ? 'policy' : undefined)
        : await span.run(() => decidePolicy(decision, span.frame));
      // This run joins the history once the store has been read (earlier runs are read from the store itself).
      // The run's note goes on its local call; it joins this wrapper's live history once the store has been read
      // (earlier runs are read from the store itself).
      const note = (fields: Omit<EscalationNote, 'function' | 'local_ms'>) => {
        if (!functionId) return;
        const value: EscalationNote = { function: functionId, local_ms: localMs, ...fields };
        if (stored) addTally(live, tallyOf(value));
        if (localCall) try { span.frame.task.runtime.callStore()?.annotate?.(localCall, 'escalation', value, 'std', false); }
        catch { /* recording never fails the call */ }
      };
      if (!reason) { note({ escalated: false }); return { ...decision, escalated: false }; }
      // `fn.on({ model })` runs a name the runtime lacks on the default model: that would be the same answer again,
      // reported as escalated. Without the model the local answer stands, marked unavailable (a local run of a program
      // written for a fallback still works).
      if (!span.frame.task.runtime.options.models?.[to]) {
        span.emit('decision_escalate', { to, reason, unavailable: true, confidence: decision.confidence, initial_value: decision.value });
        note({ escalated: false, unavailable: true, reason, to });
        return { ...decision, escalated: false, reason: `unavailable: this runtime has no model ${to} (${reason})` };
      }
      span.emit('decision_escalate', { to, reason, confidence: decision.confidence, initial_value: decision.value });
      const fallbackStarted = Date.now();
      const value = await span.run(() => fallback(...args));
      const fallbackMs = Date.now() - fallbackStarted;
      // Whether the fallback agreed with the local answer: the history the nl policy reads (a disagreement is a
      // labelled hard case for this function).
      const agreed = stableHash(value) === stableHash(decision.value);
      const [fallbackCall] = span.stage('escalate', { to, reason, agreed, ms: fallbackMs }).slice(-1);
      note({ escalated: true, reason, to, agreed, fallback_ms: fallbackMs, ...(fallbackCall ? { fallback_call: fallbackCall } : {}) });
      return { ...decision, value, escalated: true, reason, initialValue: decision.value };
    }, result => result.escalated ? `escalated to ${to} (${result.reason})` : `local (p=${result.confidence.toFixed(2)})`);
  };
}

// --- withContext ----------------------------------------------------------------------------------------------------

/**
 * Bind data or file-loaded helper entries onto `fn`, preserving `fn`'s own companion folder and existing context data.
 * Sugar over `fn.in(Context.ofCallable(fn).with(entries))`.
 */
export function withContext<F extends NatlangFunction<any, any>>(fn: F, entries: Record<string, unknown>): F {
  if (!isNatlangCallable(fn))
    throw stdError('std-arguments', 'withContext(fn, entries) needs a natlang function.');
  if (!entries || typeof entries !== 'object' || Array.isArray(entries))
    throw stdError('std-arguments', 'withContext(fn, entries) needs a record of context entries to bind.');
  return (fn as unknown as { in(context: Context): F }).in(Context.ofCallable(fn).with(entries));
}

// --- compileRule ----------------------------------------------------------------------------------------------------

export type RuleExample<I, O> = { input: I; expected: O; name?: string };
export type CompileRuleOptions<I, O> = {
  /** Labeled cases the compiled rule must satisfy before it is returned. */
  examples?: readonly RuleExample<I, O>[];
  /** Additional exact verifier over the compiled rule function. */
  check?: (fn: (item: I) => Promise<O>) => Maybe<Problems>;
  /** Maximum repair attempts when `examples` or `check` fail (default 2). */
  attempts?: number;
  /** Custom comparator for `examples` (default: canonical JSON equality). */
  same?: (received: O, expected: O) => boolean;
  /** Replacement for the shipped `compileRuleFn(rule, problems?)` factory. */
  compiler?: (rule: string, problems?: string[]) => Promise<(item: I) => Promise<O>>;
};

/**
 * Compile a plain-language rule into a reusable typed judgment function `(item: I) => Promise<O>`.
 * When `examples` or `check` are supplied, the synthesized function is verified and repaired with `semanticRepair`
 * until every check passes (throwing `std-rule-unverified` if it still fails after `attempts`).
 */
export async function compileRule<I = unknown, O = unknown>(rule: string, options: CompileRuleOptions<I, O> = {}): Promise<(item: I) => Promise<O>> {
  if (typeof rule !== 'string' || !rule.trim())
    throw stdError('std-arguments', 'compileRule(rule, options?) needs a nonempty rule string.');
  const compiler = options.compiler ?? (shippedCompileRule as unknown as (rule: string, problems?: string[]) => Promise<(item: I) => Promise<O>>);
  const examples = options.examples ?? [];
  const same = options.same ?? ((a: O, b: O) => stableHash(a) === stableHash(b));
  const attempts = options.attempts ?? 2;
  const span = new StdSpan('compileRule');
  return span.around(async () => {
    const first = await span.run(() => compiler(rule));
    span.stage('compile', { attempt: 0 });
    if (!examples.length && !options.check) return first;
    let version = 0;
    const verify = async (candidate: (item: I) => Promise<O>): Promise<string[]> => {
      const problems: string[] = [];
      if (examples.length) {
        const received = await Promise.all(examples.map(example => span.run(() => candidate(example.input))));
        examples.forEach((example, index) => {
          if (!same(received[index]!, example.expected)) {
            const label = example.name ?? `example ${index + 1}`;
            problems.push(`${label}: expected ${JSON.stringify(example.expected)}, got ${JSON.stringify(received[index])}.`);
          }
        });
      }
      if (options.check) problems.push(...await span.run(() => options.check!(candidate)));
      return problems;
    };
    const repaired = await semanticRepair(first, {
      check: verify,
      fix: (_candidate, problems) => compiler(rule, problems),
      attempts,
      identity: () => `rule-v${++version}`,
    });
    span.stage('verify', { ok: repaired.ok, attempts: repaired.attempts, problems: repaired.problems.length });
    if (!repaired.ok)
      throw stdError('std-rule-unverified', `compileRule could not satisfy its checks after ${repaired.attempts} retries: ${repaired.problems.join(' ')}`);
    return repaired.value;
  }, () => 'compiled rule');
}
