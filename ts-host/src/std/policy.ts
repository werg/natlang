/**
 * Policy & higher-order combinators (plans/semantic-fp/03_SEMANTIC_STDLIB_AND_PATTERNS.md S3;
 * ts-host/std/DECOMPOSITION.md):
 * - `escalate`: confidence- and risk-gated escalation from a local `readout: decision` function to a stronger model;
 * - `withContext`: ergonomic sugar over `fn.in(Context.ofCallable(fn).with(entries))` preserving companion helpers;
 * - `compileRule`: turn a written rule into a typed semantic function, validated and repaired against examples.
 */
import { Context } from '../runtime/contexts.js';
import { decideInFrame, type Decision } from '../runtime/runtime.js';
import { isNatlangCallable } from '../runtime/callable.js';
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
  /** Optional prior escalation statistics supplied to the `nl` policy. */
  history?: EscalationHistory;
  /** Optional local vs fallback tier cost metadata supplied to the `nl` policy. */
  cost?: TierCost;
};

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
  const decidePolicy = pluggable<[Decision<R>], string | undefined>(
    {
      crisp: decision => crispReason(decision),
      nl: async decision => (await nlJudge(decision, floor, options.risk, options.history, options.cost)) ? 'policy' : undefined,
    },
    options.policy,
    { default: 'crisp', serve: 'crisp', name: 'std.escalate', same: (a, b) => Boolean(a) === Boolean(b) },
  );

  const fallback = fn.on({ model: to });
  return async (...args: A): Promise<Escalated<R>> => {
    const span = new StdSpan('escalate');
    return span.around(async () => {
      const decision = await decideInFrame(span.frame, fn, args);
      span.stage('local', { confidence: decision.confidence, scored: decision.scored });
      const reason = options.shouldEscalate
        ? ((await span.run(() => options.shouldEscalate!(decision))) ? 'policy' : undefined)
        : await span.run(() => decidePolicy(decision));
      if (!reason) return { ...decision, escalated: false };
      // `fn.on({ model })` runs a name the runtime lacks on the default model: that would be the same answer again,
      // reported as escalated. Without the model the local answer stands, marked unavailable (a local run of a program
      // written for a fallback still works).
      if (!span.frame.task.runtime.options.models?.[to]) {
        span.emit('decision_escalate', { to, reason, unavailable: true, confidence: decision.confidence, initial_value: decision.value });
        return { ...decision, escalated: false, reason: `unavailable: this runtime has no model ${to} (${reason})` };
      }
      span.emit('decision_escalate', { to, reason, confidence: decision.confidence, initial_value: decision.value });
      const value = await span.run(() => fallback(...args));
      span.stage('escalate', { to, reason });
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
