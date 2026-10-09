/**
 * The tiered execution engine (plans/TIERED_ENGINE.md). One natural-language function runs at a ladder of tiers, chosen
 * per call: the highest admissible tier serves, its output is verified, and a failure deoptimizes to the next lower tier.
 * Evidence (served, deopted, shadow comparisons) moves a tier between `shadow`, `active` and `demoted` by the same
 * rule and settings as the compilation cases (calls/evidence.ts); the ladder is built from what the specializer stored.
 *
 * This module is platform-neutral and knows nothing of the kernel: the kernel hands it a `TierHost` whose `run` performs
 * one attempt of the existing dispatch (runtime/kernel.ts runDefinition) under a `TierAttempt` plan.
 */
import type { CallCapture } from './recorder.js';
import type { LoadedCase } from './compilations.js';
import { Deopt, isDeopt } from './dispatch.js';
import { canonicalValue } from '../native/refinement.js';
import { crispPolicy, failureKind, foldEvent, moveTier, newTierEvidence, ruleOf, setTierState, summaryOfTier, type Evidence, type EvidenceRule,
  type EvidenceState, type PromotionDecision, type TierEvidence } from './evidence.js';
import { DEFAULT_SETTINGS, type CallStoreSettings } from './types.js';

/**
 * 0 the teacher model, 1 the student model, 2 specialized natural language (trace-specialized guidance for the student),
 * 3 crisp code behind guards (the specializer's compiled cases, and registered implementations), 4 Neuralese.
 */
export type TierLevel = 0 | 1 | 2 | 3 | 4;
export type TierStateName = 'active' | 'shadow' | 'demoted';

/** How the kernel runs one attempt. Absent from `InvokeOptions` means the kernel behaves exactly as before. */
export type TierAttempt = {
  tier: string;
  /** Nothing may serve but a crisp case: when none does, the attempt ends in `Deopt` before any model runs. */
  strict: boolean;
  /** Whether the stored compilation's cases may serve (default true). */
  compiled?: boolean;
  /** Model profile name (`options.models`) for the agent, or to continue a hand-off. Default: the function's own. */
  model?: string;
  /** Specialized guidance for the model, added to its system prompt for this attempt (tier 2). */
  guidance?: string;
  /** A tier's own crisp case, tried before stored compilations; admitted by the tier's guard already. */
  cases?: LoadedCase[];
  /** Set by a tier's case when it declined the call after running (its output check failed): why. */
  declined?: string;
  /** Set by the kernel: the call's record, so the engine can read which executor served it. */
  capture?: CallCapture;
};

export type TierInput = {
  name: string;
  /** The call's arguments by parameter name. */
  args: Record<string, unknown>;
  /** Run one attempt through the kernel under a plan. */
  run(plan: Omit<TierAttempt, 'tier'>): Promise<unknown>;
  hasModel(name: string): boolean;
  /** Whether a call store records this call (tier 3 is served through its records). */
  recorded: boolean;
};

export type Verdict = boolean | string;
type Check = (args: Record<string, unknown>, output: unknown) => Verdict | Promise<Verdict>;

export interface Tier {
  readonly id: string;
  readonly level: TierLevel;
  /** The guard over inputs, evaluated before serving. */
  canServe(input: TierInput): boolean | Promise<boolean>;
  serve(input: TierInput): Promise<unknown>;
  /** The check over outputs after serving: true, or false / a reason string to deoptimize. */
  verify(input: TierInput, output: unknown): Verdict | Promise<Verdict>;
  /** Produce the output without serving it, for shadow comparison. Absent: the tier cannot be shadowed. */
  shadow?(input: TierInput): Promise<unknown>;
}

// --- Tiers -----------------------------------------------------------------------------------------------------------

/** Tiers 0 and 1: a model profile interprets the source. `model` undefined is the function's own model. */
export class ModelTier implements Tier {
  readonly id: string;
  constructor(readonly level: 0 | 1, readonly model: string | undefined, private readonly check?: Check, shadowable = false) {
    this.id = `tier${level}`;
    if (shadowable) this.shadow = input => input.run({ strict: false, compiled: false, ...(model ? { model } : {}) });
  }
  shadow?: (input: TierInput) => Promise<unknown>;
  canServe(input: TierInput): boolean { return !this.model || input.hasModel(this.model); }
  serve(input: TierInput): Promise<unknown> { return input.run({ strict: false, compiled: false, ...(this.model ? { model: this.model } : {}) }); }
  verify(input: TierInput, output: unknown): Verdict | Promise<Verdict> { return this.check ? this.check(input.args, output) : true; }
}

/**
 * Tier 2: the student model under the specializer's recorded guidance (trace-specialized instructions or examples,
 * the compilation's `instructions.md`). The specializer does not write that file yet, so the tier appears in a ladder
 * only when a compilation carries it.
 */
export class SpecializedTier implements Tier {
  readonly id = 'tier2';
  readonly level = 2;
  constructor(readonly model: string | undefined, readonly guidance: string, private readonly check?: Check, shadowable = false) {
    if (shadowable) this.shadow = input => input.run(this.plan());
  }
  shadow?: (input: TierInput) => Promise<unknown>;
  private plan(): Omit<TierAttempt, 'tier'> { return { strict: false, compiled: false, guidance: this.guidance, ...(this.model ? { model: this.model } : {}) }; }
  canServe(input: TierInput): boolean { return !this.model || input.hasModel(this.model); }
  serve(input: TierInput): Promise<unknown> { return input.run(this.plan()); }
  verify(input: TierInput, output: unknown): Verdict | Promise<Verdict> { return this.check ? this.check(input.args, output) : true; }
}

/**
 * Tier 3, stored cases: the compilation the specializer wrote for this function (calls/compilations.ts). Its cases are
 * crisp TypeScript with input guards, each with its own state in the store; the kernel admits them (`admit`) and
 * serves the first active case whose guard holds. Present in a ladder only when the compilation has an active case.
 */
export class CrispTier implements Tier {
  readonly id = 'tier3';
  readonly level = 3;
  constructor(private readonly fallbackModel?: string, private readonly check?: Check) {}
  canServe(input: TierInput): boolean { return input.recorded; }
  serve(input: TierInput): Promise<unknown> {
    return input.run({ strict: true, compiled: true, ...(this.fallbackModel ? { model: this.fallbackModel } : {}) });
  }
  verify(input: TierInput, output: unknown): Verdict | Promise<Verdict> { return this.check ? this.check(input.args, output) : true; }
}

/** A hand-written crisp implementation of a function, as a registered tier-3 case. */
export type CrispImplementation<A = Record<string, unknown>, R = unknown> = {
  /** Name of the implementation (appears in the case hash and the trace). */
  id: string;
  /** Input-shape guard: a synchronous predicate over the arguments. */
  when(args: A): boolean;
  run(args: A): R | Promise<R>;
  /** Check over the output; false or a reason deoptimizes before the output is served. */
  verify?(args: A, output: R): Verdict | Promise<Verdict>;
  /** Whether running it twice is harmless, so shadow replay may call it beside a lower tier. */
  pure?: boolean;
};

/**
 * Tier 3, registered implementation: crisp code behind a guard, served through the kernel's crisp-case path (type check,
 * refinements, records, hand-off after effects) as a synthetic case whose hash starts with `tier:`. Not a separate
 * level from the stored cases: the same kind of artifact, with its own state in the tier ledger because the store
 * keeps no case row for it.
 */
export class ImplementationTier implements Tier {
  readonly id: string;
  readonly level = 3;
  constructor(readonly implementation: CrispImplementation, private readonly fallbackModel?: string) {
    this.id = `tier3:${implementation.id}`;
    if (implementation.pure) this.shadow = async input => implementation.run(input.args);
  }
  shadow?: (input: TierInput) => Promise<unknown>;
  canServe(input: TierInput): boolean {
    if (!input.recorded) return false;
    try { return this.implementation.when(input.args) === true; } catch { return false; }
  }
  async serve(input: TierInput): Promise<unknown> {
    const implementation = this.implementation;
    let attempt: TierAttempt | undefined;
    const item: LoadedCase = { hash: `tier:${implementation.id}`, position: 0, tier: 'active', services: [],
      when: () => true,
      run: async args => {
        const output = await implementation.run(args);
        const verdict = implementation.verify ? await implementation.verify(args, output) : true;
        if (verdict !== true) {
          const reason = typeof verdict === 'string' ? verdict : 'the output check failed';
          if (attempt) attempt.declined = reason;
          throw new Deopt(reason);
        }
        return output;
      } };
    const plan: Omit<TierAttempt, 'tier'> = { strict: true, compiled: false, cases: [item], ...(this.fallbackModel ? { model: this.fallbackModel } : {}) };
    // The engine completes the plan with the tier id; the case reaches it through the shared object.
    const running = input.run(plan);
    attempt = (input as TierInput & { last?: TierAttempt }).last;
    return running;
  }
  verify(): Verdict { return true; }
}

/** Tier 4: a Neuralese soft program or adapter. Interface only: it never serves until a learning-continuum artifact exists. */
export class NeuraleseTier implements Tier {
  readonly id = 'tier4';
  readonly level = 4;
  canServe(): boolean { return false; }
  async serve(): Promise<unknown> { throw new Error('tier 4 (Neuralese soft program) is not implemented'); }
  verify(): Verdict { return true; }
}

// --- Configuration ---------------------------------------------------------------------------------------------------

/**
 * Per function: which model profiles serve, and a few per-tier options. Thresholds are not here: promotion and demotion
 * use the store's settings (`acceptanceBound`, `promotionComparisons`, `promotionLiveComparisons`, `promotionPolicy`),
 * the same ones the compilation cases use. Tiers 2 and 3 come from the specializer's compilation; nothing names them.
 */
export type TierSettings = {
  /** Model profile (`options.models`) of the frontier/teacher tier. Default: the function's own model. */
  tier0?: { model?: string };
  /** The student tier: a model profile. `start: 'shadow'` makes it earn promotion by shadow evidence. */
  tier1?: { model: string; start?: TierStateName; verify?: Check; shadow?: boolean };
  /** Specialized natural language (the compilation's `instructions.md`), run on the student. Starts in `shadow`. */
  specialized?: { enabled?: boolean; start?: TierStateName; verify?: Check; shadow?: boolean };
  /**
   * Crisp code. The compilation's cases join the ladder by themselves (`enabled: false` turns them off);
   * `implementations` registers hand-written ones, which start in `shadow` unless told otherwise.
   */
  crisp?: { enabled?: boolean; verify?: Check; implementations?: CrispImplementation[]; start?: TierStateName };
  /** Interface stub. */
  tier4?: Tier;
  /** Fraction of served calls that also run each shadow/demoted tier above them (default 0). */
  shadowRate?: number;
  /** Whether two outputs agree for shadow comparison. Default: equal canonical values. */
  same?: (served: unknown, shadow: unknown) => boolean;
};

/** What the specializer stored for a function: how many of its compiled cases are active, and its specialized guidance if any. */
export type SpecializerOutput = { cases: number; guidance?: string };

export type TierEngineOptions = { functions: Record<string, TierSettings>; random?: () => number };

// --- Evidence --------------------------------------------------------------------------------------------------------

export type TierEventName = 'served' | 'deopt' | 'shadow_equal' | 'shadow_worse' | 'promoted' | 'demoted';
export type TierEvent = { fn: string; tier: string; event: TierEventName;
  /** For `deopt`: `guard` (the tier declined; not a failure), `verify`/`error` (the tier was wrong), `infrastructure` (bad luck). */
  kind?: 'guard' | 'verify' | 'error' | 'infrastructure'; reason?: string; to?: string; call_id?: string | null;
  /** The tier's state when the event happened (so a store can be folded without the configuration). */
  state?: EvidenceState;
  /** The call belongs to the failed attempt itself (its cost was wasted). */
  own?: boolean; evidence?: Record<string, unknown> };

/**
 * The tiers' running evidence. Counters fold the events (calls/evidence.ts `foldEvent`); promotion and demotion are
 * decisions of the shared policy, applied here as events arrive (`inline`) or later by `reviewPromotions` from the
 * specializer loop (policy `nl`), and read back from the store as `promoted` / `demoted` events.
 */
export class TierLedger {
  private readonly rows = new Map<string, TierEvidence>();
  private readonly lastCall = new Map<string, string>();
  private readonly seen = new WeakMap<object, { rows: number; at: number }>();
  state(fn: string, tier: string, initial: TierStateName): TierStateName { return this.row(fn, tier, initial).state as TierStateName; }
  stats(fn: string, tier: string): Evidence | undefined { return this.rows.get(`${fn}\0${tier}`)?.evidence; }
  /** Every (function, tier) with evidence, with the last call that carried an event for it. */
  entries(): { fn: string; tier: string; row: TierEvidence; call: string | undefined }[] {
    return [...this.rows].map(([key, row]) => { const [fn, tier] = key.split('\0') as [string, string]; return { fn, tier, row, call: this.lastCall.get(key) }; });
  }
  row(fn: string, tier: string, initial: EvidenceState = 'active'): TierEvidence {
    const key = `${fn}\0${tier}`;
    let row = this.rows.get(key);
    if (!row) this.rows.set(key, row = newTierEvidence(initial));
    return row;
  }
  /**
   * Fold an event in. With `inline`, the crisp policy decides straight away and the derived `promoted` / `demoted`
   * event is returned; otherwise the decision is left to the policy loop. `promoted` / `demoted` events are decisions
   * already taken: they set the state.
   */
  apply(ev: TierEvent, rule: EvidenceRule, initial: TierStateName, inline = true): TierEvent | undefined {
    const row = this.row(ev.fn, ev.tier, ev.state ?? initial);
    if (ev.call_id) this.lastCall.set(`${ev.fn}\0${ev.tier}`, ev.call_id);
    if (ev.event === 'promoted') { setTierState(row, 'active'); return; }
    if (ev.event === 'demoted') { setTierState(row, 'demoted'); return; }
    foldEvent(row, ev);
    if (!inline || ev.tier === 'tier3' || ev.event === 'served' || (ev.event === 'deopt' && ev.kind === 'guard')) return;
    const summary = summaryOfTier(`${ev.fn}/${ev.tier}`, row, rule);
    const verdict = crispPolicy(summary);
    const evidence = { ...row.evidence };
    const moved = moveTier(row, verdict.decision);
    if (moved) return { fn: ev.fn, tier: ev.tier, event: moved === 'active' ? 'promoted' : 'demoted', evidence: { ...evidence, reason: verdict.reason, by: 'crisp' } };
  }
  /** Apply a policy's decision about a tier (reviewPromotions); returns the event to record when the state moved. */
  decide(fn: string, tier: string, verdict: PromotionDecision, by: string): TierEvent | undefined {
    const row = this.row(fn, tier);
    const evidence = { ...row.evidence };
    const moved = moveTier(row, verdict.decision);
    if (moved) return { fn, tier, event: moved === 'active' ? 'promoted' : 'demoted', evidence: { ...evidence, reason: verdict.reason, by } };
  }
  /**
   * Rebuild state from the events a store kept (annotations of kind `tier`): everything the first time, then, at most every
   * 5 s, only the decisions others took since (a policy loop in another process). `fn` limits the fold to one function.
   */
  hydrate(store: object | undefined, fn: string | undefined, initial: (tier: string) => TierStateName, rule: EvidenceRule): void {
    if (!store) return;
    const known = this.seen.get(store), now = Date.now();
    if (known && now - known.at < 5000) return;
    try {
      const rows = (store as { tierRows?: () => { call_id?: string; value: unknown }[] }).tierRows?.() ?? [];
      for (let index = known?.rows ?? 0; index < rows.length; index++) {
        const ev = rows[index]!.value as TierEvent;
        if (!ev || (fn && ev.fn !== fn)) continue;
        if (known && ev.event !== 'promoted' && ev.event !== 'demoted') continue;
        if (known && this.rows.get(`${ev.fn}\0${ev.tier}`)?.state === (ev.event === 'promoted' ? 'active' : 'demoted')) continue;
        this.apply({ ...ev, call_id: ev.call_id ?? rows[index]!.call_id }, rule, initial(ev.tier), false);
      }
      this.seen.set(store, { rows: rows.length, at: now });
    } catch { /* evidence is advisory; a store without it starts from configuration */ }
  }
}

// --- Engine ----------------------------------------------------------------------------------------------------------

export class TierEngine {
  readonly ledger = new TierLedger();
  readonly random: () => number;
  constructor(readonly options: TierEngineOptions) { this.random = options.random ?? Math.random; }
  settingsFor(name: string): TierSettings | undefined { return this.options.functions[name] ?? this.options.functions['*']; }
}

export type TierStoreLike = { annotate?(callId: string, kind: string, value: unknown, source?: string, pin?: boolean): void;
  settings?(): CallStoreSettings };

export type TierHost = {
  name: string;
  args: Record<string, unknown>;
  store?: TierStoreLike;
  /** What the specializer stored for this function revision (its active cases, its specialized guidance). */
  specialized?(): SpecializerOutput;
  /** One attempt of the kernel's dispatch under a plan; the plan's `capture` is set by the kernel. */
  run(plan: TierAttempt): Promise<unknown>;
  hasModel(name: string): boolean;
  /** Record trace events for this call. */
  emit(events: Record<string, unknown>[]): void;
};

const initialState = (settings: TierSettings, tier: string): TierStateName =>
  tier === 'tier1' ? settings.tier1?.start ?? 'active' : tier === 'tier2' ? settings.specialized?.start ?? 'shadow' :
  tier.startsWith('tier3:') ? settings.crisp?.start ?? 'shadow' : 'active';

/**
 * The ladder for a function, highest tier first, built from what the specializer produced (`found`) and the named model
 * profiles. Tier 0 is always last: it is the terminal tier.
 */
export function ladderOf(settings: TierSettings, found: SpecializerOutput = { cases: 0 }): Tier[] {
  const lower = settings.tier1?.model ?? settings.tier0?.model;
  const tiers: Tier[] = [];
  if (settings.tier4) tiers.push(settings.tier4);
  for (const implementation of settings.crisp?.implementations ?? []) tiers.push(new ImplementationTier(implementation, lower));
  if (found.cases > 0 && settings.crisp?.enabled !== false) tiers.push(new CrispTier(lower, settings.crisp?.verify));
  if (found.guidance && settings.specialized?.enabled !== false)
    tiers.push(new SpecializedTier(settings.tier1?.model ?? settings.tier0?.model, found.guidance, settings.specialized?.verify, settings.specialized?.shadow));
  if (settings.tier1) tiers.push(new ModelTier(1, settings.tier1.model, settings.tier1.verify, settings.tier1.shadow));
  tiers.push(new ModelTier(0, settings.tier0?.model));
  return tiers;
}

const message = (error: unknown): string => error instanceof Error ? error.message : String(error);

/**
 * Serve one call: try each admissible tier from the highest, verify, and deoptimize downward. Tier 0 is terminal: its
 * errors propagate and its output is served unverified-by-fallback (a failed check is recorded, not retried).
 */
export async function runTiered(engine: TierEngine, settings: TierSettings, host: TierHost): Promise<unknown> {
  const fn = host.name;
  const ladder = ladderOf(settings, host.specialized?.());
  const initial = (tier: string) => initialState(settings, tier);
  let storeSettings: CallStoreSettings = DEFAULT_SETTINGS;
  try { storeSettings = host.store?.settings?.() ?? DEFAULT_SETTINGS; } catch { /* defaults */ }
  const rule = ruleOf(storeSettings);
  const inline = storeSettings.promotionPolicy !== 'nl';
  engine.ledger.hydrate(host.store, fn, initial, rule);
  const events: TierEvent[] = [];
  const note = (ev: Omit<TierEvent, 'fn' | 'state'>): void => {
    const full: TierEvent = { fn, ...ev, state: engine.ledger.state(fn, ev.tier, initial(ev.tier)) };
    events.push(full);
    const derived = engine.ledger.apply(full, rule, initial(full.tier), inline);
    if (derived) events.push({ ...derived, state: derived.event === 'promoted' ? 'shadow' : 'active' });
  };
  const input: TierInput & { last?: TierAttempt } = { name: fn, args: host.args, recorded: !!host.store, hasModel: host.hasModel,
    run: plan => { const attempt: TierAttempt = { ...plan, tier: current }; input.last = attempt; return host.run(attempt); } };
  let current = '';
  const lastPlan = (): TierAttempt | undefined => input.last;
  const terminal = ladder[ladder.length - 1]!;
  let lastCall: string | null = null;
  const flush = (served?: string | null): void => {
    const fallback = served ?? lastCall;
    for (const ev of events) {
      const call = ev.call_id ?? fallback;
      try { if (call) host.store?.annotate?.(call, 'tier', { ...ev, call_id: call }, 'tiers', false); } catch { /* recording never fails a call */ }
    }
    host.emit(events.map(ev => ({ kind: ev.event === 'served' ? 'tier_served' : ev.event === 'deopt' ? 'tier_deopt' :
      ev.event === 'promoted' ? 'tier_promoted' : ev.event === 'demoted' ? 'tier_demoted' : 'tier_shadow', function: fn, tier: ev.tier,
      ...(ev.kind ? { reason_kind: ev.kind } : {}), ...(ev.reason ? { reason: ev.reason } : {}), ...(ev.to ? { to: ev.to } : {}),
      ...(ev.evidence ? { evidence: ev.evidence } : {}), ...(ev.event === 'shadow_equal' ? { agree: true } : ev.event === 'shadow_worse' ? { agree: false } : {}) })));
  };
  const nextBelow = (tier: Tier): string => ladder[ladder.indexOf(tier) + 1]?.id ?? terminal.id;
  // The stored cases carry their own state in the store; every other tier's is in the ledger.
  const stateOf = (tier: Tier): TierStateName => tier.id === 'tier3' ? 'active' : engine.ledger.state(fn, tier.id, initial(tier.id));

  for (const tier of ladder) {
    const last = tier === terminal;
    if (!last && stateOf(tier) !== 'active') continue;
    current = tier.id;
    if (!last) {
      let admitted = false;
      try { admitted = (await tier.canServe(input)) === true; } catch (error) { admitted = false; note({ tier: tier.id, event: 'deopt', kind: 'guard', reason: `guard threw: ${message(error)}`, to: nextBelow(tier) }); continue; }
      if (!admitted) { note({ tier: tier.id, event: 'deopt', kind: 'guard', reason: 'the guard does not admit this input', to: nextBelow(tier) }); continue; }
    }
    let output: unknown;
    input.last = undefined;
    try { output = await tier.serve(input); } catch (error) {
      const own = lastPlan()?.capture?.base.callId ?? null;
      if (own) lastCall = own;
      if (last) { flush(); throw error; }
      note(isDeopt(error) && !lastPlan()?.declined ? { tier: tier.id, event: 'deopt', kind: 'guard', reason: message(error), to: nextBelow(tier) } :
        isDeopt(error) ? { tier: tier.id, event: 'deopt', kind: 'verify', reason: message(error), to: nextBelow(tier) } :
        { tier: tier.id, event: 'deopt', kind: failureKind(error), reason: message(error), to: nextBelow(tier), call_id: own, own: !!own });
      continue;
    }
    const capture = lastPlan()?.capture;
    const callId = capture?.base.callId ?? null;
    if (callId) lastCall = callId;
    let servedBy = tier;
    // A crisp case that failed after effects handed the call to the model that follows it.
    if (capture?.executor.kind === 'crisp-agent') {
      note({ tier: tier.id, event: 'deopt', kind: failureKind(capture.executor.case_error ?? ''), reason: capture.executor.case_error ?? 'the case stopped', to: lastPlan()?.model ? 'model' : terminal.id });
      servedBy = ladder.find(item => item.level < 3 && (item as ModelTier).model === lastPlan()?.model) ?? terminal;
    }
    const verdict = await Promise.resolve(servedBy.verify(input, output)).catch(error => `verify threw: ${message(error)}`);
    if (verdict !== true && !last) {
      note({ tier: servedBy.id, event: 'deopt', kind: 'verify', reason: typeof verdict === 'string' ? verdict : 'the output check failed', to: nextBelow(servedBy), call_id: callId, own: !!callId });
      continue;
    }
    if (verdict !== true) note({ tier: servedBy.id, event: 'deopt', kind: 'verify', reason: typeof verdict === 'string' ? verdict : 'the output check failed', call_id: callId, own: false });
    // The stored cases are promoted by the store (the same rule, calls/evidence.ts); the first call one serves is the observable promotion.
    if (servedBy.id === 'tier3' && !engine.ledger.stats(fn, 'tier3')?.served)
      events.push({ fn, tier: 'tier3', event: 'promoted', state: 'active', evidence: { case_hash: capture?.executor.case_hash ?? null, by: 'compilation case tier' } });
    note({ tier: servedBy.id, event: 'served', call_id: callId, ...(capture?.executor.case_hash ? { evidence: { case_hash: capture.executor.case_hash } } : {}) });
    await shadowAbove(engine, settings, ladder, servedBy, input, output, note, stateOf);
    flush(callId);
    return output;
  }
  flush();
  throw new Error(`no tier served ${fn}`);
}

/** Run the shadow or demoted tiers above the one that served, on a sampled fraction, and count agreement. */
async function shadowAbove(engine: TierEngine, settings: TierSettings, ladder: Tier[], served: Tier, input: TierInput, output: unknown,
  note: (ev: Omit<TierEvent, 'fn' | 'state'>) => void, stateOf: (tier: Tier) => TierStateName): Promise<void> {
  const rate = settings.shadowRate ?? 0;
  if (rate <= 0) return;
  const same = settings.same ?? ((a: unknown, b: unknown) => canonicalValue(a) === canonicalValue(b));
  for (const tier of ladder) {
    if (tier === served) break;
    if (!tier.shadow || stateOf(tier) === 'active') continue;
    if (engine.random() >= rate) continue;
    let admitted = false;
    try { admitted = (await tier.canServe(input)) === true; } catch { /* not admitted */ }
    if (!admitted) continue;
    let agree = false;
    try { agree = same(output, await tier.shadow(input)); } catch { agree = false; }
    note({ tier: tier.id, event: agree ? 'shadow_equal' : 'shadow_worse' });
  }
}
