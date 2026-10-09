/**
 * The tiered execution engine (plans/TIERED_ENGINE.md). One natural-language function runs at a ladder of tiers, chosen
 * per call: the highest admissible tier serves, its output is verified, and a failure deoptimizes to the next lower tier.
 * Evidence (served, deopted, shadow comparisons) moves a tier between `shadow`, `active` and `demoted`.
 *
 * This module is platform-neutral and knows nothing of the kernel: the kernel hands it a `TierHost` whose `run` performs
 * one attempt of the existing dispatch (runtime/kernel.ts runDefinition) under a `TierAttempt` plan.
 */
import type { CallCapture } from './recorder.js';
import type { LoadedCase } from './compilations.js';
import { Deopt, isDeopt } from './dispatch.js';
import { canonicalValue } from '../native/refinement.js';

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
  /** Whether a call store records this call (tiers 2 and 3 are served through its records). */
  recorded: boolean;
};

export type Verdict = boolean | string;

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
  constructor(readonly level: 0 | 1, readonly model: string | undefined,
    private readonly check?: (args: Record<string, unknown>, output: unknown) => Verdict | Promise<Verdict>,
    private readonly shadowable = false) {
    this.id = `tier${level}`;
    if (shadowable) this.shadow = input => input.run({ strict: false, compiled: false, ...(model ? { model } : {}) });
  }
  shadow?: (input: TierInput) => Promise<unknown>;
  canServe(input: TierInput): boolean { return !this.model || input.hasModel(this.model); }
  serve(input: TierInput): Promise<unknown> { return input.run({ strict: false, compiled: false, ...(this.model ? { model: this.model } : {}) }); }
  verify(input: TierInput, output: unknown): Verdict | Promise<Verdict> { return this.check ? this.check(input.args, output) : true; }
}

/** Tier 2: the stored compilation's active cases (calls/compilations.ts). Their own case tiers decide admission. */
export class CompiledTier implements Tier {
  readonly id = 'tier2';
  readonly level = 2;
  constructor(private readonly fallbackModel?: string,
    private readonly check?: (args: Record<string, unknown>, output: unknown) => Verdict | Promise<Verdict>) {}
  canServe(input: TierInput): boolean { return input.recorded; }
  serve(input: TierInput): Promise<unknown> {
    return input.run({ strict: true, compiled: true, ...(this.fallbackModel ? { model: this.fallbackModel } : {}) });
  }
  verify(input: TierInput, output: unknown): Verdict | Promise<Verdict> { return this.check ? this.check(input.args, output) : true; }
}

/** A hand-written (later: synthesized) crisp implementation of a function. */
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
 * Tier 3: crisp code behind a guard. It is served through the kernel's crisp-case path (type check, refinements, records,
 * hand-off after effects) as a synthetic case whose hash starts with `tier:`.
 */
export class CrispTier implements Tier {
  readonly id = 'tier3';
  readonly level = 3;
  constructor(readonly implementation: CrispImplementation, private readonly fallbackModel?: string) {
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

export type TierSettings = {
  /** Model profile (`options.models`) of the frontier/teacher tier. Default: the function's own model. */
  tier0?: { model?: string };
  /** The student tier: a model profile. `start: 'shadow'` makes it earn promotion by shadow evidence. */
  tier1?: { model: string; start?: TierStateName; verify?: ModelTierCheck; shadow?: boolean };
  /** Compiled cases of the call store. `enabled: false` turns the tier off. */
  tier2?: { enabled?: boolean; verify?: ModelTierCheck };
  /** Crisp code. Starts in `shadow` unless told otherwise. */
  tier3?: { implementation: CrispImplementation; start?: TierStateName };
  /** Interface stub. */
  tier4?: Tier;
  /** Consecutive failures that demote an active tier (default 3). */
  demoteAfter?: number;
  /** Shadow comparisons a tier needs before promotion (default 10). */
  promoteAfter?: number;
  /** Largest share of worse comparisons a promoted tier may have (default 0.05). */
  bound?: number;
  /** Fraction of served calls that also run each shadow/demoted tier above them (default 0). */
  shadowRate?: number;
  /** Whether two outputs agree for shadow comparison. Default: equal canonical values. */
  same?: (served: unknown, shadow: unknown) => boolean;
};
type ModelTierCheck = (args: Record<string, unknown>, output: unknown) => Verdict | Promise<Verdict>;

export type TierEngineOptions = { functions: Record<string, TierSettings>; random?: () => number };

// --- Evidence --------------------------------------------------------------------------------------------------------

export type TierEventName = 'served' | 'deopt' | 'shadow_equal' | 'shadow_worse' | 'promoted' | 'demoted';
export type TierEvent = { fn: string; tier: string; event: TierEventName;
  /** For `deopt`: `guard` (the tier declined; not a failure), `verify`, `error`. */
  kind?: 'guard' | 'verify' | 'error'; reason?: string; to?: string; call_id?: string | null;
  /** The call belongs to the failed attempt itself (its cost was wasted). */
  own?: boolean; evidence?: Record<string, unknown> };

type Row = { state: TierStateName; served: number; deopts: number; failures: number; compared: number; worse: number };

export class TierLedger {
  private readonly rows = new Map<string, Row>();
  private readonly hydrated = new WeakSet<object>();
  state(fn: string, tier: string, initial: TierStateName): TierStateName { return this.row(fn, tier, initial).state; }
  stats(fn: string, tier: string): Row | undefined { return this.rows.get(`${fn}\0${tier}`); }
  private row(fn: string, tier: string, initial: TierStateName = 'active'): Row {
    const key = `${fn}\0${tier}`;
    let row = this.rows.get(key);
    if (!row) this.rows.set(key, row = { state: initial, served: 0, deopts: 0, failures: 0, compared: 0, worse: 0 });
    return row;
  }
  /** Fold an event in. Returns the derived `promoted` / `demoted` event this one caused, if any. */
  apply(ev: TierEvent, settings: TierSettings, initial: TierStateName): TierEvent | undefined {
    const row = this.row(ev.fn, ev.tier, initial);
    const demoteAfter = settings.demoteAfter ?? 3, promoteAfter = settings.promoteAfter ?? 10, bound = settings.bound ?? 0.05;
    if (ev.event === 'served') { row.served++; row.failures = 0; return; }
    if (ev.event === 'deopt') {
      row.deopts++;
      if (ev.kind === 'guard') return;
      row.failures++;
      if (row.state === 'active' && row.failures >= demoteAfter) {
        row.state = 'demoted'; row.compared = 0; row.worse = 0;
        return { fn: ev.fn, tier: ev.tier, event: 'demoted', evidence: { consecutive_failures: row.failures } };
      }
      return;
    }
    if (ev.event === 'shadow_equal' || ev.event === 'shadow_worse') {
      row.compared++;
      if (ev.event === 'shadow_worse') row.worse++;
      if (row.state !== 'active' && row.compared >= promoteAfter && row.worse <= bound * row.compared) {
        const evidence = { compared: row.compared, worse: row.worse };
        row.state = 'active'; row.failures = 0;
        return { fn: ev.fn, tier: ev.tier, event: 'promoted', evidence };
      }
    }
  }
  /** Rebuild state from the events a store kept (annotations of kind `tier`), once per store. */
  hydrate(store: object | undefined, fn: string, settings: TierSettings, initial: (tier: string) => TierStateName): void {
    if (!store || this.hydrated.has(store)) return;
    this.hydrated.add(store);
    try {
      const rows = (store as { tierRows?: () => { value: unknown }[] }).tierRows?.() ?? [];
      for (const { value } of rows) {
        const ev = value as TierEvent;
        if (!ev || ev.fn !== fn || ev.event === 'promoted' || ev.event === 'demoted') continue;
        this.apply(ev, settings, initial(ev.tier));
      }
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

export type TierStoreLike = { annotate?(callId: string, kind: string, value: unknown, source?: string, pin?: boolean): void };

export type TierHost = {
  name: string;
  args: Record<string, unknown>;
  store?: TierStoreLike;
  /** One attempt of the kernel's dispatch under a plan; the plan's `capture` is set by the kernel. */
  run(plan: TierAttempt): Promise<unknown>;
  hasModel(name: string): boolean;
  /** Record trace events for this call. */
  emit(events: Record<string, unknown>[]): void;
};

const initialState = (settings: TierSettings, tier: string): TierStateName =>
  tier === 'tier1' ? settings.tier1?.start ?? 'active' : tier === 'tier3' ? settings.tier3?.start ?? 'shadow' : 'active';

/** The ladder for a function, highest tier first. Tier 0 is always last: it is the terminal tier. */
export function ladderOf(settings: TierSettings): Tier[] {
  const lower = settings.tier1?.model ?? settings.tier0?.model;
  const tiers: Tier[] = [];
  if (settings.tier4) tiers.push(settings.tier4);
  if (settings.tier3) tiers.push(new CrispTier(settings.tier3.implementation, lower));
  if (settings.tier2 && settings.tier2.enabled !== false) tiers.push(new CompiledTier(lower, settings.tier2.verify));
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
  const ladder = ladderOf(settings);
  const initial = (tier: string) => initialState(settings, tier);
  engine.ledger.hydrate(host.store, fn, settings, initial);
  const events: TierEvent[] = [];
  const note = (ev: Omit<TierEvent, 'fn'>): void => {
    const full: TierEvent = { fn, ...ev };
    events.push(full);
    const derived = engine.ledger.apply(full, settings, initial(full.tier));
    if (derived) events.push(derived);
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

  for (const tier of ladder) {
    const last = tier === terminal;
    if (!last && engine.ledger.state(fn, tier.id, initial(tier.id)) !== 'active') continue;
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
        { tier: tier.id, event: 'deopt', kind: 'error', reason: message(error), to: nextBelow(tier), call_id: own, own: !!own });
      continue;
    }
    const capture = lastPlan()?.capture;
    const callId = capture?.base.callId ?? null;
    if (callId) lastCall = callId;
    let servedBy = tier;
    // A crisp case that failed after effects handed the call to the model that follows it.
    if (capture?.executor.kind === 'crisp-agent') {
      note({ tier: tier.id, event: 'deopt', kind: 'error', reason: capture.executor.case_error ?? 'the case stopped', to: lastPlan()?.model ? 'model' : terminal.id });
      servedBy = ladder.find(item => item.level < 2 && (item as ModelTier).model === lastPlan()?.model) ?? terminal;
    }
    const verdict = await Promise.resolve(servedBy.verify(input, output)).catch(error => `verify threw: ${message(error)}`);
    if (verdict !== true && !last) {
      note({ tier: servedBy.id, event: 'deopt', kind: 'verify', reason: typeof verdict === 'string' ? verdict : 'the output check failed', to: nextBelow(servedBy), call_id: callId, own: !!callId });
      continue;
    }
    if (verdict !== true) note({ tier: servedBy.id, event: 'deopt', kind: 'verify', reason: typeof verdict === 'string' ? verdict : 'the output check failed', call_id: callId, own: false });
    // Tier 2 is promoted by the store's case tiers; its first service is the observable promotion.
    if (servedBy.id === 'tier2' && !engine.ledger.stats(fn, 'tier2')?.served)
      events.push({ fn, tier: 'tier2', event: 'promoted', evidence: { case_hash: capture?.executor.case_hash ?? null, by: 'compilation case tier' } });
    note({ tier: servedBy.id, event: 'served', call_id: callId, ...(capture?.executor.case_hash ? { evidence: { case_hash: capture.executor.case_hash } } : {}) });
    await shadowAbove(engine, settings, ladder, servedBy, input, output, note);
    flush(callId);
    return output;
  }
  flush();
  throw new Error(`no tier served ${fn}`);
}

/** Run the shadow or demoted tiers above the one that served, on a sampled fraction, and count agreement. */
async function shadowAbove(engine: TierEngine, settings: TierSettings, ladder: Tier[], served: Tier, input: TierInput, output: unknown,
  note: (ev: Omit<TierEvent, 'fn'>) => void): Promise<void> {
  const rate = settings.shadowRate ?? 0;
  if (rate <= 0) return;
  const same = settings.same ?? ((a: unknown, b: unknown) => canonicalValue(a) === canonicalValue(b));
  for (const tier of ladder) {
    if (tier === served) break;
    if (!tier.shadow || engine.ledger.state(input.name, tier.id, initialState(settings, tier.id)) === 'active') continue;
    if (engine.random() >= rate) continue;
    let admitted = false;
    try { admitted = (await tier.canServe(input)) === true; } catch { /* not admitted */ }
    if (!admitted) continue;
    let agree = false;
    try { agree = same(output, await tier.shadow(input)); } catch { agree = false; }
    note({ tier: tier.id, event: agree ? 'shadow_equal' : 'shadow_worse' });
  }
}
