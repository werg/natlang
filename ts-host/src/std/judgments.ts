/**
 * The judgments @natlang/std ships (ts-host/std/*.nl), loaded from their exact bytes. Each is an ordinary natlang
 * function: a program can call it, `decide` it, rebind it with `.in`, run it on another model with `.on`, and the call
 * store, traces and specialisation see it under its source `@natlang/std/<name>.nl`.
 */
import { loadVirtualNatlang } from '../runtime/virtual-project.js';
import type { NatlangFunction } from '../runtime/surface.js';
import type { Decision } from '../runtime/runtime.js';
import type { NeuraleseRef } from '../native/neuralese.js';
import { STD_SOURCES } from './sources.generated.js';

const FILES = Object.fromEntries(Object.entries(STD_SOURCES).map(([name, text]) => [`@natlang/std/${name}`, text]));
const load = <A extends unknown[], R>(name: string) => loadVirtualNatlang(FILES, `/@natlang/std/${name}.nl`, '') as unknown as NatlangFunction<A, R>;

/** What one change of a value is for: one sentence. */
export type Intent = string;
/** One side of a merge: the value an edit produced (or `removed: true` when deleted) and what it is for. */
export type Edit<T = unknown> = { after?: T; removed?: boolean; intent: Intent };
/** Prior escalation statistics passed to `shouldEscalate.nl`. */
export type EscalationHistory = { totalCalls?: number; escalationRate?: number; disagreementRate?: number };
/** Relative local vs fallback tier cost metadata passed to `shouldEscalate.nl`. */
export type TierCost = { local?: number; fallback?: number; ratio?: number };

/** `intentOf(before?, after?)`: what the edit from `before` to `after` is for. */
export const intentOf = load<[before?: unknown, after?: unknown], Intent>('intentOf');
/** `compatible(base?, ours, theirs)` (`readout: decision`): whether one value can keep both edits. */
export const compatible = load<[base: unknown, ours: Edit, theirs: Edit], boolean>('compatible');
/** `combine(base?, ours, theirs)`: the value that makes both edits of `base`. */
export const combine = load<[base: unknown, ours: Edit, theirs: Edit], unknown>('combine');
/** `settled({ previous, current })` (`readout: decision`): whether the revision only reworded its value. */
export const settled = load<[revision: { previous: unknown; current: unknown }], boolean>('settled');
/** `beats(goal, first, second)` (`readout: decision`): whether `first` serves `goal` better than `second`. */
export const beats = load<[goal: string, first: unknown, second: unknown], boolean>('beats');
/** `meets(criterion, item)` (`readout: decision`): whether `item` satisfies `criterion`. */
export const meets = load<[criterion: string, item: unknown], boolean>('meets');
/** `relates(relation, left, right)` (`readout: decision`): whether `left` and `right` match under `relation`. */
export const relates = load<[relation: string, left: unknown, right: unknown], boolean>('relates');
/** `compileRuleFn(rule, problems?)`: synthesize a typed judgment function implementing `rule`. */
export const compileRuleFn = load<[rule: string, problems?: string[]], (item: unknown) => Promise<unknown>>('compileRule');
/** `shouldEscalate(decision, floor?, risk?, history?, cost?)` (`readout: decision`): whether a local decision should escalate. */
export const shouldEscalate = load<[decision: Decision<unknown>, floor?: number, risk?: Record<string, number>,
  history?: EscalationHistory, cost?: TierCost], boolean>('shouldEscalate');
