/**
 * The judgments @natlang/std ships (ts-host/std/*.nl), loaded from their exact bytes. Each is an ordinary natlang
 * function: a program can call it, `decide` it, rebind it with `.in`, run it on another model with `.on`, and the call
 * store, traces and specialisation see it under its source `@natlang/std/<name>.nl`.
 */
import { loadVirtualNatlang } from '../runtime/virtual-project.js';
import type { NatlangFunction } from '../runtime/surface.js';
import { STD_SOURCES } from './sources.generated.js';

const FILES = Object.fromEntries(Object.entries(STD_SOURCES).map(([name, text]) => [`@natlang/std/${name}`, text]));
const load = <A extends unknown[], R>(name: string) => loadVirtualNatlang(FILES, `/@natlang/std/${name}.nl`, '') as unknown as NatlangFunction<A, R>;

/** What one change of a value is for: one sentence. */
export type Intent = string;
/** One side of a merge: the value an edit produced and what it is for. */
export type Edit<T = unknown> = { after: T; intent: Intent };

/** `intentOf(before, after)`: what the edit from `before` to `after` is for. */
export const intentOf = load<[before: unknown, after: unknown], Intent>('intentOf');
/** `compatible(base, ours, theirs)` (`readout: decision`): whether one value can keep both edits. */
export const compatible = load<[base: unknown, ours: Edit, theirs: Edit], boolean>('compatible');
/** `combine(base, ours, theirs)`: the value that makes both edits of `base`. */
export const combine = load<[base: unknown, ours: Edit, theirs: Edit], unknown>('combine');
/** `settled({ previous, current })` (`readout: decision`): whether the revision only reworded its value. */
export const settled = load<[revision: { previous: unknown; current: unknown }], boolean>('settled');
/** `beats(goal, first, second)` (`readout: decision`): whether `first` serves `goal` better than `second`. */
export const beats = load<[goal: string, first: unknown, second: unknown], boolean>('beats');
