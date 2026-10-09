/**
 * Crisp checkers for the refinement predicates of types.ts (plans/REFINEMENT_TYPES.md). The launcher loads this
 * module's `refinements` table; each key is a predicate exactly as `Is<T, "...">` states it, and a checker decides the
 * predicate from the value alone, without a model call. A checker that returns `undefined` leaves the value to the judge.
 */
import { entriesOf } from './games/economy/ledger.js';
import type { Entry, TradeOutcome } from './types.js';

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);
const text = (value: unknown, pattern: RegExp) => typeof value === 'string' && pattern.test(value);
const integer = (value: unknown, least: number) => typeof value === 'number' && Number.isSafeInteger(value) && value >= least;

const ENGINES = ['nl', 'crisp', 'shadow'];

export const refinements: Record<string, (value: any) => boolean | undefined> = {
  "a name: a letter followed by letters, digits, '_' or '-'": value => text(value, /^[A-Za-z][A-Za-z0-9_-]*$/),
  'a non-negative safe integer': value => integer(value, 0),
  'a positive safe integer': value => integer(value, 1),
  "an event id: 'event-' followed by digits": value => text(value, /^event-\d+$/),
  "a memory entry id: 'event-' followed by digits, optionally followed by '.note-' and digits": value => text(value, /^event-\d+(\.note-\d+)?$/),
  '0 or 1': value => value === 0 || value === 1,
  '1 or 2': value => value === 1 || value === 2,
  'settings in which validate, settle and resolve are each nl, crisp or shadow, and remember and narrate are each nl or crisp': value =>
    isRecord(value) && ['validate', 'settle', 'resolve'].every(part => ENGINES.includes(value[part] as string)) &&
    ['remember', 'narrate'].every(part => ENGINES.slice(0, 2).includes(value[part] as string)),
  'entries that are, for each traded outcome in order, the goods from seller to buyer and then the money from buyer to seller, and no others': value => {
    if (!isRecord(value) || !Array.isArray(value.outcomes) || !Array.isArray(value.entries)) return false;
    const wanted: Entry[] = (value.outcomes as TradeOutcome[]).flatMap(outcome => isRecord(outcome) ? entriesOf(outcome) : []);
    return JSON.stringify(value.entries) === JSON.stringify(wanted);
  },
  // Empty or blank is decided here; whether a statement is concrete is the judge's.
  'a concrete commitment that names what the NPC will do, stated in one sentence': value =>
    typeof value !== 'string' || !value.trim() || /\n/.test(value.trim()) ? false : undefined,
};
