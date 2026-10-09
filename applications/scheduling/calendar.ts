/**
 * The `calendar` service: the exact verifier, the exact enumeration, and clock arithmetic over one frozen day. The
 * scheduler's stages may call it to check what they produce; they cannot change it. It is built from a `Problem`
 * snapshot, so a run decides on the facts it started with.
 */
import { pluggableMode, type PluggableMode, type PluggableSetting } from '@natlang/node';
import type { Problem } from './workspace.js';
import type { Hard, Offered, Placement, Verdict } from './types.js';

/** A pluggable part's mode: 'crisp', 'nl' or 'shadow' (runs both and records agreement); 'natlang' and 'natural-language' are deprecated spellings of 'nl', accepted and normalized. */
export type Implementation = Exclude<PluggableSetting, undefined>;
export type Implementations = { enumeration: Implementation };

export type CalendarService = {
  check(placements: Placement[], hard?: Hard): Verdict;
  exact(hard: Hard, limit: number): Offered;
  describe(placements: Placement[]): string[];
  implementation(name: 'enumeration'): PluggableMode;
};

export function calendarService(problem: Problem, implementations: Implementations): CalendarService {
  return {
    check: (placements, hard) => problem.check(placements, hard),
    exact: (hard, limit) => problem.enumerate(hard, limit),
    describe: placements => problem.describe(placements),
    implementation: name => pluggableMode(implementations[name]),
  };
}

/** What the model is told about the service. */
export const calendarDeclaration = `/** The verifier and clock arithmetic for the day being planned. Minutes are counted from the day's origin. */
export const calendar: {
  /**
   * Check a complete schedule against every hard constraint of the day, plus the request's (hard: its new tasks, limits
   * and commitments). Exact. ok is true only when every task is placed once, for its duration, inside its bounds and a
   * window, on the slot grid, clear of every commitment and of every other task, after the tasks it depends on. Each
   * violation names the task and the move that would fix it.
   */
  check(placements: { id: string, start: number, end: number }[], hard?: Hard): { ok: boolean, violations: { rule: string, task: string, detail: string }[] };
  /**
   * Exact enumeration: complete schedules that satisfy every constraint, up to limit, windows in order and starts
   * ascending. An empty result means that no schedule exists.
   */
  exact(hard: Hard, limit: number): Offered;
  /** The placements as lines of clock times, earliest first, e.g. "draft 09:00-09:30". */
  describe(placements: { id: string, start: number, end: number }[]): string[];
  /** Which implementation of a pluggable part is selected. */
  implementation(name: 'enumeration'): 'crisp' | 'nl' | 'shadow';
};`;
