/**
 * Decision readout for calls with a finite result type (`"a" | "b"`, `boolean`, a union of literals).
 *
 * Instead of the tool loop, the runtime asks the model's driver to score every allowed result as the reply to the
 * call's opening, and returns the most probable one. The full distribution is kept in the trace, so a decision is a
 * calibrated probability vector, not one sample: it can be scored with proper scoring rules (Brier, ranked
 * probability score) and, on a Neuralese server, differentiated. A call opts in with `readout: decision` in its
 * frontmatter, or every finite-typed call does when the model config sets `decisionReadout: 'finite-returns'`.
 */
import type { DecisionScorer } from '../contracts.js';
import type { Type, TypeEnv } from './types.js';

export type DecisionValue = string | number | boolean | null;
export const DECISION_READOUT_MODES = ['decision'] as const;
export type DecisionReadout = typeof DECISION_READOUT_MODES[number];

/** Every value of a finite type, in declaration order; null when the type is not finite. */
export function finiteValues(type: Type, env: TypeEnv): DecisionValue[] | null {
  const out: DecisionValue[] = [];
  const visit = (item: Type, depth: number): boolean => {
    if (depth > 32) return false;
    const resolved = item.kind === 'name' ? env.resolve(item) : item;
    switch (resolved.kind) {
      case 'lit': out.push(resolved.value); return true;
      case 'union': return resolved.members.every(member => visit(member, depth + 1));
      case 'prim':
        if (resolved.name === 'boolean') { out.push(true, false); return true; }
        if (resolved.name === 'null') { out.push(null); return true; }
        return false;
      default: return false;
    }
  };
  if (!visit(type, 0)) return null;
  return [...new Map(out.map(value => [JSON.stringify(value), value])).values()];
}

/** The readout's system prompt: the call's opening stays, the tool manual does not (no tools are offered). */
export const DECISION_SYSTEM_PROMPT = 'You are running one call of a natural-language function. Carry out its ' +
  "instructions yourself for this call's argument values, judging the visible inputs. Your reply is the function's " +
  'return value: one JSON value of the declared type, with nothing else.';

export const decisionPrompt = (replies: string[]) =>
  'Answer this call now, without tools. Reply with exactly one of these JSON values and nothing else: ' +
  replies.join(', ') + '.';

/** The driver's decision scorer, if its transport has one. */
export function decisionScorer(driver: unknown): DecisionScorer | undefined {
  const decide = (driver as { decide?: unknown } | null)?.decide;
  return typeof decide === 'function' ? decide as DecisionScorer : undefined;
}

/** Normalised probabilities from log-probabilities (stable softmax). */
export function softmax(logProbs: number[]): number[] {
  const top = Math.max(...logProbs);
  const weights = logProbs.map(value => Math.exp(value - top));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map(value => value / total);
}
