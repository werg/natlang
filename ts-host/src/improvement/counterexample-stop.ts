/**
 * The stop decision of the counterexample-guided search: the shared `shouldStop` policy (improveStep/shouldStop.nl, the
 * same pluggable slot and setting as the source search), here with the facts of a round. The crisp default is the rule
 * counterexampleStep.nl stated in prose: no admitted example, the repaired training suite passing, the oracle allowance
 * exhausted, or a repair that did not complete each end the search. The facts are measured by the host, never reported
 * by the model.
 */
import { pluggable, type PluggableSetting } from '../runtime/pluggable.js';
import { loadVirtualNatlang } from '../runtime/virtual-project.js';
import { AUTHORED_IMPROVER } from './authored-source.js';

export type CounterexampleStopFacts = { search: 'counterexample'; round: number; admitted: number; remainingChecks: number;
  repair?: { eligible: boolean; trainingQuality: number; disposition: string } };
export type CounterexampleStopDecision = { stop: boolean; reason: string };

/** The rule as written in the step: stop on no admitted example, a passing suite, an exhausted allowance or an incomplete repair. */
export function crispCounterexampleStop(facts: CounterexampleStopFacts): CounterexampleStopDecision {
  if (facts.admitted === 0) return { stop: true, reason: 'No counterexample was admitted; the source is retained.' };
  if (facts.repair?.trainingQuality === 1) return { stop: true, reason: 'The repaired training suite passes.' };
  if (facts.remainingChecks === 0) return { stop: true, reason: 'The independent oracle allowance is exhausted.' };
  if (facts.repair?.eligible !== true) return { stop: true, reason: 'The repair did not produce a completed eligible result.' };
  return { stop: false, reason: 'Continue with further counterexamples.' };
}
/** A decision is a boolean stop with a reason. The round limit is the loop's own measure, so no further bound is needed here. */
export function verifyCounterexampleStop(decision: CounterexampleStopDecision): CounterexampleStopDecision {
  if (typeof decision?.stop !== 'boolean' || typeof decision.reason !== 'string')
    throw new Error('A stop decision is {stop: boolean, reason: string}.');
  return decision;
}
let shouldStopNl: ((facts: CounterexampleStopFacts) => Promise<CounterexampleStopDecision>) | undefined;
/** The shared policy for one round: `crisp` (default), `nl` or `shadow`. */
export function counterexampleStop(setting: PluggableSetting, facts: CounterexampleStopFacts): Promise<CounterexampleStopDecision> {
  return pluggable<[CounterexampleStopFacts], CounterexampleStopDecision>({
    crisp: async value => crispCounterexampleStop(value),
    nl: async value => verifyCounterexampleStop(await (shouldStopNl ??= loadVirtualNatlang(AUTHORED_IMPROVER, 'improveStep/shouldStop.nl') as never)(value)),
  }, setting, { name: 'shouldStop', default: 'crisp', serve: 'crisp' })(facts);
}
