/**
 * The crisp side of the `selectGuidanceFacts` pluggable: a fixed kind order. It serves when the setting is `crisp`, and
 * whenever the natural-language selection fails or returns something other than distinct ids of visible facts.
 * Packing into the token budget is the engine's, after selection, so the budget does not change the order here.
 */
import type { GuidanceConsumer, GuidanceFact } from './types.js';

/** What a crisp selection records as its policy: the sha256 of this string. */
export const CRISP_SELECTION_IDENTITY = 'natlang.select-guidance-facts.crisp/1';

const ROOT_ORDER = ['expected_result', 'expected_stop', 'expected_files', 'expected_effects', 'reference_solution',
  'attempt_feedback', 'reference_failure', 'minimum_sequence', 'decisive', 'teacher_trajectory', 'fit_hindsight'];
const CHILD_ORDER = ['child_value', 'attempt_feedback'];

/**
 * The ids of `facts` (already visible to `consumer`) in kind order, stable within a kind. A child whose prompt matches
 * more than one distinct verified value has no verified answer of its own: its child_value facts are dropped.
 */
export function crispSelection(consumer: GuidanceConsumer, facts: readonly GuidanceFact[], budgetTokens: number): string[] {
  void budgetTokens;
  const order = consumer.role === 'child' ? CHILD_ORDER : ROOT_ORDER;
  let chosen = facts.filter(fact => order.includes(fact.kind));
  if (new Set(chosen.filter(fact => fact.kind === 'child_value').map(fact => fact.text)).size > 1)
    chosen = chosen.filter(fact => fact.kind !== 'child_value');
  return chosen.map((fact, index) => ({ fact, index }))
    .sort((a, b) => order.indexOf(a.fact.kind) - order.indexOf(b.fact.kind) || a.index - b.index)
    .map(({ fact }) => fact.id);
}
