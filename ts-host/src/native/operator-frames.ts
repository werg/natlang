/**
 * Learned Neuralese operators: the sites the runtime compacts with and how a consumer is shown their state
 * (plans/neuralese/LEARNED_COMPACTION_OPERATORS.md; operators in training/neuralese/natlang_neuralese/model/learned_operators.py,
 * served by `POST /v1/neuralese/operator`, client in model/native-operator.ts).
 *
 * Whether the runtime compacts with a learned operator or keeps the crisp text cut is the `compaction` setting of the
 * native agent (a pluggable part: `crisp` keeps the cut, `nl` uses the learned operator).
 */
import { neuraleseRefSentinel, type NeuraleseRef } from './neuralese.js';

export type ValueOperator = 'compact-value' | 'compact-params' | 'extract';
export type CompactSite =
  | { operator: ValueOperator; instructions: string; value: string; note?: string }
  | { operator: 'compact-history'; messages: unknown[]; context?: unknown[]; tools?: unknown[]; note?: string };
export type Compactor = (site: CompactSite) => Promise<NeuraleseRef>;

/**
 * How a consumer is shown an operator's state: open + state + close. Must equal the Python registry's
 * `consumer_open`/`consumer_close` (tests/neuralese/test_learned_operators.py checks this file).
 */
export const OPERATOR_FRAMES: Record<CompactSite['operator'], { open: string; close: string }> = {
  'compact-value': { open: 'Compacted value: ', close: '\n\n' },
  'compact-params': { open: 'Compacted argument: ', close: '\n\n' },
  'compact-history': { open: 'Earlier steps of this session, compacted: ', close: '\n\nContinue from the most recent steps.' },
  'extract': { open: 'Extracts from the documents, one per document:\n', close: '\n\n' },
};

/** The framed sentinel of an operator's state, as text the driver turns into a native-state part. */
export function framedState(operator: CompactSite['operator'], ref: NeuraleseRef): string {
  const frame = OPERATOR_FRAMES[operator];
  return frame.open + neuraleseRefSentinel(ref) + frame.close;
}
