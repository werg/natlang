/**
 * The runtime's operator sites as data, for training rows built outside a run (`ts-host/scripts/operator-sites.mjs`,
 * scripts/neuralese_data/operator_rows.py; plans/neuralese/OPERATORS_RESTORE_20261117.md §9). Every string here comes
 * from the code the agent runs at that site (agent.ts `views` and the tool-result loop): the operator call's messages,
 * the block the consumer sees with the state in place (split around the state), the crisp text the consumer sees
 * without it, and whether the runtime would compact the value at all.
 */
import { listingLine, renderValue } from './agent.js';
import { shownCharsFor, SCOPE_VALUE_CHARS } from './cutoff.js';
import { framedState, OPERATOR_MIN_CHARS, argumentInstructions, operatorPurpose, valueInstructions, viewSiteMessages,
  type ValueOperator } from './operator-frames.js';
import { PageStore } from './pages.js';
import { viewNote } from './prompt.js';
import { neuraleseRefSentinel, type NeuraleseRef } from './neuralese.js';

export type OperatorSiteRequest = {
  operator: Exclude<ValueOperator, 'extract'>;
  /** compact-value: the whole tool output; compact-params: the argument value (any JSON value). */
  value: unknown;
  /** The calling function's instructions (the call body). */
  purpose: string;
  /** compact-value: the tool whose output it is. */
  tool?: string;
  /** compact-params: the argument's name and type text. */
  argument?: { name: string; type: string };
  /** The call's context budget in prompt tokens (sizes the shown page; null or absent: the floor). */
  context_tokens?: number | null;
  /** compact-value: the result's index in the call's transcript (names where all of it is). */
  entry?: number;
};

const STATE_ID = 'nzs1_' + 'a'.repeat(52);

/** One operator site as the runtime renders it. */
export function operatorSite(request: OperatorSiteRequest) {
  const budget = shownCharsFor(request.context_tokens);
  const purpose = operatorPurpose(request.purpose);
  const ref = { $neuralese: { id: STATE_ID } } as unknown as NeuraleseRef;
  const sentinel = neuraleseRefSentinel(ref);
  const split = (block: string) => {
    const at = block.indexOf(sentinel);
    return { before: block.slice(0, at), after: block.slice(at + sentinel.length) };
  };
  if (request.operator === 'compact-value') {
    if (typeof request.value !== 'string') throw new TypeError('compact-value reads a text output');
    const shown = new PageStore().show(request.value, `transcript.entry(${request.entry ?? 0}).output`, budget);
    const value = request.value;
    const tool = request.tool ?? 'tool';
    return { operator: request.operator, messages: viewSiteMessages(value, valueInstructions(tool, purpose)), tools: [],
      compacted: shown !== value && value.length >= OPERATOR_MIN_CHARS, text: shown,
      state: split(shown + '\n\n' + framedState('compact-value', ref).trimEnd()) };
  }
  if (request.operator !== 'compact-params') throw new TypeError(`no runtime site for operator ${String(request.operator)}`);
  const { name, type } = request.argument ?? { name: 'value', type: 'string' };
  const scope = Math.max(SCOPE_VALUE_CHARS, budget);
  const shown = renderValue(request.value, { holder: name, budget: scope });
  const value = JSON.stringify(request.value);
  return { operator: request.operator, messages: viewSiteMessages(value, argumentInstructions(name, type, purpose)), tools: [],
    compacted: shown.includes('<<cut off:') && value.length >= OPERATOR_MIN_CHARS, text: listingLine(name, type, shown),
    state: split(listingLine(name, type, sentinel + viewNote(name))) };
}
