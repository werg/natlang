/**
 * Learned Neuralese operators on a native server (plans/neuralese/LEARNED_COMPACTION_OPERATORS.md; the operators are
 * training/neuralese/natlang_neuralese/model/learned_operators.py, served by `POST /v1/neuralese/operator`).
 *
 * An operator is one write site: its learned prompt, the value (or the conversation) it compacts, and K learned slot
 * positions whose native state is the result. Writing it is one prefill on the server, never sampling. The runtime
 * shows the state where it would otherwise show a text cut-off, framed by the operator's consumer text below, and the
 * full value stays reachable (the variable, `read_page`, `transcript`).
 *
 * Whether the runtime compacts with a learned operator or keeps the crisp text cut is the `compaction` setting of the
 * native agent (a pluggable part: `crisp` keeps the cut, `nl` uses the learned operator).
 */
import { neuraleseRef, neuraleseRefSentinel, type NeuraleseRef } from '../native/neuralese.js';
import type { NeuraleseStore } from '../native/neuralese-store.js';
import { fetchModel } from '../model/chat-completion.js';
import { serves, type NeuraleseServerInfo } from '../model/neuralese-info.js';
import { archiveNativeState } from '../model/native-state-client.js';

export const OPERATOR_CAPABILITY = 'operator/1';
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

/** The learned operators of one native server that advertises `operator/1`. */
export function serverCompactor(options: { endpoint: string; headers?: Record<string, string>; store: NeuraleseStore;
  info: NeuraleseServerInfo }): Compactor {
  const base = options.endpoint.replace(/\/$/, '');
  return async site => {
    if (!serves(options.info, OPERATOR_CAPABILITY))
      throw Object.assign(new Error('neuralese-operator-unavailable: the server does not serve learned operators'),
        { code: 'neuralese-operator-unavailable' });
    if (!options.store.putNativeState || !options.store.getNativeState)
      throw new Error('neuralese-native-archive-unavailable: learned operators need a native-capable archive');
    const body = site.operator === 'compact-history'
      ? { operator: site.operator, messages: site.messages, context: site.context ?? [], tools: site.tools, note: site.note ?? '' }
      : { operator: site.operator, instructions: site.instructions, value: site.value, note: site.note ?? '' };
    // model-call-kind: operator-write -- training/model-call-kinds.json
    const response = await fetchModel(`${base}/v1/neuralese/operator`, { method: 'POST',
      headers: { 'content-type': 'application/json', ...options.headers }, body: JSON.stringify(body) });
    if (!response.ok) throw new Error(`native operator failed (${response.status}): ${(await response.text()).slice(0, 1000)}`);
    const result = await response.json() as { id?: string; operator?: { name?: string } };
    if (typeof result.id !== 'string' || result.operator?.name !== site.operator)
      throw new Error('neuralese-operator-integrity: the server did not return this operator\'s native state');
    await archiveNativeState({ endpoint: options.endpoint, headers: options.headers, store: options.store }, result.id);
    return neuraleseRef('Neuralese<string>', result.id);
  };
}
