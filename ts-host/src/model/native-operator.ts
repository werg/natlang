/**
 * The client of a native server's learned Neuralese operators (`POST /v1/neuralese/operator`, capability `operator/1`):
 * one prefill of the operator site on the server, whose K slot states are archived and returned as a Neuralese value.
 */
import { neuraleseRef } from '../native/neuralese.js';
import type { Compactor } from '../native/operator-frames.js';
import type { NeuraleseStore } from '../native/neuralese-store.js';
import { fetchModel } from './chat-completion.js';
import { serves, type NeuraleseServerInfo } from './neuralese-info.js';
import { archiveNativeState } from './native-state-client.js';

export const OPERATOR_CAPABILITY = 'operator/1';

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
