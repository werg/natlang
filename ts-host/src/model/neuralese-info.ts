/**
 * What a Neuralese server reads (`GET /v1/neuralese/info`, spec/NEURALESE_PORT.md "Wire protocol"), and the startup
 * check of a declared reader against it (plans/neuralese/DECISIONS.md 2026-10-09: a consumer's reader type is declared
 * by configuration and checked at startup, not discovered by a failing request).
 */
import { fetchModel, openAIEndpointRoot } from './chat-completion.js';

/** A server's `/v1/neuralese/info`: the dialects it speaks, and the fields hosts read. Other fields are kept as sent. */
export type NeuraleseServerInfo = { dialects: string[]; width?: number; dtype?: string; max_block_length?: number;
  capabilities?: string[] } & Record<string, unknown>;

/**
 * Whether a server serves `capability` (spec/NEURALESE_PORT.md "Capabilities"), from its `capabilities` list. A server
 * from before the list is judged by its older fields: `adapters: "lora"` (the fork) loads LoRAs and serves no grad,
 * embed, optim, adapter creation or streaming; otherwise (the reference) everything but LoRA loading.
 */
export function serves(info: Record<string, unknown>, capability: string): boolean {
  if (Array.isArray(info.capabilities)) return info.capabilities.includes(capability);
  const lora = info.adapters === 'lora';
  if (capability === 'adapters.lora-load') return lora;
  const referenceOnly = ['grad', 'grad.order2', 'embed', 'optim', 'adapters.create', 'adapters.direct', 'adapters.lora-export',
    'adapters.projection', 'chat.stream', 'template.value-type', 'template.argument-path', 'parts.value-type'];
  return !(lora && referenceOnly.includes(capability));
}

/**
 * The server's `/v1/neuralese/info`. A server without the endpoint is no Neuralese server: that fails with
 * `neuralese-unsupported-backend`.
 */
export async function neuraleseServerInfo(endpoint: string, options: { apiKey?: string; headers?: Record<string, string>;
    signal?: AbortSignal } = {}): Promise<NeuraleseServerInfo> {
  const url = openAIEndpointRoot(endpoint) + '/v1/neuralese/info';
  let response: Response;
  try {
    response = await fetchModel(url, { signal: options.signal,
      headers: { ...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}), ...options.headers } });
  } catch (error) {
    throw new Error(`neuralese-unsupported-backend: ${url} could not be reached (${error instanceof Error ? error.message : String(error)})`,
      { cause: error });
  }
  if (!response.ok) throw new Error(`neuralese-unsupported-backend: ${url} answered HTTP ${response.status}; the server ` +
    'does not declare Neuralese dialects');
  const info = await response.json() as Partial<NeuraleseServerInfo>;
  if (!Array.isArray(info.dialects) || !info.dialects.every(dialect => typeof dialect === 'string'))
    throw new Error(`neuralese-unsupported-backend: ${url} names no dialects`);
  return info as NeuraleseServerInfo;
}

/**
 * Check that a server reads `dialect`; returns its info. A server that does not speak it fails with
 * `neuralese-dialect-mismatch` (values are never reused across dialects, NEURALESE_DIALECTS.md).
 */
export async function checkNeuraleseReader(endpoint: string, dialect: string,
    options: Parameters<typeof neuraleseServerInfo>[1] = {}): Promise<NeuraleseServerInfo> {
  const info = await neuraleseServerInfo(endpoint, options);
  if (!info.dialects.includes(dialect))
    throw Object.assign(new Error(`neuralese-dialect-mismatch: the reader is declared as dialect ${JSON.stringify(dialect)}, ` +
      `but the server at ${openAIEndpointRoot(endpoint)} speaks ${info.dialects.map(name => JSON.stringify(name)).join(', ') || 'none'}`),
      { code: 'neuralese-dialect-mismatch' });
  return info;
}
