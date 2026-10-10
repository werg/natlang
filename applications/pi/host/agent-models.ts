/**
 * The agent's model as pi-ai models with one provider, "agent": pi-ai's OpenAI-compatible provider, or natlang's own
 * model transport (host/natlang-provider.ts), whose model declares the reader it is (text or a Neuralese dialect).
 * Platform-neutral: the Node host reads the settings from its flags and the profile (main.ts), the browser host from
 * its options (browser.ts), where the endpoint may be the in-page WebAssembly Neuralese engine's.
 */
import { createModels, createProvider, type Models } from '@earendil-works/pi-ai';
import { openAICompletionsApi } from '@earendil-works/pi-ai/api/openai-completions.lazy';
import { neuraleseServerModelTurn, openAICompatibleModelTurn, type NeuraleseStore } from 'natlang:runtime';
import { natlangApi, type AgentReader, type ModelDriver } from './natlang-provider.ts';

export type AgentTransport = 'pi-ai' | 'natlang';

export type AgentModelSettings = {
  /** The server's root URL (a trailing `/v1` is accepted). */
  endpoint: string;
  modelId: string;
  apiKey?: string;
  /** `pi-ai` (its OpenAI-compatible provider, the default) or `natlang` (natlang's model transport). */
  transport?: AgentTransport;
  /** With the natlang transport: what the model reads, as declared and checked (`declareReader`). Default text. */
  reader?: AgentReader;
  /** Default 65536. */
  contextWindow?: number;
  /** Default 16384. */
  maxTokens?: number;
  /** The runtime's Neuralese store, which a Neuralese server's blocks are archived in. */
  store?: NeuraleseStore;
};

export type AgentModels = { models: Models; ref: { provider: string; modelId: string } };

/** pi-ai models whose provider "agent" serves `settings.modelId` at `settings.endpoint`. */
export function agentModels(settings: AgentModelSettings): AgentModels {
  const { modelId, apiKey, store } = settings;
  const transport = settings.transport ?? 'pi-ai';
  if (transport !== 'natlang' && settings.reader && settings.reader.kind !== 'text')
    throw new Error('a Neuralese reader needs the natlang transport');
  const reader: AgentReader = settings.reader ?? { kind: 'text' };
  const root = settings.endpoint.replace(/\/+$/, '').replace(/\/v1$/, '');
  const baseUrl = `${root}/v1`;
  // One natlang driver per thinking setting, sent as the template argument `enable_thinking`; for a Neuralese server
  // also one per owner (the conversation's provider session ID), whose blocks its requests hold. No `preserve_thinking`:
  // earlier turns' reasoning renders by the served backbone's declared history-reasoning policy, its template default
  // (training/neuralese/natlang_neuralese/serve/chat.py BACKBONE_HISTORY_REASONING: LFM2.5 last turn only, Mellum kept).
  const drivers = new Map<string, ModelDriver>();
  const driver = (reasoning: boolean, owner?: string): ModelDriver => {
    const request = { endpoint: root, model: modelId, apiKey,
      request: { chat_template_kwargs: { enable_thinking: reasoning } } };
    const key = reader.kind === 'neuralese' ? `${reasoning}\0${owner ?? ''}` : String(reasoning);
    if (!drivers.has(key)) drivers.set(key, reader.kind === 'neuralese' ?
      neuraleseServerModelTurn({ ...request, store, ...(owner ? { owner } : {}) }) : openAICompatibleModelTurn(request));
    return drivers.get(key)!;
  };
  const models = createModels();
  models.setProvider(createProvider({
    id: 'agent', name: 'Agent', baseUrl,
    // pi-ai's OpenAI-compatible provider needs some key; the natlang transport sends one only when it is given (its
    // drivers, and the view requests of host/views.ts, read it from here).
    auth: { apiKey: { name: 'agent key', resolve: async () => ({ auth: transport === 'natlang' ? (apiKey ? { apiKey } : {}) :
      { apiKey: apiKey ?? 'none' } }) } },
    models: [{ id: modelId, name: modelId, api: transport === 'natlang' ? 'natlang-model-turn' : 'openai-completions', provider: 'agent',
      baseUrl, input: ['text'], reasoning: true, reader,
      contextWindow: settings.contextWindow ?? 65536, maxTokens: settings.maxTokens ?? 16384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      compat: { thinkingFormat: 'chat-template', chatTemplateKwargs: { enable_thinking: { $var: 'thinking.enabled' } },
        supportsDeveloperRole: false, supportsStore: false, supportsReasoningEffort: false,
        maxTokensField: 'max_tokens' } } as never],
    api: transport === 'natlang' ? natlangApi((_model, { reasoning, owner }) => driver(reasoning, owner), store) : openAICompletionsApi(),
  }));
  return { models, ref: { provider: 'agent', modelId } };
}
