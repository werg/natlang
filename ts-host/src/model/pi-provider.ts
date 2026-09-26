/** Optional provider backend. Natlang retains the conversation, tools, retries, and execution loop. */
import { mkdir, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { join } from 'node:path';
import { builtinModels } from '@earendil-works/pi-ai/providers/all';
import { cleanupSessionResources, defaultProviderAuthContext } from '@earendil-works/pi-ai';
import type { Api, AssistantMessage, Context, Credential, CredentialStore, Message,
  ModelsApiStreamOptions, ModelsSimpleStreamOptions, Tool } from '@earendil-works/pi-ai';
import { modelTools } from './chat-completion.js';
import { defaultNatlangConfigDirectory } from '../package/store.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';

const emptyUsage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };

/** One file per provider makes refresh writes independent across processes. */
class FileCredentials implements CredentialStore {
  constructor(private readonly directory: string) {}
  private path(id: string) { return join(this.directory, `${encodeURIComponent(id)}.json`); }
  async read(id: string): Promise<Credential | undefined> {
    try { return JSON.parse(await readFile(this.path(id), 'utf8')) as Credential; }
    catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return; throw error; }
  }
  async list() {
    try {
      const files = await readdir(this.directory);
      const rows = await Promise.all(files.filter(file => file.endsWith('.json')).map(async file => {
        const providerId = decodeURIComponent(file.slice(0, -5));
        const credential = await this.read(providerId);
        return credential ? { providerId, type: credential.type } : null;
      }));
      return rows.filter((row): row is NonNullable<typeof row> => row !== null);
    } catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []; throw error; }
  }
  private async locked<T>(id: string, action: () => Promise<T>): Promise<T> {
    await mkdir(this.directory, { recursive: true, mode: 0o700 });
    const lock = `${this.path(id)}.lock`;
    const deadline = Date.now() + 30000;
    while (true) {
      try { await mkdir(lock); break; }
      catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
        try { if (Date.now() - (await stat(lock)).mtimeMs > 600000) await rm(lock, { recursive: true }); }
        catch (failure) { if ((failure as NodeJS.ErrnoException).code !== 'ENOENT') throw failure; }
        if (Date.now() >= deadline) throw new Error(`timed out waiting for credential lock for ${id}`);
        await new Promise(resolveWait => setTimeout(resolveWait, 100));
      }
    }
    try { return await action(); } finally { await rm(lock, { recursive: true, force: true }); }
  }
  async modify(id: string, fn: (current: Credential | undefined) => Promise<Credential | undefined>) {
    return this.locked(id, async () => {
      const current = await this.read(id), next = await fn(current);
      if (next === undefined) return current;
      const temporary = `${this.path(id)}.${process.pid}.${randomUUID()}.tmp`;
      try { await writeFile(temporary, JSON.stringify(next), { mode: 0o600, flag: 'wx' }); await rename(temporary, this.path(id)); }
      finally { await rm(temporary, { force: true }); }
      return next;
    });
  }
  async delete(id: string) { await this.locked(id, async () => { await rm(this.path(id), { force: true }); }); }
}

export function piModels(environment: NodeJS.ProcessEnv = process.env, scopedEnv: Record<string, unknown> = {}) {
  const base = defaultProviderAuthContext();
  return builtinModels({ credentials: new FileCredentials(join(defaultNatlangConfigDirectory(environment), 'credentials')),
    authContext: { env: async name => {
      const value = scopedEnv[name] ?? environment[name];
      return typeof value === 'string' && value.trim() ? value : undefined;
    }, fileExists: base.fileExists } });
}

function piContext(request: ModelTurnRequest, model: { api: string; provider: string; id: string }): Context {
  const messages: Message[] = [];
  const names = new Map<string, string>();
  for (const raw of request.messages as Record<string, unknown>[]) {
    const role = raw.role, content = String(raw.content ?? '');
    const timestamp = Date.now();
    if (role === 'system') messages.push({ role, content, timestamp });
    else if (role === 'user') messages.push({ role, content, timestamp });
    else if (role === 'assistant') {
      const calls = (raw.tool_calls ?? []) as Array<{ id: string; function: { name: string; arguments: string } }>;
      const blocks: AssistantMessage['content'] = [];
      const reasoning = raw.reasoning_content ?? raw.reasoning;
      if (typeof reasoning === 'string' && reasoning) blocks.push({ type: 'thinking', thinking: reasoning });
      if (content) blocks.push({ type: 'text', text: content });
      for (const call of calls) {
        names.set(call.id, call.function.name);
        blocks.push({ type: 'toolCall', id: call.id, name: call.function.name,
          arguments: JSON.parse(call.function.arguments || '{}') as Record<string, never> });
      }
      messages.push({ role, content: blocks, api: model.api as AssistantMessage['api'], provider: model.provider,
        model: model.id, usage: emptyUsage, stopReason: calls.length ? 'toolUse' : 'stop', timestamp });
    } else if (role === 'tool') messages.push({ role: 'toolResult', toolCallId: String(raw.tool_call_id),
      toolName: names.get(String(raw.tool_call_id)) ?? 'unknown', content: [{ type: 'text', text: content }],
      isError: false, timestamp });
  }
  const tools: Tool[] = modelTools(request.tools).map(tool => ({ name: tool.function.name,
    description: String(tool.function.description ?? ''), parameters: tool.function.parameters as Tool['parameters'] }));
  return { messages, tools };
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (value === undefined) return {};
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`${label} must be an object`);
  return value as Record<string, unknown>;
}

function requiredToolChoice(api: string): string | undefined {
  if (['anthropic-messages', 'bedrock-converse-stream', 'google-generative-ai', 'google-vertex'].includes(api))
    return 'any';
  if (['openai-completions', 'openai-responses', 'openai-codex-responses', 'azure-openai-responses',
    'mistral-conversations', 'pi-messages'].includes(api)) return 'required';
  return undefined;
}

export function createPiModelBackend(provider: string, modelId: string, environment: NodeJS.ProcessEnv = process.env,
  apiKeyEnv?: string, headers?: Record<string, string>, piOptions?: Record<string, unknown>,
  modelOptions?: Record<string, unknown>, mode: 'native' | 'simple' = 'native',
  piPayload?: Record<string, unknown>) {
  if (!['native', 'simple'].includes(mode)) throw new Error('Pi mode must be native or simple');
  const modelOverride = record(modelOptions, 'modelOptions');
  const requestOverride = record(piOptions, 'piOptions');
  const payloadOverride = record(piPayload, 'piPayload');
  const scopedEnv = record(requestOverride.env, 'piOptions.env');
  for (const [name, value] of Object.entries(scopedEnv)) if (typeof value !== 'string')
    throw new TypeError(`piOptions.env.${name} must be a string`);
  const models = piModels(environment, scopedEnv);
  if (!models.getProvider(provider)) throw new Error(`Pi has no provider ${provider}; run natlang models`);
  for (const key of ['id', 'provider', 'api']) if (Object.hasOwn(modelOverride, key))
    throw new Error(`modelOptions cannot override ${key}; select a different provider or model instead`);
  const templateId = modelOverride.inherit;
  if (templateId !== undefined && (typeof templateId !== 'string' || !templateId))
    throw new Error('modelOptions.inherit must name a model in the selected provider catalog');
  const { inherit: _inherit, ...modelSettings } = modelOverride;
  if (requestOverride.maxTokens !== undefined &&
    (!Number.isSafeInteger(requestOverride.maxTokens) || Number(requestOverride.maxTokens) < 1))
    throw new Error('piOptions.maxTokens must be a positive integer');
  if (requestOverride.deferred) throw new Error('piOptions.deferred needs a deferred-result driver and is not supported by natlang turns');
  let model: ReturnType<typeof models.getModel>;
  const selectModel = async () => {
    if (model) return model;
    let selected = models.getModel(provider, modelId);
    if (!selected) {
      const refreshed = await models.refresh({ providers: [provider] });
      const failure = refreshed.errors.get(provider);
      if (failure) throw new Error(`could not refresh ${provider} models: ${failure.message}`, { cause: failure });
      selected = models.getModel(provider, modelId);
    }
    if (!selected && templateId) selected = models.getModel(provider, templateId);
    if (!selected) throw new Error(`Pi has no model ${provider}/${modelId}; run natlang models ${provider} --refresh or set modelOptions.inherit`);
    model = { ...selected, id: modelId, ...modelSettings,
      headers: { ...selected.headers, ...record(modelSettings.headers, 'modelOptions.headers') },
      samplingParams: { ...selected.samplingParams, ...record(modelSettings.samplingParams, 'modelOptions.samplingParams') },
    } as typeof selected;
    return model;
  };
  return {
    async prepare() {
      if (!await models.getAuth(provider, { ...(apiKeyEnv && environment[apiKeyEnv] ? { apiKey: environment[apiKeyEnv] } : {}),
        env: scopedEnv as Record<string, string> }))
        throw new Error(`provider ${provider} has no credentials; run natlang auth login ${provider} or set its API key environment variable`);
      await selectModel();
    },
    async turn(request: ModelTurnRequest): Promise<ModelTurn> {
      const model = await selectModel();
      const context = piContext(request, model);
      const required = request.tool_choice === 'required' ? requiredToolChoice(model.api) : undefined;
      const configuredMax = requestOverride.maxTokens as number | undefined;
      const maxTokens = request.max_tokens === null ? configuredMax :
        configuredMax === undefined ? request.max_tokens : Math.min(request.max_tokens, configuredMax);
      const options = { ...requestOverride,
        ...(request.temperature === undefined ? {} : { temperature: request.temperature }),
        ...(maxTokens === undefined ? {} : { maxTokens }),
        samplingParams: { ...record(requestOverride.samplingParams, 'piOptions.samplingParams'),
          ...(request.seed === null ? {} : { seed: request.seed }) },
        headers: { ...record(requestOverride.headers, 'piOptions.headers'), ...headers },
        ...(Object.keys(payloadOverride).length ? { onPayload: async (payload: unknown, model: unknown) => {
          const hook = requestOverride.onPayload;
          const transformed = typeof hook === 'function' ? await hook(payload, model) : undefined;
          return { ...record(transformed ?? payload, 'provider request payload'), ...payloadOverride };
        } } : {}),
        ...(required ? { toolChoice: required } : {}),
        ...(apiKeyEnv && environment[apiKeyEnv] ? { apiKey: environment[apiKeyEnv] } : {})
      } as ModelsApiStreamOptions<Api>;
      const reply = mode === 'simple' && request.tool_choice !== 'required' ?
        await models.completeSimple(model, context, options as ModelsSimpleStreamOptions) :
        await models.complete(model, context, options);
      if (reply.stopReason === 'error' || reply.stopReason === 'aborted') throw new Error(reply.errorMessage ?? `provider ${provider} failed`);
      const calls = reply.content.filter(block => block.type === 'toolCall');
      const text = reply.content.filter(block => block.type === 'text').map(block => block.text).join('');
      const reasoning = reply.content.filter(block => block.type === 'thinking').map(block => block.thinking).join('\n');
      const raw_calls = calls.map(call => ({ id: call.id, type: 'function', function: { name: call.name,
        arguments: JSON.stringify(call.arguments) } }));
      return { calls: calls.map(call => [call.name, call.arguments as Record<string, unknown>]), raw_calls, text,
        ...(reasoning ? { reasoning } : {}), prompt_tokens: reply.usage.input,
        completion_tokens: reply.usage.output, ...(reply.stopReason === 'length' ? { truncated: true } : {}),
        raw_response: { provider, model: modelId, stop_reason: reply.stopReason, usage: reply.usage } };
    },
    close() { if (typeof requestOverride.sessionId === 'string') cleanupSessionResources(requestOverride.sessionId); },
  };
}
