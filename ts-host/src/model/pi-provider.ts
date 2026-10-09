/** Optional provider backend. Natlang retains the conversation, tools, retries, and execution loop. */
import { mkdir, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { join } from 'node:path';
import { builtinModels } from '@earendil-works/pi-ai/providers/all';
import { cleanupSessionResources, defaultProviderAuthContext } from '@earendil-works/pi-ai';
import type { Api, AssistantMessage, Context, Credential, CredentialStore, Message,
  ModelsApiStreamOptions, ModelsSimpleStreamOptions, Tool } from '@earendil-works/pi-ai';
import { modelTools } from './chat-completion.js';
import { defaultNatlangConfigDirectory } from '../package/store.js';
import type { ModelStreamProgress, ModelStreamProgressSink, ModelTurn, ModelTurnRequest } from '../contracts.js';

const emptyUsage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };

const CONTROL_CHARACTERS = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/u;
const REPLY_DIAGNOSTIC_EXCERPT_BYTES = 256;

function sha256Text(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

/** Keep only a bounded UTF-8 prefix and suffix for private trajectory diagnostics. */
function boundedUtf8Preview(value: string, limit = 1024): Record<string, unknown> {
  let prefix = '', prefixBytes = 0;
  for (const character of value) {
    const size = Buffer.byteLength(character, 'utf8');
    if (prefixBytes + size > limit) break;
    prefix += character; prefixBytes += size;
  }
  const tail: { character: string; bytes: number }[] = [];
  let tailBytes = 0;
  for (const character of value) {
    const bytes = Buffer.byteLength(character, 'utf8');
    tail.push({ character, bytes }); tailBytes += bytes;
    while (tailBytes > limit) tailBytes -= tail.shift()!.bytes;
  }
  return { encoding: 'final parsed tool arguments JSON string', preview_limit_utf8_bytes: limit,
    prefix_utf8: prefix, prefix_bytes: prefixBytes,
    tail_utf8: tail.map(item => item.character).join(''), tail_bytes: tailBytes,
    truncated: Buffer.byteLength(value, 'utf8') > limit };
}

function containsControlCharacter(value: unknown): boolean {
  if (typeof value === 'string') return CONTROL_CHARACTERS.test(value);
  if (Array.isArray(value)) return value.some(containsControlCharacter);
  if (value && typeof value === 'object') return Object.values(value).some(containsControlCharacter);
  return false;
}

/** JSON-escaped excerpt, capped by encoded bytes so control characters cannot affect logs or terminals. */
function escapedExcerpt(value: string, maxBytes = REPLY_DIAGNOSTIC_EXCERPT_BYTES): string {
  let out = '', used = 0;
  for (const character of value) {
    const escaped = JSON.stringify(character).slice(1, -1);
    const size = Buffer.byteLength(escaped, 'utf8');
    if (used + size > maxBytes) break;
    out += escaped;
    used += size;
  }
  return out;
}

function piReplyDiagnostic(reply: AssistantMessage, visibleText: string, reasoning: string,
  calls: AssistantMessage['content']): Record<string, unknown> | undefined {
  const emptyWithoutAction = visibleText.length === 0 && calls.length === 0;
  const hasControl = containsControlCharacter(reply.content);
  if (!emptyWithoutAction && !hasControl) return;

  const contentJson = JSON.stringify(reply.content);
  return {
    version: 'pi-sdk-parsed-reply-diagnostic/1',
    source: 'pi-sdk-assistant-message-content',
    sdk_stop_reason: reply.stopReason,
    usage_input_tokens: reply.usage.input,
    usage_output_tokens: reply.usage.output,
    visible_text_empty: visibleText.length === 0,
    visible_text_utf8_bytes: Buffer.byteLength(visibleText, 'utf8'),
    visible_text_sha256: sha256Text(visibleText),
    reasoning_present: reasoning.length > 0,
    reasoning_utf8_bytes: Buffer.byteLength(reasoning, 'utf8'),
    tool_call_count: calls.length,
    content_block_count: reply.content.length,
    content_block_types: reply.content.slice(0, 32).map(block => block.type),
    parsed_sdk_content_sha256: sha256Text(contentJson),
    control_character_present: hasControl,
    ...(hasControl ? { parsed_sdk_content_excerpt_escaped: escapedExcerpt(contentJson) } : {}),
    ...(visibleText.length && CONTROL_CHARACTERS.test(visibleText) ?
      { visible_text_excerpt_escaped: escapedExcerpt(visibleText) } : {}),
    ...(reasoning.length && CONTROL_CHARACTERS.test(reasoning) ?
      { reasoning_excerpt_escaped: escapedExcerpt(reasoning) } : {}),
  };
}

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
    async turn(request: ModelTurnRequest, signal?: AbortSignal, onProgress?: ModelStreamProgressSink): Promise<ModelTurn> {
      signal?.throwIfAborted();
      const model = await selectModel();
      const context = piContext(request, model);
      const required = request.tool_choice === 'required' ? requiredToolChoice(model.api) : undefined;
      const configuredMax = requestOverride.maxTokens as number | undefined;
      const maxTokens = request.max_tokens === null ? configuredMax :
        configuredMax === undefined ? request.max_tokens : Math.min(request.max_tokens, configuredMax);
      const options = { ...requestOverride,
        ...(signal ? { signal } : {}),
        // Codex Responses rejects temperature, including natlang's usual zero default.
        ...(request.temperature === undefined || model.api === 'openai-codex-responses' ? {} :
          { temperature: request.temperature }),
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
      const stream = mode === 'simple' && request.tool_choice !== 'required' ?
        models.streamSimple(model, context, options as ModelsSimpleStreamOptions) :
        models.stream(model, context, options);
      const started = Date.now(), startedAt = new Date(started).toISOString();
      const counts = { deltaEvents: 0, deltaBytes: 0, textDeltaEvents: 0, textDeltaBytes: 0,
        thinkingDeltaEvents: 0, thinkingDeltaBytes: 0, toolCallDeltaEvents: 0, toolCallDeltaBytes: 0 };
      const deltaFingerprints = new Map<string, { occurrences: number; utf8Bytes: number }>();
      const MAX_TRACKED_DELTA_FINGERPRINTS = 4096;
      let untrackedDeltaEvents = 0, adjacentDuplicateDeltaEvents = 0;
      let previousDeltaFingerprint = '', currentRepeatRun = 0, longestRepeatRun = 0;
      const emit = (status: ModelStreamProgress['status']) => {
        if (!onProgress) return;
        const progress: ModelStreamProgress = { status, ...counts, startedAt,
          observedAt: new Date().toISOString(), elapsedMs: Date.now() - started };
        try { onProgress(progress); } catch { /* progress observers must not affect model turns */ }
      };
      let lastEmitted = 0;
      const finalMessage = stream.result();
      void finalMessage.catch(() => undefined);
      let reply: AssistantMessage;
      try {
        for await (const event of stream) {
          let kind: 'text' | 'thinking' | 'toolcall' | undefined;
          let delta: string | undefined;
          if (event.type === 'text_delta') { kind = 'text'; delta = event.delta; }
          else if (event.type === 'thinking_delta') { kind = 'thinking'; delta = event.delta; }
          else if (event.type === 'toolcall_delta') { kind = 'toolcall'; delta = event.delta; }
          if (!kind || !delta) continue;
          const bytes = Buffer.byteLength(delta, 'utf8');
          if (!bytes) continue;
          counts.deltaEvents++; counts.deltaBytes += bytes;
          if (kind === 'text') { counts.textDeltaEvents++; counts.textDeltaBytes += bytes; }
          else if (kind === 'thinking') { counts.thinkingDeltaEvents++; counts.thinkingDeltaBytes += bytes; }
          else {
            counts.toolCallDeltaEvents++; counts.toolCallDeltaBytes += bytes;
            const fingerprint = sha256Text(delta), prior = deltaFingerprints.get(fingerprint);
            if (prior) { prior.occurrences++; }
            else if (deltaFingerprints.size < MAX_TRACKED_DELTA_FINGERPRINTS)
              deltaFingerprints.set(fingerprint, { occurrences: 1, utf8Bytes: bytes });
            else untrackedDeltaEvents++;
            if (fingerprint === previousDeltaFingerprint) {
              adjacentDuplicateDeltaEvents++; currentRepeatRun++;
            } else { previousDeltaFingerprint = fingerprint; currentRepeatRun = 1; }
            longestRepeatRun = Math.max(longestRepeatRun, currentRepeatRun);
          }
          const now = Date.now();
          if (lastEmitted === 0 || now - lastEmitted >= 15_000) { emit('progress'); lastEmitted = now; }
        }
        reply = await finalMessage;
        if (reply.stopReason === 'error' || reply.stopReason === 'aborted')
          throw new Error(reply.errorMessage ?? `provider ${provider} failed`);
        emit('completed');
      } catch (error) {
        emit('failed');
        throw error;
      }
      const calls = reply.content.filter(block => block.type === 'toolCall');
      const text = reply.content.filter(block => block.type === 'text').map(block => block.text).join('');
      const reasoning = reply.content.filter(block => block.type === 'thinking').map(block => block.thinking).join('\n');
      const replyDiagnostic = piReplyDiagnostic(reply, text, reasoning, calls);
      const raw_calls = calls.map(call => ({ id: call.id, type: 'function', function: { name: call.name,
        arguments: JSON.stringify(call.arguments) } }));
      // Delta events describe the streamed path; the completed AssistantMessage is authoritative
      // for what the collector saved. Keep payload-free measurements of both so revised argument
      // deltas are not mistaken for an equally large final tool call.
      const piStreamObservation = {
        version: 'pi-stream-final-comparison/1',
        streamed_delta_events: counts.deltaEvents,
        streamed_delta_utf8_bytes: counts.deltaBytes,
        streamed_tool_call_delta_events: counts.toolCallDeltaEvents,
        streamed_tool_call_delta_utf8_bytes: counts.toolCallDeltaBytes,
        streamed_text_delta_utf8_bytes: counts.textDeltaBytes,
        streamed_thinking_delta_utf8_bytes: counts.thinkingDeltaBytes,
        final_call_count: raw_calls.length,
        final_raw_calls_json_utf8_bytes: Buffer.byteLength(JSON.stringify(raw_calls), 'utf8'),
        final_calls: raw_calls.map((call, index) => ({ index, name: call.function.name,
          arguments_utf8_bytes: Buffer.byteLength(call.function.arguments, 'utf8'),
          arguments_sha256: sha256Text(call.function.arguments),
          arguments_preview: boundedUtf8Preview(call.function.arguments) })),
        tool_delta_repetition: {
          fingerprint: 'sha256 of exact delta string; no delta contents retained',
          unique_fingerprint_limit: MAX_TRACKED_DELTA_FINGERPRINTS,
          tracked_unique_fingerprints: deltaFingerprints.size,
          untracked_delta_events_after_limit: untrackedDeltaEvents,
          repeated_events_within_tracked_fingerprints: [...deltaFingerprints.values()]
            .reduce((sum, item) => sum + Math.max(0, item.occurrences - 1), 0),
          adjacent_duplicate_events: adjacentDuplicateDeltaEvents,
          longest_adjacent_repeat_run: longestRepeatRun,
          top_repeated_fingerprints: [...deltaFingerprints.entries()]
            .filter(([, item]) => item.occurrences > 1)
            .sort((a, b) => b[1].occurrences - a[1].occurrences || a[0].localeCompare(b[0]))
            .slice(0, 8).map(([sha256, item]) => ({ sha256, occurrences: item.occurrences,
              delta_utf8_bytes: item.utf8Bytes }))
        },
        comparison_scope: 'stream delta byte totals vs final parsed sdk call arguments; no delta-to-call identity asserted'
      };
      return { calls: calls.map(call => [call.name, call.arguments as Record<string, unknown>]), raw_calls, text,
        ...(reasoning ? { reasoning } : {}), prompt_tokens: reply.usage.input,
        completion_tokens: reply.usage.output, ...(reply.stopReason === 'length' ? { truncated: true } : {}),
        raw_response: { provider, model: modelId, stop_reason: reply.stopReason, usage: reply.usage,
          pi_stream_observation: piStreamObservation,
          ...(replyDiagnostic ? { pi_reply_diagnostic: replyDiagnostic } : {}) } };
    },
    close() { if (typeof requestOverride.sessionId === 'string') cleanupSessionResources(requestOverride.sessionId); },
  };
}
