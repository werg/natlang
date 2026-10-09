/**
 * The `ai` service: pi-ai behind the declared types. Model lookup, one model turn (streamed into `pi.live` when asked),
 * deferred polling, failure classification, token estimates and pi's argument validation. Provider transports, auth,
 * catalogs, transcript conversion and the overflow and retry pattern tables stay inside pi-ai.
 */
import { copyJson, type Context } from '@earendil-works/chord';
import type { Api, AssistantMessage, Message, Model, SimpleStreamOptions } from '@earendil-works/pi-ai';
import { estimateMessageTokens } from '@earendil-works/pi-ai/utils/estimate';
import { isContextOverflow } from '@earendil-works/pi-ai/utils/overflow';
import { isRetryableAssistantError, retryDelayMs } from '@earendil-works/pi-ai/utils/retry';
import { validateToolArguments } from '@earendil-works/pi-ai/utils/validation';
import type { TaskRuntime } from '../vendor/durable/src/types.ts';
import { streamResponse } from '../vendor/durable/src/harness/generation.ts';
import type { DeferredHandle, ModelInfo, ModelRef, RetryPolicy, StreamOptions, ThinkingLevel } from '../types.ts';
import { ONCE_EFFECTS, type NeuraleseStore } from '@natlang/node';
import { plain } from './durable.ts';
import { blockPositions, modelReader, isNeuraleseContent, neuraleseBlockIds, neuraleseBlocks, noteBlockMeta } from './natlang-provider.ts';
import { ProviderDoc } from '../vendor/durable/src/harness/provider.ts';
import { collectViewBlocks, forceStoredViews, forceView, recallNote, representationKey, storedCall, storedCalls, VIEW_FAILED, VIEW_UNAVAILABLE,
  type ForcedView, type ViewDocs } from './views.ts';

/**
 * A provider's message as the Session line stores it: strict JSON, or a rejection as pi-durable's commit would give
 * (a message that is not JSON faults the generation; it is never cleaned into a valid-looking one).
 */
function storable(message: AssistantMessage, phase: { failed?: string }): AssistantMessage {
  try { return copyJson(message, { omitUndefinedProperties: true }) as unknown as AssistantMessage; }
  catch (error) {
    phase.failed = `the provider returned a message that cannot be stored: ${error instanceof Error ? error.message : String(error)}`;
    throw new Error(`The provider returned a message that cannot be stored (${error instanceof Error ? error.message : String(error)}), ` +
      'so this request failed. Do not work around it, retry it, or build a message yourself: end this call with ' +
      'return_result status "failed" and this reason. The harness faults the generation.');
  }
}

/** The answer of a model that does not read Neuralese to a request whose messages hold blocks. */
function unreadable(model: Model<Api>): AssistantMessage {
  return { role: 'assistant', content: [], api: model.api, provider: model.provider, model: model.id,
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    stopReason: 'error', timestamp: Date.now(),
    errorMessage: `neuralese-unsupported-backend: the messages hold Neuralese blocks, but ${model.provider}/${model.id} reads text ` +
      '(declare a Neuralese reader for it: --agent-transport natlang --agent-reader DIALECT)' };
}

/** The answer when a stored view call could not be forced for a Neuralese reader (views.ts): the request is not sent. */
function unforced(model: Model<Api>, error: unknown, aborted: boolean): AssistantMessage {
  return { ...unreadable(model), stopReason: aborted ? 'aborted' : 'error',
    errorMessage: error instanceof Error ? error.message : String(error) };
}

type Runtime = TaskRuntime<unknown, unknown, unknown, Record<string, unknown>>;

/** Options of one turn: the pinned stream options plus the thinking level, the provider session and a token cap. */
export type TurnOptions = StreamOptions & { thinkingLevel: ThinkingLevel; sessionId: string; maxTokens?: number };

/**
 * `streamAttempt`: for a generation task, the attempt its request streams into pi.live (pi-durable's generation
 * request always streams; compaction never does), so a turn the executor sends without `live` still streams.
 * `store`: the runtime's Neuralese store, where token estimates read each block's length and blocks are archived.
 * `collect`: the task's requests are its conversation's context (a generation task), so a request to a Neuralese reader
 * collects the conversation's blocks on the server when the context's head moved (views.ts `collectViewBlocks`).
 * `agent`: the conversation's agent model, the reader token estimates are for unless they name another;
 * `owner`: the conversation's provider session ID, which views forced while preparing an estimate are held under.
 *
 * For a model whose reader is a Neuralese dialect, `turn` sends each stored view call (views.ts) as its block, forced
 * once per call and dialect under the conversation's owner (its provider session ID); a call that cannot be forced
 * fails the turn (`neuralese-view-unavailable`: retryable; `neuralese-view-failed`: not), never falling back to text.
 */
export function aiService(runtime: Runtime, context: Context, streamAttempt?: () => number, phase: { failed?: string } = {},
    store?: NeuraleseStore, service: { collect?: boolean; agent?: ModelRef; owner?: string } = {}) {
  const resolve = (ref: ModelRef): Model<Api> => {
    const model = ref && runtime.models.getModel(ref.provider, ref.modelId);
    if (!model) throw new Error(`Model ${ref?.provider}/${ref?.modelId} is not available`);
    return model;
  };
  const options = (turn: TurnOptions): SimpleStreamOptions => {
    const { thinkingLevel, ...rest } = turn;
    return { ...rest, signal: runtime.signal, ...(thinkingLevel === 'off' ? {} : { reasoning: thinkingLevel }) } as SimpleStreamOptions;
  };
  const viewDocs = (): ViewDocs => ({ conversationId: runtime.conversationId, read: runtime,
    commit: (change, at) => runtime.commit(async tx => { await change(tx); return undefined; }, at) });
  /** The agent's model as `blockMeta` resolved it, when no `agent` was given. */
  let agentRef: ModelRef | undefined = service.agent;
  /** The model an estimate is for (default: the agent's), and its reader; no model reads text. */
  const estimated = (ref?: ModelRef) => {
    const target = ref ?? agentRef;
    const model = target && typeof target.provider === 'string' ? runtime.models.getModel(target.provider, target.modelId) : undefined;
    return { model, reader: modelReader(model) };
  };
  /**
   * The synchronous index of forced views (stored call → block per representation), filled from the durable memo by
   * `blockMeta`; and why a call could not be forced, for the estimate's error.
   */
  const forcedViews = new Map<string, ForcedView>();
  const unforcedViews = new Map<string, string>();
  const viewKey = (call: string, dialect: string) => `${call}\0${representationKey(dialect)}`;
  const viewPositions = (call: string, dialect: string): number => {
    const forced = forcedViews.get(viewKey(call, dialect));
    if (forced) return forced.length;
    const cause = unforcedViews.get(viewKey(call, dialect));
    throw new Error(`neuralese-unknown-view-length: the stored view call ${call} is not forced for the ${JSON.stringify(dialect)} reader, ` +
      `so its block's length is unknown${cause ? ` (forcing it failed: ${cause})` : '; call await ai.blockMeta(messages) before ' +
      'estimating them, which forces it'}. The estimate counts the block the reader is sent, never the text form.`);
  };
  return {
    // A phase's request and poll are external effects: an executor that runs the same call again within the phase
    // (a retried eval, a function called twice) gets the provider's earlier answer, not a second request.
    [ONCE_EFFECTS]: ['turn', 'poll'],
    model(ref: ModelRef): ModelInfo | null {
      const model = ref && typeof ref.provider === 'string' && typeof ref.modelId === 'string' ? runtime.models.getModel(ref.provider, ref.modelId) : undefined;
      if (!model) return null;
      return { provider: model.provider, modelId: model.id, name: model.name, contextWindow: model.contextWindow ?? 0,
        maxTokens: model.maxTokens ?? 0, reasoning: Boolean(model.reasoning) };
    },
    async turn(model: ModelRef, messages: Message[], turn: TurnOptions, live?: { attempt: number }): Promise<AssistantMessage> {
      // A generation always has the user's input in its context: an empty list is a slip in the eval, not a request.
      if (!Array.isArray(messages) || messages.length === 0) throw new Error('ai.turn got no messages. ' +
        'The context through the cutoff is never empty: check whether the eval cleared the list it was building from ' +
        '(for example messages.length = 0 on the same array it then copies back), and pass the context\'s messages.');
      const resolved = resolve(model);
      const reader = modelReader(resolved);
      // A Neuralese block reaches only a model that reads its dialect (natlang-provider.ts); there is no text fallback.
      if (reader.kind !== 'neuralese' && messages.some(message => neuraleseBlocks(message))) return storable(unreadable(resolved), phase);
      let sent = messages;
      if (reader.kind === 'neuralese') {
        // Stored view calls at this reader's dialect, under the conversation's owner (views.ts).
        const docs = viewDocs();
        const views = { models: runtime.models, model: resolved, owner: turn.sessionId, store };
        try { sent = await forceStoredViews(messages, docs, views, context); }
        catch (error) { return storable(unforced(resolved, error, Boolean(runtime.signal?.aborted)), phase); }
        if (service.collect) {
          // A generation's request is the context: when its head moved (the first request, a compaction, a reset), the
          // owner's blocks are collected to the ones it references. Housekeeping: a failure is reported, not fatal.
          try {
            const head = (await runtime.context(runtime.conversationId, context)).head?.id ?? 0;
            await collectViewBlocks(docs, views, new Set(neuraleseBlockIds(sent)), head, context);
          } catch (error) { runtime.report(error); }
        }
      }
      const attempt = live?.attempt ?? streamAttempt?.();
      return storable(await (attempt !== undefined ? streamResponse(runtime as never, resolved, sent, options(turn), attempt, context) :
        runtime.models.completeSimple(resolved, { messages: sent }, options(turn))), phase);
    },
    poll(model: ModelRef, handle: DeferredHandle): Promise<AssistantMessage> {
      return runtime.models.fetchDeferred(resolve(model), handle as never, { signal: runtime.signal }).then(message => storable(message, phase));
    },
    async cancel(model: ModelRef, handle: DeferredHandle): Promise<void> {
      await runtime.models.cancelDeferred(resolve(model), handle as never);
    },
    failure(message: AssistantMessage): { overflow: boolean; retryable: boolean } {
      // A stored view that could not be forced says itself whether a retry may succeed (views.ts).
      const view = message.errorMessage?.startsWith(VIEW_UNAVAILABLE) ? true : message.errorMessage?.startsWith(VIEW_FAILED) ? false : undefined;
      return { overflow: message.stopReason === 'error' && view === undefined && isContextOverflow(message),
        retryable: message.stopReason === 'error' && (view ?? isRetryableAssistantError(message)) };
    },
    retryDelayMs(policy: RetryPolicy, attempt: number): number { return retryDelayMs(policy, attempt); },
    estimateTokens(messages: Message[], model?: ModelRef): number[] {
      // pi-ai's estimate knows text and images; a Neuralese block occupies its length in positions, read from the store.
      // The estimate is the reader's: for a Neuralese reader a stored view call is the block it is sent (its length,
      // from the index blockMeta filled) and the recall note; a text reader reads the text form, which pi-ai counts.
      const { reader } = estimated(model);
      return messages.map(message => {
        const views = reader.kind === 'neuralese' && Array.isArray(message.content) && message.content.some(part => storedCall(part));
        if (!views && !neuraleseBlocks(message)) return estimateMessageTokens(message);
        let positions = blockPositions(message, store);
        const text = (message.content as unknown[]).flatMap(part => {
          if (isNeuraleseContent(part)) return [];
          const ref = storedCall(part);
          if (!ref || reader.kind !== 'neuralese') return [part];
          positions += viewPositions(ref.call, reader.dialect);
          return [{ type: 'text', text: recallNote(ref.call) }];
        });
        return estimateMessageTokens({ ...message, content: text } as Message) + positions;
      });
    },
    async blockMeta(messages: Message[], model?: ModelRef): Promise<string[]> {
      const ids = neuraleseBlockIds(messages).filter(id => !store?.peek?.(id));
      const calls = storedCalls(messages);
      if (!ids.length && !calls.length) return [];
      let ref = model ?? agentRef;
      if (!ref) {
        ref = (await (runtime as { agent?: (context: Context) => Promise<{ model?: ModelRef }> }).agent?.(context))?.model;
        agentRef = ref;
      }
      const { model: resolved, reader } = estimated(ref);
      // Stored view calls this Neuralese reader has not forced are forced now (the memo when there is one), so the
      // estimate counts their blocks; one that cannot be forced stays unknown, and its estimate fails naming it.
      const unknownViews: string[] = [];
      if (reader.kind === 'neuralese' && resolved) {
        const pending = calls.filter(call => !forcedViews.has(viewKey(call, reader.dialect)));
        const owner = pending.length ? service.owner ?? (await runtime.snapshot(ProviderDoc, runtime.conversationId, context))?.sessionId : undefined;
        await Promise.all(pending.map(async call => {
          const key = viewKey(call, reader.dialect);
          try {
            if (!owner) throw new Error('the conversation has no provider session ID yet, which its blocks are held under');
            forcedViews.set(key, await forceView(viewDocs(), call, { models: runtime.models, model: resolved, owner, store }, context));
            unforcedViews.delete(key);
          } catch (error) {
            unforcedViews.set(key, error instanceof Error ? error.message : String(error));
            unknownViews.push(`view:${call}`);
          }
        }));
      }
      // Blocks the store already describes need nothing; the rest are asked of the agent model's server, once each.
      if (!ids.length || !store || !resolved || reader.kind !== 'neuralese') return [...ids, ...unknownViews];
      return [...await noteBlockMeta(resolved.baseUrl, ids, store, { signal: runtime.signal }), ...unknownViews];
    },
    validateArguments(toolName: string, parameters: Record<string, unknown>, args: Record<string, unknown>): { args: Record<string, unknown> } | { error: string } {
      try {
        const call = { type: 'toolCall' as const, id: 'validate', name: toolName, arguments: args } as never;
        return { args: plain(validateToolArguments({ name: toolName, description: '', parameters } as never, call)) as Record<string, unknown> };
      } catch (error) { return { error: error instanceof Error ? error.message : String(error) }; }
    },
  };
}

export type AiService = ReturnType<typeof aiService>;

export const AI_DECLARATION = `/** pi-ai: the agent's model provider. Types are those of types.ts. */
/** What the harness needs to know about a model; null when the catalog does not have it. */
export function model(ref: ModelRef): ModelInfo | null;
/**
 * Run one model turn over messages and return the finished message. Never throws for provider trouble: a failure is
 * stopReason "error" with errorMessage, and cancelling the task gives "aborted". options: the pinned stream options plus
 * thinkingLevel, sessionId (the conversation's provider session) and an optional maxTokens. With live, the host
 * publishes the growing answer into pi.live.generation for that attempt, and finishes publishing before this returns.
 * Messages holding a Neuralese block, sent to a model that reads text, fail with "neuralese-unsupported-backend" (there
 * is no text fallback). To a model that reads Neuralese, a text part with a stored call (its stored field) is sent as
 * the call's block; when the block cannot be written the answer is stopReason "error" with an errorMessage starting
 * "neuralese-view-unavailable" (failure says retryable) or "neuralese-view-failed". Rejects only when the model is not in
 * the catalog.
 */
export function turn(model: ModelRef, messages: Message[], options: StreamOptions & { thinkingLevel: ThinkingLevel; sessionId: string; maxTokens?: number }, live?: { attempt: number }): Promise<AssistantMessage>;
/** Check a deferred response once; still pending: stopReason "deferred" with a new handle. */
export function poll(model: ModelRef, handle: DeferredHandle): Promise<AssistantMessage>;
/** Best-effort cancel of a deferred response; may reject. */
export function cancel(model: ModelRef, handle: DeferredHandle): Promise<void>;
/** For a stopReason "error" message: overflow, the request was too long for the model; retryable, a transient provider or network failure (not quota or billing). */
export function failure(message: AssistantMessage): { overflow: boolean; retryable: boolean };
/** The wait before retry attempt n: min(baseDelayMs * 2^(n-1), maxAgentDelayMs ?? 60000). */
export function retryDelayMs(policy: RetryPolicy, attempt: number): number;
/**
 * pi's token estimate of each message as model (default: the agent's model) reads it: characters / 3.5, rounded up; an
 * image counts as 4800 characters, a Neuralese block as its length (the context positions it occupies), from the
 * runtime's Neuralese store. For a model that reads Neuralese, a text part with a stored call (its stored field) counts
 * as the call's block (its length) and the recall note sent after it, as turn sends them. Throws
 * neuralese-unknown-block-length, naming the block, when the store has no metadata for it, and
 * neuralese-unknown-view-length, naming the call, when the call's view is not forced (blockMeta forces it).
 */
export function estimateTokens(messages: Message[], model?: ModelRef): number[];
/**
 * Make messages known to estimateTokens for model (default: the agent's model): metadata of Neuralese blocks the
 * runtime's store lacks is fetched once from the model's server and kept in the store; for a model that reads
 * Neuralese, each stored call is forced (its view written once, or read from the conversation's memo). Returns what is
 * still unknown: block IDs (the block is lost: neither the store nor the server has it) and "view:CALL" for a stored
 * call whose view could not be forced (estimateTokens then says why); empty when everything is known.
 */
export function blockMeta(messages: Message[], model?: ModelRef): Promise<string[]>;
/** Validate and coerce tool arguments against the tool's JSON Schema parameters, as pi does; error is pi's exact message. */
export function validateArguments(toolName: string, parameters: Record<string, unknown>, args: Record<string, unknown>): { args: Record<string, unknown> } | { error: string };`;
