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
import { ONCE_EFFECTS } from '@natlang/node';
import { plain } from './durable.ts';

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

type Runtime = TaskRuntime<unknown, unknown, unknown, Record<string, unknown>>;

/** Options of one turn: the pinned stream options plus the thinking level, the provider session and a token cap. */
export type TurnOptions = StreamOptions & { thinkingLevel: ThinkingLevel; sessionId: string; maxTokens?: number };

/**
 * `streamAttempt`: for a generation task, the attempt its request streams into pi.live (pi-durable's generation
 * request always streams; compaction never does), so a turn the executor sends without `live` still streams.
 */
export function aiService(runtime: Runtime, context: Context, streamAttempt?: () => number, phase: { failed?: string } = {}) {
  const resolve = (ref: ModelRef): Model<Api> => {
    const model = ref && runtime.models.getModel(ref.provider, ref.modelId);
    if (!model) throw new Error(`Model ${ref?.provider}/${ref?.modelId} is not available`);
    return model;
  };
  const options = (turn: TurnOptions): SimpleStreamOptions => {
    const { thinkingLevel, ...rest } = turn;
    return { ...rest, signal: runtime.signal, ...(thinkingLevel === 'off' ? {} : { reasoning: thinkingLevel }) } as SimpleStreamOptions;
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
    turn(model: ModelRef, messages: Message[], turn: TurnOptions, live?: { attempt: number }): Promise<AssistantMessage> {
      // A generation always has the user's input in its context: an empty list is a slip in the eval, not a request.
      if (!Array.isArray(messages) || messages.length === 0) return Promise.reject(new Error('ai.turn got no messages. ' +
        'The context through the cutoff is never empty: check whether the eval cleared the list it was building from ' +
        '(for example messages.length = 0 on the same array it then copies back), and pass the context\'s messages.'));
      const resolved = resolve(model);
      const attempt = live?.attempt ?? streamAttempt?.();
      return (attempt !== undefined ? streamResponse(runtime as never, resolved, messages, options(turn), attempt, context) :
        runtime.models.completeSimple(resolved, { messages }, options(turn))).then(message => storable(message, phase));
    },
    poll(model: ModelRef, handle: DeferredHandle): Promise<AssistantMessage> {
      return runtime.models.fetchDeferred(resolve(model), handle as never, { signal: runtime.signal }).then(message => storable(message, phase));
    },
    async cancel(model: ModelRef, handle: DeferredHandle): Promise<void> {
      await runtime.models.cancelDeferred(resolve(model), handle as never);
    },
    failure(message: AssistantMessage): { overflow: boolean; retryable: boolean } {
      return { overflow: message.stopReason === 'error' && isContextOverflow(message),
        retryable: message.stopReason === 'error' && isRetryableAssistantError(message) };
    },
    retryDelayMs(policy: RetryPolicy, attempt: number): number { return retryDelayMs(policy, attempt); },
    estimateTokens(messages: Message[]): number[] { return messages.map(message => estimateMessageTokens(message)); },
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
 * Rejects only when the model is not in the catalog.
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
/** pi's token estimate of each message: characters / 3.5, rounded up; an image counts as 4800 characters. */
export function estimateTokens(messages: Message[]): number[];
/** Validate and coerce tool arguments against the tool's JSON Schema parameters, as pi does; error is pi's exact message. */
export function validateArguments(toolName: string, parameters: Record<string, unknown>, args: Record<string, unknown>): { args: Record<string, unknown> } | { error: string };`;
