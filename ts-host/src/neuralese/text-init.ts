/**
 * In-context text initialisation of soft bodies (owner 2026-10-10; server side `natlang_neuralese/text_init.py`).
 *
 * A soft body (a combinator's instructions, a soft function's or soft skill's body, a system-prompt bank entry) sits
 * where a text-instructed call has its instructions. Its text initialisation is the value under which the soft call is
 * that text-instructed call: the server takes the instruction tokens as the call's own rendering tokenizes them and
 * inverts its read transport over their embeddings (`POST /v1/neuralese/init_body`). It never encodes the bare
 * sentence through the port (`/encode` writes values that calls read back) or splices raw token embeddings through a
 * transport that changes them.
 *
 * `initBodyInContext` renders the soft call exactly as the runtime does, with a placeholder body: it runs the call
 * once through a driver that captures the chat request instead of sending it. The server then builds the body and
 * gates it: at initialisation the soft call must reproduce the text-instructed call (next-token agreement and KL over
 * the prompt's generation position and a greedy reply). A failed gate is an error unless the caller accepts it.
 */
import { chatCompletionModelTurn, fetchModel } from '../model/chat-completion.js';

type Json = Record<string, unknown>;

export type TextInitGate = { passed: boolean; agreement: number; kl: number; bit_exact: boolean; positions: number;
  max_abs_logit_delta: number; reply_preview?: string };
export type TextInitResult = { id: string; init: { exact: boolean; aligned: boolean; exact_transport: boolean; tokens: number };
  gate: TextInitGate | null;
  /** `call`: the rendered call holds the body; `system-message`: it did not (a piece the sample call does not show),
   * and the body was initialised as the system message of a minimal call. */
  context: 'call' | 'system-message' };

export class TextInitGateError extends Error {
  readonly code = 'neuralese-text-init-gate';
  constructor(readonly text: string, readonly result: TextInitResult) {
    super(`the soft body initialised from ${JSON.stringify(text.slice(0, 80))} does not reproduce the text-instructed call ` +
      `(agreement ${result.gate?.agreement}, KL ${result.gate?.kl}); initialise on heads whose read transport is the identity ` +
      '(raw-token or latent-sketch profiles), or pass acceptFailedGate to keep it as a recorded diagnostic');
    this.name = 'TextInitGateError';
  }
}

class Captured extends Error {
  constructor(readonly body: Json) { super('captured'); this.name = 'Captured'; }
}

async function post(endpoint: string, path: string, body: unknown, headers: Record<string, string> = {}): Promise<Json> {
  const response = await fetchModel(endpoint.replace(/\/$/, '') + path, { method: 'POST',
    headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error(`text init: ${path} failed (${response.status}): ${(await response.text()).slice(0, 500)}`);
  return await response.json() as Json;
}

/**
 * The chat request the runtime sends for `call` (a thunk invoking a callable inside a runtime run), without sending it.
 * `run(driver)` must execute the call with `driver` as the runtime's model.
 */
export async function captureCallRequest(run: (driver: unknown) => Promise<unknown>): Promise<Json> {
  // Chat calls send their body through the transport; decision readouts send their messages to `decide`.
  const driver = Object.assign(chatCompletionModelTurn(async body => { throw new Captured(body); }), { neuralese: true,
    decide: async (request: { messages: unknown[] }) => { throw new Captured({ messages: request.messages }); } });
  try {
    await run(driver);
  } catch (error) {
    let cause: unknown = error;
    for (let depth = 0; cause && depth < 8; depth++) {
      if (cause instanceof Captured) return cause.body;
      cause = (cause as { cause?: unknown }).cause;
    }
    throw error;
  }
  throw new Error('text init: the call finished without a model request');
}

/**
 * Initialise the soft body for `text` in the context of the call that `render(bodyId, driver)` makes: a placeholder
 * block is registered (token embeddings of the text), the call is rendered with it, and the server returns the body.
 */
export async function initBodyInContext(options: { endpoint: string; headers?: Record<string, string>; text: string;
  type: string; render: (bodyId: string, driver: unknown) => Promise<unknown>; gate?: boolean; acceptFailedGate?: boolean;
  replyTokens?: number; fallbackToSystemMessage?: boolean }): Promise<TextInitResult> {
  const { endpoint, headers = {}, text, type } = options;
  const placeholder = String((await post(endpoint, '/v1/neuralese/embed', { text, type }, headers)).id);
  let request = await captureCallRequest(driver => options.render(placeholder, driver));
  let context: TextInitResult['context'] = 'call';
  if (!JSON.stringify(request.messages ?? []).includes(placeholder)) {
    if (!options.fallbackToSystemMessage) throw new Error(`text init: the rendered call does not show the body of ${JSON.stringify(text.slice(0, 80))}`);
    request = { messages: [{ role: 'system', content: [{ type: 'neuralese', id: placeholder }] }, { role: 'user', content: 'Begin.' }] };
    context = 'system-message';
  }
  const answer = await post(endpoint, '/v1/neuralese/init_body', { messages: request.messages, tools: request.tools,
    placeholder, text, type, gate: options.gate ?? true, ...(options.replyTokens ? { reply_tokens: options.replyTokens } : {}) }, headers);
  const result: TextInitResult = { id: String((answer.block as Json).id), init: answer.init as TextInitResult['init'],
    gate: (answer.gate ?? null) as TextInitGate | null, context };
  if (result.gate && !result.gate.passed && !options.acceptFailedGate) throw new TextInitGateError(text, result);
  return result;
}
