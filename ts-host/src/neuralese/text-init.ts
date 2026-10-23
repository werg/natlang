/**
 * In-context text initialisation of soft bodies (owner 2026-10-10; server side `natlang_neuralese/text_init.py`).
 *
 * A native soft body (a combinator's instructions, a soft function's or soft skill's body, a system-prompt bank entry)
 * is initialized by capturing the actual native call with a placeholder state, then asking the native server to
 * produce state in that call's context. The resulting nzs1 artifact retains the server's source capture and exact
 * positional import receipts.
 *
 * `initBodyInContext` renders the soft call exactly as the runtime does, with a placeholder body: it runs the call
 * once through a driver that captures the actual chat request instead of sending it. The server then performs native
 * text-to-state initialization and returns the resulting typed native-state artifact and provenance.
 */
import { chatCompletionModelTurn } from '../model/chat-completion.js';
import { neuraleseServerInfo } from '../model/neuralese-info.js';
import { nativeTextOperation, type NativeSourceCapture } from '../model/native-state-client.js';
import type { NeuraleseStore } from '../native/neuralese-store.js';

type Json = Record<string, unknown>;

export type TextInitResult = { id: string; init: { exact: boolean; aligned: boolean; exact_transport: boolean; tokens: number; source_capture?: NativeSourceCapture; source_text_sha256?: string };
  gate: null; context: 'call' };

class Captured extends Error {
  constructor(readonly body: Json) { super('captured'); this.name = 'Captured'; }
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
export async function initBodyInContext(options: { endpoint: string; headers?: Record<string, string>; store?: NeuraleseStore; text: string;
  type: string; render: (bodyId: string, driver: unknown) => Promise<unknown>; placeholderOccurrence?: number }): Promise<TextInitResult> {
  const { endpoint, headers = {}, text, type } = options;
  const server = await neuraleseServerInfo(endpoint, { headers });
  if (!(server.native_state as { enabled?: boolean } | undefined)?.enabled)
    throw Object.assign(new Error('neuralese-native-state-unavailable: text body initialization requires the native state producer'),
      { code: 'neuralese-native-state-unavailable' });
  if (!options.store?.putNativeState || !options.store.getNativeState)
    throw Object.assign(new Error('neuralese-native-archive-unavailable: contextual native bodies need a native-capable archive'),
      { code: 'neuralese-native-archive-unavailable' });
  const client = { endpoint, headers, store: options.store, info: server };
  const placeholder = (await nativeTextOperation(client, 'write-text', { text, type,
    producer: { source_kind: 'native-body-placeholder' } })).result.id;
  const request = await captureCallRequest(driver => options.render(placeholder, driver));
  if (!JSON.stringify(request.messages ?? []).includes(placeholder))
    throw new Error('native body init: the rendered call must contain its exact native placeholder occurrence');
  const { result } = await nativeTextOperation(client, 'init-body', { messages: request.messages,
    tools: request.tools, placeholder, text, type, gate: false,
    ...(options.placeholderOccurrence === undefined ? {} : { placeholder_occurrence: options.placeholderOccurrence }) });
  if (!result.init || result.init.exact !== false || result.init.aligned !== true ||
      result.init.exact_transport !== true || result.init.contextual !== true || result.gate !== null)
    throw new Error('native body init: server must return contextual capture of the actual native call');
  return { id: result.id, init: { ...result.init, source_capture: result.source_capture, source_text_sha256: result.source_text_sha256 } as TextInitResult['init'], gate: null, context: 'call' };
}
