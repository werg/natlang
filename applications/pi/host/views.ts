/**
 * Stored calls of `view`, forced per reader (plans/neuralese/DECISIONS.md 2026-10-09, "representation chosen by use",
 * step 3). A long tool output is not turned into a summary when it arrives, because its readers are not known then. It
 * is stored as a call of the builtin `view(value, instructions?)`, with two inputs: the whole output (`ToolOutputs`, also
 * `recall`'s store) and the agent's intent when it made the call (`ViewIntents`, records.py `intent()`). The tool result
 * message keeps what text readers read, the companion's shape of the output, in a text part whose `stored` field
 * references the call (types.ts `StoredCallRef`).
 *
 * Each reader forces the call at its own representation:
 * - a text reader reads that text part as it is. pi never runs view's crisp instance: the text form of a long output is
 *   the companion's shape (head and tail, or `shape.nl`'s exact lines and gist), which keeps every shown line exact and
 *   fits a budget, and nobody would read a second text form beside it;
 * - a Neuralese reader (the agent model's declared reader, natlang-provider.ts `AgentReader`) is sent view's Neuralese
 *   instance in its dialect in place of that part, followed by the note the harness bench trains with (records.py
 *   `recall_note`). `forceStoredViews` runs in the ai service's `turn` (ai.ts), for every request to such a model.
 *
 * The Neuralese instance is written by the agent model's own server (`POST /v1/neuralese/view`, ts-host
 * neuralese/view.ts `serverViewer`), the server that reads the block. Forcing is memoized durably per (call,
 * representation) in the conversation's `ViewForcings` documents (block ID, length, dialect), so a call is written once
 * whatever the turns, retries or restarts. The block is archived in the runtime's Neuralese store and pinned on the
 * server under the conversation's owner ID (its provider session ID, `x-natlang-owner`); `collectViewBlocks` unpins
 * what left the context and collects the owner's blocks with the referenced set when the context's head changes
 * (the first request, and after each compaction or reset). A block the server lost is restored from the archive.
 *
 * Failures are loud: a Neuralese reader never gets the text form instead. `ViewForcingError.transient` says whether the
 * request may succeed later (the server unreachable, 429, 5xx); the ai service turns it into a failed provider message
 * that pi's retry policy retries (transient) or that fails the turn.
 */
import type { Context } from '@earendil-works/chord';
import type { Api, Model, Models } from '@earendil-works/pi-ai';
import { HttpNeuraleseStore, OWNER_HEADER, serverViewer, withRestoredBlocks, type NeuraleseStore } from 'natlang:runtime';
import { defineDoc, defineDocFamily } from '../vendor/durable/src/documents.ts';
import type { ConversationId, DocumentReader, Tx } from '../vendor/durable/src/types.ts';
import type { StoredCallRef } from '../types.ts';
import { modelReader } from './natlang-provider.ts';

/** Tool outputs longer than this many characters are stored as view calls (records.py VIEW_CHARS: the bench's views). */
export const VIEW_LIMIT = 2_000;

/** The full text of a long tool output, by tool call ID: view's `value` input, and what `recall(handle)` returns. */
export const ToolOutputs = defineDocFamily<{ tool: string; text: string }, { tool: string; text: string }>({
  kind: 'pi.companion.output', version: 1, scope: 'conversation', history: 'latest', fork: 'current', family: true,
  initial: seed => seed,
});

/**
 * The agent's intent for a tool call, by call ID (records.py `intent()`): view's `instructions` input. Recorded after
 * every response, whatever the agent model reads (a text document, cheap), so a conversation that switches to a
 * Neuralese reader can force the views of earlier calls.
 */
export const ViewIntents = defineDocFamily<{ intent: string }, { intent: string }>({
  kind: 'pi.view.intent', version: 1, scope: 'conversation', history: 'latest', fork: 'current', family: true,
  initial: seed => seed,
});

/** A forced view: the block, its length (context positions) and dialect. */
export type ForcedView = { id: string; length: number; dialect: string };

/** The forced results of one stored view call, by representation (`representationKey`). */
export const ViewForcings = defineDocFamily<{ forced: Record<string, ForcedView> }, null>({
  kind: 'pi.view.forced', version: 1, scope: 'conversation', history: 'latest', fork: 'current', family: true,
  initial: () => ({ forced: {} }),
});

/**
 * The view blocks the conversation pinned on its server (block ID → call ID), and the context head of its last
 * collection. A fork has a new owner, so it starts empty.
 */
export const ViewBlocks = defineDoc<{ pinned: Record<string, string>; collectedHead?: number }>({
  kind: 'pi.view.blocks', version: 1, scope: 'conversation', history: 'latest', fork: 'initial', initial: () => ({ pinned: {} }),
});

/** The memo key of a representation. */
export const representationKey = (dialect: string) => `neuralese:${dialect}`;

/** What the agent wanted from a tool call when it made it: the call, then the turn's reasoning and text (records.py `intent`). */
export function viewIntent(assistant: { content: readonly { type: string; thinking?: string; text?: string }[] },
    call: { name: string; arguments: unknown }): string {
  const reasoning = assistant.content.filter(part => part.type === 'thinking').map(part => part.thinking ?? '').join('\n').trim();
  const text = assistant.content.filter(part => part.type === 'text').map(part => part.text ?? '').join('\n').trim();
  const lines = ['The agent made this tool call and reads its output next:', `${call.name} ${JSON.stringify(call.arguments)}`];
  if (reasoning || text) lines.push('Its reasoning when it made the call:', [reasoning, text].filter(Boolean).join('\n'));
  return lines.join('\n');
}

/** The text after a view block, saying how to get the whole output (records.py `recall_note`). */
export const recallNote = (call: string) => `  // view of the output; recall("${call}") returns all of it`;

/** The stored call a content part references, if it is the text form of one. */
export function storedCall(part: unknown): StoredCallRef | undefined {
  const found = part as { type?: unknown; stored?: { function?: unknown; call?: unknown } } | null;
  return found?.type === 'text' && found.stored?.function === 'view' && typeof found.stored.call === 'string' ?
    found.stored as StoredCallRef : undefined;
}

/** Where a conversation's stored calls live: its documents, read and committed through a task runtime or a handle. */
export type ViewDocs = {
  readonly conversationId: ConversationId;
  readonly read: DocumentReader;
  commit(change: (tx: Tx) => Promise<void>, context: Context): Promise<void>;
};

/**
 * The reader that forces: the model (its declared reader and server), the catalog its auth comes from, the owner ID
 * its blocks are held and pinned under, and the runtime's Neuralese store, which archives every block.
 */
export type ViewReader = { models: Models; model: Model<Api>; owner: string; store?: NeuraleseStore };

/** A view that could not be forced; `transient`: the server was unreachable or overloaded, and a retry may succeed. */
export class ViewForcingError extends Error {
  readonly transient: boolean;
  constructor(message: string, transient: boolean) {
    super(message);
    this.name = 'ViewForcingError';
    this.transient = transient;
  }
}
/** The error codes a forcing failure's message starts with. */
export const VIEW_UNAVAILABLE = 'neuralese-view-unavailable';
export const VIEW_FAILED = 'neuralese-view-failed';

/** A failure of the server or the network that a later attempt may not meet. */
const TRANSIENT = /\((429|5\d\d)\)|fetch failed|ECONNREFUSED|ECONNRESET|EPIPE|socket hang up|other side closed|timed? ?out/i;

/** The agent server's root and the headers of its requests: the model's auth and the owner. */
async function serverAccess(reader: ViewReader): Promise<{ root: string; headers: Record<string, string> }> {
  const auth = (await reader.models.getAuth(reader.model))?.auth;
  const headers: Record<string, string> = {};
  for (const [name, value] of Object.entries(auth?.headers ?? {})) if (typeof value === 'string') headers[name] = value;
  if (auth?.apiKey) headers.authorization = `Bearer ${auth.apiKey}`;
  headers[OWNER_HEADER] = reader.owner;
  return { root: (auth?.baseUrl ?? reader.model.baseUrl).replace(/\/+$/, '').replace(/\/v1$/, ''), headers };
}

/** Forcings in flight in this process, by owner, call and representation: the hook's pre-force and a request share one. */
const inflight = new Map<string, Promise<ForcedView>>();

/**
 * The stored view call `call` at `reader`'s Neuralese dialect: the memo when there is one, else written once (a
 * pre-force already running is joined), archived, pinned and memoized. Throws `ViewForcingError`.
 */
export async function forceView(docs: ViewDocs, call: string, reader: ViewReader, context: Context): Promise<ForcedView> {
  const declared = modelReader(reader.model);
  if (declared.kind !== 'neuralese') throw new Error(`forceView: ${reader.model.provider}/${reader.model.id} reads text, ` +
    'and a text reader reads the stored call\'s text form, which the message holds');
  const key = representationKey(declared.dialect);
  const memo = (await docs.read.snapshot(ViewForcings, docs.conversationId, call, context))?.forced?.[key];
  if (memo) return { ...memo };
  const flight = `${reader.owner}\0${call}\0${key}`;
  let running = inflight.get(flight);
  if (!running) {
    running = writeView(docs, call, declared.dialect, reader, context).finally(() => inflight.delete(flight));
    inflight.set(flight, running);
  }
  return running;
}

async function writeView(docs: ViewDocs, call: string, dialect: string, reader: ViewReader, context: Context): Promise<ForcedView> {
  const fail = (transient: boolean, cause: string) => new ViewForcingError(`${transient ? VIEW_UNAVAILABLE : VIEW_FAILED}: the ` +
    `view of tool call ${call}'s output for the ${JSON.stringify(dialect)} reader of ${reader.model.provider}/${reader.model.id} ` +
    `was not written: ${cause}`, transient);
  const output = await docs.read.snapshot(ToolOutputs, docs.conversationId, call, context);
  if (!output) throw fail(false, 'the conversation has no stored output for the call (pi.companion.output)');
  const intent = await docs.read.snapshot(ViewIntents, docs.conversationId, call, context);
  if (!intent) throw fail(false, 'the conversation recorded no intent for the call (pi.view.intent). The companion records ' +
    'it after each response that makes tool calls, so the call was made before the companion recorded intents (or ' +
    'without the companion). Continue with a text reader, or start a new conversation with the Neuralese reader.');
  const store = reader.store;
  if (!store) throw fail(false, 'the natlang runtime has no Neuralese store to archive the block in (its neuralese.store option)');
  let forced: ForcedView;
  let remote: HttpNeuraleseStore;
  try {
    const { root, headers } = await serverAccess(reader);
    remote = new HttpNeuraleseStore(root, headers);
    const ref = await serverViewer({ endpoint: root, headers, store })({ value: output.text, instructions: intent.intent });
    if (!ref) throw fail(false, 'the server wrote no block');
    const id = ref.$neuralese.id;
    const meta = store.peek?.(id) ?? await store.meta(id);
    if (!meta) throw fail(false, `the server wrote ${id}, but the runtime's store has no metadata for it`);
    if (meta.dialect !== dialect) throw fail(false, `neuralese-dialect-mismatch: the server wrote ${id} in dialect ` +
      `${JSON.stringify(meta.dialect)}; the model's declared reader is ${JSON.stringify(dialect)}`);
    // Pinned under the owner until it leaves the context (collectViewBlocks); restored from the archive if it is gone.
    await withRestoredBlocks(remote, store, [id], () => remote.pin(id));
    forced = { id, length: meta.length, dialect };
  } catch (error) {
    if (error instanceof ViewForcingError) throw error;
    const message = error instanceof Error ? error.message : String(error);
    throw fail(TRANSIENT.test(message) || TRANSIENT.test(String((error as { cause?: unknown })?.cause ?? '')), message);
  }
  const key = representationKey(dialect);
  let winner = forced;
  await docs.commit(async tx => {
    const doc = await tx.doc(ViewForcings, docs.conversationId, call, null);
    const existing = doc.forced[key];
    if (existing) { winner = { id: existing.id, length: existing.length, dialect: existing.dialect }; return; }
    doc.forced[key] = forced as never;
    (await tx.doc(ViewBlocks, docs.conversationId)).pinned[forced.id] = call;
  }, context);
  // Another process memoized the call first: its block is the call's result, and ours is not held for it.
  if (winner.id !== forced.id) await remote!.unpin(forced.id);
  return winner;
}

/** The stored view calls `messages` reference, each once, in order. */
export function storedCalls(messages: readonly { content?: unknown }[]): string[] {
  const calls = new Set<string>();
  for (const message of messages) if (Array.isArray(message.content))
    for (const part of message.content) { const ref = storedCall(part); if (ref) calls.add(ref.call); }
  return [...calls];
}

/**
 * `messages` as `reader` reads them: each text part that is a stored view call becomes the call's block (forced once per
 * call and dialect) followed by the recall note. Messages without such parts are returned unchanged.
 */
export async function forceStoredViews<M extends { content?: unknown }>(messages: readonly M[], docs: ViewDocs, reader: ViewReader,
    context: Context): Promise<M[]> {
  const calls = storedCalls(messages);
  if (!calls.length) return [...messages];
  // Every forcing settles before the request fails: the failure reported is the first call's, and none stays in flight
  // for a retry to join.
  const settled = await Promise.allSettled(calls.map(call => forceView(docs, call, reader, context)));
  const failed = settled.find(result => result.status === 'rejected');
  if (failed) throw failed.reason;
  const forced = new Map(calls.map((call, index) => [call, (settled[index] as PromiseFulfilledResult<ForcedView>).value]));
  return messages.map(message => !Array.isArray(message.content) || !message.content.some(part => storedCall(part)) ? message :
    { ...message, content: (message.content as unknown[]).flatMap(part => {
      const ref = storedCall(part);
      return ref ? [{ type: 'neuralese', id: forced.get(ref.call)!.id }, { type: 'text', text: recallNote(ref.call) }] : [part];
    }) });
}

/**
 * Collect the owner's blocks on the agent server when the conversation's context head is not the one of its last
 * collection: unpin the view blocks the context no longer references, and keep exactly `referenced` (the blocks of the
 * request about to be sent, which is the context) held by the owner. Pinned blocks of other calls stay.
 */
export async function collectViewBlocks(docs: ViewDocs, reader: ViewReader, referenced: ReadonlySet<string>, head: number,
    context: Context): Promise<void> {
  const state = await docs.read.snapshot(ViewBlocks, docs.conversationId, context);
  if (state?.collectedHead === head) return;
  const { root, headers } = await serverAccess(reader);
  const remote = new HttpNeuraleseStore(root, headers);
  const released = Object.keys(state?.pinned ?? {}).filter(id => !referenced.has(id));
  for (const id of released) await remote.unpin(id);
  await remote.collect(referenced);
  await docs.commit(async tx => {
    const doc = await tx.doc(ViewBlocks, docs.conversationId);
    for (const id of released) delete doc.pinned[id];
    doc.collectedHead = head;
  }, context);
}
