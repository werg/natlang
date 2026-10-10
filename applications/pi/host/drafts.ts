/**
 * The user's input as a stream (plans/STREAMING.md §3): one draft per conversation, a pi-durable document
 * (`pi.draft`, conversation scope, never an entry), changed by edit deltas from the harness UI while the user types.
 * Helpers read the draft, debounced, and answer with offers (relevant context, warnings, clarifying questions) in a
 * second document (`pi.draft.offers`) the UI shows beside the draft. An offer reaches the agent only when the user takes
 * it: taking adds its text to the draft. Sending the draft is the one event that enters the transcript: it is
 * submitted as the conversation's input, and the draft and its offers are cleared.
 *
 * Platform-neutral: the Node host opens it on its harness (index.ts exports it), the browser host as `BrowserPi.drafts`.
 * The helpers' policy is theirs (the companion's is natural language behind `pluggable()`, extensions/companion/
 * draft.ts); this file is the mechanism: documents, deltas, debounce, cancellation, and the rule that offers commit
 * only for the draft version they were computed from.
 */
import type { Context } from '@earendil-works/chord';
import { BACKGROUND_CONTEXT } from '@earendil-works/chord/context';
import type { Message } from '@earendil-works/pi-ai';
import { defineDoc } from '../vendor/durable/src/documents.ts';
import type { Harness } from '../vendor/durable/src/harness/harness.ts';
import type { HarnessOptions, InputSubmissionDraft, Submission } from '../vendor/durable/src/harness/types.ts';
import type { ConversationId, DocumentReader } from '../vendor/durable/src/types.ts';
import type { ExecutionEnv } from '../vendor/durable/src/env/index.ts';
import type { DraftOffer } from '../types.ts';

/**
 * One edit of the draft: the characters `from` up to `to` (UTF-16 offsets, as a text field reports them) replaced by
 * `insert`; or the whole text replaced.
 */
export type DraftDelta = { from: number; to: number; insert: string } | { text: string };

/** The draft: its text and a version that every change increments. A fork starts without one. */
export const DraftDoc = defineDoc<{ text: string; version: number }>({
  kind: 'pi.draft', version: 1, scope: 'conversation', history: 'latest', fork: 'initial', initial: () => ({ text: '', version: 0 }),
});

/** An offer as the UI shows it: with an ID to take it by, and the helper that made it. */
export type ShownOffer = DraftOffer & { id: string; helper: string };

/** The offers for the draft, computed from draft version `version` (older than the draft's while new ones are pending). */
export const DraftOffers = defineDoc<{ version: number; offers: ShownOffer[] }>({
  kind: 'pi.draft.offers', version: 1, scope: 'conversation', history: 'latest', fork: 'initial', initial: () => ({ version: 0, offers: [] }),
});

/** What a helper reads: the draft, the conversation's workspace and recent messages. Abort `signal` when superseded. */
export type DraftInput = {
  conversationId: ConversationId;
  text: string;
  version: number;
  /** The agent's working directory and the conversation's environment there (undefined without one). */
  cwd: string;
  env: ExecutionEnv | undefined;
  read: DocumentReader;
  /** The conversation's latest model messages (at most RECENT_MESSAGES). */
  recent: readonly Message[];
  signal: AbortSignal;
  context: Context;
};

/** A helper beside the draft: its offers for one draft version. */
export type DraftHelper = { name: string; offers(input: DraftInput): Promise<DraftOffer[]> };

export type DraftsOptions = {
  helpers?: readonly DraftHelper[];
  /** The harness's environment factory (`HarnessOptions.env`), so helpers see the workspace the agent's tools see. */
  env?: HarnessOptions['env'];
  /** Quiet time after the last edit before helpers run (default 400 ms). */
  debounceMs?: number;
  onReport?(error: unknown): void;
  /** Called after each change this API commits to a conversation's draft or offers. */
  onChange?(conversationId: ConversationId): void;
};

export type DraftState = { text: string; version: number; offers: ShownOffer[]; offersVersion: number };

export type Drafts = {
  /** Apply edit deltas in order, in one commit; returns the draft's new version. An out-of-range delta changes nothing. */
  edit(conversationId: ConversationId, delta: DraftDelta | readonly DraftDelta[], context?: Context): Promise<number>;
  /** The draft and its offers. */
  read(conversationId: ConversationId, context?: Context): Promise<DraftState>;
  /** Take offer `offerId`: its insert (or text) is added to the draft as a paragraph; returns the draft's new version. */
  take(conversationId: ConversationId, offerId: string, context?: Context): Promise<number>;
  /**
   * Send the draft as the conversation's input (`whenBusy` as for any input) and clear it with its offers. An empty
   * draft is not sent. If the submission fails, the draft is restored unless the user typed meanwhile.
   */
  send(conversationId: ConversationId, options?: { whenBusy?: InputSubmissionDraft['whenBusy'] }, context?: Context): Promise<Submission>;
  /** Resolves when no helper run is pending or running (for the conversation, or for all). */
  idle(conversationId?: ConversationId): Promise<void>;
  /** Stop pending helper runs. */
  close(): void;
};

const RECENT_MESSAGES = 8;
const MAX_OFFERS = 6;

/** The draft text after `deltas`, or a RangeError naming the first delta that does not fit. */
export function applyDraftDeltas(text: string, deltas: readonly DraftDelta[]): string {
  for (const [index, delta] of deltas.entries()) {
    if ('text' in delta) { text = String(delta.text); continue; }
    const { from, to, insert } = delta;
    if (!Number.isInteger(from) || !Number.isInteger(to) || from < 0 || from > to || to > text.length || typeof insert !== 'string')
      throw new RangeError(`draft delta ${index} replaces ${from}..${to} with ${typeof insert}, but the draft has ${text.length} characters; ` +
        'a delta is { from, to, insert } with 0 <= from <= to <= length, or { text }');
    text = text.slice(0, from) + insert + text.slice(to);
  }
  return text;
}

/** Open the drafts of `harness`'s conversations. */
export function openDrafts(harness: Harness, options: DraftsOptions = {}): Drafts {
  const helpers = options.helpers ?? [];
  const debounce = options.debounceMs ?? 400;
  /** Per conversation: the debounce timer, the running helper pass and its controller. */
  const runs = new Map<ConversationId, { timer?: ReturnType<typeof setTimeout>; controller?: AbortController; running?: Promise<void> }>();
  let closed = false;
  const changed = (conversationId: ConversationId) => {
    try { options.onChange?.(conversationId); } catch (error) { options.onReport?.(error); }
  };
  const conversationOf = async (conversationId: ConversationId, context: Context) => {
    const conversation = await harness.conversation(conversationId, context);
    if (!conversation) throw new Error(`conversation ${conversationId} does not exist`);
    return conversation;
  };

  /** One helper pass for the draft as it stands; offers commit only while the draft is still that version. */
  const pass = async (conversationId: ConversationId, controller: AbortController, context: Context) => {
    const conversation = await conversationOf(conversationId, context);
    const draft = await harness.snapshot(DraftDoc, conversationId, context) ?? { text: '', version: 0 };
    let offers: ShownOffer[] = [];
    if (draft.text.trim()) {
      const agent = await conversation.agent(context);
      const env = options.env ? await options.env({ conversationId, read: harness, ...(agent.cwd === undefined ? {} : { cwd: agent.cwd }) }, context) : undefined;
      const view = await conversation.context(context);
      const input: DraftInput = { conversationId, text: draft.text, version: draft.version, cwd: agent.cwd ?? env?.cwd ?? '/',
        env, read: harness, recent: view.messages.slice(-RECENT_MESSAGES), signal: controller.signal, context };
      const results = await Promise.allSettled(helpers.map(async helper => (await helper.offers(input)).map(offer => ({ ...offer, helper: helper.name }))));
      if (controller.signal.aborted) return;
      for (const result of results) {
        if (result.status === 'fulfilled') offers.push(...result.value.filter(validOffer).map(offer => JSON.parse(JSON.stringify(offer)) as ShownOffer));
        else options.onReport?.(result.reason);
      }
      offers = offers.slice(0, MAX_OFFERS).map((offer, index) => ({ ...offer, id: `${draft.version}.${index}` }));
    }
    if (controller.signal.aborted) return;
    const committed = await conversation.commit(async tx => {
      if ((await tx.doc(DraftDoc, conversationId)).version !== draft.version) return false;
      const doc = await tx.doc(DraftOffers, conversationId);
      doc.version = draft.version;
      doc.offers = offers as never;
      return true;
    }, context);
    if (committed) changed(conversationId);
  };

  /** Run the helpers `debounce` ms after the last change; a newer change supersedes a pass that is still running. */
  const schedule = (conversationId: ConversationId) => {
    if (closed || !helpers.length) return;
    const run = runs.get(conversationId) ?? {};
    runs.set(conversationId, run);
    clearTimeout(run.timer);
    run.controller?.abort(new Error('the draft changed'));
    run.timer = setTimeout(() => {
      run.timer = undefined;
      const controller = new AbortController();
      run.controller = controller;
      const running: Promise<void> = pass(conversationId, controller, BACKGROUND_CONTEXT)
        .catch(error => { if (!controller.signal.aborted) options.onReport?.(error); })
        .finally(() => {
          if (run.running === running) run.running = undefined;
          if (!run.timer && !run.running) runs.delete(conversationId);
        });
      run.running = running;
    }, debounce);
  };

  return {
    async edit(conversationId, delta, context = BACKGROUND_CONTEXT) {
      const deltas = Array.isArray(delta) ? delta as readonly DraftDelta[] : [delta as DraftDelta];
      const conversation = await conversationOf(conversationId, context);
      const version = await conversation.commit(async tx => {
        const doc = await tx.doc(DraftDoc, conversationId);
        doc.text = applyDraftDeltas(doc.text, deltas);
        return ++doc.version;
      }, context);
      changed(conversationId);
      schedule(conversationId);
      return version;
    },
    async read(conversationId, context = BACKGROUND_CONTEXT) {
      const draft = await harness.snapshot(DraftDoc, conversationId, context);
      const offers = await harness.snapshot(DraftOffers, conversationId, context);
      return { text: draft?.text ?? '', version: draft?.version ?? 0, offers: (offers?.offers ?? []) as ShownOffer[], offersVersion: offers?.version ?? 0 };
    },
    async take(conversationId, offerId, context = BACKGROUND_CONTEXT) {
      const conversation = await conversationOf(conversationId, context);
      const version = await conversation.commit(async tx => {
        const offers = await tx.doc(DraftOffers, conversationId);
        const at = offers.offers.findIndex(offer => offer.id === offerId);
        if (at < 0) throw new Error(`the draft has no offer ${JSON.stringify(offerId)} (it may have been replaced by newer offers)`);
        const offer = offers.offers[at]!;
        const doc = await tx.doc(DraftDoc, conversationId);
        const added = offer.insert ?? offer.text;
        doc.text = doc.text.trim() ? `${doc.text.replace(/\s+$/, '')}\n\n${added}` : added;
        offers.offers.splice(at, 1);
        return ++doc.version;
      }, context);
      changed(conversationId);
      schedule(conversationId);
      return version;
    },
    async send(conversationId, sendOptions = {}, context = BACKGROUND_CONTEXT) {
      const conversation = await conversationOf(conversationId, context);
      const run = runs.get(conversationId);
      if (run) {
        clearTimeout(run.timer);
        run.timer = undefined;
        run.controller?.abort(new Error('the draft was sent'));
        if (!run.running) runs.delete(conversationId);
      }
      // Taken and cleared in one commit, so an edit cannot fall between what is sent and what is cleared.
      const sent = await conversation.commit(async tx => {
        const doc = await tx.doc(DraftDoc, conversationId);
        const text = doc.text;
        if (!text.trim()) return undefined;
        doc.text = '';
        const version = ++doc.version;
        const offers = await tx.doc(DraftOffers, conversationId);
        offers.version = version;
        offers.offers = [];
        return { text, version };
      }, context);
      if (!sent) throw new Error('the draft is empty: there is nothing to send');
      changed(conversationId);
      try {
        return await conversation.submit({ type: 'input', content: sent.text,
          ...(sendOptions.whenBusy ? { whenBusy: sendOptions.whenBusy } : {}) }, context);
      } catch (error) {
        await conversation.commit(async tx => {
          const doc = await tx.doc(DraftDoc, conversationId);
          if (doc.version === sent.version) { doc.text = sent.text; doc.version++; }
        }, context);
        changed(conversationId);
        throw error;
      }
    },
    async idle(conversationId) {
      for (;;) {
        const pending = [...runs.entries()].filter(([id]) => conversationId === undefined || id === conversationId).map(([, run]) => run);
        if (!pending.length) return;
        await Promise.all(pending.map(run => run.running ?? new Promise(resolve => setTimeout(resolve, Math.max(1, debounce)))));
      }
    },
    close() {
      closed = true;
      for (const run of runs.values()) { clearTimeout(run.timer); run.controller?.abort(new Error('drafts closed')); }
      runs.clear();
    },
  };
}

/** Whether a helper's offer has the declared shape. */
function validOffer(offer: unknown): offer is DraftOffer {
  const found = offer as Partial<DraftOffer> | null;
  return Boolean(found) && ['context', 'warning', 'question'].includes(found!.kind as string) && typeof found!.text === 'string' &&
    found!.text.trim() !== '' && (found!.insert === undefined || typeof found!.insert === 'string');
}
