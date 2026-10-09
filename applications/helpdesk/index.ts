/**
 * A multi-user support desk. Each ticket is its own event loop (`KeyedEventLoop`), so one customer's slow model call
 * never holds up another ticket. Natural language judges the conversation, as follow-up work after a commit
 * (`context.after`): triage (urgency, then topic and summary), the details still missing and a draft reply for the
 * agent, how long the ticket may wait, and, when a ticket is overdue, what its escalation does. The state machine,
 * the stores and the timers are crisp (DECOMPOSITION.md). Tickets persist per key and resume after a restart,
 * deadlines included.
 *
 * Two policies are pluggable (`crisp | nl | shadow`; the crisp default is the table and the comparator below): how long
 * a ticket may wait (`deadlineMode`, read from a written service policy in `nl`) and the agents' queue order
 * (`inboxMode`, read from written ranking rules in `nl`). The escalation plan is natural language with a crisp floor.
 */
import { mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { KeyedEventLoop, pluggable, untrusted, type AppEvent, type Commit, type PluggableSetting } from '@natlang/node';
import urgencyOf from './triage/urgency.nl';
import summarize from './triage/summarize.nl';
import missingDetails from './reply/missingDetails.nl';
import draftReply from './reply/draft.nl';
import deadlineFor from './deadline/deadlineFor.nl';
import escalationPlan from './escalate/plan.nl';
import rank from './inbox/rank.nl';
import type { EscalationFacts, EscalationPlan, Line, Message, RankedTicket, Ticket, Triage, Urgency } from './types.js';

export type * from './types.js';
/** What customers and agents send. Each needs a unique `id`: a repeated ID is applied once. */
export type DeskRequest =
  | { id: string, kind: 'message', ticket: string, author: string, text: string }
  | { id: string, kind: 'reply', ticket: string, author: string, text: string }
  | { id: string, kind: 'close', ticket: string };
type Followup =
  | { id: string, kind: 'triaged', ticket: string, through: number, triage: Triage, allowedMs: number }
  | { id: string, kind: 'drafted', ticket: string, through: number, draft: string, missing: string[] }
  | { id: string, kind: 'planned', ticket: string, plan: EscalationPlan };
type Wake = { id: string, kind: 'wake', at: number };
type DeskEvent = AppEvent & (DeskRequest | Followup | Wake);

/** How long a customer may wait for a first answer, by urgency (ms). Before triage a ticket counts as normal. */
export type ResponseTimes = Record<Urgency, number>;
export const RESPONSE_TIMES: ResponseTimes = { urgent: 15 * 60_000, normal: 4 * 3_600_000, low: 24 * 3_600_000 };

/** The ranking rules that `inboxMode: "nl"` reads when the desk states none: the crisp comparator in words. */
export const DEFAULT_RANKING_POLICY = 'Escalated tickets come first. Then the ticket due soonest, a ticket with no due time last. ' +
  'Ties go to the more urgent ticket, then to the smaller id.';

/** A service policy in words that says what `RESPONSE_TIMES` (or the given table) says, for `deadlineMode: "nl"` without a written policy. */
export function describeResponseTimes(times: ResponseTimes): string {
  const minutes = (urgency: Urgency) => Math.max(1, Math.round(times[urgency] / 60_000));
  return `urgent tickets: ${minutes('urgent')} minutes. normal tickets: ${minutes('normal')} minutes. low tickets: ${minutes('low')} minutes. ` +
    `Default when nothing else applies: ${minutes('normal')} minutes.`;
}

export type SavedTicket = { state: Ticket, revision: number, seenEventIds: string[] };

/** One JSON file per ticket, replaced atomically on every commit. */
export class TicketStore {
  constructor(readonly directory: string) {}
  private path(id: string) { return join(this.directory, `${encodeURIComponent(id)}.json`); }
  async load(id: string): Promise<SavedTicket | undefined> {
    try { return JSON.parse(await readFile(this.path(id), 'utf8')) as SavedTicket; }
    catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined; throw error; }
  }
  async save(id: string, saved: SavedTicket): Promise<void> {
    await mkdir(this.directory, { recursive: true });
    const temporary = `${this.path(id)}.${process.pid}.tmp`;
    await writeFile(temporary, JSON.stringify(saved));
    await rename(temporary, this.path(id));
  }
  async ids(): Promise<string[]> {
    const names = await readdir(this.directory).catch(() => [] as string[]);
    return names.filter(name => name.endsWith('.json')).map(name => decodeURIComponent(name.slice(0, -5)));
  }
}

const blankTicket = (id: string): Ticket => ({ id, customer: '', messages: [], status: 'open', triage: null, triaged: 0, draft: null,
  missing: [], allowedMs: null, due: null, escalated: false, escalation: null });
/** A stored ticket with the fields later versions added (tickets saved before them lack those). */
const filled = (state: Ticket): Ticket => ({ ...blankTicket(state.id), ...state });

const conversation = (ticket: Ticket): Line[] => ticket.messages.map(({ from, text }) => ({ from, text: untrusted(text, `${from} message`) }));
/** When the customer started waiting: their first message since an agent last answered. */
function waitingSince(ticket: Ticket): number | null {
  let since: number | null = null;
  for (const message of ticket.messages) since = message.from === 'agent' ? null : since ?? message.at;
  return since;
}

/** The queue order of the open tickets, as a comparator: escalated first, then the nearest deadline, then the id. */
export const crispInboxOrder = (a: RankedTicket, b: RankedTicket): number => Number(b.escalated) - Number(a.escalated) ||
  (a.due ?? Infinity) - (b.due ?? Infinity) || a.id.localeCompare(b.id);

/** True when `ids` lists every id of `tickets` exactly once. */
export function isRanking(ids: unknown, tickets: RankedTicket[]): ids is string[] {
  return Array.isArray(ids) && ids.length === tickets.length && new Set(ids).size === ids.length &&
    ids.every(id => tickets.some(ticket => ticket.id === id));
}

/** The escalation plan without a model: the pager by urgency and the summary as the note. It never holds a reply back for an agent to send. */
export function crispEscalationPlan(facts: EscalationFacts): EscalationPlan {
  return { notify: facts.urgency === 'urgent' ? 'on-call' : 'supervisor', holding_reply: null,
    note: `Waited ${facts.waited_minutes} minutes of ${facts.allowed_minutes} allowed. ${facts.topic}: ${facts.summary}` };
}

export type DeskOptions = {
  store: TicketStore,
  /** Runs natural-language work in a task, e.g. `(fn, signal) => runtime.run(fn, { signal })`. */
  run: <T>(fn: () => Promise<T>, signal: AbortSignal) => Promise<T>,
  /** The crisp deadlines, by urgency. */
  responseTimes?: ResponseTimes,
  /** How a ticket's allowed wait is decided: the `responseTimes` table (`crisp`, the default), `deadlineFor` over `deadlinePolicy` (`nl`), or both with the table served (`shadow`). */
  deadlineMode?: PluggableSetting,
  /** The desk's written service policy, read by `deadlineFor`; default: `responseTimes` in words. */
  deadlinePolicy?: string,
  /** How the agents' queue is ordered: the comparator (`crisp`, the default), `rank` over `rankingPolicy` (`nl`), or both with the comparator served (`shadow`). */
  inboxMode?: PluggableSetting,
  /** The desk's written ranking rules, read by `rank`; default: `DEFAULT_RANKING_POLICY`. */
  rankingPolicy?: string,
  /** How an escalation is planned: `escalate/plan` (`nl`, the default; the crisp plan stands in when it fails), the crisp plan, or both with the crisp plan served. */
  escalationMode?: PluggableSetting,
  /** Called once when a ticket is escalated (after the escalation is stored), with its plan, e.g. to page the person the plan names. */
  onEscalate?: (ticket: Ticket, plan: EscalationPlan) => void | Promise<void>,
  onFailure?: (ticket: string, error: unknown) => void,
  clock?: () => number,
};

export class HelpDesk {
  readonly loops: KeyedEventLoop<Ticket, Ticket, DeskEvent>;
  private readonly times: ResponseTimes;
  private readonly allowance: (triage: Triage) => Promise<number>;
  private readonly ordering: (tickets: RankedTicket[]) => Promise<string[]>;
  private readonly planning: (facts: EscalationFacts, messages: Line[]) => Promise<EscalationPlan>;
  private readonly inboxNl: boolean;

  constructor(private readonly options: DeskOptions) {
    this.times = options.responseTimes ?? RESPONSE_TIMES;
    const { store, run } = options;
    const servesCrisp = { serve: 'crisp' as const, default: 'crisp' as const };
    const deadlinePolicy = options.deadlinePolicy ?? describeResponseTimes(this.times);
    const rankingPolicy = options.rankingPolicy ?? DEFAULT_RANKING_POLICY;
    this.allowance = pluggable<[Triage], number>({
      crisp: triage => this.times[triage.urgency],
      nl: async triage => (await deadlineFor(triage.urgency, triage.topic, deadlinePolicy)) * 60_000,
    }, options.deadlineMode, { ...servesCrisp, name: 'helpdesk.deadline' });
    this.inboxNl = options.inboxMode !== undefined && options.inboxMode !== 'crisp';
    this.ordering = pluggable<[RankedTicket[]], string[]>({
      crisp: tickets => [...tickets].sort(crispInboxOrder).map(ticket => ticket.id),
      nl: async tickets => {
        const ids = await rank(tickets, rankingPolicy);
        if (!isRanking(ids, tickets)) throw new Error(`the ranking lists every offered id exactly once (${tickets.map(ticket => ticket.id).join(', ')}); it listed ${JSON.stringify(ids)}`);
        return ids;
      },
    }, options.inboxMode, { ...servesCrisp, name: 'helpdesk.inbox', same: (exact, judged) => exact.join('\n') === judged.join('\n') });
    this.planning = pluggable<[EscalationFacts, Line[]], EscalationPlan>({
      crisp: facts => crispEscalationPlan(facts),
      nl: async (facts, messages) => {
        const plan = await escalationPlan(facts, messages);
        // A holding reply goes to the customer through an agent; the desk offers one only to a customer who kept writing.
        return facts.late_messages >= 2 ? plan : { ...plan, holding_reply: null };
      },
    }, options.escalationMode, { default: 'nl', serve: 'crisp', name: 'helpdesk.escalation',
      same: (exact, judged) => exact.notify === judged.notify });

    this.loops = new KeyedEventLoop<Ticket, Ticket, DeskEvent>({
      key: event => (event as DeskRequest | Followup).ticket,
      initialState: id => blankTicket(id),
      restore: async id => { const saved = await store.load(id); return saved && { ...saved, state: filled(saved.state) }; },
      reduce: (ticket, event, context) => {
        const now = context.now;
        switch (event.kind) {
          case 'message': {
            const next: Ticket = { ...ticket, customer: ticket.customer || event.author, status: 'open', draft: null, missing: [],
              messages: [...ticket.messages, { from: 'customer', author: event.author, text: event.text, at: now }] };
            next.due = this.deadline(next);
            // Every new message is triaged with the whole conversation; results apply in conversation order.
            const through = next.messages.length, messages = conversation(next);
            context.after(signal => run(async () => {
              const urgency = await urgencyOf(messages);
              const { topic, summary } = await summarize(messages, urgency);
              const triage: Triage = { urgency, topic, summary };
              return { id: `${ticket.id}:triaged:${through}`, kind: 'triaged' as const, ticket: ticket.id, through, triage,
                allowedMs: await this.allowedWait(ticket.id, triage) };
            }, signal));
            return next;
          }
          case 'triaged': {
            if (event.through <= ticket.triaged) return ticket;
            const next: Ticket = { ...ticket, triage: event.triage, triaged: event.through, allowedMs: event.allowedMs };
            next.due = this.deadline(next);
            if (next.status === 'open' && event.through === next.messages.length) {
              const messages = conversation(next), through = event.through;
              context.after(signal => run(async () => {
                const missing = await missingDetails(messages, event.triage.summary);
                return { id: `${ticket.id}:drafted:${through}`, kind: 'drafted' as const, ticket: ticket.id, through, missing,
                  draft: await draftReply(messages, event.triage.summary, missing) };
              }, signal));
            }
            return next;
          }
          case 'drafted':
            // A draft is for the conversation it read; once the conversation moved on it no longer fits.
            return event.through === ticket.messages.length && ticket.status === 'open' ? { ...ticket, draft: event.draft, missing: event.missing } : ticket;
          case 'reply':
            return { ...ticket, status: 'answered', draft: null, missing: [], due: null, escalated: false, escalation: null,
              messages: [...ticket.messages, { from: 'agent', author: event.author, text: event.text, at: now }] };
          case 'close':
            return { ...ticket, status: 'closed', draft: null, missing: [], due: null };
          case 'planned':
            // The plan is for the escalation it was made for; an answer in the meantime ends that escalation.
            return ticket.escalated && !ticket.escalation ? { ...ticket, escalation: event.plan } : ticket;
          case 'wake': {
            if (ticket.status !== 'open' || ticket.escalated || ticket.due === null || ticket.due > now) return ticket;
            const escalated = { ...ticket, escalated: true };
            const facts = this.escalationFacts(escalated, now), messages = conversation(escalated);
            context.after(signal => run(async () => {
              const plan = await this.planned(ticket.id, facts, messages);
              try { await options.onEscalate?.(escalated, plan); }
              catch (error) { options.onFailure?.(ticket.id, error); }
              return { id: `${ticket.id}:planned:${now}`, kind: 'planned' as const, ticket: ticket.id, plan };
            }, signal));
            return escalated;
          }
        }
      },
      view: ticket => ticket,
      wakeAt: ticket => ticket.status === 'open' && !ticket.escalated ? ticket.due : null,
      onCommit: async (id, commit: Commit<Ticket, DeskEvent>) => {
        // The committed event is not yet among the loop's applied IDs; store it with them, in the same write.
        const loop = await this.loops.loop(id);
        await store.save(id, { state: commit.state, revision: commit.revision,
          seenEventIds: [...loop.seenEventIds, commit.event.id] });
      },
      onFailure: (id, failure) => options.onFailure?.(id, failure.error),
      clock: options.clock,
    });
  }

  /** The first-response deadline: from when the customer started waiting, by the allowed wait of the latest triage (a normal ticket's before). */
  private deadline(ticket: Ticket): number | null {
    const since = ticket.status === 'open' ? waitingSince(ticket) : null;
    return since === null ? null : since + (ticket.allowedMs ?? this.times.normal);
  }

  /** How long the ticket may wait: the pluggable policy, with the crisp table standing in when the natural-language side fails. */
  private async allowedWait(ticket: string, triage: Triage): Promise<number> {
    try {
      const allowed = await this.allowance(triage);
      if (Number.isFinite(allowed) && allowed > 0) return allowed;
      throw new Error(`the allowed wait is a positive number of milliseconds, not ${allowed}`);
    } catch (error) {
      this.options.onFailure?.(ticket, error);
      return this.times[triage.urgency];
    }
  }

  private escalationFacts(ticket: Ticket, now: number): EscalationFacts {
    const since = waitingSince(ticket) ?? now, due = ticket.due ?? now;
    return { urgency: ticket.triage?.urgency ?? 'normal', topic: ticket.triage?.topic ?? '', summary: ticket.triage?.summary ?? '',
      missing: ticket.missing, waited_minutes: Math.max(0, Math.floor((now - since) / 60_000)),
      allowed_minutes: Math.max(0, Math.round((ticket.allowedMs ?? this.times.normal) / 60_000)),
      late_messages: ticket.messages.filter(message => message.from === 'customer' && message.at > due).length };
  }

  /** The escalation plan; the crisp plan stands in when the natural-language side fails, so a page is never lost to a model failure. */
  private async planned(ticket: string, facts: EscalationFacts, messages: Line[]): Promise<EscalationPlan> {
    try { return await this.planning(facts, messages); }
    catch (error) {
      this.options.onFailure?.(ticket, error);
      return crispEscalationPlan(facts);
    }
  }

  /**
   * Resume after a restart: open the loop of every stored ticket someone is waiting on, so its deadline is armed
   * again. (A ticket's loop otherwise starts with its next event.)
   */
  async start(): Promise<void> {
    for (const id of await this.options.store.ids()) {
      const saved = await this.options.store.load(id);
      if (saved?.state.status === 'open' && saved.state.due !== null && !saved.state.escalated) await this.loops.loop(id);
    }
  }

  /** Apply a customer's or agent's request to its ticket; resolves with the ticket after it (null for a repeated ID). */
  async send(request: DeskRequest): Promise<Ticket | null> {
    if (!request.ticket) throw new Error('a request names its ticket');
    const transition = await this.loops.dispatch(request as DeskEvent);
    return transition ? transition.view : null;
  }

  /** A ticket as stored: its latest committed state. */
  async ticket(id: string): Promise<Ticket | undefined> {
    const saved = await this.options.store.load(id);
    return saved && filled(saved.state);
  }

  /**
   * The agents' queue: tickets a customer is waiting on. The order is the inbox policy: escalated first, then by
   * deadline (`crisp`), or the desk's written rules (`nl`). The desk checks that the order lists every open ticket once;
   * a natural-language order that does not is replaced by the crisp one.
   */
  async inbox(): Promise<Ticket[]> {
    const tickets: Ticket[] = [];
    for (const id of await this.options.store.ids()) {
      const ticket = await this.ticket(id);
      if (ticket?.status === 'open') tickets.push(ticket);
    }
    const ranked: RankedTicket[] = tickets.map(ticket => ({ id: ticket.id, urgency: ticket.triage?.urgency ?? null,
      topic: ticket.triage?.topic ?? '', summary: ticket.triage?.summary ?? '', escalated: ticket.escalated, due: ticket.due }));
    let ids: string[];
    try {
      ids = this.inboxNl ? await this.options.run(() => this.ordering(ranked), new AbortController().signal) : await this.ordering(ranked);
    } catch (error) {
      this.options.onFailure?.('inbox', error);
      ids = [...ranked].sort(crispInboxOrder).map(ticket => ticket.id);
    }
    return ids.map(id => tickets.find(ticket => ticket.id === id)!);
  }

  close(): Promise<void> { return this.loops.close(); }
}
