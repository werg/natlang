/**
 * A multi-user support desk. Each ticket is its own event loop (`KeyedEventLoop`), so one customer's slow model call
 * never holds up another ticket. Natural language makes two judgments, both as follow-up work after a commit
 * (`context.after`): triage (how urgent, what about) and a draft reply for the agent. The first-response deadline is
 * ticket state; `wakeAt` escalates a ticket that no agent has answered in time. Tickets persist per key and resume
 * after a restart, deadlines included.
 */
import { mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { KeyedEventLoop, type AppEvent, type Commit } from '@natlang/node';
import triage from './triage.nl';
import draftReply from './draft_reply.nl';

export type Urgency = 'low' | 'normal' | 'urgent';
export type Message = { from: 'customer' | 'agent', author: string, text: string, at: number };
export type Triage = { urgency: Urgency, topic: string, summary: string };
export type Ticket = {
  id: string,
  customer: string,
  messages: Message[],
  status: 'open' | 'answered' | 'closed',
  triage: Triage | null,
  /** How many messages the triage read: a triage of more messages replaces one of fewer, never the reverse. */
  triaged: number,
  /** A suggested reply for the agent, for the conversation as it stands (dropped when it moves on). */
  draft: string | null,
  /** When an agent must have answered the customer; null while nobody is waiting. */
  due: number | null,
  escalated: boolean,
};
/** What customers and agents send. Each needs a unique `id`: a repeated ID is applied once. */
export type DeskRequest =
  | { id: string, kind: 'message', ticket: string, author: string, text: string }
  | { id: string, kind: 'reply', ticket: string, author: string, text: string }
  | { id: string, kind: 'close', ticket: string };
type Followup =
  | { id: string, kind: 'triaged', ticket: string, through: number, triage: Triage }
  | { id: string, kind: 'drafted', ticket: string, through: number, draft: string };
type Wake = { id: string, kind: 'wake', at: number };
type DeskEvent = AppEvent & (DeskRequest | Followup | Wake);

/** How long a customer may wait for a first answer, by urgency (ms). Before triage a ticket counts as normal. */
export type ResponseTimes = Record<Urgency, number>;
export const RESPONSE_TIMES: ResponseTimes = { urgent: 15 * 60_000, normal: 4 * 3_600_000, low: 24 * 3_600_000 };

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

const conversation = (ticket: Ticket) => ticket.messages.map(({ from, text }) => ({ from, text }));
/** When the customer started waiting: their first message since an agent last answered. */
function waitingSince(ticket: Ticket): number | null {
  let since: number | null = null;
  for (const message of ticket.messages) since = message.from === 'agent' ? null : since ?? message.at;
  return since;
}

export type DeskOptions = {
  store: TicketStore,
  /** Runs natural-language work in a task, e.g. `(fn, signal) => runtime.run(fn, { signal })`. */
  run: <T>(fn: () => Promise<T>, signal: AbortSignal) => Promise<T>,
  responseTimes?: ResponseTimes,
  /** Called once when a ticket is escalated (after the escalation is stored), e.g. to page a supervisor. */
  onEscalate?: (ticket: Ticket) => void | Promise<void>,
  onFailure?: (ticket: string, error: unknown) => void,
  clock?: () => number,
};

export class HelpDesk {
  readonly loops: KeyedEventLoop<Ticket, Ticket, DeskEvent>;
  private readonly times: ResponseTimes;

  constructor(private readonly options: DeskOptions) {
    this.times = options.responseTimes ?? RESPONSE_TIMES;
    const { store, run } = options;
    this.loops = new KeyedEventLoop<Ticket, Ticket, DeskEvent>({
      key: event => (event as DeskRequest | Followup).ticket,
      initialState: id => ({ id, customer: '', messages: [], status: 'open', triage: null, triaged: 0, draft: null,
        due: null, escalated: false }),
      restore: id => store.load(id),
      reduce: (ticket, event, context) => {
        const now = context.now;
        switch (event.kind) {
          case 'message': {
            const next: Ticket = { ...ticket, customer: ticket.customer || event.author, status: 'open', draft: null,
              messages: [...ticket.messages, { from: 'customer', author: event.author, text: event.text, at: now }] };
            next.due = this.deadline(next);
            // Every new message is triaged with the whole conversation; results apply in conversation order.
            const through = next.messages.length, messages = conversation(next);
            context.after(signal => run(async () => ({ id: `${ticket.id}:triaged:${through}`, kind: 'triaged' as const,
              ticket: ticket.id, through, triage: await triage(messages) }), signal));
            return next;
          }
          case 'triaged': {
            if (event.through <= ticket.triaged) return ticket;
            const next: Ticket = { ...ticket, triage: event.triage, triaged: event.through };
            next.due = this.deadline(next);
            if (next.status === 'open' && event.through === next.messages.length) {
              const messages = conversation(next), through = event.through;
              context.after(signal => run(async () => ({ id: `${ticket.id}:drafted:${through}`, kind: 'drafted' as const,
                ticket: ticket.id, through, draft: await draftReply(messages, event.triage.summary) }), signal));
            }
            return next;
          }
          case 'drafted':
            // A draft is for the conversation it read; once the conversation moved on it no longer fits.
            return event.through === ticket.messages.length && ticket.status === 'open' ? { ...ticket, draft: event.draft } : ticket;
          case 'reply':
            return { ...ticket, status: 'answered', draft: null, due: null, escalated: false,
              messages: [...ticket.messages, { from: 'agent', author: event.author, text: event.text, at: now }] };
          case 'close':
            return { ...ticket, status: 'closed', draft: null, due: null };
          case 'wake': {
            if (ticket.status !== 'open' || ticket.escalated || ticket.due === null || ticket.due > now) return ticket;
            const escalated = { ...ticket, escalated: true };
            if (options.onEscalate) context.after(async () => { await options.onEscalate!(escalated); });
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

  /** The first-response deadline: from when the customer started waiting, by the triaged urgency (normal before). */
  private deadline(ticket: Ticket): number | null {
    const since = ticket.status === 'open' ? waitingSince(ticket) : null;
    return since === null ? null : since + this.times[ticket.triage?.urgency ?? 'normal'];
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
  async ticket(id: string): Promise<Ticket | undefined> { return (await this.options.store.load(id))?.state; }

  /**
   * The agents' queue: tickets a customer is waiting on, escalated first, then by deadline. Exact code over the
   * stored tickets; natural language decided only each ticket's urgency.
   */
  async inbox(): Promise<Ticket[]> {
    const tickets: Ticket[] = [];
    for (const id of await this.options.store.ids()) {
      const ticket = (await this.options.store.load(id))?.state;
      if (ticket?.status === 'open') tickets.push(ticket);
    }
    return tickets.sort((a, b) => Number(b.escalated) - Number(a.escalated) ||
      (a.due ?? Infinity) - (b.due ?? Infinity) || a.id.localeCompare(b.id));
  }

  close(): Promise<void> { return this.loops.close(); }
}
