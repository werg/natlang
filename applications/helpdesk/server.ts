/**
 * The support desk over HTTP: customers write and agents answer through the same JSON API, and the agents' queue is
 * one request. A deployment puts authentication in front; this example trusts its callers.
 *
 *   POST /tickets/:id/messages   { id, author, text }   a customer writes
 *   POST /tickets/:id/replies    { id, author, text }   an agent answers
 *   POST /tickets/:id/close      { id }
 *   GET  /tickets/:id                                   the ticket, with triage and the draft reply
 *   GET  /inbox                                         tickets a customer is waiting on, escalated first
 */
import { createServer, type IncomingMessage, type Server } from 'node:http';
import { join } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { HelpDesk, TicketStore, type DeskRequest, type Ticket } from './index.js';

const KINDS: Record<string, DeskRequest['kind']> = { messages: 'message', replies: 'reply', close: 'close' };

async function body(request: IncomingMessage): Promise<Record<string, unknown>> {
  let text = '';
  for await (const chunk of request) text += chunk;
  const value = text ? JSON.parse(text) : {};
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('the body is a JSON object');
  return value as Record<string, unknown>;
}

/** Serve a desk on `port` (0 picks a free one). */
export async function serveHelpDesk(desk: HelpDesk, port = 0, host = '127.0.0.1'): Promise<{ url: string, server: Server }> {
  const server = createServer(async (request, response) => {
    const reply = (status: number, value: unknown) => {
      response.writeHead(status, { 'content-type': 'application/json' });
      response.end(JSON.stringify(value));
    };
    try {
      const url = new URL(request.url ?? '/', 'http://desk');
      const parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
      if (request.method === 'GET' && parts.length === 1 && parts[0] === 'inbox') return reply(200, await desk.inbox());
      if (request.method === 'GET' && parts.length === 2 && parts[0] === 'tickets') {
        const ticket = await desk.ticket(parts[1]!);
        return ticket ? reply(200, ticket) : reply(404, { error: 'no such ticket' });
      }
      const kind = KINDS[parts[2] ?? ''];
      if (request.method === 'POST' && parts.length === 3 && parts[0] === 'tickets' && kind) {
        const fields = await body(request);
        if (typeof fields.id !== 'string' || !fields.id) return reply(400, { error: 'a request needs a unique string id' });
        if (kind !== 'close' && (typeof fields.author !== 'string' || typeof fields.text !== 'string' || !fields.text.trim()))
          return reply(400, { error: 'a message needs author and text' });
        const ticket: Ticket | null = await desk.send({ ...fields, kind, ticket: parts[1]! } as DeskRequest);
        return ticket ? reply(200, ticket) : reply(200, { duplicate: fields.id });
      }
      reply(404, { error: 'not found' });
    } catch (error) {
      reply(400, { error: String((error as Error)?.message ?? error) });
    }
  });
  await new Promise<void>(resolve => server.listen(port, host, resolve));
  const address = server.address();
  return { url: `http://${host}:${typeof address === 'object' && address ? address.port : port}`, server };
}

/** `natlang run applications/helpdesk -- --port 8080`: the desk with the launcher's model, until interrupted. */
export default async function main(context: TargetContext): Promise<number> {
  const index = context.args.indexOf('--port');
  const port = index >= 0 ? Number(context.args[index + 1]) : 0;
  const desk = new HelpDesk({ store: new TicketStore(join(context.stateDirectory, 'tickets')),
    run: (fn, signal) => context.runtime.run(fn, { signal }),
    onEscalate: ticket => { context.io.output.write(`escalated ${ticket.id}: ${ticket.triage?.summary ?? 'no triage yet'}\n`); },
    onFailure: (ticket, error) => { context.io.error.write(`ticket ${ticket}: ${String((error as Error)?.message ?? error)}\n`); } });
  await desk.start();
  const { url, server } = await serveHelpDesk(desk, port);
  context.io.output.write(`help desk at ${url}\n`);
  await new Promise<void>(resolve => process.once('SIGINT', resolve));
  await new Promise<void>(resolve => server.close(() => resolve()));
  await desk.close();
  return 0;
}
