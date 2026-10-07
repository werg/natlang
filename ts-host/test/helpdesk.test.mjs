import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime } from '../dist/index.js';
import { HelpDesk, TicketStore } from '../../applications/dist/helpdesk/index.js';
import { serveHelpDesk } from '../../applications/dist/helpdesk/server.js';
import { scriptedModel } from './support/natlang.mjs';

const settle = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check, ms = 3000) {
  for (const started = Date.now(); Date.now() - started < ms; await settle(10)) if (await check()) return;
  assert.fail('timed out waiting');
}

/**
 * A scripted interpreter: triage reads the conversation (urgent when the product is unusable), a draft names the
 * ticket's topic. `hold(word or test)` makes a triage of a matching conversation wait until released.
 */
function desk(root, options = {}) {
  const held = new Map();
  const model = scriptedModel(async opening => {
    if (/next reply/.test(opening)) return 'return "Sorry about that. Could you send the invoice number?"';
    for (const [match, gate] of held) if (match(opening)) await gate.promise;
    // Phrases from the customers' messages only: the opening also holds the triage instructions.
    const urgency = /cannot log in|charged twice/.test(opening) ? 'urgent' : /dark mode|for the menu/.test(opening) ? 'low' : 'normal';
    return `return { urgency: "${urgency}", topic: "accounts", summary: "The customer needs help with their account." }`;
  });
  const runtime = createNatlangRuntime({ model: model.driver });
  const escalations = [], failures = [];
  const helpdesk = new HelpDesk({ store: new TicketStore(join(root, 'tickets')), run: (fn, signal) => runtime.run(fn, { signal }),
    responseTimes: { urgent: 150, normal: 60_000, low: 600_000 }, onEscalate: ticket => { escalations.push(ticket.id); },
    onFailure: (ticket, error) => failures.push([ticket, String(error)]), ...options });
  const hold = match => { let release; const promise = new Promise(resolve => { release = resolve; });
    held.set(typeof match === 'string' ? text => text.includes(match) : match, { promise }); return release; };
  return { helpdesk, escalations, failures, hold };
}

test('tickets are served side by side: one customer\'s pending triage holds up nobody else', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-helpdesk-'));
  const { helpdesk, hold, failures } = desk(root);
  try {
    const release = hold('printer');
    const first = await helpdesk.send({ id: 'a1', kind: 'message', ticket: 'A', author: 'ana', text: 'My printer setting is gone.' });
    assert.equal(first.triage, null, 'the step returns before its triage');
    await helpdesk.send({ id: 'b1', kind: 'message', ticket: 'B', author: 'ben', text: 'I cannot log in since this morning.' });
    await until(async () => (await helpdesk.ticket('B'))?.draft);
    const b = await helpdesk.ticket('B');
    assert.equal(b.triage.urgency, 'urgent'); assert.match(b.draft, /invoice number/);
    assert.equal((await helpdesk.ticket('A')).triage, null, 'A is still waiting on its triage');
    // A's customer writes again while A's first triage is pending; the conversation keeps its order.
    await helpdesk.send({ id: 'a2', kind: 'message', ticket: 'A', author: 'ana', text: 'Also the scanner.' });
    release();
    await until(async () => (await helpdesk.ticket('A'))?.triaged === 2);
    const a = await helpdesk.ticket('A');
    assert.deepEqual(a.messages.map(m => m.text), ['My printer setting is gone.', 'Also the scanner.']);
    assert.equal(a.triage.urgency, 'normal');
    assert.deepEqual(failures, []);
  } finally { await helpdesk.close(); await rm(root, { recursive: true, force: true }); }
});

test('a triage of an earlier conversation never replaces a later one', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-helpdesk-'));
  const { helpdesk, hold } = desk(root);
  try {
    // The triage of the first message alone is held; the triage of both messages finishes first.
    const release = hold(text => text.includes('first note') && !text.includes('update'));
    await helpdesk.send({ id: 'c1', kind: 'message', ticket: 'C', author: 'cy', text: 'first note: a suggestion for the menu' });
    await helpdesk.send({ id: 'c2', kind: 'message', ticket: 'C', author: 'cy', text: 'update: I was charged twice' });
    await until(async () => (await helpdesk.ticket('C'))?.triaged === 2);
    release();
    await settle(150);
    const c = await helpdesk.ticket('C');
    assert.equal(c.triaged, 2); assert.equal(c.triage.urgency, 'urgent', 'the triage of both messages stands');
  } finally { await helpdesk.close(); await rm(root, { recursive: true, force: true }); }
});

test('an unanswered urgent ticket escalates once at its deadline; an agent\'s answer clears it', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-helpdesk-'));
  const { helpdesk, escalations } = desk(root);
  try {
    await helpdesk.send({ id: 'd1', kind: 'message', ticket: 'D', author: 'dee', text: 'I was charged twice for March.' });
    await until(async () => (await helpdesk.ticket('D'))?.escalated, 2000);
    await settle(200);
    assert.deepEqual(escalations, ['D'], 'escalated once');
    assert.equal((await helpdesk.inbox())[0].id, 'D', 'escalated tickets lead the queue');
    const answered = await helpdesk.send({ id: 'd2', kind: 'reply', ticket: 'D', author: 'agent-1', text: 'Refunded the second charge.' });
    assert.equal(answered.status, 'answered'); assert.equal(answered.due, null); assert.equal(answered.escalated, false);
    assert.deepEqual(await helpdesk.inbox(), []);
    assert.equal(await helpdesk.send({ id: 'd2', kind: 'reply', ticket: 'D', author: 'agent-1', text: 'again' }), null,
      'a repeated request ID is applied once');
  } finally { await helpdesk.close(); await rm(root, { recursive: true, force: true }); }
});

test('after a restart, stored tickets resume: deadlines are armed again and applied requests are not repeated', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-helpdesk-'));
  const before = desk(root, { responseTimes: { urgent: 400, normal: 60_000, low: 600_000 } });
  await before.helpdesk.send({ id: 'e1', kind: 'message', ticket: 'E', author: 'eve', text: 'I cannot log in at all.' });
  await until(async () => (await before.helpdesk.ticket('E'))?.triage);
  await before.helpdesk.close();
  assert.deepEqual(before.escalations, [], 'closed before its deadline');
  const after = desk(root, { responseTimes: { urgent: 400, normal: 60_000, low: 600_000 } });
  try {
    await after.helpdesk.start();
    await until(async () => (await after.helpdesk.ticket('E'))?.escalated, 2000);
    assert.deepEqual(after.escalations, ['E'], 'the restored deadline fired');
    assert.equal(await after.helpdesk.send({ id: 'e1', kind: 'message', ticket: 'E', author: 'eve', text: 'I cannot log in at all.' }), null);
    assert.equal((await after.helpdesk.ticket('E')).messages.length, 1);
  } finally { await after.helpdesk.close(); await rm(root, { recursive: true, force: true }); }
});

test('customers and agents use the desk over HTTP', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-helpdesk-'));
  const { helpdesk } = desk(root);
  const { url, server } = await serveHelpDesk(helpdesk);
  const post = (path, body) => fetch(url + path, { method: 'POST', body: JSON.stringify(body) }).then(async r => [r.status, await r.json()]);
  try {
    const [status, ticket] = await post('/tickets/F/messages', { id: 'f1', author: 'fay', text: 'A suggestion: dark mode.' });
    assert.equal(status, 200); assert.equal(ticket.customer, 'fay');
    assert.equal((await post('/tickets/F/messages', { author: 'fay', text: 'no id' }))[0], 400);
    await until(async () => (await (await fetch(`${url}/tickets/F`)).json()).draft);
    assert.equal((await (await fetch(`${url}/inbox`)).json())[0].triage.urgency, 'low');
    assert.equal((await post('/tickets/F/replies', { id: 'f2', author: 'agent-2', text: 'Noted, thank you!' }))[1].status, 'answered');
    assert.deepEqual(await (await fetch(`${url}/inbox`)).json(), []);
    assert.equal((await fetch(`${url}/tickets/nope`)).status, 404);
  } finally {
    await new Promise(resolve => server.close(resolve));
    await helpdesk.close(); await rm(root, { recursive: true, force: true });
  }
});
