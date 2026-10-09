import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime } from '../dist/index.js';
import { HelpDesk, TicketStore, crispEscalationPlan, crispInboxOrder, describeResponseTimes, isRanking, RESPONSE_TIMES } from '../../applications/dist/helpdesk/index.js';
import { sentenceCount } from '../../applications/dist/helpdesk/refinements.js';
import { serveHelpDesk } from '../../applications/dist/helpdesk/server.js';
import { appCrisp, scriptedModel } from './support/natlang.mjs';
import { readFileSync } from 'node:fs';

const settle = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check, ms = 3000) {
  for (const started = Date.now(); Date.now() - started < ms; await settle(10)) if (await check()) return;
  assert.fail('timed out waiting');
}

/**
 * A scripted interpreter: urgency reads the conversation (urgent when the product is unusable), summarize and the reply
 * stages answer by their instruction text. `hold(word or test)` makes a stage that reads a matching conversation wait
 * until released. `script(opening)` may answer a stage first (return code, null to fail the stage, undefined to pass).
 */
function desk(root, { script, ...options } = {}) {
  const held = new Map();
  const model = scriptedModel(async opening => {
    const scripted = script?.(opening);
    if (scripted !== undefined) return scripted;
    if (/Write the support agent's next reply/.test(opening)) return 'return "Sorry about that. Could you send the invoice number?"';
    if (/List what the agent needs from the customer/.test(opening)) return 'return ["invoice number"]';
    for (const [match, gate] of held) if (match(opening)) await gate.promise;
    if (/Plan the escalation/.test(opening))
      return 'return { notify: facts.urgency == "urgent" ? "on-call" : "supervisor", note: "Waited past the deadline. The customer needs help with their account.", holding_reply: null }';
    if (/Summarize the open request/.test(opening)) return 'return { topic: "account access", summary: "The customer needs help with their account." }';
    // Phrases from the customers' messages only: the opening also holds the stage instructions.
    return `return "${/cannot log in|charged twice/.test(opening) ? 'urgent' : /dark mode|for the menu/.test(opening) ? 'low' : 'normal'}"`;
  });
  const runtime = createNatlangRuntime({ model: model.driver });
  const escalations = [], plans = [], failures = [];
  const helpdesk = new HelpDesk({ store: new TicketStore(join(root, 'tickets')), run: (fn, signal) => runtime.run(fn, { signal }),
    responseTimes: { urgent: 150, normal: 60_000, low: 600_000 }, onEscalate: (ticket, plan) => { escalations.push(ticket.id); plans.push(plan); },
    onFailure: (ticket, error) => failures.push([ticket, String(error)]), ...options });
  const hold = match => { let release; const promise = new Promise(resolve => { release = resolve; });
    held.set(typeof match === 'string' ? text => text.includes(match) : match, { promise }); return release; };
  return { helpdesk, escalations, plans, failures, hold, model };
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

const withDesk = async (options, body) => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-helpdesk-'));
  const made = desk(root, options);
  try { await body(made); } finally { await made.helpdesk.close(); await rm(root, { recursive: true, force: true }); }
};
const customer = (helpdesk, ticket, text, id = `${ticket}-${Math.random()}`) =>
  helpdesk.send({ id, kind: 'message', ticket, author: ticket.toLowerCase(), text });

test('triage runs as separate stages and the ticket keeps the details still missing', () => withDesk({}, async ({ helpdesk, model, failures }) => {
  await customer(helpdesk, 'G', 'A note: please add dark mode.');
  await until(async () => (await helpdesk.ticket('G'))?.draft);
  const g = await helpdesk.ticket('G');
  assert.deepEqual(g.triage, { urgency: 'low', topic: 'account access', summary: 'The customer needs help with their account.' });
  assert.deepEqual(g.missing, ['invoice number']);
  assert.equal(g.allowedMs, 600_000, 'the crisp table decides the allowed wait by default');
  const stages = ['Decide the urgency', 'Summarize the open request', 'List what the agent needs', "Write the support agent's next reply"]
    .map(phrase => model.openings.findIndex(opening => opening.includes(phrase)));
  assert.ok(stages.every(index => index >= 0) && stages.every((index, at) => at === 0 || index > stages[at - 1]), `stages in order: ${stages}`);
  assert.match(model.openings[0], /untrusted data from customer message/, 'customer text reaches the model as quoted data');
  assert.deepEqual(failures, []);
}));

test('an escalation carries a plan, stored on the ticket and given to the pager', () => withDesk({}, async ({ helpdesk, plans, escalations }) => {
  await customer(helpdesk, 'H', 'I was charged twice for March.');
  await until(async () => (await helpdesk.ticket('H'))?.escalation, 2000);
  const h = await helpdesk.ticket('H');
  assert.deepEqual(escalations, ['H']);
  assert.equal(plans[0].notify, 'on-call');
  assert.deepEqual(h.escalation, plans[0]);
  assert.equal(h.escalation.holding_reply, null);
  const answered = await helpdesk.send({ id: 'h2', kind: 'reply', ticket: 'H', author: 'agent', text: 'Refunded.' });
  assert.equal(answered.escalation, null, 'an answer ends the escalation');
}));

test('a plan that fails is replaced by the crisp plan, so the page still goes out', () =>
  withDesk({ script: opening => /Plan the escalation/.test(opening) ? null : undefined }, async ({ helpdesk, plans, failures }) => {
    await customer(helpdesk, 'I', 'I cannot log in at all.');
    await until(async () => (await helpdesk.ticket('I'))?.escalation, 3000);
    assert.equal(plans[0].notify, 'on-call');
    assert.match(plans[0].note, /Waited \d+ minutes of \d+ allowed/);
    assert.equal(failures.length, 1);
  }));

test('a holding reply is offered only when the customer kept writing after the deadline', () => withDesk({
  script: opening => /Plan the escalation/.test(opening)
    ? 'return { notify: "supervisor", note: "Late.", holding_reply: "We are sorry for the wait. An agent is on it." }' : undefined,
}, async ({ helpdesk, plans }) => {
  await customer(helpdesk, 'J', 'I cannot log in at all.');
  await until(async () => (await helpdesk.ticket('J'))?.escalation, 3000);
  assert.equal(plans[0].holding_reply, null);
}));

test('deadlineMode nl reads the written policy; a bad answer falls back to the table; shadow serves the table', async () => {
  const policy = 'urgent: 2 minutes. Default: 10 minutes.';
  const asks = answer => opening => /may wait for an agent's first answer/.test(opening) ? `return ${answer}` : undefined;
  await withDesk({ deadlineMode: 'nl', deadlinePolicy: policy, script: asks(2) }, async ({ helpdesk, model }) => {
    await customer(helpdesk, 'K', 'Please add dark mode.');
    await until(async () => (await helpdesk.ticket('K'))?.triaged === 1);
    const k = await helpdesk.ticket('K');
    assert.equal(k.allowedMs, 120_000); assert.equal(k.due, k.messages[0].at + 120_000);
    assert.ok(model.openings.some(opening => opening.includes(policy)), 'the stage reads the written policy');
  });
  await withDesk({ deadlineMode: 'nl', script: asks(0) }, async ({ helpdesk, failures }) => {
    await customer(helpdesk, 'L', 'Please add dark mode.');
    await until(async () => (await helpdesk.ticket('L'))?.triaged === 1);
    assert.equal((await helpdesk.ticket('L')).allowedMs, 600_000, 'the crisp table stands in');
    assert.equal(failures.length >= 1, true);
  });
  await withDesk({ deadlineMode: 'shadow', script: asks(2) }, async ({ helpdesk, model }) => {
    await customer(helpdesk, 'M', 'Please add dark mode.');
    await until(async () => (await helpdesk.ticket('M'))?.triaged === 1);
    assert.equal((await helpdesk.ticket('M')).allowedMs, 600_000, 'shadow serves the table');
    assert.ok(model.openings.some(opening => /may wait for an agent's first answer/.test(opening)), 'and also asks the policy');
  });
});

test('inboxMode nl orders by the written rules; an order that is not a permutation falls back to the comparator', async () => {
  const both = async (options, check) => withDesk(options, async made => {
    await customer(made.helpdesk, 'P', 'Please add dark mode.');
    await customer(made.helpdesk, 'Q', 'Please add dark mode.');
    await until(async () => (await made.helpdesk.ticket('P'))?.triaged === 1 && (await made.helpdesk.ticket('Q'))?.triaged === 1);
    await check(made);
  });
  const ask = answer => opening => /Order the ids of tickets/.test(opening) ? `return ${answer}` : undefined;
  await both({ script: ask('["Q","P"]') }, async ({ helpdesk, model }) => {
    assert.deepEqual((await helpdesk.inbox()).map(t => t.id), ['P', 'Q'], 'crisp by default');
    assert.equal(model.openings.some(opening => /Order the ids of tickets/.test(opening)), false, 'no model call in crisp mode');
  });
  await both({ inboxMode: 'nl', rankingPolicy: 'Security first.', script: ask('["Q","P"]') }, async ({ helpdesk, model }) => {
    assert.deepEqual((await helpdesk.inbox()).map(t => t.id), ['Q', 'P']);
    assert.ok(model.openings.some(opening => opening.includes('Security first.')));
  });
  await both({ inboxMode: 'nl', script: ask('["Q"]') }, async ({ helpdesk, failures }) => {
    assert.deepEqual((await helpdesk.inbox()).map(t => t.id), ['P', 'Q']);
    assert.match(failures.at(-1)[1], /every offered id exactly once/);
  });
  await both({ inboxMode: 'shadow', script: ask('["Q","P"]') }, async ({ helpdesk }) => {
    assert.deepEqual((await helpdesk.inbox()).map(t => t.id), ['P', 'Q'], 'shadow serves the comparator');
  });
});

test('the crisp defaults are the previous behaviour, and the crisp helpers are exact', () => {
  const ticket = (id, due, escalated = false) => ({ id, urgency: 'normal', topic: '', summary: '', escalated, due });
  assert.deepEqual([ticket('b', 5), ticket('a', 5), ticket('c', null), ticket('d', 9, true)].sort(crispInboxOrder).map(t => t.id), ['d', 'a', 'b', 'c']);
  assert.equal(isRanking(['a', 'b'], [ticket('a', 1), ticket('b', 2)]), true);
  assert.equal(isRanking(['a', 'a'], [ticket('a', 1), ticket('b', 2)]), false);
  assert.equal(isRanking(['a'], [ticket('a', 1), ticket('b', 2)]), false);
  assert.equal(isRanking(['a', 'z'], [ticket('a', 1), ticket('b', 2)]), false);
  assert.deepEqual(RESPONSE_TIMES, { urgent: 900_000, normal: 14_400_000, low: 86_400_000 });
  assert.match(describeResponseTimes(RESPONSE_TIMES), /urgent tickets: 15 minutes\. normal tickets: 240 minutes\. low tickets: 1440 minutes/);
  const plan = crispEscalationPlan({ urgency: 'low', topic: 'billing', summary: 'Asks about an invoice.', missing: [], waited_minutes: 30, allowed_minutes: 20, late_messages: 0 });
  assert.deepEqual([plan.notify, plan.holding_reply], ['supervisor', null]);
});

test('every predicate of types.ts has a crisp checker, and the checkers decide exactly', async () => {
  const types = readFileSync(new URL('../../applications/helpdesk/types.ts', import.meta.url), 'utf8');
  const declared = new Set([...types.matchAll(/Is<[^"]*"([^"]*)"/g)].map(match => match[1].replace(/\s+/g, ' ').trim()));
  const crisp = await appCrisp('helpdesk');
  assert.deepEqual([...declared].filter(key => !(key in crisp)), []);
  assert.deepEqual(Object.keys(crisp).filter(key => !declared.has(key)), []);
  assert.equal(crisp['two to four words naming a product area']('account access'), true);
  assert.equal(crisp['two to four words naming a product area']('accounts'), false);
  assert.equal(crisp['two to four words naming a product area']('one two three four five'), false);
  assert.equal(crisp['a positive whole number of minutes'](0), false);
  assert.equal(crisp['a positive whole number of minutes'](2.5), false);
  assert.equal(crisp['a positive whole number of minutes'](90), true);
  const draft = crisp['two to five sentences of plain text addressed to the customer'];
  assert.equal(draft('Sorry about that. Could you send the invoice number?'), true);
  assert.equal(draft('One sentence only.'), false);
  assert.equal(draft('Version 2.1 is out. Please update. **Bold** is markup.'), false);
  assert.equal(sentenceCount('It costs 3.5 dollars. Is that fine?'), 2);
  const summary = crisp['one sentence that states what is wrong or wanted and what the customer already tried'];
  assert.equal(summary('Cannot log in; already reset the password.'), true);
  assert.equal(summary('Cannot log in. Reset the password.'), false);
});
