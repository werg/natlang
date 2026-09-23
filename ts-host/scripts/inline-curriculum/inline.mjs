// Inline families: semantic judgments over several computed values, union and structured results, live
// Date/Map values, and the choice between an existing named function and a new inline one.
import { capitalize, curriculumCase, evalCall, literal, nlFile, nonceWords, Random, returnCall } from './lib.mjs';

const GATE_POLICY = 'Roll back a service whose error rate is above 5% unless its release notes announce those specific failures as expected. ' +
  'Hold a service whose median latency is above 300 ms, or whose error rate is above 5% with failures the notes announce. Otherwise proceed.';

/**
 * Multi-capture inference: exact metrics per service, then a judgment that combines them with the release notes
 * and a policy. The notes decide whether the same failures are announced, unrelated, or announced for another path.
 */
export function releaseGate(seed, index) {
  const rng = new Random(seed, `gate:${index}`);
  const [failing, slow, healthy] = nonceWords(rng, 3);
  const failingPath = rng.pick(['/v1/export', '/v1/reports', '/legacy/sync']);
  const otherPath = rng.pick(['/checkout', '/cart/items', '/login']);
  const requests = (errors, slowMs, errorPath) => Array.from({ length: 40 }, (_, i) => ({
    status: i < errors ? 503 : 200, ms: slowMs ? rng.int(320, 700) : rng.int(40, 240), path: i < errors ? errorPath : rng.pick(['/', '/search', '/account']) }));
  const shape = `release${index}`;
  const announce = `This release switches off ${failingPath}; clients still calling it receive 503 until they upgrade, which is expected.`;
  const variants = {
    announced: { path: failingPath, notes: announce, decision: 'hold' },
    unrelated: { path: failingPath, notes: 'This release changes the colour of the checkout button and updates translations.', decision: 'rollback' },
    other_path: { path: otherPath, notes: announce, decision: 'rollback' },
  };
  return Object.entries(variants).map(([variant, spec]) => {
    const data = {
      [failing]: { requests: requests(6, false, spec.path), notes: spec.notes },
      [slow]: { requests: requests(0, true, ''), notes: 'Adds a second database replica for reporting queries.' },
      [healthy]: { requests: requests(1, false, '/search'), notes: 'Minor dependency updates.' },
    };
    const expected = { [failing]: spec.decision, [slow]: 'hold', [healthy]: 'proceed' };
    return curriculumCase({ family: 'inline_multi_capture', shape, variant, pairGroup: `gate:${shape}`,
      slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
      evidence: { world: [`${failing} fails ${spec.path}`, spec.notes], retrieved: ['error rate', 'median latency', 'failing paths', 'release notes'], background: [] },
      minimumSequence: ['compute each service\'s metrics exactly', 'judge each service from its metrics, notes, and the policy in its own child'],
      reference: { root: [evalCall(`const policy = ${JSON.stringify(GATE_POLICY)};
const decisions: Record<string, Decision> = {};
for (const service of release.services()) {
  const rows = release.requests(service);
  const failures = rows.filter(row => row.status >= 500);
  const latencies = rows.map(row => row.ms).sort((a, b) => a - b);
  const facts = { service, errorRate: failures.length / rows.length, medianMs: latencies[Math.floor(latencies.length / 2)],
    failingPaths: [...new Set(failures.map(row => row.path))], notes: release.release_notes(service) };
  decisions[service] = await nl\`Decide under policy whether to proceed with, hold, or roll back the service described by facts.\`(facts);
}
return decisions;`), returnCall(expected)],
        children: Object.entries(expected).map(([service, value]) => ({ match: JSON.stringify(service), value })) },
      root: { name: 'release_gate', args: {}, returns: 'Record<string, Decision>',
        instructions: `For every service in release.services(), compute from release.requests(service) its error rate (the share of requests with status 500 or above), its median latency in ms, and the paths of its failing requests. Then decide for each service, in a separate judgment that sees those computed values and the service's release notes, under this policy: ${GATE_POLICY} Return a record from service name to decision.` },
      files: {
        'release_gate/release.ts': `const DATA: Record<string, { requests: { status: number, ms: number, path: string }[], notes: string }> = ${literal(data)};
/** The services in this release. */
export function services(): string[] { return Object.keys(DATA); }
/** Requests to service since the release: HTTP status, latency in ms, and request path. */
export function requests(service: string): { status: number, ms: number, path: string }[] { return DATA[service]?.requests ?? []; }
/** The release notes for service. */
export function release_notes(service: string): string { return DATA[service]?.notes ?? ''; }
`,
        'types.ts': 'export type Decision = "proceed" | "hold" | "rollback";\n' },
      inputs: {}, expected });
  });
}

const MESSAGES = [
  { text: 'I was charged twice for the March invoice and the second charge is still pending.', triage: { kind: 'bug', component: 'billing' } },
  { text: 'The invoice PDF still shows our old VAT number although I changed it in settings yesterday.', triage: { kind: 'bug', component: 'billing' } },
  { text: 'Searching for an exact order number returns nothing even though the order exists.', triage: { kind: 'bug', component: 'search' } },
  { text: 'Search results stop loading after the second page and the spinner never ends.', triage: { kind: 'bug', component: 'search' } },
  { text: 'The password reset link says it has expired even when I open it right away.', triage: { kind: 'bug', component: 'login' } },
  { text: 'Codes from my authenticator app are rejected on the sign-in page every time.', triage: { kind: 'bug', component: 'login' } },
  { text: 'Is there a discount if we pay yearly instead of monthly?', triage: { kind: 'question', topic: 'pricing' } },
  { text: 'What would the team plan cost us for twelve seats?', triage: { kind: 'question', topic: 'pricing' } },
  { text: 'How do I connect the app to our Slack workspace?', triage: { kind: 'question', topic: 'setup' } },
  { text: 'Which DNS records do we add to use our own domain?', triage: { kind: 'question', topic: 'setup' } },
  { text: 'Where are our customer records stored, and can we choose the EU region?', triage: { kind: 'question', topic: 'privacy' } },
  { text: 'How long do you keep deleted files before they are gone for good?', triage: { kind: 'question', topic: 'privacy' } },
  { text: 'Thanks for the quick help last week, have a great weekend!', triage: { kind: 'noise' } },
  { text: 'I am out of the office until Monday with no access to email.', triage: { kind: 'noise' } },
  { text: 'Congratulations on the launch, the new logo looks great.', triage: { kind: 'noise' } },
];

/** A union-of-records result for each message, collected with Promise.all and keyed back to its message. */
export function triageUnion(seed, index) {
  const rng = new Random(seed, `triage:${index}`);
  const shape = `inbox${index}`;
  const draw = () => rng.sample(MESSAGES, 5).map((message, i) => ({ id: `M${i + 1}`, ...message }));
  let first = draw(), second = draw();
  for (let tries = 0; tries < 20 && JSON.stringify(first.map(m => m.triage)) === JSON.stringify(second.map(m => m.triage)); tries++) second = draw();
  return [['a', first], ['b', second]].map(([variant, messages]) => {
    const expected = Object.fromEntries(messages.map(m => [m.id, m.triage]));
    const plain = messages.map(({ id, text }) => ({ id, text }));
    return curriculumCase({ family: 'inline_union_target', shape, variant, pairGroup: `triage:${shape}`,
      slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
      evidence: { world: messages.map(m => `${m.id}: ${JSON.stringify(m.triage)}`), retrieved: [], background: [] },
      minimumSequence: ['triage every message in its own child, concurrently', 'key the results back to message ids'],
      reference: { root: [evalCall(`const messages = inbox();
const triaged = await Promise.all(messages.map(message => nl\`Triage message: a bug report with its component, a question with its topic, or noise.\`(message)));
return Object.fromEntries(messages.map((message, i) => [message.id, triaged[i]]));`), returnCall(expected)],
        children: plain.map(m => ({ match: JSON.stringify(m.id), value: expected[m.id] })) },
      root: { name: 'triage_inbox', args: {}, returns: 'Record<string, Triage>',
        instructions: 'Triage every message from inbox(): a bug report names the component it concerns, a question names its topic, and anything else is noise. Judge each message separately rather than all of them in one pass. Return a record from message id to its triage.' },
      files: { 'triage_inbox/inbox.ts': `const MESSAGES = ${literal(plain)};\n/** Today's support messages. */\nexport default function inbox(): { id: string, text: string }[] { return MESSAGES; }\n`,
        'types.ts': 'export type Component = "billing" | "search" | "login";\nexport type Topic = "pricing" | "setup" | "privacy";\n' +
          'export type Triage = { kind: "bug", component: Component } | { kind: "question", topic: Topic } | { kind: "noise" };\n' },
      inputs: {}, expected });
  });
}

/**
 * Live Date and Map values: overdue contacts computed exactly from a Map of Dates, with a note that may defer
 * the follow-up to a date that has or has not arrived.
 */
export function followUpDue(seed, index) {
  const rng = new Random(seed, `followup:${index}`);
  const [held, overdue, recent, quiet] = nonceWords(rng, 4).map(capitalize);
  const today = '2026-03-01';
  const last = { [held]: '2026-01-20', [overdue]: '2026-02-03', [recent]: '2026-02-24', [quiet]: '2026-02-12' };
  const shape = `contacts${index}`;
  const fair = rng.pick(['trade fair', 'board meeting', 'product launch']);
  const variants = {
    deferred: { note: `Asked us to hold off until after their ${fair} on March 9.`, due: false },
    deferral_passed: { note: `Asked us to hold off until after their ${fair} on February 20.`, due: true },
    unrelated: { note: 'Prefers email to phone calls and reads it in the mornings.', due: true },
  };
  return Object.entries(variants).map(([variant, spec]) => {
    const notes = { [held]: spec.note, [overdue]: 'Renewal is due in April.', [recent]: 'Happy with the onboarding so far.', [quiet]: 'Asked for the new price list.' };
    const expected = [held, overdue, quiet].filter(name => name !== held || spec.due).sort();
    return curriculumCase({ family: 'stateful_dates', shape, variant, pairGroup: `dates:${shape}`,
      slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'optional',
      evidence: { world: [`today ${today}`, ...Object.entries(last).map(([n, d]) => `${n} last ${d}`), spec.note], retrieved: ['days since each contact', 'the deferral note'], background: [] },
      minimumSequence: ['compute days since each last contact from the Map of Dates', 'read the notes of overdue customers', 'apply the deferral'],
      reference: { root: [evalCall(`const now = crm.today();
const overdue = [...crm.last_contacts()].filter(([, date]) => (now.getTime() - date.getTime()) / 86400000 > 14).map(([name]) => name);
overdue.map(name => name + ': ' + crm.note(name))`), returnCall(expected)] },
      root: { name: 'due_follow_ups', args: {}, returns: 'string[]',
        instructions: 'A customer is due for a follow-up when more than 14 days have passed between their last contact (crm.last_contacts()) and crm.today(), unless their note (crm.note(name)) asks us to wait until a date that is still after today. Return the names of the customers who are due, sorted alphabetically.' },
      files: { 'due_follow_ups/crm.ts': `const LAST: [string, string][] = ${literal(Object.entries(last))};
const NOTES: Record<string, string> = ${literal(notes)};
/** When each customer was last contacted. */
export function last_contacts(): Map<string, Date> { return new Map(LAST.map(([name, day]) => [name, new Date(day + "T12:00:00Z")])); }
/** The date of this review. */
export function today(): Date { return new Date("${today}T12:00:00Z"); }
/** The account manager's note on customer. */
export function note(customer: string): string { return NOTES[customer] ?? ""; }
` },
      inputs: {}, expected });
  });
}

const RECEIPTS = [
  { text: 'CAFE LUMEN\n2x Espresso 5,60\n1x Croissant 3,20\nTOTAL 8,80 EUR\nThank you!', total: 8.8, currency: 'EUR' },
  { text: 'Harbor Books - receipt #4471\nPaperback  $14.99\nBookmark    $2.50\nSales tax   $1.40\nAmount paid: $18.89', total: 18.89, currency: 'USD' },
  { text: 'Northline Rail\nSingle ticket London - Leeds\nFare paid £42.10 (card)', total: 42.1, currency: 'GBP' },
  { text: 'Bäckerei Sonne\nBrot 4,20 €\nKuchen 3,90 €\nSumme: 8,10 €', total: 8.1, currency: 'EUR' },
  { text: 'QuickPark garage, 3 hours\nRate $4.00/hour\nTotal charged: $12.00', total: 12, currency: 'USD' },
  { text: 'The Crown pub\n2 pints £11.40\nCrisps £1.60\nTotal £13.00\nService not included', total: 13, currency: 'GBP' },
  { text: 'Hotel Aster, 1 night\nRoom 118,00 EUR\nCity tax 4,50 EUR\nGrand total 122,50 EUR', total: 122.5, currency: 'EUR' },
  { text: 'Metro taxi #88\nDistance 7.2 mi\nFare $21.35 + tip $4.00 = $25.35 paid', total: 25.35, currency: 'USD' },
];

/** Structured extraction: each receipt's total and currency need a typed record, so `nl<T>` is required. */
export function receiptTotals(seed, index) {
  const rng = new Random(seed, `receipts:${index}`);
  const shape = `receipts${index}`;
  const draw = () => rng.sample(RECEIPTS, 4).map((receipt, i) => ({ id: `R${i + 1}`, ...receipt }));
  let first = draw(), second = draw();
  const sums = receipts => {
    const out = {};
    for (const r of receipts) out[r.currency] = Math.round(((out[r.currency] ?? 0) + r.total) * 100) / 100;
    return out;
  };
  for (let tries = 0; tries < 20 && JSON.stringify(sums(first)) === JSON.stringify(sums(second)); tries++) second = draw();
  return [['a', first], ['b', second]].map(([variant, receipts]) => {
    const expected = sums(receipts);
    const plain = receipts.map(({ id, text }) => ({ id, text }));
    return curriculumCase({ family: 'inline_structured_extract', shape, variant, pairGroup: `receipts:${shape}`,
      slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
      evidence: { world: receipts.map(r => `${r.id}: ${r.total} ${r.currency}`), retrieved: [], background: ['decimal commas in European receipts'] },
      minimumSequence: ['extract each receipt\'s total and currency in its own typed child', 'sum exactly per currency'],
      reference: { root: [evalCall(`const receipts = expenses.receipts();
const read = await Promise.all(receipts.map(receipt => nl<Amount>\`Read the final amount paid on receipt and its currency.\`(receipt)));
const totals: Record<string, number> = {};
for (const amount of read) totals[amount.currency] = Math.round(((totals[amount.currency] ?? 0) + amount.total) * 100) / 100;
return totals;`), returnCall(expected)],
        children: receipts.map(r => ({ match: JSON.stringify(r.id), value: { total: r.total, currency: r.currency } })) },
      root: { name: 'expense_totals', args: {}, returns: 'Record<string, number>',
        instructions: 'Add up the final amount paid on every receipt from expenses.receipts(), per currency (EUR, USD, or GBP), rounded to cents. Read each receipt in its own judgment. Return a record from currency code to total.' },
      files: { 'expense_totals/expenses.ts': `const RECEIPTS = ${literal(plain)};\n/** The scanned receipts of this trip, as text. */\nexport function receipts(): { id: string, text: string }[] { return RECEIPTS; }\n`,
        'types.ts': 'export type Amount = { total: number, currency: "EUR" | "USD" | "GBP" };\n' },
      inputs: {}, expected });
  });
}

const TICKETS = [
  { text: 'Checkout is down for every customer since 9:00, we are losing orders.', level: 'high', competitor: false },
  { text: 'Our nightly export deleted last month\'s invoices from the archive.', level: 'high', competitor: false },
  { text: 'The CSV export puts dates in the wrong column; we fix it by hand for now.', level: 'medium', competitor: false },
  { text: 'We are comparing you with Zendesk; can your reports show first-response time like theirs do?', level: 'low', competitor: true },
  { text: 'After moving from Freshdesk, our old ticket links no longer open, but search still finds the tickets.', level: 'medium', competitor: true },
  { text: 'The logo on the login page is slightly blurry on large screens.', level: 'low', competitor: false },
  { text: 'Someone outside our company could open our admin dashboard without logging in.', level: 'high', competitor: false },
  { text: 'Intercom offers a free tier for small teams; do you have anything similar?', level: 'low', competitor: true },
];

/**
 * A callable folder already has the right judgment (`urgency`), so a new inline function is gratuitous; the
 * neighbouring task asks a question no helper answers, so an inline judgment per ticket is the right design.
 */
export function namedVersusInline(seed, index) {
  const rng = new Random(seed, `named:${index}`);
  const tickets = rng.sample(TICKETS, 5).map((ticket, i) => ({ id: `T${i + 1}`, ...ticket }));
  const plain = tickets.map(({ id, text }) => ({ id, text }));
  const shape = `support${index}`;
  const files = {
    'support_queue/tickets.ts': `const TICKETS = ${literal(plain)};\n/** The open support tickets. */\nexport default function tickets(): { id: string, text: string }[] { return TICKETS; }\n`,
    'support_queue/urgency.nl': nlFile({ args: { ticket: 'Ticket' }, returns: 'Level',
      description: 'Rate how urgent a support ticket is.',
      instructions: 'Rate the urgency of ticket. high: an outage, data loss, or a security problem. medium: a broken feature that has a workaround. low: a question, a request, or a cosmetic issue.' }),
    'types.ts': 'export type Ticket = { id: string, text: string };\nexport type Level = "low" | "medium" | "high";\n',
  };
  const high = tickets.filter(t => t.level === 'high').map(t => t.id);
  const competitor = tickets.filter(t => t.competitor).map(t => t.id);
  return [
    curriculumCase({ family: 'named_versus_inline', shape, variant: 'named', slice: 'nested_scoped', domain: 'other', mode: 'single_call',
      inline: 'avoid', named: 'required',
      evidence: { world: tickets.map(t => `${t.id}: ${t.level}`), retrieved: [], background: [] },
      minimumSequence: ['rate every ticket with the existing urgency function', 'keep the high ones'],
      reference: { root: [evalCall(`const open = tickets();
const levels = await Promise.all(open.map(ticket => urgency(ticket)));
return open.filter((ticket, i) => levels[i] === 'high').map(ticket => ticket.id);`), returnCall(high)],
        children: plain.map(t => ({ match: JSON.stringify(t.id), value: tickets.find(x => x.id === t.id).level })) },
      root: { name: 'support_queue', args: {}, returns: 'string[]',
        instructions: 'Return the ids of the open tickets that are high urgency, in ticket order. Judge each ticket separately.' },
      files, inputs: {}, expected: high }),
    curriculumCase({ family: 'named_versus_inline', shape, variant: 'inline', slice: 'inline_placement', domain: 'other', mode: 'single_call',
      inline: 'required',
      evidence: { world: tickets.map(t => `${t.id}: competitor ${t.competitor}`), retrieved: [], background: ['Zendesk, Freshdesk, and Intercom are competing help-desk products'] },
      minimumSequence: ['judge every ticket in its own inline child', 'keep the ones that mention a competitor'],
      reference: { root: [evalCall(`const open = tickets();
const mentions = await Promise.all(open.map(ticket => nl\`Does ticket mention a competing help-desk product by name?\`(ticket)));
return open.filter((ticket, i) => mentions[i]).map(ticket => ticket.id);`), returnCall(competitor)],
        children: plain.map(t => ({ match: JSON.stringify(t.id), value: tickets.find(x => x.id === t.id).competitor })) },
      root: { name: 'support_queue', args: {}, returns: 'string[]',
        instructions: 'Return the ids of the open tickets that mention a competing help-desk product by name, in ticket order. Judge each ticket separately.' },
      files, inputs: {}, expected: competitor }),
  ];
}
