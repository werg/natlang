// Follow-up and repair families: a policy read after measuring, a counterexample that revises a rule,
// parallel per-item labels, repair of an untyped inline lambda, and idempotent retry after a lost ack.
import { Random, capitalize, curriculumCase, evalCall, literal, nonceWords, returnCall } from './lib.mjs';

/** Measure first, then read a runbook policy whose threshold or exception decides. */
export function policyAfterMeasure(seed, index) {
  const rng = new Random(seed, `measure:${index}`);
  const samples = Array.from({ length: 40 }, (_, minute) => ({ minute, ms: minute >= 20 && minute < 25 ? rng.int(900, 1300) : rng.int(180, 520) }));
  const sorted = samples.map(s => s.ms).sort((a, b) => a - b);
  const p95 = sorted[Math.ceil(0.95 * sorted.length) - 1];
  const policies = {
    strict: { text: 'Page the on-call engineer when p95 latency is above 800 ms.', decision: 'page' },
    lenient: { text: `Page the on-call engineer when p95 latency is above ${Math.ceil((p95 + 100) / 100) * 100} ms; below that, hold and review in the morning.`, decision: 'hold' },
    maintenance: { text: 'Page the on-call engineer when p95 latency is above 800 ms, unless every sample above 800 ms falls inside the announced maintenance window, minutes 20 to 24.', decision: 'hold' },
    maintenance_missed: { text: 'Page the on-call engineer when p95 latency is above 800 ms, unless every sample above 800 ms falls inside the announced maintenance window, minutes 30 to 34.', decision: 'page' },
  };
  const shape = `latency${index}`;
  return Object.entries(policies).map(([variant, policy]) => curriculumCase({ family: 'policy_after_measure', shape, variant,
    pairGroup: `measure:${shape}`, slice: 'observation_followup', domain: 'other', mode: 'followup',
    evidence: { world: [policy.text], retrieved: [`p95 = ${p95} ms`, policy.text], background: [] },
    decisive: [{ marker: policy.text, source: 'eval', note: 'the runbook policy' }],
    plausibleActions: ['page on the measured spike', 'hold'], minimumSequence: ['compute p95', 'read the policy', 'apply it to the measurement'],
    reference: { root: [evalCall(`const values = samples.map(s => s.ms).sort((a, b) => a - b);
const p95 = values[Math.ceil(0.95 * values.length) - 1];
console.log('p95', p95, 'above 800 at minutes', samples.filter(s => s.ms > 800).map(s => s.minute).join(','));
console.log(runbook.policy());`), returnCall(policy.decision)] },
    root: { name: 'latency_alert', args: { samples: 'Sample[]' }, returns: '"page" | "hold"',
      instructions: 'Decide whether to page the on-call engineer about the latency samples. Measure their p95 latency (the smallest sample that at least 95% of samples do not exceed), then apply the paging policy from runbook.policy().' },
    files: { 'latency_alert/runbook.ts': `/** The current paging policy, in prose. */\nexport function policy(): string { return ${JSON.stringify(policy.text)}; }\n`,
      'types.ts': 'export type Sample = { minute: number, ms: number };\n' },
    inputs: { samples }, expected: policy.decision }));
}

/** A merge rule that looks right on early rows; a later row is a counterexample in one variant and a confirmation in the other. */
export function counterexampleRevision(seed, index) {
  const rng = new Random(seed, `dedupe:${index}`);
  const used = new Set();
  const first = nonceWords(rng, 10, used).map(capitalize), last = nonceWords(rng, 5, used).map(capitalize);
  const email = (i, j) => `${first[i].toLowerCase()}.${last[j].toLowerCase()}@mail.example`;
  const base = [
    { id: 'C01', name: `${first[0]} ${last[0]}`, email: email(0, 0), note: 'Signed up in March.' },
    { id: 'C02', name: `${first[1]} ${last[1]}`, email: email(1, 1), note: 'Prefers phone contact.' },
    { id: 'C03', name: `${first[0]} ${last[0]}`, email: email(0, 0), note: 'Placed a second order from a new device.' },
    { id: 'C04', name: `${first[2]} ${last[2]}`, email: email(2, 2), note: 'Asked for paper invoices.' },
    { id: 'C05', name: `${first[3]} ${last[3]}`, email: email(3, 3), note: 'Newsletter subscriber.' },
    { id: 'C06', name: `${first[1]} ${last[1]}`, email: email(1, 1), note: 'Updated the delivery address.' },
    { id: 'C07', name: `${first[4]} ${last[4]}`, email: email(4, 4), note: 'Signed up at the spring fair.' },
    { id: 'C08', name: `${first[5]} ${last[0]}`, email: email(5, 0), note: 'Referred by a friend.' },
    { id: 'C09', name: `${first[6]} ${last[1]}`, email: email(6, 1), note: 'Returned one item.' },
    { id: 'C10', name: `${first[7]} ${last[2]}`, email: email(7, 2), note: 'Pays by bank transfer.' },
  ];
  const variants = {
    shared_inbox: { row: { id: 'C11', name: `${first[8]} ${last[3]}`, email: email(3, 3),
      note: `Uses the family inbox shared with ${first[3]}; a different person with a separate account.` } },
    same_person: { row: { id: 'C11', name: `${first[3]} ${last[3]}`, email: email(3, 3),
      note: `${first[3]} again, writing from the same address after changing banks.` } },
  };
  const shape = `customers${index}`;
  return Object.entries(variants).map(([variant, v]) => {
    const records = [...base, { id: 'C12', name: `${first[9]} ${last[4]}`, email: `${first[9].toLowerCase()}@mail.example`, note: 'New this week.' }];
    records.splice(10, 0, v.row);
    const groups = new Map();
    for (const r of records) {
      const key = variant === 'shared_inbox' && r.id === 'C11' ? 'C11' : r.email;
      groups.set(key, [...(groups.get(key) ?? []), r.id]);
    }
    const expected = [...groups.values()].map(ids => ids.sort()).sort((a, b) => a[0].localeCompare(b[0]));
    const module = `const RECORDS = ${literal(records)};
/** Customer records in file order, 4 per page; pages are numbered from 1 to pages(). */
export function page(n: number): { id: string, name: string, email: string, note: string }[] {
  if (!Number.isInteger(n) || n < 1 || n > ${Math.ceil(records.length / 4)}) throw new RangeError('pages run from 1 to ${Math.ceil(records.length / 4)}');
  return RECORDS.slice((n - 1) * 4, n * 4);
}
/** The number of pages of records. */
export function pages(): number { return ${Math.ceil(records.length / 4)}; }
`;
    return curriculumCase({ family: 'counterexample_revision', shape, variant, pairGroup: `dedupe:${shape}`,
      slice: 'observation_followup', domain: 'other', mode: 'followup',
      evidence: { world: records.map(r => `${r.id} ${r.name} <${r.email}> ${r.note}`), retrieved: [v.row.note], background: ['People in one household can share an email inbox.'] },
      decisive: [{ marker: v.row.note, source: 'eval', note: 'the late record that tests the email rule' }],
      plausibleActions: ['group by email address', 'separate the record that shares an inbox'],
      minimumSequence: ['group the early records by email', 'read every record', 'revise the rule where a note contradicts it'],
      reference: { root: [evalCall(`const all: { id: string, name: string, email: string, note: string }[] = [];
const count = customers.pages();
for (let p = 1; p <= count; p++) all.push(...customers.page(p));
for (const r of all) console.log(r.id, r.name, r.email, r.note);`), returnCall(expected)] },
      root: { name: 'dedupe_customers', args: {}, returns: 'string[][]',
        instructions: 'Group the customer records in customers so that each group holds the records of one person. Records of one person usually share an email address. Return the groups as sorted id lists, ordered by their first id, including single-record groups.' },
      files: { 'dedupe_customers/customers.ts': module }, inputs: {}, expected });
  });
}

const FEEDBACK = {
  bug: ['The export button does nothing when I click it.', 'Totals on the invoice page are off by one cent.', 'The app logs me out every few minutes.'],
  praise: ['Setting up the account took two minutes, lovely.', 'Your support team solved my problem the same day.', 'The new dashboard is so much clearer.'],
  question: ['Is there a way to share a report with a colleague?', 'Do you offer a discount for charities?', 'Which browsers do you support?'],
};
/** Parallel per-item labels whose association with their items must be kept. */
export function parallelLabels(seed, index) {
  const rng = new Random(seed, `labels:${index}`);
  const draw = () => rng.shuffle(Object.entries(FEEDBACK).flatMap(([label, texts]) => rng.sample(texts, rng.int(1, 3)).map(text => ({ label, text }))))
    .map((item, i) => ({ ...item, id: `F${i + 1}` }));
  const shape = `feedback${index}`;
  return [['a', draw()], ['b', draw()]].map(([variant, items]) => {
    const expected = Object.fromEntries(items.map(item => [item.id, item.label]));
    const plain = items.map(({ id, text }) => ({ id, text }));
    return curriculumCase({ family: 'parallel_labels', shape, variant, pairGroup: `labels:${shape}`,
      slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
      evidence: { world: items.map(i => `${i.id} ${i.label}: ${i.text}`), retrieved: [], background: [] },
      plausibleActions: [], minimumSequence: ['label every item in its own inline child, concurrently', 'map ids to labels'],
      reference: { root: [evalCall(`const items = feedback();
const labels = await Promise.all(items.map(item => nl<Label>\`Label item as a bug report, praise, or a question.\`(item)));
return Object.fromEntries(items.map((item, i) => [item.id, labels[i]]));`), returnCall(expected)],
        children: plain.map(item => ({ match: JSON.stringify(item.id), value: expected[item.id] })) },
      root: { name: 'label_feedback', args: {}, returns: 'Record<string, Label>',
        instructions: 'Label every item from feedback() as "bug", "praise", or "question". Judge each item on its own, in a separate judgment, rather than all of them in one pass. Return a record from item id to label.' },
      files: { 'label_feedback/feedback.ts': `const ITEMS = ${literal(plain)};\n/** This week's feedback items. */\nexport default function feedback(): { id: string, text: string }[] { return ITEMS; }\n`,
        'types.ts': 'export type Label = "bug" | "praise" | "question";\n' },
      inputs: {}, expected });
  });
}

const REPORTS = [
  { text: 'After the rollback, error rates returned to baseline and customers confirm checkout works.', status: 'recovered' },
  { text: 'The restart helped a little, but a third of requests still time out.', status: 'degraded' },
  { text: 'Every request to the payment service fails; the service does not respond at all.', status: 'down' },
  { text: 'Latency is still twice the normal level, although no requests are failing.', status: 'degraded' },
  { text: 'Monitoring has been green for an hour and the incident channel was closed.', status: 'recovered' },
];
/**
 * A seeded eval uses an untyped inline lambda's result as an object. Nothing says what its fields are, so the
 * compiler rejects it and proposes an annotation; the repair types the lambda (or answers directly).
 */
export function inlineTypeRepair(seed, index) {
  const report = REPORTS[index % REPORTS.length];
  const shape = `status${index}`;
  return [curriculumCase({ family: 'inline_type_repair', shape, variant: 'seeded', slice: 'inline_placement', domain: 'other',
    mode: 'followup', inline: 'optional',
    evidence: { world: [report.text], retrieved: ['nl-unknown-return'], background: [] },
    decisive: [{ marker: 'nl-unknown-return', source: 'error', note: 'the compiler diagnostic for the untyped inline lambda' }],
    plausibleActions: ['annotate the inline lambda\'s target type', 'judge the report directly'],
    minimumSequence: ['read the diagnostic', 'give the lambda its type or answer directly'],
    reference: { root: [evalCall('const outcome: Status = await nl`Decide from report whether the service has recovered, is degraded, or is down.`(report);\noutcome'),
      returnCall(report.status)], children: [{ match: 'Decide from report', value: report.status }] },
    root: { name: 'incident_status', args: { report: 'string' }, returns: 'Status',
      instructions: 'Classify the service status described in report.' },
    files: { 'types.ts': 'export type Status = "recovered" | "degraded" | "down";\n' },
    failureSeed: { kind: 'compile', code: 'const assessment = await nl`Decide from report whether the service has recovered, is degraded, or is down, and quote the evidence.`(report);\nassessment.status' },
    inputs: { report: report.text }, expected: report.status })];
}

/** Idempotent retry: a send succeeds but its acknowledgement is lost; the same command is retried, never a new one. */
export function idempotentRetry(seed, index) {
  const rng = new Random(seed, `retry:${index}`);
  const people = nonceWords(rng, 4).map(capitalize);
  const reminders = people.map((name, i) => ({ id: `R${i + 1}`, to: `${name.toLowerCase()}@clinic.example`, text: `Hello ${name}, your appointment is tomorrow at ${9 + i}:00.` }));
  const failKey = reminders[rng.int(0, reminders.length - 1)].id;
  const commands = reminders.map(r => ({ key: r.id, to: r.to, text: r.text }));
  return [curriculumCase({ family: 'idempotent_retry', shape: `reminders${index}`, variant: failKey, slice: 'folder_failure', domain: 'other', mode: 'followup',
    evidence: { world: ['mail delivers a key at most once'], retrieved: ['delivery succeeded but its acknowledgement was lost'], background: [] },
    decisive: [{ marker: 'acknowledgement was lost', source: 'error', note: 'the failed acknowledgement' }],
    plausibleActions: ['report the reminder as failed', 'retry the same command', 'send a changed command'],
    minimumSequence: ['send every reminder', 'see the lost acknowledgement', 'retry that command unchanged'],
    reference: { root: [evalCall(`const failures: string[] = [];
for (const r of reminders) {
  try { mail.send({ key: r.id, to: r.to, text: r.text }); } catch (error) { failures.push(r.id + ': ' + String(error)); }
}
failures`), evalCall(`for (const r of reminders.filter(r => r.id === ${JSON.stringify(failKey)})) mail.send({ key: r.id, to: r.to, text: r.text });
'retried'`), returnCall(reminders.map(r => r.id))] },
    root: { name: 'send_reminders', args: { reminders: 'Reminder[]' }, returns: 'string[]',
      instructions: 'Send every reminder with mail.send({ key, to, text }), using the reminder id as the key. mail delivers each key at most once, so sending the same command again is always safe. Return the ids of the reminders whose delivery is confirmed, in input order.' },
    files: { 'types.ts': 'export type Reminder = { id: string, to: string, text: string };\n' },
    inputs: { reminders }, expected: reminders.map(r => r.id),
  }), ].map(record => ({ ...record, semantics: { ...record.semantics,
    effects: { 'mail.send': { kind: 'deliver_once_ack_loss', fail_key: failKey, expected_delivered: commands } } } }));
}

/**
 * A live class instance from a callable module: reserve order lines, and roll back when one line is short.
 * The fingerprint of the stock after the call is part of the result, so a rollback cannot be claimed without
 * doing it.
 */
export function liveInventory(seed, index) {
  const rng = new Random(seed, `inventory:${index}`);
  const skus = nonceWords(rng, 4).map(word => `SKU-${word.toUpperCase()}`);
  const order = skus.slice(0, 3).map(sku => ({ sku, quantity: rng.int(2, 6) }));
  const shortSku = order[rng.int(1, 2)].sku;
  const fingerprint = stock => Object.entries(stock).sort().map(([sku, n]) => `${sku}=${n}`).join(';');
  const variants = {
    enough: Object.fromEntries(skus.map(sku => [sku, 10])),
    short: Object.fromEntries(skus.map(sku => [sku, sku === shortSku ? 1 : 10])),
  };
  const shape = `order${index}`;
  return Object.entries(variants).map(([variant, stock]) => {
    const ok = variant === 'enough';
    const after = ok ? Object.fromEntries(Object.entries(stock).map(([sku, n]) => [sku, n - (order.find(l => l.sku === sku)?.quantity ?? 0)])) : stock;
    const expected = { reserved: ok ? order.map((_, i) => `RSV-${i + 1}`) : [], short: ok ? null : shortSku, stock_after: fingerprint(after) };
    const module = `export class Inventory {
  private readonly stock: Record<string, number> = ${literal(stock)};
  private readonly holds = new Map<string, { sku: string, quantity: number }>();
  private next = 1;
  /** Units of sku that can still be reserved. */
  available(sku: string): number { return this.stock[sku] ?? 0; }
  /** Reserve quantity units of sku; returns the reservation id, or throws when there are not enough. */
  reserve(sku: string, quantity: number): string {
    if (this.available(sku) < quantity) throw new Error('only ' + this.available(sku) + ' of ' + sku + ' available, ' + quantity + ' requested');
    this.stock[sku] -= quantity;
    const id = 'RSV-' + this.next++;
    this.holds.set(id, { sku, quantity });
    return id;
  }
  /** Cancel a reservation and return its units to stock. */
  release(id: string): void {
    const hold = this.holds.get(id);
    if (!hold) throw new Error('no reservation ' + id);
    this.stock[hold.sku] += hold.quantity;
    this.holds.delete(id);
  }
  /** A fingerprint of the current stock levels. */
  fingerprint(): string { return Object.entries(this.stock).sort().map(([sku, n]) => sku + '=' + n).join(';'); }
}
const shared = new Inventory();
/** The warehouse inventory; every call returns the same live instance. */
export function open(): Inventory { return shared; }
`;
    const code = `const inv = inventory.open();
const reserved: string[] = [];
let short: string | null = null;
for (const line of order) {
  if (inv.available(line.sku) < line.quantity) { short = line.sku; break; }
  reserved.push(inv.reserve(line.sku, line.quantity));
}
if (short) { for (const id of reserved) inv.release(id); reserved.length = 0; }
console.log({ reserved, short, stock_after: inv.fingerprint() });`;
    return curriculumCase({ family: 'live_inventory', shape, variant, pairGroup: `inventory:${shape}`,
      slice: 'nested_scoped', domain: 'actor', mode: 'followup',
      evidence: { world: [fingerprint(stock)], retrieved: [expected.stock_after], background: [] },
      decisive: [{ marker: expected.stock_after, source: 'eval', note: 'the stock fingerprint after reserving or rolling back' }],
      plausibleActions: ['reserve every line', 'roll back and report the short line'],
      minimumSequence: ['open the live inventory', 'reserve line by line', 'on a shortage release what was reserved', 'read the fingerprint'],
      reference: { root: [evalCall(code), returnCall(expected)] },
      root: { name: 'reserve_order', args: { order: 'Line[]' }, returns: 'Reservation',
        instructions: `Reserve every line of order in the inventory from inventory.open(), in order.
If some line cannot be fully reserved, release everything this order already reserved, so the order holds nothing; short is that line's sku (null when every line was reserved), and reserved lists the reservation ids still held.
stock_after is the inventory's fingerprint() once you are done.` },
      files: { 'reserve_order/inventory.ts': module,
        'types.ts': 'export type Line = { sku: string, quantity: number };\nexport type Reservation = { reserved: string[], short: string | null, stock_after: string };\n' },
      inputs: { order }, expected });
  });
}

const ASKS = [
  { text: 'We ask for 4,000 dollars for equipment, plus 600 for travel.', total: 4600 },
  { text: 'The project needs 2,500 dollars in total.', total: 2500 },
  { text: 'Our request is 7,200 dollars; no other costs.', total: 7200 },
  { text: 'We need 3,000 dollars for printing and another 1,500 for the venue.', total: 4500 },
  { text: 'Funding of 5,500 dollars would cover everything.', total: 5500 },
  { text: 'Nine hundred dollars for materials is all we need.', total: 900 },
  { text: 'We request 6,000 dollars, of which 1,000 is a contingency we will return if unused.', total: 6000 },
  { text: 'The workshop costs 1,800 dollars, plus 1,200 for a translator.', total: 3000 },
];
/**
 * Late binding: one saved inline judgment captures a `let` budget that is reassigned between two
 * screening rounds; each call must see the budget current at call time.
 */
export function lateBinding(seed, index) {
  const rng = new Random(seed, `late:${index}`);
  const applications = rng.sample(ASKS, 5).map((ask, i) => ({ id: `A${i + 1}`, ask: ask.text, total: ask.total }));
  const limits = [rng.pick([5000, 6000, 7500]), rng.pick([2500, 3000, 4000])];
  const plain = applications.map(({ id, ask }) => ({ id, ask }));
  const expected = limits.map(limit => applications.filter(a => a.total <= limit).map(a => a.id));
  const code = `let budget = limits[0];
const fits: (application: Application) => Promise<boolean> = nl\`Does the total amount application asks for, including every extra cost it mentions, fit within budget dollars?\`;
const rounds: string[][] = [];
for (const limit of limits) {
  budget = limit;
  const kept: string[] = [];
  for (const application of applications) if (await fits(application)) kept.push(application.id);
  rounds.push(kept);
}
return rounds;`;
  return [curriculumCase({ family: 'inline_late_binding', shape: `grants${index}`, variant: 'rounds', slice: 'inline_placement', domain: 'other',
    mode: 'single_call', inline: 'required',
    evidence: { world: applications.map(a => `${a.id}: ${a.ask} (total ${a.total})`), retrieved: [], background: ['An inline function reads a captured let at call time.'] },
    plausibleActions: [], minimumSequence: ['define one judgment that mentions budget', 'reassign budget between rounds and call it again'],
    reference: { root: [evalCall(code), returnCall(expected)],
      children: limits.flatMap(limit => applications.map(a => ({ match: [JSON.stringify(a.id), `budget: number = ${limit}`], value: a.total <= limit }))) },
    root: { name: 'screen_grants', args: { applications: 'Application[]', limits: 'number[]' }, returns: 'string[][]',
      instructions: 'Screen the applications once for each budget in limits, in order. An application passes a round when everything it asks for, extras included, fits within that round\'s budget. Use one judgment of whether an application fits within the current budget for every round. Return, for each round, the ids that pass, in application order.' },
    files: { 'types.ts': 'export type Application = { id: string, ask: string };\n' },
    inputs: { applications: plain, limits }, expected })];
}
