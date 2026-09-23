// Application and interface-coverage families: inline placement and near neighbours, late rows past the
// opening preview, contract diagnosis with warranted and unwarranted edits, directory reducers, eval
// repair, child results that change the parent's plan, and hierarchical callable discovery.
import { Random, blockedCall, capitalize, curriculumCase, evalCall, literal, nonceWords, returnCall } from './lib.mjs';

/** Aggregate rows exactly, then apply a prose policy by subgroup; the aggregate and subgroups can disagree. */
export function cohortPolicy(seed, index) {
  const rng = new Random(seed, `cohort:${index}`);
  const groups = ['new', 'returning'];
  const make = (cohort, group, n, rate) => Array.from({ length: n }, (_, i) => ({ cohort, group, converted: i < Math.round(n * rate) }));
  // A shared prefix keeps the opening identical; the variants differ in rows past the preview.
  const prefix = rng.shuffle([...make('A', 'new', 10, 0.3), ...make('B', 'new', 10, 0.3), ...make('A', 'returning', 10, 0.5), ...make('B', 'returning', 10, 0.5)]);
  const tails = {
    b_better_everywhere: [...make('A', 'new', 20, 0.2), ...make('B', 'new', 20, 0.35), ...make('A', 'returning', 20, 0.5), ...make('B', 'returning', 20, 0.65)],
    // B is better within each group but worse pooled, because it saw mostly new customers.
    simpson: [...make('A', 'new', 4, 0.25), ...make('B', 'new', 36, 0.35), ...make('A', 'returning', 32, 0.75), ...make('B', 'returning', 8, 1)],
    mixed: [...make('A', 'new', 20, 0.2), ...make('B', 'new', 20, 0.4), ...make('A', 'returning', 20, 0.7), ...make('B', 'returning', 20, 0.5)],
    b_worse_everywhere: [...make('A', 'new', 20, 0.4), ...make('B', 'new', 20, 0.2), ...make('A', 'returning', 20, 0.7), ...make('B', 'returning', 20, 0.5)],
  };
  const policy = 'Judge the claim within each customer group, not over all customers pooled. The claim is supported if variant B converts at least 2 percentage points better than A in every group, contradicted if B converts at least 2 points worse in every group, and uncertain otherwise.';
  const verdictOf = rows => {
    const rate = (cohort, group) => { const sel = rows.filter(r => r.cohort === cohort && r.group === group); return 100 * sel.filter(r => r.converted).length / sel.length; };
    const diffs = groups.map(group => rate('B', group) - rate('A', group));
    return diffs.every(d => d >= 2) ? 'supported' : diffs.every(d => d <= -2) ? 'contradicted' : 'uncertain';
  };
  const shape = `cohort${index}`;
  return Object.entries(tails).map(([variant, tail]) => {
    const rows = [...prefix, ...rng.shuffle(tail)];
    const expected = verdictOf(rows);
    const code = `const rate = (cohort: string, group: string) => {
  const sel = rows.filter(r => r.cohort === cohort && r.group === group);
  return 100 * sel.filter(r => r.converted).length / sel.length;
};
const diffs = ['new', 'returning'].map(g => ({ group: g, a: rate('A', g), b: rate('B', g), diff: rate('B', g) - rate('A', g) }));
diffs`;
    return curriculumCase({ family: 'inline_cohort_policy', shape, variant, pairGroup: `cohort:${shape}`,
      slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'optional',
      evidence: { world: [policy], retrieved: [`${rows.length} rows`], background: ['Pooled rates can reverse subgroup rates (Simpson\'s paradox).'] },
      plausibleActions: [], minimumSequence: ['compute per-group conversion rates in code', 'apply the policy to the computed differences'],
      reference: { root: [evalCall(code), returnCall(expected)] },
      root: { name: 'check_claim', args: { claim: 'string', policy: 'string', rows: 'Visit[]' }, returns: 'Verdict',
        instructions: 'Decide whether the experiment rows support claim, following policy.' },
      files: { 'types.ts': 'export type Visit = { cohort: "A" | "B", group: "new" | "returning", converted: boolean };\nexport type Verdict = "supported" | "contradicted" | "uncertain";\n' },
      inputs: { claim: 'Variant B converts better than variant A.', policy, rows }, expected });
  });
}

const TICKETS = {
  live: ['Checkout has returned a 500 error for every customer since 9:10 this morning.', 'Customers in Spain cannot log in; the login page times out.',
    'Invoices sent today show the wrong currency, and customers are writing in about it.', 'The mobile app crashes on launch for everyone who installed today\'s update.'],
  request: ['It would be great if the dashboard had a dark mode.', 'Could you add an export to spreadsheet button?',
    'Please consider supporting two-factor authentication by text message.'],
  resolved: ['Yesterday\'s search outage is fixed; the customer confirmed it works again.', 'The broken password reset link was repaired last week and verified.',
    'Thanks, the missing receipts arrived after your fix.'],
};
function inbox(tickets) {
  return `const TICKETS: { id: string, priority: number, text: string }[] = ${literal(tickets)};
/** Today's open tickets, oldest first. */
export default function inbox(): { id: string, priority: number, text: string }[] { return TICKETS; }
`;
}
const REVIEW_EACH = `export type Ticket = { id: string, priority: number, text: string };
/** Judge each ticket separately and return the ids of those judged true, in input order. */
export default async function review_each(tickets: Ticket[], judge: (ticket: Ticket) => Promise<boolean>): Promise<string[]> {
  const kept: string[] = [];
  for (const ticket of tickets) if (await judge(ticket)) kept.push(ticket.id);
  return kept;
}
`;

/**
 * A per-item semantic filter through a typed callback helper (inline required), and its near neighbour
 * whose criterion is an exact field test (an inline child is gratuitous).
 */
export function reviewEach(seed, index) {
  const rng = new Random(seed, `review:${index}`);
  const shape = `inbox${index}`;
  const draw = () => {
    const kinds = rng.shuffle(['live', 'live', 'request', 'resolved', rng.pick(['live', 'request', 'resolved']), rng.pick(['request', 'resolved'])]);
    return kinds.map((kind, i) => ({ id: `T${100 + i}`, priority: rng.int(1, 5), text: rng.pick(TICKETS[kind]), kind }));
  };
  // The second draw must change both answers, or the pair would not be counterfactual.
  const live = tickets => tickets.filter(t => t.kind === 'live').map(t => t.id).join();
  const urgent = tickets => tickets.filter(t => t.priority >= 3).map(t => t.id).join();
  const first = draw();
  let second = draw();
  for (let tries = 0; tries < 50 && (live(second) === live(first) || urgent(second) === urgent(first)); tries++) second = draw();
  const cases = [];
  for (const [variant, tickets] of [['a', first], ['b', second]]) {
    const plain = tickets.map(({ kind, ...ticket }) => ticket);
    const semantic = tickets.filter(t => t.kind === 'live').map(t => t.id);
    cases.push(curriculumCase({ family: 'inline_review_each', shape, variant, pairGroup: `review:${shape}`,
      slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
      evidence: { world: tickets.map(t => `${t.id} (${t.kind}): ${t.text}`), retrieved: semantic, background: [] },
      plausibleActions: [], minimumSequence: ['load the inbox', 'judge each ticket in an inline child through review_each'],
      reference: { root: [evalCall('const kept = await review_each(inbox(), nl`Does ticket report a problem that customers are experiencing right now?`);\nkept'), returnCall(semantic)],
        children: plain.map(t => ({ match: JSON.stringify(t.id), value: semantic.includes(t.id) })) },
      root: { name: 'live_incidents', args: {}, returns: 'string[]',
        instructions: 'From the tickets in inbox(), keep the ones that report a problem customers are experiencing right now: not feature requests, and not problems already fixed. Judge the tickets one at a time with review_each. Return the kept ticket ids in inbox order.' },
      files: { 'live_incidents/inbox.ts': inbox(plain), 'live_incidents/review_each.ts': REVIEW_EACH }, inputs: {}, expected: semantic }));
    const threshold = 3;
    const crisp = plain.filter(t => t.priority >= threshold).map(t => t.id);
    cases.push(curriculumCase({ family: 'inline_avoid_crisp', shape, variant, pairGroup: `crisp:${shape}`,
      slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'avoid',
      evidence: { world: plain.map(t => `${t.id} p${t.priority}`), retrieved: crisp, background: [] },
      plausibleActions: [], minimumSequence: ['filter by the priority field in code'],
      reference: { root: [evalCall(`inbox().filter(t => t.priority >= ${threshold}).map(t => t.id)`), returnCall(crisp)] },
      root: { name: 'urgent_queue', args: {}, returns: 'string[]',
        instructions: `From the tickets in inbox(), keep the ones whose priority is ${threshold} or more. review_each is available for judging tickets one at a time. Return the kept ticket ids in inbox order.` },
      files: { 'urgent_queue/inbox.ts': inbox(plain), 'urgent_queue/review_each.ts': REVIEW_EACH }, inputs: {}, expected: crisp }));
  }
  return cases;
}

const ROLLOUT_LATE = {
  regression: { note: 'Customer checkout errors in eu-west doubled after the new build reached 50% of traffic.', decision: 'halt' },
  recovered: { note: 'The eu-west error spike from minute 12 has returned to baseline and stayed there for twenty minutes.', decision: 'continue' },
  internal: { note: 'A synthetic test tenant reported errors; no customer traffic was affected.', decision: 'continue' },
};
/** A long health log whose decisive row is past the opening preview. */
export function pagedLateRow(seed, index) {
  const rng = new Random(seed, `rollout:${index}`);
  const regions = ['us-east', 'us-west', 'eu-west', 'ap-south'];
  const early = Array.from({ length: 34 }, (_, minute) => ({ minute, region: rng.pick(regions), status: 'ok',
    note: rng.pick(['latency normal', 'error rate at baseline', 'traffic ramping as planned', 'no alerts', 'cache hit rate steady']) }));
  early[12] = { minute: 12, region: 'eu-west', status: 'warn', note: 'Error rate spiked briefly in eu-west while caches warmed.' };
  const shape = `rollout${index}`;
  return Object.entries(ROLLOUT_LATE).map(([variant, late]) => {
    const tail = Array.from({ length: 6 }, (_, i) => ({ minute: 34 + i, region: rng.pick(regions), status: 'ok', note: 'no alerts' }));
    tail.splice(3, 0, { minute: 37, region: 'eu-west', status: variant === 'regression' ? 'fail' : 'ok', note: late.note });
    const checks = [...early, ...tail].map((row, i) => ({ ...row, minute: i }));
    return curriculumCase({ family: 'paged_late_row', shape, variant, pairGroup: `rollout:${shape}`,
      slice: 'observation_followup', domain: 'other', mode: 'followup',
      evidence: { world: checks.map(c => `${c.minute} ${c.region} ${c.status}: ${c.note}`), retrieved: [late.note], background: [] },
      decisive: [{ marker: late.note, source: 'page', note: 'the late health check decides' }],
      plausibleActions: ['continue from the early rows', 'halt on the early warning', 'read the rest and decide'],
      minimumSequence: ['read every health check, including the rows the opening cuts off', 'judge customer impact'],
      reference: { root: [evalCall('checks.slice(30).map(c => `${c.minute} ${c.region} ${c.status}: ${c.note}`).join("\\n")'), returnCall(late.decision)] },
      root: { name: 'rollout_gate', args: { checks: 'HealthCheck[]' }, returns: '"continue" | "halt"',
        instructions: 'Decide whether the rollout may continue. Halt when a health check shows customers are currently harmed by the new build; a problem that has recovered, or one that touches no customer traffic, does not stop the rollout.' },
      files: { 'types.ts': 'export type HealthCheck = { minute: number, region: string, status: string, note: string };\n' },
      inputs: { checks }, expected: late.decision });
  });
}

/** A helper with a documented contract: a real defect that must be fixed, or a correct helper that must be left alone. */
export function contractDiagnosis(seed, index) {
  const rng = new Random(seed, `contract:${index}`);
  const lines = Array.from({ length: rng.int(4, 6) }, (_, i) => ({ sku: `K${i + 1}`, quantity: rng.int(1, 5), unit_cents: rng.int(200, 1500), discount_cents: rng.int(0, 300) }));
  lines[rng.int(0, lines.length - 1)].discount_cents = 99_999; // a discount larger than the line
  const clampTotal = lines.reduce((sum, l) => sum + Math.max(0, l.quantity * l.unit_cents - l.discount_cents), 0);
  const doc = '/** The amount of one invoice line in cents: quantity times unit price, less the line discount. A discount never makes a line negative; such a line is 0. */';
  const buggy = `${doc}\nexport default function line_total(line: { sku: string, quantity: number, unit_cents: number, discount_cents: number }): number {\n  return line.quantity * line.unit_cents - line.discount_cents;\n}\n`;
  const correct = `${doc}\nexport default function line_total(line: { sku: string, quantity: number, unit_cents: number, discount_cents: number }): number {\n  return Math.max(0, line.quantity * line.unit_cents - line.discount_cents);\n}\n`;
  const shape = `invoice${index}`;
  const variants = { defect: { source: buggy, edits: 'required' }, sound: { source: correct, edits: 'forbidden' } };
  return Object.entries(variants).map(([variant, v]) => {
    const reference = variant === 'defect' ? [evalCall('lines.map(l => line_total(l))'), ['read_function', { name: 'line_total' }],
      ['edit_function', { name: 'line_total', find: 'return line.quantity * line.unit_cents - line.discount_cents;',
        replace_with: 'return Math.max(0, line.quantity * line.unit_cents - line.discount_cents);' }],
      evalCall('lines.reduce((sum, l) => sum + line_total(l), 0)'), returnCall(clampTotal)] :
      [evalCall('lines.map(l => line_total(l))'), evalCall('lines.reduce((sum, l) => sum + line_total(l), 0)'), returnCall(clampTotal)];
    return curriculumCase({ family: 'contract_diagnosis', shape, variant, pairGroup: `contract:${shape}`,
      slice: 'folder_failure', domain: 'other', mode: variant === 'defect' ? 'followup' : 'single_call', edits: v.edits,
      evidence: { world: [doc], retrieved: ['line totals'], background: [] },
      decisive: variant === 'defect' ? [{ marker: 'line.quantity * line.unit_cents - line.discount_cents;', source: 'function_source', note: 'the source misses the clamp' }] : [],
      plausibleActions: ['sum line_total as given', 'fix line_total to match its contract', 'clamp in the caller'],
      minimumSequence: variant === 'defect' ? ['run line_total on the lines', 'see a negative line', 'read and fix the helper', 'recompute'] :
        ['run line_total on the lines', 'sum them'],
      reference: { root: reference },
      root: { name: 'invoice_total', args: { lines: 'Line[]' }, returns: 'number',
        instructions: 'Return the invoice total in cents: the sum of line_total over lines. line_total must follow its documented contract; if it does not, fix line_total itself.' },
      files: { 'invoice_total/line_total.ts': v.source, 'types.ts': 'export type Line = { sku: string, quantity: number, unit_cents: number, discount_cents: number };\n' },
      inputs: { lines }, expected: clampTotal });
  });
}

const TICKET_FILES = {
  confirmed: 'Status: resolved\nThe customer replied that the export works again.\n',
  unconfirmed: 'Status: resolved\nThe fix is deployed; the customer has not replied yet.\n',
  open: 'Status: open\nStill reproducing the sync failure on the customer\'s laptop.\n',
  wontfix: 'Status: closed\nDeclined: this is how the product is meant to work, and the customer accepted that.\n',
};
/** A directory reducer: read the criteria, then move the matching ticket files. Criteria differ by variant. */
export function folderCriteria(seed, index) {
  const rng = new Random(seed, `folder:${index}`);
  const names = nonceWords(rng, 5).map(name => `tickets/${name}.md`);
  const kinds = rng.shuffle(['confirmed', 'unconfirmed', 'open', 'wontfix', rng.pick(['confirmed', 'unconfirmed'])]);
  const tickets = Object.fromEntries(names.map((name, i) => [name, TICKET_FILES[kinds[i]]]));
  const criteria = {
    customer_confirmed: { text: 'Archive a ticket once its problem is resolved and the customer has confirmed the fix.\n', keep: ['confirmed'] },
    no_action_left: { text: 'Archive every ticket on which nobody has anything left to do: fixed and confirmed by the customer, or closed with the customer\'s agreement.\n', keep: ['confirmed', 'wontfix'] },
  };
  // Criteria files are padded to one size: the opening's file listing shows sizes, and must not tell the variants apart.
  const width = Math.max(...Object.values(criteria).map(c => c.text.length));
  for (const c of Object.values(criteria)) c.text = c.text.trimEnd().padEnd(width - 1) + '\n';
  const shape = `tickets${index}`;
  return Object.entries(criteria).map(([variant, c]) => {
    const moved = names.filter((_, i) => c.keep.includes(kinds[i]));
    const expectedFiles = Object.fromEntries(Object.entries({ 'criteria.md': c.text, ...tickets })
      .map(([path, text]) => [moved.includes(path) ? path.replace('tickets/', 'archive/') : path, text]));
    const result = moved.map(path => path.split('/').pop()).sort();
    const moves = moved.map(path => `await folder.file(${JSON.stringify(path)}).moveTo(${JSON.stringify(path.replace('tickets/', 'archive/'))});`).join('\n');
    return curriculumCase({ family: 'folder_criteria_reducer', shape, variant, pairGroup: `folder:${shape}`,
      slice: 'folder_failure', domain: 'other', mode: 'followup',
      evidence: { world: [c.text, ...Object.entries(tickets).map(([p, t]) => `${p}: ${t.trim()}`)], retrieved: [c.text], background: [] },
      decisive: [{ marker: c.text.trim(), source: 'file', note: 'the archiving criteria' }],
      plausibleActions: ['archive every resolved ticket', 'archive only confirmed ones', 'also archive agreed closures'],
      minimumSequence: ['read criteria.md', 'read each ticket', 'move the matching tickets'],
      reference: { root: [['read_file', { path: 'criteria.md' }],
        evalCall(`const texts: Record<string, string> = {};\nfor (const f of await folder.files('tickets/*.md')) texts[f.relativePath] = await f.readText();\ntexts`),
        evalCall(moves || '0'), returnCall(result)] },
      root: { name: 'archive_tickets', kind: 'directory-reducer', args: {}, returns: 'string[]',
        instructions: 'Move the tickets in tickets/ that criteria.md says to archive into archive/, keeping their file names. Return the file names you moved, sorted alphabetically.' },
      folderFiles: { 'criteria.md': c.text, ...tickets }, expectedFiles, inputs: {}, expected: result });
  });
}

const EVIDENCE = {
  basic_sufficient: { basic: ['The parcel tracking shows delivery to the customer\'s door with a signed photo of the customer holding it.'],
    detailed: [], finding: 'fraud' },
  needs_detail: { basic: ['The customer says the parcel never arrived.'],
    detailed: ['The courier\'s GPS log shows the van never stopped at the customer\'s street that day.', 'The depot found the parcel on a shelf the next morning.'], finding: 'legitimate' },
  undetermined: { basic: ['The customer says the parcel never arrived.'],
    detailed: ['The courier\'s scanner failed that day, so there is no delivery record either way.'], finding: null },
};
/** A named child judges sufficiency; an insufficient answer should lead to more reading, then a decision or a blocker. */
export function childSufficiency(seed, index) {
  const rng = new Random(seed, `child:${index}`);
  const claim = `R-${rng.int(1000, 9999)}`;
  const shape = `refund${index}`;
  return Object.entries(EVIDENCE).map(([variant, e]) => {
    const assess = (list, sufficient, finding) => ({ match: JSON.stringify(list[0]).slice(1, 40), value: { sufficient, finding } });
    const children = variant === 'basic_sufficient' ? [assess(e.basic, true, 'fraud')] :
      [assess(e.basic, false, null), assess(e.detailed, variant === 'needs_detail', e.finding)];
    const root = variant === 'basic_sufficient' ?
      [evalCall(`const first = await assess(records.basic(${JSON.stringify(claim)}));\nfirst`), returnCall(e.finding)] :
      [evalCall(`const first = await assess(records.basic(${JSON.stringify(claim)}));\nfirst`),
        evalCall(`const second = await assess(records.detailed(${JSON.stringify(claim)}));\nsecond`),
        variant === 'needs_detail' ? returnCall(e.finding) : blockedCall('Neither the basic nor the detailed records settle whether the parcel was delivered.')];
    const module = `/** The first-line records for a refund claim. */
export function basic(claim: string): string[] { return ${literal(e.basic)}; }
/** The detailed courier and depot records for a refund claim; slower to obtain. */
export function detailed(claim: string): string[] { return ${literal(e.detailed)}; }
`;
    return curriculumCase({ family: 'child_sufficiency', shape, variant, pairGroup: `child:${shape}`,
      slice: 'nested_scoped', domain: 'logic', mode: 'followup',
      evidence: { world: [...e.basic, ...e.detailed], retrieved: variant === 'basic_sufficient' ? e.basic : e.detailed, background: [] },
      decisive: [{ marker: (variant === 'basic_sufficient' ? e.basic : e.detailed)[0].slice(0, 40), source: 'child', note: 'the evidence the decision rests on' }],
      plausibleActions: ['decide from the first assessment', 'read the detailed records', 'report a blocker'],
      minimumSequence: ['assess the basic records', 'if insufficient, assess the detailed records', 'decide or report a blocker'],
      reference: { root, children },
      root: { name: 'review_refund', args: { claim: 'string' }, returns: '"fraud" | "legitimate"',
        instructions: 'Decide whether refund claim is fraud or legitimate. Assess records.basic(claim) with assess first. If assess finds that evidence insufficient, assess records.detailed(claim). If the evidence still does not settle it, report that you are blocked.' },
      files: { 'review_refund/records.ts': module, 'review_refund/assess.nl': `---
description: Judge whether evidence settles a refund claim, and which way.
args:
  evidence: "string[]"
returns: "Assessment"
---
Decide whether evidence is enough to tell whether the parcel was delivered to the customer. If it is,
finding is "fraud" when it was delivered and "legitimate" when it was not; if it is not enough, sufficient
is false and finding is null.
`, 'types.ts': 'export type Assessment = { sufficient: boolean, finding: "fraud" | "legitimate" | null };\n' },
      inputs: { claim }, expected: e.finding, ...(e.finding === null ? { operation: 'blocked' } : {}) });
  });
}

/** Hierarchical callables: the right leaf is nested, and a flat-name guess fails. */
export function moduleDiscovery(seed, index) {
  const rng = new Random(seed, `modules:${index}`);
  const regions = { north: 8, south: 12, east: 5, west: 20 };
  const months = { 1: 0, 6: 10, 11: 25, 12: 30 };
  const shape = `pricing${index}`;
  const files = {
    'quote/pricing/tax.ts': `const RATES: Record<string, number> = ${literal(regions)};
/** The sales tax rate for a region, in percent. */
export function rate(region: string): number {
  if (!(region in RATES)) throw new Error('unknown region ' + region);
  return RATES[region];
}
/** Add a region's sales tax to an amount in cents, rounding to the nearest cent. */
export function with_tax(amount_cents: number, region: string): number { return Math.round(amount_cents * (1 + rate(region) / 100)); }
`,
    'quote/pricing/discounts/seasonal.ts': `const PERCENT: Record<number, number> = ${literal(months)};
/** The seasonal discount for a month (1 to 12), in percent. */
export function percent(month: number): number { return PERCENT[month] ?? 0; }
`,
    'quote/pricing/discounts/loyalty.ts': `/** The loyalty discount in percent for a customer with this many years of membership. */
export default function loyalty(years: number): number { return Math.min(15, years * 3); }
`,
  };
  return [0, 1].map(variant => {
    const region = rng.pick(Object.keys(regions)), month = Number(rng.pick(Object.keys(months))), years = rng.int(0, 7);
    const subtotal = rng.int(20, 400) * 100;
    const discount = Math.max(months[month], Math.min(15, years * 3));
    const discounted = Math.round(subtotal * (1 - discount / 100));
    const expected = Math.round(discounted * (1 + regions[region] / 100));
    const code = `const discount = Math.max(pricing.discounts.seasonal.percent(order.month), pricing.discounts.loyalty(order.years));
const discounted = Math.round(order.subtotal_cents * (1 - discount / 100));
pricing.tax.with_tax(discounted, order.region)`;
    return curriculumCase({ family: 'scoped_module_discovery', shape, variant: `v${variant}`, splitGroup: `modules:${shape}`,
      slice: 'nested_scoped', domain: 'other', mode: 'single_call', inline: 'avoid',
      evidence: { world: ['tax, seasonal, and loyalty tables'], retrieved: [String(expected)], background: [] },
      plausibleActions: [], minimumSequence: ['read the callable listing', 'call the nested leaves'],
      reference: { root: [evalCall(code), returnCall(expected)] },
      root: { name: 'quote', args: { order: 'Order' }, returns: 'number',
        instructions: 'Price order in cents: take the larger of the seasonal discount for its month and the loyalty discount for its membership years, round the discounted subtotal to a whole cent, then add the sales tax for its region.' },
      files: { ...files, 'types.ts': 'export type Order = { subtotal_cents: number, month: number, years: number, region: string };\n' },
      inputs: { order: { subtotal_cents: subtotal, month, years, region } }, expected });
  });
}
