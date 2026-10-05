// Dataset workbench: one source gives the data, the case composes what to do with it. Items come from labeled text
// datasets (labeled.mjs) and get crisp metadata (day, channel, amount); the semantic criteria are ad hoc buckets that
// regroup the datasets' gold labels behind natural definitions ("a card that is lost, stolen or not working" covers
// eight banking intents), so a criterion is not a label name and keyword matching does not decide it. Operations
// combine the per-item judgments with exact code: counts, ids, per-channel groups, the first match, an iterateOn scan
// that stops at the k-th match, and routing a mixed inbox (items of several datasets) by a rules table. Expected
// results come from the gold labels; paired worlds swap a few items. (Owner 2026-10-05: combinatorics of data-derived
// tasks; many inline lambdas.)
import { SOURCES } from './acquire.mjs';
import { sourceRecordId } from './folder-data.mjs';
import { DATASETS, rowsOf } from './labeled.mjs';
import { Random, curriculumCase, evalCall, literal, nlFile, returnCall } from './lib.mjs';

/** Buckets per dataset: name, the gold labels they cover, the lambda's question (with its noun) and the request's
 * wording. Buckets of one dataset may overlap. */
const BUCKETS = {
  banking77: [
    { name: 'card_trouble', labels: ['card_not_working', 'card_swallowed', 'compromised_card', 'lost_or_stolen_card', 'declined_card_payment',
      'card_arrival', 'virtual_card_not_working', 'contactless_not_working'],
      ask: 'Is the customer in message dealing with a payment card that is lost, stolen, compromised, not arriving or not working?',
      about: 'deal with a payment card that is lost, stolen, compromised, not arriving or not working' },
    { name: 'top_ups', labels: ['automatic_top_up', 'pending_top_up', 'top_up_failed', 'top_up_limits', 'top_up_reverted', 'topping_up_by_card',
      'top_up_by_cash_or_cheque', 'verify_top_up', 'top_up_by_bank_transfer_charge', 'top_up_by_card_charge'],
      ask: 'Is message about adding money to the account (topping it up)?', about: 'are about adding money to the account' },
    { name: 'unexpected_charges', labels: ['card_payment_fee_charged', 'extra_charge_on_statement', 'transaction_charged_twice', 'transfer_fee_charged',
      'cash_withdrawal_charge', 'exchange_charge'],
      ask: 'Is message a complaint or question about a fee or an extra or duplicate charge?', about: 'are about a fee or an extra or duplicate charge' },
    { name: 'stuck_transfers', labels: ['pending_transfer', 'transfer_not_received_by_recipient', 'failed_transfer', 'declined_transfer',
      'balance_not_updated_after_bank_transfer', 'transfer_timing'],
      ask: 'Is message about a money transfer that is late, pending, failed or not received?', about: 'are about a transfer that is late, failed or not received' },
    { name: 'access_identity', labels: ['unable_to_verify_identity', 'verify_my_identity', 'why_verify_identity', 'verify_source_of_funds',
      'passcode_forgotten', 'pin_blocked', 'change_pin'],
      ask: 'Is message about proving identity, or about a PIN or passcode?', about: 'are about proving identity or a PIN or passcode' },
    { name: 'unrecognised', labels: ['card_payment_not_recognised', 'direct_debit_payment_not_recognised', 'cash_withdrawal_not_recognised'],
      ask: 'Does the customer in message report a payment or withdrawal they do not recognise?', about: 'report a payment or withdrawal they do not recognise' },
  ],
  sst2: [
    { name: 'praise', labels: ['positive'], ask: 'Does message speak well of the film overall?', about: 'speak well of the film overall' },
    { name: 'pan', labels: ['negative'], ask: 'Does message speak badly of the film overall?', about: 'speak badly of the film overall' },
  ],
  ag_news: [
    { name: 'economy', labels: ['Business'], ask: 'Is message mainly about companies, markets or the economy?', about: 'are mainly about companies, markets or the economy' },
    { name: 'science_tech', labels: ['Sci/Tech'], ask: 'Is message mainly about science, technology or the internet?', about: 'are mainly about science, technology or the internet' },
    { name: 'sport', labels: ['Sports'], ask: 'Is message mainly about sport?', about: 'are mainly about sport' },
    { name: 'world', labels: ['World'], ask: 'Is message mainly about international politics or world events?', about: 'are mainly about international politics or world events' },
  ],
  sms_spam: [
    { name: 'unsolicited', labels: ['spam'], ask: 'Is message an unsolicited promotion, prize claim or scam?', about: 'are unsolicited promotions, prize claims or scams' },
  ],
};
// The emotion dataset is left out: its hashtag-derived labels are too noisy for an exact oracle.
/** Each source's kind of item: a crisp field, and the subject of a routing rule. */
const KINDS = { banking77: 'bank message', sst2: 'film comment', ag_news: 'news item', sms_spam: 'text message' };
const SETTINGS = {
  banking77: 'customer messages to a bank', sst2: 'comments about films',
  ag_news: 'news items in a feed', sms_spam: 'text messages to a shared phone',
};
const CHANNELS = ['email', 'chat', 'app'];
const TEAMS = ['cards', 'payments', 'trust', 'community', 'editorial', 'triage'];

function membership(bucket, label) { return bucket.labels.includes(label); }

function drawItems(rng, sources, count, split) {
  const items = [];
  for (let i = 0; i < count; i++) {
    const name = rng.pick(sources);
    const row = rng.pick(rowsOf(name, split));
    items.push({ source: name, text: row.text, label: row.label, sourceId: sourceRecordId(name, row.text, '') });
  }
  return items;
}

function decorate(rng, rows) {
  return rows.map((row, i) => ({ id: `M${i + 1}`, kind: KINDS[row.source], day: rng.int(1, 20), channel: rng.pick(CHANNELS), amount: rng.int(5, 900), ...row }));
}

// Operations: (rng, items, bucket) -> { text, returns, expected(items), code } --------------------------------------
const q = bucket => bucket.ask.replace(/`/g, '\\`');
const OPS = {
  count: (rng, b) => ({ text: `How many messages ${b.about}?`, returns: 'number',
    expected: items => items.filter(x => membership(b, x.label)).length,
    code: `const hits = await Promise.all(all.map(message => nl<boolean>\`${q(b)}\`(message)));\nreturn hits.filter(Boolean).length;` }),
  ids_recent: (rng, b) => { const day = rng.int(6, 14);
    return { text: `Return the ids of the messages from day ${day} on that ${b.about}, in id order.`, returns: 'string[]',
      expected: items => items.filter(x => x.day >= day && membership(b, x.label)).map(x => x.id),
      code: `const recent = all.filter(message => message.day >= ${day});\nconst hits = await Promise.all(recent.map(message => nl<boolean>\`${q(b)}\`(message)));\nreturn recent.filter((_, i) => hits[i]).map(message => message.id);` }; },
  per_channel: (rng, b) => ({ text: `Count the messages that ${b.about}, per channel; leave out channels with none.`, returns: 'Record<string, number>',
    expected: items => { const out = {}; for (const x of items) if (membership(b, x.label)) out[x.channel] = (out[x.channel] ?? 0) + 1; return out; },
    code: `const hits = await Promise.all(all.map(message => nl<boolean>\`${q(b)}\`(message)));\nconst out: Record<string, number> = {};\nall.forEach((message, i) => { if (hits[i]) out[message.channel] = (out[message.channel] ?? 0) + 1; });\nreturn out;` }),
  amount: (rng, b) => ({ text: `What is the total amount across the messages that ${b.about}?`, returns: 'number',
    expected: items => items.filter(x => membership(b, x.label)).reduce((s, x) => s + x.amount, 0),
    code: `const hits = await Promise.all(all.map(message => nl<boolean>\`${q(b)}\`(message)));\nreturn all.filter((_, i) => hits[i]).reduce((s, message) => s + message.amount, 0);` }),
  first: (rng, b) => ({ text: `Going through the messages in id order, return the id of the first one that ${b.about.replace(/^are /, 'is ')}, or null. Stop judging once you find it.`, returns: 'string | null',
    expected: items => items.find(x => membership(b, x.label))?.id ?? null, sequential: true,
    code: `const fits = nl<(message: Message) => Promise<boolean>>\`${q(b)}\`;\nfor (const message of all) if (await fits(message)) return message.id;\nreturn null;` }),
  kth: (rng, b) => { const k = rng.int(2, 3);
    return { text: `Scan the messages in id order and stop as soon as ${k} of them ${b.about}; return the id of the message where you stopped, or null if fewer than ${k} do. Use iterateOn over a scan state.`, returns: 'string | null',
      expected: items => items.filter(x => membership(b, x.label))[k - 1]?.id ?? null, sequential: true, iterate: true,
      code: `const fits = nl<(message: Message) => Promise<boolean>>\`${q(b)}\`;
const step = async (s: { next: number, hits: number, stop: string | null }) => {
  const message = all[s.next];
  const hits = s.hits + ((await fits(message)) ? 1 : 0);
  return { next: s.next + 1, hits, stop: hits >= ${k} ? message.id : null };
};
const final = await iterateOn(step, { next: 0, hits: 0, stop: null }).withLimit({maxSteps: 128}).until(s => s.stop !== null || s.next >= all.length);
return final.stop;` }; },
};

/** Routing a mixed inbox: rules map buckets (of different datasets) to teams, first match wins, else the default. */
function routingOp(rng, rules, fallback) {
  const listing = rules.map((r, i) => `${i + 1}. ${KINDS[r.source]}s that ${r.bucket.about} go to "${r.team}"`).join('; ');
  return { text: `Route every message to a team by these rules, the first rule that applies wins: ${listing}; anything else goes to "${fallback}". Return a record from message id to team.`,
    returns: 'Record<string, string>',
    expected: items => Object.fromEntries(items.map(x => [x.id, rules.find(r => r.source === x.source && membership(r.bucket, x.label))?.team ?? fallback])),
    code: `const teams = await Promise.all(all.map(message => routeOne(message)));\nreturn Object.fromEntries(all.map((message, i) => [message.id, teams[i]]));`,
    preamble: `const rules = inbox.rules();\nconst applies = nl<(message: Message, rule: string) => Promise<boolean>>\`Does message fall under rule? Answer from what message says.\`;
const routeOne = async (message: Message) => { for (const rule of rules) if (rule.kind === message.kind && await applies(message, rule.when)) return rule.team; return ${JSON.stringify(fallback)}; };` };
}

function serviceSource(items, rules) {
  const plain = items.map(({ id, kind, day, channel, amount, text }) => ({ id, kind, day, channel, amount, text }));
  return `const MESSAGES = ${literal(plain)};
${rules ? `const RULES = ${literal(rules.map(r => ({ kind: KINDS[r.source], when: `${KINDS[r.source]}s that ${r.bucket.about}`, team: r.team })))};\n/** The routing rules, in priority order: a rule applies to messages of its kind. */\nexport function rules(): { kind: string, when: string, team: string }[] { return RULES; }\n` : ''}/** Every message, in id order. */
export function messages(): Message[] { return MESSAGES; }
/** The messages that arrived on channel. */
export function by_channel(channel: string): Message[] { return MESSAGES.filter(m => m.channel === channel); }
/** The messages from day start to day end, inclusive. */
export function between(start: number, end: number): Message[] { return MESSAGES.filter(m => m.day >= start && m.day <= end); }
`;
}

const NEAR_MISS = nlFile({ args: { message: 'Message' }, returns: 'string', description: 'Summarize a message in a few words.',
  instructions: 'Summarize message in at most eight words.' });

/** A single-dataset case with one bucket and one operation, or a mixed inbox routed by rules. */
export function datasetWorkbench(seed, index, split = 'train') {
  const rng = new Random(seed, `dataset-workbench:${index}`);
  const names = Object.keys(BUCKETS);
  for (let attempt = 0; attempt < 40; attempt++) {
    const route = rng.next() < 0.3;
    const sources = route ? rng.sample(names, rng.int(2, 3)) : [rng.pick(names)];
    const count = rng.int(6, 10);
    let op, bucket = null, rules = null;
    if (route) {
      const teams = rng.sample(TEAMS, sources.length + 1);
      rules = sources.map((source, i) => ({ source, bucket: rng.pick(BUCKETS[source]), team: teams[i] }));
      op = routingOp(rng, rules, teams[sources.length]);
    } else {
      bucket = rng.pick(BUCKETS[sources[0]]);
      op = OPS[rng.pick(Object.keys(OPS))](rng, bucket);
    }
    const base = decorate(rng, drawItems(rng, sources, count, split));
    const decides = x => route ? rules.some(r => r.source === x.source && membership(r.bucket, x.label)) : membership(bucket, x.label);
    // World b: two to three items replaced by others from the same datasets with the opposite verdict.
    const swap = new Set(rng.sample(base.map(x => x.id), rng.int(2, 3)));
    const other = base.map(x => {
      if (!swap.has(x.id)) return x;
      const pool = rowsOf(x.source, split).filter(row => decidesRow(row) !== decides(x));
      function decidesRow(row) { return route ? rules.some(r => r.source === x.source && membership(r.bucket, row.label)) : membership(bucket, row.label); }
      if (!pool.length) return x;
      const row = rng.pick(pool);
      return { ...x, text: row.text, label: row.label, sourceId: sourceRecordId(x.source, row.text, '') };
    });
    const ea = op.expected(base), eb = op.expected(other);
    if (JSON.stringify(ea) === JSON.stringify(eb)) continue;
    if (!route && [base, other].every(items => !items.some(decides))) continue;
    const shape = `dataset${index}`;
    const setting = sources.map(s => SETTINGS[s]).join(' and ');
    return [['a', base], ['b', other]].map(([variant, items]) => {
      const expected = op.expected(items);
      const askOf = x => route ? rules.map(r => ({ match: [r.bucket.about.slice(0, 30), JSON.stringify(x.id)], value: r.source === x.source && membership(r.bucket, x.label) }))
        : [{ match: [bucket.ask.slice(0, 40), JSON.stringify(x.id)], value: membership(bucket, x.label) }];
      const record = curriculumCase({ family: 'dataset_workbench', shape, variant, pairGroup: `dataset:${shape}`,
        slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required', ...(op.iterate ? { iterate: 'required' } : {}),
        evidence: { world: items.map(x => `${x.id}: ${x.source} ${x.label}`), retrieved: [], background: [] },
        minimumSequence: ['read the messages from the inbox service', 'judge each message in its own nl call against the criterion',
          op.iterate ? 'scan with iterateOn and stop at the k-th match' : 'combine the judgments exactly in code'],
        reference: { root: [evalCall(`const all = inbox.messages();\n${op.preamble ? op.preamble + '\n' : ''}${op.code}`), returnCall(expected)],
          children: items.flatMap(askOf) },
        root: { name: 'inbox_desk', args: {}, returns: op.returns,
          instructions: `The inbox holds ${setting}. ${op.text} Read the messages with inbox.messages(); judge each one from what it says.` },
        files: { 'inbox_desk/inbox.ts': serviceSource(items, rules), 'inbox_desk/gist.nl': NEAR_MISS,
          'types.ts': 'export type Message = { id: string, kind: string, day: number, channel: "email" | "chat" | "app", amount: number, text: string };\n' },
        inputs: {}, expected });
      record.license = [...new Set(sources.map(s => SOURCES[s].license))].join('; ');
      record.gold_sources = sources.map(s => `${s}-labels`);
      record.dataset = sources.join('+');
      record.dataset_records = items.map(x => x.sourceId);
      return record;
    });
  }
  throw new Error(`no discriminating dataset workbench case for ${seed}:${index}`);
}
