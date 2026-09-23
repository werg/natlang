// Failure-and-repair families: a rejected loop rewritten within the eval rules, a child that honestly cannot
// answer, and an application event that must not be applied twice when only its view failed.
import { blockedCall, curriculumCase, evalCall, failedCall, literal, nonceWords, Random, returnCall } from './lib.mjs';

/**
 * A seeded eval pages through a ledger with a `while` loop, which eval rejects. With a page count the repair is a
 * counted loop; without one it is `iterateOn` until an empty page.
 */
export function loopRewrite(seed, index) {
  const rng = new Random(seed, `loop:${index}`);
  const categories = ['travel', 'meals', 'software', 'office'];
  const pages = rng.int(4, 6);
  const entries = Array.from({ length: pages * 5 }, (_, i) => ({ id: `E${i + 1}`, category: rng.pick(categories), amount: rng.int(5, 400) }));
  const pageOf = n => entries.slice((n - 1) * 5, n * 5);
  const target = rng.pick(categories);
  const expected = entries.filter(e => e.category === target).reduce((sum, e) => sum + e.amount, 0);
  const seedCode = `let total = 0;
let n = 1;
while (true) {
  const rows = ledger.page(n);
  if (rows.length === 0) break;
  for (const row of rows) if (row.category === ${JSON.stringify(target)}) total += row.amount;
  n++;
}
total`;
  const module = counted => `const ENTRIES = ${literal(entries)};
/** Page n of the ledger (1-based, five entries per page); an empty list after the last page. */
export function page(n: number): { id: string, category: string, amount: number }[] { return ENTRIES.slice((n - 1) * 5, n * 5); }
${counted ? `/** The number of ledger pages. */\nexport function page_count(): number { return ${pages}; }\n` : ''}`;
  const shape = `ledger${index}`;
  const common = { slice: 'folder_failure', domain: 'other', mode: 'followup', inline: 'avoid',
    evidence: { world: [`${pages} pages`, `${target} total ${expected}`], retrieved: ['the loop rejection'], background: [] },
    decisive: [{ marker: 'loops are not allowed here', source: 'error', note: 'the eval rejects the while loop' }],
    plausibleActions: ['retry the same loop', 'rewrite it as a counted loop', 'rewrite it with iterateOn'],
    root: { name: 'category_total', args: {}, returns: 'number',
      instructions: `Add up the amounts of every ledger entry in category "${target}", across all pages of ledger.page(n).` },
    inputs: {}, expected, failureSeed: { kind: 'compile', code: seedCode } };
  return [
    curriculumCase({ ...common, family: 'loop_rewrite', shape, variant: 'counted',
      minimumSequence: ['read the rejection', 'loop over the known page count'],
      reference: { root: [evalCall(`let total = 0;
const pages = ledger.page_count();
for (let n = 1; n <= pages; n++) for (const row of ledger.page(n)) if (row.category === ${JSON.stringify(target)}) total += row.amount;
return total;`), returnCall(expected)] },
      files: { 'category_total/ledger.ts': module(true) } }),
    curriculumCase({ ...common, family: 'loop_rewrite', shape, variant: 'open_ended', iterate: 'required',
      minimumSequence: ['read the rejection', 'page with iterateOn until an empty page'],
      reference: { root: [evalCall(`type Scan = { n: number, total: number, done: boolean };
const scan = (state: Scan): Scan => {
  const rows = ledger.page(state.n);
  const add = rows.filter(row => row.category === ${JSON.stringify(target)}).reduce((sum, row) => sum + row.amount, 0);
  return { n: state.n + 1, total: state.total + add, done: rows.length === 0 };
};
const final = await iterateOn(scan, { n: 1, total: 0, done: false }).until(state => state.done);
return final.total;`), returnCall(expected)] },
      files: { 'category_total/ledger.ts': module(false) } }),
  ];
  void pageOf;
}

const INVOICES = [
  { text: 'Invoice 2211 from Fernwood Supplies. 40 reams of paper, 12 toner cartridges. Total due: 1,284.50 EUR by April 30.', total: 1284.5 },
  { text: 'Invoice 87 - Kestrel Cleaning. March office cleaning, 8 visits. Amount due 640.00 EUR.', total: 640 },
  { text: 'Invoice A-19 from Brightline IT. Laptop repair and two replacement batteries. Please pay 312.75 EUR within 14 days.', total: 312.75 },
  { text: 'Invoice 5530, Oakridge Catering. Lunch for the board meeting on March 12, 14 guests. Total 455.00 EUR.', total: 455 },
];
const UNPRICED = [
  'Invoice 9002 from Larkspur Print. 500 brochures, 200 posters. The amount will follow in a separate statement once the paper surcharge is known.',
  'Invoice 31 - Tidewater Movers. Office move on March 3, two vans, four movers. Pricing: see the attached quote (not included here).',
];

/**
 * An honest opt-out: one invoice states no total, so its child cannot produce a number and says so; the
 * parent records null for it instead of computing a figure from line items.
 */
export function childOptOut(seed, index) {
  const rng = new Random(seed, `optout:${index}`);
  const priced = rng.sample(INVOICES, 3);
  const shape = `invoices${index}`;
  const variants = {
    one_unpriced: [...priced.slice(0, 2), { text: rng.pick(UNPRICED), total: null }],
    all_priced: priced,
  };
  return Object.entries(variants).map(([variant, list]) => {
    const invoices = rng.shuffle(list).map((invoice, i) => ({ id: `I${i + 1}`, ...invoice }));
    const expected = Object.fromEntries(invoices.map(invoice => [invoice.id, invoice.total]));
    const plain = invoices.map(({ id, text }) => ({ id, text }));
    return curriculumCase({ family: 'child_opt_out', shape, variant, pairGroup: `optout:${shape}`,
      slice: 'nested_scoped', domain: 'other', mode: 'single_call', inline: 'required',
      evidence: { world: invoices.map(i => `${i.id}: ${i.total}`), retrieved: [], background: [] },
      minimumSequence: ['read each invoice in its own child', 'record null where the child reports no stated total'],
      reference: { root: [evalCall(`const totals: Record<string, number | null> = {};
for (const invoice of payables.invoices()) {
  try {
    totals[invoice.id] = await nl<number>\`Read the total amount due that invoice states, in euros.\`(invoice);
  } catch (error) {
    totals[invoice.id] = null;
  }
}
return totals;`), returnCall(expected)],
        children: invoices.map(i => i.total === null ?
          { match: JSON.stringify(i.id), call: blockedCall('The invoice does not state its total amount.') } :
          { match: JSON.stringify(i.id), value: i.total }) },
      root: { name: 'invoice_totals', args: {}, returns: 'Record<string, number | null>',
        instructions: 'Record the total amount due that each invoice from payables.invoices() states, in euros. Read each invoice in its own judgment. When an invoice does not state its total, record null for it; never compute a figure it does not state. Return a record from invoice id to amount.' },
      files: { 'invoice_totals/payables.ts': `const INVOICES = ${literal(plain)};\n/** This month's invoices, as text. */\nexport function invoices(): { id: string, text: string }[] { return INVOICES; }\n` },
      inputs: {}, expected });
  });
}

/**
 * Application event sequencing: an event is committed, then its view is rendered. A transient view failure is
 * retried without applying the event again (its revision proves it); a rejected event is reported as failed.
 */
export function eventRetry(seed, index) {
  const rng = new Random(seed, `event:${index}`);
  const [card] = nonceWords(rng, 1);
  const start = rng.int(10, 40);
  const shape = `board${index}`;
  const column = rng.pick(['review', 'done', 'blocked']);
  const variants = {
    view_timeout: { reject: false, expected: { revision: start + 1, column }, operation: undefined },
    rejected: { reject: true, expected: null, operation: 'blocked' },
  };
  return Object.entries(variants).map(([variant, spec]) => {
    const module = `type Card = { id: string, column: string };
const cards: Card[] = [{ id: ${JSON.stringify(card)}, column: "doing" }];
let revision = ${start};
let renders = 0;
const REJECT = ${spec.reject};
/** Apply a move event: move card to column. Commits it and returns the new board revision. */
export function commit_move(event: { card: string, to: string }): { revision: number } {
  if (REJECT) throw new Error("event rejected: card " + event.card + " is locked by an open audit");
  const found = cards.find(item => item.id === event.card);
  if (!found) throw new Error("no such card");
  found.column = event.to;
  revision += 1;
  return { revision };
}
/** Render the board view: its revision and each card's column. */
export function render(): { revision: number, cards: Card[] } {
  renders += 1;
  if (renders === 1) throw new Error("view renderer timed out; the board state is unaffected");
  return { revision, cards: cards.map(item => ({ ...item })) };
}
`;
    const applyCode = `const applied = board.commit_move({ card: ${JSON.stringify(card)}, to: ${JSON.stringify(column)} });
let view;
try { view = board.render(); } catch (error) { view = String(error); }
({ applied, view })`;
    const reference = spec.reject ?
      [evalCall(applyCode), failedCall(`The move was rejected: card ${card} is locked by an open audit.`)] :
      [evalCall(applyCode), evalCall(`const view2 = board.render();
return { revision: view2.revision, column: view2.cards.find(item => item.id === ${JSON.stringify(card)})!.column };`), returnCall(spec.expected)];
    return curriculumCase({ family: 'event_retry', shape, variant, pairGroup: `event:${shape}`,
      slice: 'folder_failure', domain: 'actor', mode: 'followup', inline: 'avoid',
      evidence: { world: [spec.reject ? 'the event is rejected' : 'the first render times out'], retrieved: [], background: [] },
      decisive: [{ marker: spec.reject ? 'locked by an open audit' : 'view renderer timed out', source: 'error', note: spec.reject ? 'the event is rejected' : 'only the view failed' }],
      plausibleActions: ['apply the event again', 'render again', 'report failure'],
      minimumSequence: spec.reject ? ['apply the event', 'see the rejection', 'report it'] : ['apply the event once', 'see the view time out', 'render again without re-applying'],
      reference: { root: reference },
      root: { name: 'move_card', args: {}, returns: '{ revision: number, column: string }',
        instructions: `Move card ${card} to column "${column}" with board.commit_move, then render the board with board.render() and return the rendered revision and the card's column. The move must be applied exactly once.` },
      files: { 'move_card/board.ts': module },
      inputs: {}, expected: spec.expected, ...(spec.operation ? { operation: spec.operation } : {}) });
  });
}
