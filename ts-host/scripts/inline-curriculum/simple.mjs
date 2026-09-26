/**
 * Simple families: everything a case needs arrives as its arguments, so the opening shows it and nothing has to be
 * fetched. They exercise the core of a natlang call with the least around it: control flow in code, typed nl
 * judgments, one eval, a typed result. A ladder, from plain code to a saved nl function used in a loop.
 */
import { Random, curriculumCase, evalCall, returnCall } from './lib.mjs';

const REVIEWS = [
  { text: 'Works exactly as described; I would buy it again.', recommends: true },
  { text: 'Broke after two days and support never answered.', recommends: false },
  { text: 'Great battery life, and my kids love it. Recommended.', recommends: true },
  { text: 'Too small for anything but a phone, and it smells of plastic.', recommends: false },
  { text: 'Setup took five minutes and it has run flawlessly since.', recommends: true },
  { text: 'Returned it: the colour looks nothing like the photos.', recommends: false },
  { text: 'Solid, quiet, and cheaper than the brand name. Happy with it.', recommends: true },
  { text: 'The app keeps logging me out; not worth the money.', recommends: false },
];
const MESSAGES = [
  { text: 'This is the third time my order is late. I want a refund today.', tone: 'angry' },
  { text: 'Could you let me know when the new size is back in stock? Thanks!', tone: 'calm' },
  { text: 'Nobody reads these emails, do you? Fix my account NOW.', tone: 'angry' },
  { text: 'Just checking whether my parcel left the warehouse yet.', tone: 'calm' },
  { text: 'You charged me twice and your chat bot hung up on me. Unbelievable.', tone: 'angry' },
  { text: 'Loving the product so far, one small question about the warranty.', tone: 'calm' },
];
const TICKETS = [
  { text: 'Checkout fails for every customer since this morning.', urgent: true },
  { text: 'Last night\'s backup deleted a week of invoices.', urgent: true },
  { text: 'Anyone can open the admin page without logging in.', urgent: true },
  { text: 'The logo looks blurry on large screens.', urgent: false },
  { text: 'Please add a dark mode to the dashboard.', urgent: false },
  { text: 'The export puts dates in the wrong column; we fix it by hand for now.', urgent: false },
];
const RECEIPTS = [
  { text: 'CAFE LUMEN - 2x Espresso 5,60 - TOTAL 8,80 EUR', total: 8.8, currency: 'EUR' },
  { text: 'Harbor Books: paperback $14.99, tax $1.40, amount paid $16.39', total: 16.39, currency: 'USD' },
  { text: 'Northline Rail single ticket, fare paid £42.10', total: 42.1, currency: 'GBP' },
  { text: 'Taxi Porto, 18,50 EUR incl. tip', total: 18.5, currency: 'EUR' },
  { text: 'Deli on 5th: sandwich and coffee, total USD 12.25', total: 12.25, currency: 'USD' },
  { text: 'Museum entry for two, paid 24,00 EUR at the desk', total: 24, currency: 'EUR' },
];
const ASKS = [
  { text: 'The workshop costs 1,800 dollars, plus 1,200 for a translator.', total: 3000 },
  { text: 'We need 2,500 dollars in total.', total: 2500 },
  { text: 'Equipment for 4,000 dollars; no other costs.', total: 4000 },
  { text: 'Travel 900 dollars and a venue for 600.', total: 1500 },
  { text: 'A 5,500-dollar grant covers everything.', total: 5500 },
  { text: 'Printing 700 dollars, postage 300, design 1,000.', total: 2000 },
];

const simple = (family, fields) => curriculumCase({ family, slice: 'inline_placement', domain: 'other', mode: 'single_call',
  evidence: { world: [], retrieved: [], background: [] }, ...fields });

/** Level 0: plain code over an argument. */
export function simpleSum(seed, index) {
  const rng = new Random(seed, `simple-sum:${index}`);
  const invoices = Array.from({ length: rng.int(4, 7) }, (_, i) =>
    ({ id: `INV-${i + 1}`, status: rng.pick(['paid', 'open']), amount: rng.int(20, 900) }));
  const expected = invoices.filter(item => item.status === 'paid').reduce((sum, item) => sum + item.amount, 0);
  return [simple('simple_sum', { shape: `invoices${index}`, variant: 'total', inline: 'avoid',
    minimumSequence: ['add up the paid amounts in code', 'return the total'],
    reference: { root: [evalCall('return invoices.filter(item => item.status === "paid").reduce((sum, item) => sum + item.amount, 0);'),
      returnCall(expected)] },
    root: { name: 'paid_total', args: { invoices: 'Invoice[]' }, returns: 'number',
      instructions: 'Return the total amount of the invoices whose status is "paid".' },
    files: { 'types.ts': 'export type Invoice = { id: string, status: "paid" | "open", amount: number };\n' },
    inputs: { invoices }, expected })];
}

/** Level 1: one typed judgment about one argument. */
export function simpleTone(seed, index) {
  const rng = new Random(seed, `simple-tone:${index}`);
  const message = rng.pick(MESSAGES);
  return [simple('simple_tone', { shape: `message${index}`, variant: 'tone', inline: 'optional',
    minimumSequence: ['judge the message\'s tone', 'return it'],
    reference: { root: [returnCall(message.tone)] },
    root: { name: 'message_tone', args: { message: 'string' }, returns: '"angry" | "calm"',
      instructions: 'Is the customer who wrote message angry or calm?' },
    inputs: { message: message.text }, expected: message.tone })];
}

/** Level 2: a typed nl judgment for each item of a list, then exact code. */
export function simpleCountPositive(seed, index) {
  const rng = new Random(seed, `simple-count:${index}`);
  const reviews = rng.sample(REVIEWS, rng.int(4, 6)).map((review, i) => ({ id: `R${i + 1}`, ...review }));
  const expected = reviews.filter(review => review.recommends).length;
  return [simple('simple_count_positive', { shape: `reviews${index}`, variant: 'count', inline: 'required',
    minimumSequence: ['judge each review with an nl function', 'count the recommending ones in code'],
    reference: { root: [evalCall('const recommends = await Promise.all(reviews.map(review => nl<boolean>`Does the writer of review recommend the product?`(review)));\nreturn recommends.filter(Boolean).length;'),
      returnCall(expected)],
    children: reviews.map(review => ({ match: JSON.stringify(review.id), value: review.recommends })) },
    root: { name: 'recommending_reviews', args: { reviews: 'Review[]' }, returns: 'number',
      instructions: 'How many of reviews recommend the product? Judge each review separately with an nl function, and count in code.' },
    files: { 'types.ts': 'export type Review = { id: string, text: string };\n' },
    inputs: { reviews: reviews.map(({ id, text }) => ({ id, text })) }, expected })];
}

/** Level 3: branch in code on a typed judgment. */
export function simpleDeadline(seed, index) {
  const rng = new Random(seed, `simple-deadline:${index}`);
  const ticket = { id: `T${index + 1}`, ...rng.pick(TICKETS), opened_day: rng.int(1, 20) };
  const expected = ticket.opened_day + (ticket.urgent ? 1 : 5);
  return [simple('simple_deadline', { shape: `ticket${index}`, variant: 'deadline', inline: 'optional',
    minimumSequence: ['judge whether the ticket is urgent', 'compute the deadline in code'],
    reference: { root: [evalCall(`const urgent = await nl<boolean>\`Is ticket urgent: an outage, lost data, or a security hole?\`(ticket);\nreturn ticket.opened_day + (urgent ? 1 : 5);`),
      returnCall(expected)], children: [{ match: JSON.stringify(ticket.id), value: ticket.urgent }] },
    root: { name: 'ticket_deadline', args: { ticket: 'Ticket' }, returns: 'number',
      instructions: 'An urgent ticket (an outage, lost data, or a security hole) is due one day after it was opened; any other ticket five days after. Return the day ticket is due.' },
    files: { 'types.ts': 'export type Ticket = { id: string, text: string, opened_day: number };\n' },
    inputs: { ticket: { id: ticket.id, text: ticket.text, opened_day: ticket.opened_day } }, expected })];
}

/** Level 4: a typed record extracted from each item, then exact aggregation. */
export function simpleReceiptSum(seed, index) {
  const rng = new Random(seed, `simple-receipts:${index}`);
  const receipts = rng.sample(RECEIPTS, rng.int(3, 5)).map((receipt, i) => ({ id: `P${i + 1}`, ...receipt }));
  const expected = {};
  for (const r of receipts) expected[r.currency] = Math.round(((expected[r.currency] ?? 0) + r.total) * 100) / 100;
  return [simple('simple_receipt_sum', { shape: `receipts${index}`, variant: 'sum', inline: 'required',
    minimumSequence: ['read each receipt\'s amount and currency with a typed nl function', 'sum per currency in code'],
    reference: { root: [evalCall('const amounts = await Promise.all(receipts.map(receipt => nl<Amount>`Read the amount paid on receipt and its currency.`(receipt)));\nconst totals: Record<string, number> = {};\nfor (const amount of amounts) totals[amount.currency] = Math.round(((totals[amount.currency] ?? 0) + amount.total) * 100) / 100;\nreturn totals;'),
      returnCall(expected)],
    children: receipts.map(r => ({ match: JSON.stringify(r.id), value: { total: r.total, currency: r.currency } })) },
    root: { name: 'receipt_totals', args: { receipts: 'Receipt[]' }, returns: 'Record<string, number>',
      instructions: 'Total the amount paid on receipts per currency. Read each receipt with an nl function that returns its amount and currency, and add them up in code, rounded to cents.' },
    files: { 'types.ts': 'export type Receipt = { id: string, text: string };\nexport type Amount = { total: number, currency: "EUR" | "USD" | "GBP" };\n' },
    inputs: { receipts: receipts.map(({ id, text }) => ({ id, text })) }, expected })];
}

/** Level 5: one saved nl function with a signature, used in a loop. */
export function simpleFits(seed, index) {
  const rng = new Random(seed, `simple-fits:${index}`);
  const applications = rng.sample(ASKS, rng.int(4, 5)).map((ask, i) => ({ id: `A${i + 1}`, ...ask }));
  const budget = rng.pick([2000, 2500, 3000, 4000]);
  const expected = applications.filter(application => application.total <= budget).map(application => application.id);
  return [simple('simple_fits', { shape: `grants${index}`, variant: 'fits', inline: 'required',
    minimumSequence: ['create one nl function with a signature', 'call it for each application in a loop', 'return the ids that fit'],
    reference: { root: [evalCall('const fits: (application: Application) => Promise<boolean> = nl`Does everything application asks for, in total, fit within budget dollars?`;\nconst kept: string[] = [];\nfor (const application of applications) if (await fits(application)) kept.push(application.id);\nreturn kept;'),
      returnCall(expected)],
    children: applications.map(application => ({ match: JSON.stringify(application.id), value: application.total <= budget })) },
    root: { name: 'fundable', args: { applications: 'Application[]', budget: 'number' }, returns: 'string[]',
      instructions: 'Which applications ask for no more than budget dollars in total? Make one nl function with a signature, (application: Application) => Promise<boolean>, and call it for each application in a loop. Return the ids that fit, in order.' },
    files: { 'types.ts': 'export type Application = { id: string, ask: string };\n' },
    inputs: { applications: applications.map(({ id, text }) => ({ id, ask: text })), budget }, expected })];
}

export const SIMPLE_FAMILIES = { simple_sum: simpleSum, simple_tone: simpleTone, simple_count_positive: simpleCountPositive,
  simple_deadline: simpleDeadline, simple_receipt_sum: simpleReceiptSum, simple_fits: simpleFits };
