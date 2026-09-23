// Small natlang calls that exercise what applications ask of a model: judgment, extraction,
// exact computation, child calls, and honest failure. `check` returns true or a reason.
const fn = (args, returns, body) =>
  `---\nargs:\n${Object.entries(args).map(([name, type]) => `  ${name}: ${JSON.stringify(type)}`).join('\n')}\nreturns: ${JSON.stringify(returns)}\n---\n${body.trim()}\n`;

const RUBRIC = 'billing: charges, invoices and payments; technical: outages, errors and bugs; spam: advertising';
const classify = fn({ ticket: 'string', rubric: 'string' }, 'Label', `
Pick the label from rubric that fits ticket. If no rule in the rubric covers
the ticket, report an error saying what is missing.`);
const labelTypes = 'export type Label = "billing" | "technical" | "spam";\n';

export const CASES = [
  { id: 'classify', root: 'classify.nl', files: { 'classify.nl': classify, 'types.ts': labelTypes },
    args: ['Production database is down, all customers get 500 errors.', RUBRIC],
    check: ({ value }) => value === 'technical' || `expected "technical", got ${JSON.stringify(value)}` },
  { id: 'classify-uncovered', root: 'classify.nl', files: { 'classify.nl': classify, 'types.ts': labelTypes },
    args: ['Please delete my account and all my data.', 'billing: charges and invoices; spam: advertising'],
    check: ({ error }) => Boolean(error) || 'expected an error: no rule covers account deletion' },
  { id: 'is-urgent', root: 'is_urgent.nl', files: { 'is_urgent.nl': fn({ ticket: 'string' }, 'boolean', `
A ticket is urgent when many customers are affected right now, money is being
lost, or security is at risk. Requests, suggestions and cosmetic problems are
not urgent.`) },
    args: ['Checkout has been failing for every customer since 9am; we are losing orders.'],
    check: ({ value }) => value === true || `expected true, got ${JSON.stringify(value)}` },
  { id: 'extract-contact', root: 'contact.nl', files: {
    'contact.nl': fn({ signature: 'string' }, 'Contact', `
Extract the sender's name and email address from signature. Use null for an
email address that does not appear.`),
    'types.ts': 'export type Contact = { name: string, email: string | null };\n' },
    args: ['Thanks again,\n-- \nDr. Maria Okafor | Head of Research\nmaria.okafor@lumen-labs.org | +44 20 7946 0958'],
    check: ({ value }) => (value?.email === 'maria.okafor@lumen-labs.org' && /Maria Okafor/.test(value?.name ?? '')) ||
      `got ${JSON.stringify(value)}` },
  { id: 'invoice-total', root: 'total.nl', files: {
    'total.nl': fn({ items: 'Item[]' }, 'number', `
Add up price times quantity for items. If that subtotal is over 100, take 10%
off. Round to cents.`),
    'types.ts': 'export type Item = { name: string, price: number, quantity: number };\n' },
    args: [[{ name: 'pen', price: 2.5, quantity: 12 }, { name: 'lamp', price: 45, quantity: 2 }, { name: 'mug', price: 7.99, quantity: 1 }]],
    check: ({ value }) => value === 115.19 || `expected 115.19, got ${JSON.stringify(value)}` },
  { id: 'child-calls', root: 'tally.nl', files: {
    'tally.nl': fn({ reviews: 'string[]' }, 'Tally', `
Judge each review with sentiment and count how many are positive and how many
are negative.`),
    'tally/sentiment.nl': fn({ review: 'string' }, '"positive" | "negative"', `
Decide whether review is positive or negative overall.`),
    'types.ts': 'export type Tally = { positive: number, negative: number };\n' },
    args: [['Arrived fast and works perfectly.', 'Broke after two days. Avoid.', 'Honestly the best purchase this year.']],
    check: ({ value }) => (value?.positive === 2 && value?.negative === 1) || `got ${JSON.stringify(value)}` },
  { id: 'missing-rate', root: 'convert.nl', files: { 'convert.nl': fn({ amount_usd: 'number', currency: 'string' }, 'number', `
Convert amount_usd into currency using the exchange rate given in the order
notes.`) },
    args: [120, 'EUR'],
    check: ({ error }) => Boolean(error) || 'expected a blocker: no exchange rate was given' },
  { id: 'pick-slot', root: 'slot.nl', files: { 'slot.nl': fn({ slots: 'string[]', constraints: 'string' }, 'string', `
Pick the one slot from slots that satisfies every constraint.`) },
    args: [['Mon 08:00', 'Mon 14:00', 'Tue 10:00', 'Wed 16:30'],
      'Not before 9am. Not on Tuesday. Must end by 4pm for a one-hour meeting.'],
    check: ({ value }) => value === 'Mon 14:00' || `expected "Mon 14:00", got ${JSON.stringify(value)}` },
  // Keyword matching gets these wrong; they need the model's own judgment.
  { id: 'classify-judgment', root: 'classify.nl', files: { 'classify.nl': classify, 'types.ts': labelTypes },
    args: ['Every time I open the invoices page it spins forever and then shows a blank screen.', RUBRIC],
    check: ({ value }) => value === 'technical' || `expected "technical", got ${JSON.stringify(value)}` },
  { id: 'urgent-judgment', root: 'is_urgent.nl', files: { 'is_urgent.nl': fn({ ticket: 'string' }, 'boolean', `
A ticket is urgent when many customers are affected right now, money is being
lost, or security is at risk. Requests, suggestions and cosmetic problems are
not urgent.`) },
    args: ['The pricing page shows the Pro plan at $1/year instead of $100/year, and signups have tripled this morning.'],
    check: ({ value }) => value === true || `expected true, got ${JSON.stringify(value)}` },
  { id: 'sarcasm', root: 'tally.nl', files: {
    'tally.nl': fn({ reviews: 'string[]' }, 'Tally', `
Judge each review with sentiment and count how many are positive and how many
are negative.`),
    'tally/sentiment.nl': fn({ review: 'string' }, '"positive" | "negative"', `
Decide whether review is positive or negative overall.`),
    'types.ts': 'export type Tally = { positive: number, negative: number };\n' },
    args: [['Great, it stopped working after one day. Perfect.', 'Not bad at all, I would buy it again.']],
    check: ({ value }) => (value?.positive === 1 && value?.negative === 1) || `got ${JSON.stringify(value)}` },
  // A directory reducer applied to a folder: the result and the committed files are both checked.
  { id: 'reducer-archive', root: 'archive.nl', files: { 'archive.nl': `---
kind: directory-reducer
args: {}
returns: "string[]"
---
Move every note in todo/ whose checklist is fully checked into done/, keeping
its file name. Return the file names of the notes you moved.
` },
    folder: { 'README.md': 'Personal notes.\n', 'todo/groceries.md': '# Groceries\n- [x] milk\n- [x] eggs\n',
      'todo/taxes.md': '# Taxes\n- [x] gather receipts\n- [ ] file return\n', 'todo/garden.md': '# Garden\n- [x] seeds\n- [x] water\n- [x] weed\n' },
    args: [],
    check: async ({ value, folder }) => {
      const names = (value ?? []).map(name => String(name).split('/').pop()).sort();
      if (JSON.stringify(names) !== '["garden.md","groceries.md"]') return `got ${JSON.stringify(value)}`;
      const exists = async path => folder.file(path).exists();
      for (const [path, expected] of [['done/garden.md', true], ['done/groceries.md', true], ['todo/garden.md', false],
        ['todo/groceries.md', false], ['todo/taxes.md', true]])
        if (await exists(path) !== expected) return `${path} should ${expected ? '' : 'not '}exist`;
      return true;
    } },
  // Natural code reaches for try/catch here (JSON.parse on text that may be malformed).
  { id: 'config-port', root: 'port.nl', files: { 'port.nl': fn({ configs: 'string[]' }, 'number[]', `
Each config is JSON text with a port field. Return each config's port, using
8080 for a config that is not valid JSON or has no numeric port.`) },
    args: [['{"port": 9000}', '{"port": 3000,}', '{"host": "a"}', 'port=22', '{"port": 443}']],
    check: ({ value }) => JSON.stringify(value) === '[9000,8080,8080,8080,443]' || `got ${JSON.stringify(value)}` },
];
