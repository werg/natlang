import { crisp } from './builders.mjs';

export const crispExamples = [
  crisp({ id: 'structured-calculation', name: 'Structured calculation', category: 'Starter',
    level: 'Beginner', description: 'Return a checked record with a sum and maximum.',
    concepts: ['typed records', 'inputs'], root: 'math/summarize.ts',
    args: { a: 'number', b: 'number' }, returns: 'Summary',
    files: { 'math/types.ts': 'type Summary = { sum: number, larger: number };\n' },
    code: 'return { sum: a + b, larger: Math.max(a, b) };',
    inputs: { a: 3, b: 5 }, expected: { sum: 8, larger: 5 } }),
  crisp({ id: 'checked-greeting', name: 'Checked greeting', category: 'Starter',
    level: 'Beginner', description: 'Read a typed person record and compose a greeting.',
    concepts: ['custom types', 'text'], root: 'people/greet.ts',
    args: { person: 'Person' }, returns: 'string',
    files: { 'people/types.ts': 'type Person = { name: string };\n' },
    code: 'return "Hello, " + person.name + "!";',
    inputs: { person: { name: 'Ada' } }, expected: 'Hello, Ada!' }),
  crisp({ id: 'invoice-totals', name: 'Invoice totals', category: 'Data',
    level: 'Beginner', description: 'Compute subtotal, tax, and grand total from line items.',
    concepts: ['arrays', 'records', 'rounding'], root: 'invoice/total.ts',
    args: { lines: 'Line[]', tax_rate: 'number' }, returns: 'Invoice',
    files: { 'invoice/types.ts': 'type Line = { label: string, price: number, quantity: number };\ntype Invoice = { subtotal: number, tax: number, total: number };\n' },
    code: `const subtotal = lines.reduce((n, line) => n + line.price * line.quantity, 0);
const tax = Math.round(subtotal * tax_rate * 100) / 100;
return { subtotal, tax, total: Math.round((subtotal + tax) * 100) / 100 };`,
    inputs: { lines: [{ label: 'book', price: 12, quantity: 2 }, { label: 'pen', price: 3, quantity: 1 }], tax_rate: 0.1 },
    expected: { subtotal: 27, tax: 2.7, total: 29.7 } }),
  crisp({ id: 'tiered-discount', name: 'Tiered discount', category: 'Data',
    level: 'Beginner', description: 'Apply a discount tier and minimum charge.',
    concepts: ['branches', 'numeric bounds'], root: 'pricing/quote.ts',
    args: { amount: 'number', member: 'boolean' }, returns: 'Quote',
    files: { 'pricing/types.ts': 'type Quote = { discount: number, due: number };\n' },
    code: `const rate = member ? (amount >= 100 ? 0.2 : 0.1) : 0;
const discount = Math.round(amount * rate * 100) / 100;
return { discount, due: Math.max(0, Math.round((amount - discount) * 100) / 100) };`,
    inputs: { amount: 120, member: true }, expected: { discount: 24, due: 96 } }),
  crisp({ id: 'word-histogram', name: 'Word histogram', category: 'Data',
    level: 'Intermediate', description: 'Normalize words and count their frequency.',
    concepts: ['dictionary', 'tokenization'], root: 'text/histogram.ts',
    args: { text: 'string' }, returns: 'Record<string, number>',
    code: `const counts: Record<string, number> = {};
for (const word of text.toLowerCase().match(/[a-z]+/g) ?? []) counts[word] = (counts[word] ?? 0) + 1;
return counts;`,
    inputs: { text: 'Cats, cats and dogs!' }, expected: { cats: 2, and: 1, dogs: 1 } }),
  crisp({ id: 'stable-deduplication', name: 'Stable deduplication', category: 'Data',
    level: 'Beginner', description: 'Keep the first occurrence of each item.',
    concepts: ['arrays', 'sets'], root: 'lists/unique.ts',
    args: { items: 'string[]' }, returns: 'string[]',
    code: 'return [...new Set(items)];',
    inputs: { items: ['red', 'blue', 'red', 'green', 'blue'] }, expected: ['red', 'blue', 'green'] }),
  crisp({ id: 'shared-tags', name: 'Shared tags', category: 'Data',
    level: 'Beginner', description: 'Find sorted tags present in both sets.',
    concepts: ['sets', 'sorting'], root: 'lists/intersection.ts',
    args: { left: 'string[]', right: 'string[]' }, returns: 'string[]',
    code: 'const rightSet = new Set(right); return [...new Set(left.filter(tag => rightSet.has(tag)))].sort();',
    inputs: { left: ['blue', 'green', 'blue', 'amber'], right: ['green', 'blue'] }, expected: ['blue', 'green'] }),
  crisp({ id: 'csv-measurements', name: 'CSV measurements', category: 'Data',
    level: 'Intermediate', description: 'Parse a small numeric CSV and report its mean.',
    concepts: ['parsing', 'validation'], root: 'data/measurements.ts',
    args: { csv: 'string' }, returns: 'Measurement',
    files: { 'data/types.ts': 'type Measurement = { count: number, sum: number, mean: number };\n' },
    code: String.raw`const rows = csv.trim().split(/\r?\n/).slice(1);
const values = rows.map(row => Number(row.split(',')[1]));
if (values.some(n => !Number.isFinite(n))) throw new Error('invalid measurement');
const sum = values.reduce((a, b) => a + b, 0);
return { count: values.length, sum, mean: values.length ? sum / values.length : 0 };`,
    inputs: { csv: 'name,value\na,4\nb,8\nc,6' }, expected: { count: 3, sum: 18, mean: 6 } }),
  crisp({ id: 'rolling-average', name: 'Rolling average', category: 'Data',
    level: 'Intermediate', description: 'Calculate fixed-width moving averages.',
    concepts: ['windows', 'arrays'], root: 'series/rolling.ts',
    args: { values: 'number[]', width: 'number' }, returns: 'number[]',
    code: `const out = [];
for (let i = 0; i + width <= values.length; i++) {
  out.push(values.slice(i, i + width).reduce((a, b) => a + b, 0) / width);
}
return out;`,
    inputs: { values: [2, 4, 6, 8], width: 2 }, expected: [3, 5, 7] }),
  crisp({ id: 'balanced-brackets', name: 'Balanced brackets', category: 'Algorithms',
    level: 'Intermediate', description: 'Validate nested brackets using a stack.',
    concepts: ['stack', 'state'], root: 'algorithms/brackets.ts',
    args: { text: 'string' }, returns: 'boolean',
    code: `const opens = new Set(['(', '[', '{']);
const closing: Record<string, string> = { ')': '(', ']': '[', '}': '{' };
const stack: string[] = [];
for (const char of text) {
  if (opens.has(char)) stack.push(char);
  else if (closing[char] && stack.pop() !== closing[char]) return false;
}
return stack.length === 0;`,
    inputs: { text: '{[a + (b)]}' }, expected: true }),
  crisp({ id: 'shortest-route', name: 'Shortest route', category: 'Algorithms',
    level: 'Advanced', description: 'Use breadth-first search to find a shortest route.',
    concepts: ['graph', 'queue', 'search'], root: 'algorithms/route.ts',
    args: { graph: 'Record<string, string[]>', start: 'string', goal: 'string' }, returns: 'string[]',
    code: `const queue = [[start]];
const seen = new Set([start]);
for (const path of queue) {
  const node = path[path.length - 1];
  if (node === goal) return path;
  for (const next of graph[node] ?? []) if (!seen.has(next)) {
    seen.add(next); queue.push([...path, next]);
  }
}
return [];`,
    inputs: { graph: { A: ['B', 'C'], B: ['D'], C: ['D'], D: [] }, start: 'A', goal: 'D' },
    expected: ['A', 'B', 'D'] }),
  crisp({ id: 'dependency-order', name: 'Dependency order', category: 'Algorithms',
    level: 'Advanced', description: 'Topologically order tasks and detect cycles.',
    concepts: ['dependency graph', 'cycle detection'], root: 'algorithms/dependencies.ts',
    args: { tasks: 'Task[]' }, returns: 'string[]',
    files: { 'algorithms/types.ts': 'type Task = { id: string, needs: string[] };\n' },
    code: `const pending = new Map(tasks.map(task => [task.id, task.needs]));
const order: string[] = [];
while (pending.size) {
  const ready = [...pending].filter(([, needs]) => needs.every(id => order.includes(id))).map(([id]) => id).sort();
  if (!ready.length) throw new Error('dependency cycle or missing task');
  for (const id of ready) { order.push(id); pending.delete(id); }
}
return order;`,
    inputs: { tasks: [{ id: 'publish', needs: ['build', 'test'] }, { id: 'test', needs: ['build'] },
      { id: 'build', needs: [] }] }, expected: ['build', 'test', 'publish'] }),
  crisp({ id: 'merge-intervals', name: 'Merge intervals', category: 'Algorithms',
    level: 'Intermediate', description: 'Coalesce overlapping closed intervals.',
    concepts: ['sorting', 'intervals'], root: 'algorithms/intervals.ts',
    args: { ranges: 'Range[]' }, returns: 'Range[]',
    files: { 'algorithms/types.ts': 'type Range = { start: number, end: number };\n' },
    code: `const sorted = [...ranges].sort((a, b) => a.start - b.start);
const merged = [];
for (const range of sorted) {
  const last = merged[merged.length - 1];
  if (last && range.start <= last.end) last.end = Math.max(last.end, range.end);
  else merged.push({ ...range });
}
return merged;`,
    inputs: { ranges: [{ start: 5, end: 7 }, { start: 1, end: 3 }, { start: 2, end: 6 }] },
    expected: [{ start: 1, end: 7 }] }),
  crisp({ id: 'event-reconciliation', name: 'Event reconciliation', category: 'Workflows',
    level: 'Advanced', description: 'Match requests and acknowledgements by stable ID.',
    concepts: ['event streams', 'idempotency'], root: 'workflow/reconcile.ts',
    args: { sent: 'string[]', acknowledged: 'string[]' }, returns: 'Reconciliation',
    files: { 'workflow/types.ts': 'type Reconciliation = { confirmed: string[], pending: string[], unknown: string[] };\n' },
    code: `const sentSet = new Set(sent), acknowledgedSet = new Set(acknowledged);
return { confirmed: [...sentSet].filter(id => acknowledgedSet.has(id)).sort(),
  pending: [...sentSet].filter(id => !acknowledgedSet.has(id)).sort(),
  unknown: [...acknowledgedSet].filter(id => !sentSet.has(id)).sort() };`,
    inputs: { sent: ['a', 'b', 'c'], acknowledged: ['a', 'c', 'x'] },
    expected: { confirmed: ['a', 'c'], pending: ['b'], unknown: ['x'] } }),
  crisp({ id: 'ticket-reducer', name: 'Ticket state reducer', category: 'Workflows',
    level: 'Intermediate', description: 'Apply ordered events to a ticket state.',
    concepts: ['state machine', 'event order'], root: 'workflow/ticket.ts',
    args: { events: 'string[]' }, returns: 'TicketState',
    files: { 'workflow/types.ts': 'type TicketState = { status: string, escalated: boolean, transitions: number };\n' },
    code: `let status = 'new', escalated = false, transitions = 0;
for (const event of events) {
  if (event === 'open' && status === 'new') { status = 'open'; transitions++; }
  else if (event === 'escalate' && status === 'open') { escalated = true; transitions++; }
  else if (event === 'resolve' && status === 'open') { status = 'resolved'; transitions++; }
}
return { status, escalated, transitions };`,
    inputs: { events: ['open', 'escalate', 'resolve', 'escalate'] },
    expected: { status: 'resolved', escalated: true, transitions: 3 } }),
  crisp({ id: 'priority-queue', name: 'Priority queue', category: 'Workflows',
    level: 'Intermediate', description: 'Select pending work by priority and age.',
    concepts: ['ranking', 'tie breaks'], root: 'workflow/priority.ts',
    args: { tasks: 'QueueTask[]' }, returns: 'string[]',
    files: { 'workflow/types.ts': 'type QueueTask = { id: string, priority: number, age: number, blocked: boolean };\n' },
    code: `return tasks.filter(task => !task.blocked)
  .sort((a, b) => b.priority - a.priority || b.age - a.age || a.id.localeCompare(b.id))
  .map(task => task.id);`,
    inputs: { tasks: [{ id: 'a', priority: 2, age: 1, blocked: false },
      { id: 'b', priority: 3, age: 0, blocked: true },
      { id: 'c', priority: 2, age: 4, blocked: false }] }, expected: ['c', 'a'] }),
  crisp({ id: 'safe-rollout', name: 'Safe rollout gate', category: 'Workflows',
    level: 'Intermediate', description: 'Decide whether a deployment passes fixed health checks.',
    concepts: ['policy', 'decision record'], root: 'workflow/rollout.ts',
    args: { tests_passed: 'boolean', error_rate: 'number', latency_ms: 'number' }, returns: 'Gate',
    files: { 'workflow/types.ts': 'type Gate = { proceed: boolean, reasons: string[] };\n' },
    code: `const reasons = [];
if (!tests_passed) reasons.push('tests failed');
if (error_rate > 0.02) reasons.push('error rate high');
if (latency_ms > 500) reasons.push('latency high');
return { proceed: reasons.length === 0, reasons };`,
    inputs: { tests_passed: true, error_rate: 0.03, latency_ms: 420 },
    expected: { proceed: false, reasons: ['error rate high'] } }),
  crisp({ id: 'record-diff', name: 'Record change summary', category: 'Data',
    level: 'Intermediate', description: 'List added, removed, and changed record fields.',
    concepts: ['records', 'change detection'], root: 'data/diff.ts',
    args: { before: 'Record<string, string>', after: 'Record<string, string>' }, returns: 'Changes',
    files: { 'data/types.ts': 'type Changes = { added: string[], removed: string[], changed: string[] };\n' },
    code: `const left = before, right = after;
return { added: Object.keys(right).filter(key => !(key in left)).sort(),
  removed: Object.keys(left).filter(key => !(key in right)).sort(),
  changed: Object.keys(left).filter(key => key in right && left[key] !== right[key]).sort() };`,
    inputs: { before: { owner: 'Ada', state: 'open' }, after: { owner: 'Lin', priority: 'high' } },
    expected: { added: ['priority'], removed: ['state'], changed: ['owner'] } }),
  crisp({ id: 'markdown-outline', name: 'Markdown outline', category: 'Writing',
    level: 'Intermediate', description: 'Extract headings with their levels.',
    concepts: ['text parsing', 'records'], root: 'text/outline.ts',
    args: { markdown: 'string' }, returns: 'Heading[]',
    files: { 'text/types.ts': 'type Heading = { level: number, title: string };\n' },
    code: String.raw`return markdown.split(/\r?\n/).flatMap(line => {
  const match = /^(#{1,6})\s+(.+)$/.exec(line);
  return match ? [{ level: match[1].length, title: match[2].trim() }] : [];
});`,
    inputs: { markdown: '# Plan\nNotes\n## Tasks\n### Tests' },
    expected: [{ level: 1, title: 'Plan' }, { level: 2, title: 'Tasks' }, { level: 3, title: 'Tests' }] }),
  crisp({ id: 'redact-identifiers', name: 'Redact identifiers', category: 'Writing',
    level: 'Intermediate', description: 'Replace emails and long numeric IDs before display.',
    concepts: ['regex', 'privacy'], root: 'text/redact.ts',
    args: { text: 'string' }, returns: 'string',
    code: String.raw`return text.replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, '[email]')
  .replace(/\b\d{6,}\b/g, '[id]');`,
    inputs: { text: 'Contact ada@example.com about ticket 1234567.' },
    expected: 'Contact [email] about ticket [id].' }),
  crisp({ id: 'form-validation', name: 'Form validation', category: 'Workflows',
    level: 'Beginner', description: 'Return field errors for a signup form.',
    concepts: ['validation', 'structured errors'], root: 'forms/validate.ts',
    args: { name: 'string', email: 'string', age: 'number' }, returns: 'Record<string, string>',
    code: String.raw`const errors: Record<string, string> = {};
if (!name.trim()) errors.name = 'required';
if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) errors.email = 'invalid email';
if (age < 18) errors.age = 'must be 18 or older';
return errors;`,
    inputs: { name: 'Ada', email: 'bad-address', age: 17 },
    expected: { email: 'invalid email', age: 'must be 18 or older' } }),
];
