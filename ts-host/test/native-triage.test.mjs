import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NativeRuntime } from '../dist/index.js';

test('triage algorithm runs as TypeScript with lexical function imports', async () => {
  const tickets = [
    'I was charged twice for my March invoice.',
    'CHEAP WATCHES!!! Visit our store today for 90% off.',
    'PRODUCTION IS DOWN. Customers cannot log in at all.',
    'Export to CSV produces an empty file, no rush.',
    'Payments are failing for every customer right now.',
  ];
  const source = { $lambda: {
    type: '(tickets: string[], rubric: string) => Report',
    types: { Report: '{ urgent: number, by_label: Record<string, number>, summary: string }' },
    instructions: 'Classify the tickets, select urgent real issues, and report the totals.',
    args: { tickets, rubric: 'billing: charges and payments; technical: outages; spam: advertising' },
    codebase: {
      classify: { args: { ticket: 'string', rubric: 'string' }, returns: 'string',
        code: 'return /WATCHES/.test(ticket) ? "spam" : /charged|Payments/.test(ticket) ? "billing" : "technical";' },
      is_urgent: { args: { ticket: 'string' }, returns: 'boolean',
        code: 'return /DOWN|Payments are failing/.test(ticket);' },
      summarize: { args: { tickets: 'string[]' }, returns: 'string',
        code: 'return tickets.join(" ").split(/\\s+/).slice(0, 60).join(" ");' },
    },
  } };
  const runtime = new NativeRuntime({ agent: async session => {
    const execution = await session.applyAsync('eval', { code:
      'const labels = await Promise.all(tickets.map(ticket => classify(ticket, rubric)));\n' +
      'const real = tickets.filter((ticket, index) => labels[index] !== "spam");\n' +
      'const urgentFlags = await Promise.all(real.map(ticket => is_urgent(ticket)));\n' +
      'const urgentTickets = real.filter((ticket, index) => urgentFlags[index]);\n' +
      'const by_label = Object.fromEntries([...new Set(labels)].map(label => [label, labels.filter(item => item === label).length]));\n' +
      'const summary = urgentTickets.length ? await summarize(urgentTickets) : "Nothing urgent today.";\n' +
      '({ urgent: urgentTickets.length, by_label, summary })' });
    assert.equal(execution.kind, 'ok', execution.text);
    const marked = session.apply('mark_lines', { start: 1, end: 1 });
    assert.equal(marked.kind, 'ok', marked.text);
  } });
  const result = await runtime.runRoot(source);
  assert.equal(result.outcome.kind, 'done', result.outcome.detail);
  assert.equal(result.value.urgent, 2);
  assert.deepEqual(result.value.by_label, { billing: 2, spam: 1, technical: 2 });
  assert.ok(result.value.summary.includes('cannot log in') && result.value.summary.includes('Payments are failing'));
  assert.ok(result.value.summary.split(/\s+/).length <= 60);
});
