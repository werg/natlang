import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { NativeRuntime } from '../dist/native/runtime.js';
import { loadFunctionFile } from '../dist/native/source.js';
import { coerce, dump } from '../dist/native/values.js';
import { TypeEnv } from '../dist/native/types.js';

test('native host executes checked triage source with nested Map, Iterate and crisp library calls', async () => {
  const root = loadFunctionFile(fileURLToPath(new URL('../../examples/triage/main.nl', import.meta.url)));
  const tickets = [
    'I was charged twice for my March invoice.',
    'CHEAP WATCHES!!! Visit our store today for 90% off.',
    'PRODUCTION IS DOWN. Customers cannot log in at all.',
    'Export to CSV produces an empty file, no rush.',
    'Payments are failing for every customer right now.',
  ];
  root.args.tickets = coerce(tickets, root.type.params.fields.find(field => field.name === 'tickets').type,
    new TypeEnv().child(root.types), 'args/tickets');
  root.args.rubric = 'billing: charges and payments; technical: outages; spam: advertising';
  const callLog = [];
  const runtime = new NativeRuntime({ agent: async session => {
    const name = session.lam.functionName;
    const call = async args => {
      const result = await session.applyAsync('call', args);
      assert.ok(['done', 'ok'].includes(result.kind), `${name}: ${result.text}`);
      callLog.push({ owner: name, ...args });
    };
    const write = (path, type, value) => {
      const result = session.apply('write', { path, type, value });
      assert.equal(result.kind, 'ok', result.text);
    };
    if (name === 'classify') {
      const ticket = session.lam.args.ticket;
      write('return', 'Label', ticket.includes('WATCHES') ? 'spam' :
        /charged|Payments/.test(ticket) ? 'billing' : 'technical');
    } else if (name === 'is_urgent') {
      write('return', 'Bool', /DOWN|Payments are failing/.test(session.lam.args.ticket));
    } else if (name === 'shorten') {
      write('return', 'Text', session.lam.args.text.split(/\s+/).slice(0, 35).join(' '));
    } else if (name === 'summarize') {
      write('let/draft', 'Text', ('Customers cannot log in because production is down. Payments are failing for every customer. ' +
        'The March invoice had a duplicate charge. ').repeat(12));
      await call({ function: 'shorten', to: 'return', init: 'let/draft', until: 'is_short', max: 3 });
    } else if (name === 'main') {
      await call({ function: 'classify', to: 'let/labels', over: 'args/tickets', inputs: { rubric: 'args/rubric' } });
      const selected = session.apply('run_code', { code: "locals.labels.map(label => label !== 'spam')", engine: 'typescript-host' });
      assert.equal(selected.kind, 'ok', selected.text);
      write('let/not_spam', 'Bool[]', selected.value);
      await call({ function: 'select_by_flags', to: 'let/real', inputs: { items: 'args/tickets', flags: 'let/not_spam' } });
      await call({ function: 'is_urgent', to: 'let/flags', over: 'let/real' });
      await call({ function: 'select_by_flags', to: 'let/urgent', inputs: { items: 'let/real', flags: 'let/flags' } });
      await call({ function: 'summarize', to: 'return/summary', inputs: { tickets: 'let/urgent' } });
      await call({ function: 'count_true', to: 'return/urgent', inputs: { flags: 'let/flags' } });
      await call({ function: 'group_count', to: 'return/by_label', inputs: { values: 'let/labels' } });
    } else throw new Error(`unhandled function ${name}`);
    assert.equal(session.finish(), true, name);
  } });
  const result = await runtime.runRoot(root);
  assert.equal(result.outcome.kind, 'done', result.outcome.detail);
  const report = dump(result.value);
  assert.equal(report.urgent, 2);
  assert.deepEqual(report.by_label, { billing: 2, spam: 1, technical: 2 });
  assert.ok(report.summary.includes('cannot log in') && report.summary.includes('Payments are failing'));
  assert.ok(report.summary.split(/\s+/).length <= 60);
  assert.equal(callLog.filter(call => call.owner === 'main' && call.function === 'classify').length, 1);
  assert.equal(callLog.filter(call => call.owner === 'main' && call.function === 'is_urgent').length, 1);
  assert.equal(callLog.filter(call => call.owner === 'main' && call.function === 'summarize').length, 1);
  assert.equal(runtime.episodesStarted, 12);
});
