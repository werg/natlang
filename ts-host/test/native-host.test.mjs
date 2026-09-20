import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { NativeNatlangHost, TypeScriptEnvironment } from '../dist/index.js';

test('public native host runs a program without starting Python', async () => {
  const host = new NativeNatlangHost();
  const result = await host.run({ source: { kind: 'program', program: { $lambda: {
    type: 'Lambda<{ item: Num }, Num>', code: 'return args.item * 4;', engine: 'typescript-host' } } },
    inputs: { item: 3 } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 12);
  host.close();
});

test('native file loader links companion functions and lexical types', async () => {
  const file = fileURLToPath(new URL('../../examples/triage/main.nl', import.meta.url));
  const host = new NativeNatlangHost();
  let looked = false;
  const result = await host.run({ source: { kind: 'file', path: file },
    inputs: { tickets: [], rubric: 'none' }, modelTurn: request => {
      looked = true;
      assert.ok(request.tools.some(tool => tool.function.name === 'call'));
      return { calls: [['report_blocker', { missing: 'No tickets were supplied for triage.' }]], completion_tokens: 8 };
    }, options: { model: { max_turns: 2 } } });
  assert.equal(looked, true);
  assert.equal(result.outcome.kind, 'quiesced');
  host.close();
});

test('retained native eval sees the same application object across runs', async () => {
  const state = { count: 0 };
  const environment = new TypeScriptEnvironment({ mode: 'retained', host: state });
  const host = new NativeNatlangHost({ environment });
  const source = { kind: 'program', program: { $lambda: { type: 'Lambda<{}, Num>',
    code: 'host.count++; return host.count;', engine: 'typescript-host' } } };
  assert.equal((await host.run({ source })).value, 1);
  assert.equal((await host.run({ source })).value, 2);
  assert.equal(state.count, 2);
  host.close(); environment.close();
});
