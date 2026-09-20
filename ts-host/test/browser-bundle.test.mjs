import test from 'node:test';
import assert from 'node:assert/strict';

test('browser bundle runs typed crisp and model programs without Node builtins', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try {
    globalThis.process = undefined;
    api = await import('../dist/browser/natlang.js');
  } finally { globalThis.process = nodeProcess; }
  const shared = { count: 2 };
  const host = new api.BrowserNatlangHost({ host: shared });
  try {
    const crisp = await host.run({ source: { kind: 'program', program: { $lambda: {
      type: 'Lambda<{}, Num>', code: 'host.count += 3; return host.count;' } } } });
    assert.equal(crisp.outcome.kind, 'done');
    assert.equal(crisp.value, 5);
    assert.equal(shared.count, 5);
    assert.equal(crisp.trace[0].engine_contracts['typescript-host'].authority, 'shared-browser-host');
    let turns = 0;
    const natural = await host.run({ source: { kind: 'program', program: { $lambda: {
      type: 'Lambda<{}, Num>', instructions: 'Return the current count.' } } },
    modelTurn: () => ++turns === 1 ? { calls: [['run_code', { engine: 'typescript-host', code: 'host.count' }]],
      completion_tokens: 1 } : turns === 2 ? { calls: [['write', { path: 'return', type: 'Num', value: 5 }]],
      completion_tokens: 1 } : { calls: [], text: 'done', completion_tokens: 1 } });
    assert.equal(natural.outcome.kind, 'done');
    assert.equal(natural.value, 5);
    assert(natural.trace.some(event => event.kind === 'action'));
  } finally { host.close(); }
});

test('browser retained eval preserves variables between tool calls', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try { globalThis.process = undefined; api = await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = nodeProcess; }
  const environment = new api.TypeScriptEnvironment({ mode: 'retained' });
  try {
    assert.equal(environment.execute({ code: 'var tally = 2; tally', scope: {}, body: false,
      path: 'eval', effectful: false }).result, 2);
    assert.equal(environment.execute({ code: 'tally += 3', scope: {}, body: false,
      path: 'eval', effectful: false }).result, 5);
  } finally { environment.close(); }
});
