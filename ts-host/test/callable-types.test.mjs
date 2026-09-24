import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';

/** Run root.nl with scripted root tool calls; returns the opening listing and each tool result. */
async function script(files, calls) {
  const results = [];
  let listing = '', step = 0;
  const model = async request => {
    if (!listing) listing = JSON.parse(request.messages[2].tool_calls[0].function.arguments).code;
    else results.push(String(request.messages.at(-1).content));
    const call = calls[step++];
    return call ? { calls: [call] } : { calls: [['return_result', { status: 'failed', reason: 'The scripted test has no more calls.' }]] };
  };
  const runtime = createNatlangRuntime({ model, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: {}\nreturns: number\n---\nUse the helpers.\n', ...files }, 'root.nl');
  let value;
  try { value = await runtime.run(() => fn()); } catch {}
  return { listing, results, value };
}

test('a callable module\'s type aliases can be used in eval annotations', async () => {
  const { listing, results } = await script({
    'root/stock.ts': 'export type Level = { sku: string, units: number };\nexport function levels(): Level[] { return [{ sku: "a", units: 2 }]; }\n',
  }, [['eval', { code: 'const m: Level[] = stock.levels();\nm.length' }]]);
  assert.match(listing, /type Level = \{ sku: string, units: number \};/);
  assert.match(results[0], /^1\b/);
});

test('a module class is listed by its public members and its instances are live values', async () => {
  const { listing, results, value } = await script({
    'root/counter.ts': `export class Counter {
  private count = 0;
  /** Add one and return the new count. */
  bump(): number { return ++this.count; }
}
const shared = new Counter();
/** The shared counter. */
export function open(): Counter { return shared; }
`,
  }, [['eval', { code: 'const c = counter.open();\nc.bump();\nc.bump()' }], ['eval', { code: 'return counter.open().bump();' }],
    ['return_result', { status: 'success', value: 3 }]]);
  assert.match(listing, /declare class Counter \{\n  \/\*\* Add one and return the new count\. \*\/\n  bump\(\): number;\n\}/);
  assert.doesNotMatch(listing, /count = 0|Live</);
  assert.match(results[0], /^2\b/);
  assert.equal(value, 3);
});
