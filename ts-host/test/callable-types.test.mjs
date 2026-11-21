import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';

/** Run root.nl with scripted root tool calls; returns the opening listing and each tool result. */
async function script(files, calls, options = {}) {
  const results = [];
  let listing = '', opening = '', step = 0;
  const model = async request => {
    if (!listing) {
      opening = request.messages[1].content;
      listing = JSON.parse(request.messages[2].tool_calls[0].function.arguments).code;
    }
    else results.push(String(request.messages.at(-1).content));
    const call = calls[step++];
    return call ? { calls: [call] } : { calls: [['return_result', { status: 'failed', reason: 'The scripted test has no more calls.' }]] };
  };
  const runtime = createNatlangRuntime({ ...options, model, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: {}\nreturns: number\n---\nUse the helpers.\n', ...files }, 'root.nl');
  let value;
  try { value = await runtime.run(() => fn()); } catch {}
  return { listing, opening, results, value };
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

test('a function of the folder named transcript takes the name from the eval built-in', async () => {
  const { results, value } = await script({ 'root/transcript.ts': 'export function answer(): number { return 42; }\n' },
    [['eval', { code: 'const n = transcript.answer();\nn' }], ['return_result', { status: 'success', value: 42 }]]);
  assert.match(results[0], /42/);
  assert.doesNotMatch(results[0], /duplicated/);
  assert.equal(value, 42);
});

test('relative type imports and renamed re-exports retain their own transitive aliases at runtime', async () => {
  const files = {
    'root.nl': '---\nargs: {}\nreturns: PublicReport\n---\nReport one record.\n',
    'types.ts': `
      export type Entry = string;
      export type { Report as PublicReport } from './contracts/index.js';
      import type { Entry as OtherEntry } from './records.js';
      export type Inputs = OtherEntry[];
    `,
    'contracts/index.ts': `
      import type { Entry } from '../records.js';
      type State = "ready" | "held";
      export type Report = { entry: Entry; state: State; label: "Entry" };
    `,
    'records.ts': 'type Count = number; export type Entry = { total: Count };',
  };
  const value = { entry: { total: 7 }, state: 'ready', label: 'Entry' };
  let turns = 0;
  const runtime = createNatlangRuntime({
    model: async () => {
      turns++;
      return { calls: [['return_result', { status: 'success', value }]] };
    },
    seed: { mode: 'backend' },
  });
  const fn = loadVirtualNatlang(files, 'root.nl');
  assert.deepEqual(await runtime.run(() => fn()), value);
  assert.equal(turns, 1);
});

test('wide callable scopes show entry points and preserve scoped reads and deep calls', async () => {
  const files = {
    'root/primary/api.ts': 'export type Row = { units: number }; export function read(): Row { return { units: 7 }; }',
    'root/secondary/api.ts': 'export type Row = { label: string }; export function read(): Row { return { label: "other" }; }',
    'root/primary/deep/counter.ts': 'export function count(): number { return 41 }',
    'root/primary/readNote.nl': '---\nargs: {}\nreturns: string\n---\nRead the fixture note.\n',
  };
  for (let index = 0; index < 120; index++) {
    files[`root/primary/deep/leaf${index}.ts`] = 'export function value(): number { return 1 }';
  }
  const { listing, opening, results, value } = await script(files, [
    ['read_code', { name: 'primary.deep' }],
    ['read_code', { name: 'secondary.api.Row' }],
    ['read_code', { name: 'primary.api.Row' }],
    ['eval', { code: 'return primary.deep.counter.count();' }],
    ['return_result', { status: 'success', value: 41 }],
  ], {
    services: { noteRead: { read: () => 'fixture' } },
    serviceScopes: { noteRead: ['root/primary/readNote.nl'], unavailableRead: ['root/secondary/readNote.nl'] },
  });
  assert.match(listing, /Entry points are listed here/);
  assert.match(opening, /Host services bound only in the listed callable scopes:\n- noteRead: root\/primary\/readNote\.nl/);
  assert.ok(opening.indexOf('Host services bound') < opening.indexOf('Instructions:'));
  assert.doesNotMatch(opening, /unavailableRead/);
  assert.match(listing, /namespace primary/);
  assert.match(listing, /function read\(\): Row/);
  assert.doesNotMatch(listing, /leaf119|type Row =/);
  assert.match(results[0], /primary\.deep\.counter/);
  assert.match(results[1], /type Row = \{ label: string \}/);
  assert.match(results[2], /type Row = \{ units: number \}/);
  assert.match(results[3], /41/);
  assert.equal(value, 41);
});

test('a planning call with scoped services elsewhere sees its direct service boundary', async () => {
  const actions = [['return_result', { status: 'success', value: 1 }]];
  const services = { calendarCatalog: { list: () => [] } };
  const scoped = await script({}, actions, {
    services, serviceScopes: { calendarCatalog: ['another-call.nl'] },
  });
  assert.match(scoped.opening, /This call has no directly bound host services\. Work from its inputs/);
  const direct = await script({}, actions, { services });
  assert.match(direct.opening, /In eval you can use calendarCatalog/);
  assert.doesNotMatch(direct.opening, /no directly bound host services/);
});

for (const layout of ['conflicting siblings', 'caller shadow']) test(
  `eval locals retain callee alias dependencies with ${layout}`, async () => {
  // Both the Gmail and unrelated event handoff scopes define ProviderRecord. The outer
  // GmailGoalReport alias is unique, but its reference belongs to the Gmail scope.
  const files = {
    'root.nl': '---\nargs: {}\nreturns: string\n---\nVerify the carried case.\n',
    'root/gmail/types.ts': 'type ProviderRecord = { subject: string }; type GmailGoalReport = { provider: ProviderRecord | null };',
    'root/gmail/recover.nl': '---\nargs: {}\nreturns: GmailGoalReport\n---\nReturn the retained draft.\n',
    'root/gmail/recoverProvider.nl': `---
args: {}
returns: 'GmailGoalReport["provider"]'
---
Return the retained provider.
`,
    'root/other/types.ts': 'type ProviderRecord = { eventId: number };',
    'root/other/history.nl': '---\nargs: {}\nreturns: ProviderRecord\n---\nReturn the event history.\n',
  };
  if (layout === 'caller shadow') files['types.ts'] = 'type ProviderRecord = { eventId: number };';
  const provider = { subject: 'Tire delivery' };
  const steps = [
    ['eval', { code: 'const draft = await gmail.recover(); draft.provider.subject;' }],
    ['eval', { code: 'const providers = await Promise.all([1, 2].map(async index => gmail.recoverProvider())); providers.length;' }],
    ['eval', { code: 'return [draft.provider.subject, providers[1].subject].join(" then ");', finish: true }],
  ];
  let turns = 0;
  const runtime = createNatlangRuntime({
    model: async request => {
      const opening = request.messages[1].content;
      if (opening.includes('inside this call: recoverProvider('))
        return { calls: [['return_result', { status: 'success', value: provider }]] };
      if (opening.includes('inside this call: recover('))
        return { calls: [['return_result', { status: 'success', value: { provider } }]] };
      assert.ok(steps[turns], String(request.messages.at(-1).content));
      return { calls: [steps[turns++]] };
    },
    seed: { mode: 'backend' },
  });
  const fn = loadVirtualNatlang(files, 'root.nl');
  assert.equal(await runtime.run(() => fn()), 'Tire delivery then Tire delivery');
  assert.equal(turns, 3);
});
