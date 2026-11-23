// `refine<Is<T, "p">>(value)` is lowered to `refine(value, "p")` in every compiler, not only project builds: eval code and
// callable-folder TypeScript (both failed with refinement-predicate-invalid).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { CallStore, createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';

const callName = request => /You are inside this call: ([\w@$.:-]+)\(/.exec(String(request.messages?.[1]?.content ?? ''))?.[1];
function run(files, code, verdict = true) {
  const judged = [], shown = [];
  let turns = 0;
  const driver = Object.assign(async request => {
    if (callName(request) !== 'r') return { calls: [['return_result', { status: 'failed', reason: 'unexpected call' }]] };
    if (turns++) { shown.push(String(request.messages.at(-1).content)); return { calls: [['return_result', { status: 'failed', reason: 'the check failed' }]] }; }
    return { calls: [['eval', { code, finish: true }]] };
  }, { decide: async request => { judged.push(request.messages.map(m => String(m.content)).join('\n')); return { log_probs: verdict ? [0, -6] : [-6, 0] }; } });
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' }, calls: CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-calls-'))) });
  const root = loadVirtualNatlang({ 'r.nl': '---\nargs:\n  note: string\nreturns: string\n---\nCheck note.\n', ...files }, 'r.nl');
  return { result: runtime.run(() => root('hello there')), judged, shown };
}

for (const verdict of [true, false]) test(`refine<Is<T, "p">>(value) in eval is judged against the type's predicate (${verdict})`, async () => {
  const { result, judged, shown } = run({}, 'type Greeting = Is<string, "a greeting">;\nreturn await refine<Greeting>(note);', verdict);
  if (verdict) assert.equal(await result, 'hello there'); else await assert.rejects(result);
  assert.equal(judged.length, 1);
  assert.match(judged[0], /Property: the value is a greeting/);
  if (!verdict) assert.match(shown.join('\n'), /refinement-unsatisfied/);
});

const CHECK_TS = 'import { refine, type Is } from "@natlang/node";\n' +
  'export default async function check(note: string): Promise<Is<string, "a greeting">> { return await refine<Is<string, "a greeting">>(note); }\n';
test('refine<Is<T, "p">>(value) in a callable-folder TypeScript helper is lowered and judged', async () => {
  const { result, judged } = run({ 'r/check.ts': CHECK_TS }, 'return await check(note);');
  assert.equal(await result, 'hello there');
  assert.ok(judged.length >= 1);
  assert.match(judged[0], /Property: the value is a greeting/);
});
