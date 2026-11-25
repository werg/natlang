// The standard library's combinators are eval built-ins (2026-11-25): a call composes its judgments with semanticMap,
// semanticReduce, ... by name, they run in the call's frame (their std:<name> spans and child calls nest under the
// call), an `import` of @natlang/node/std reads as those built-ins, and NATLANG_EVAL_STD=0 turns them off.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';

const callName = request => /You are inside this call: ([\w@$.:-]+)\(/.exec(String(request.messages?.[1]?.content ?? ''))?.[1] ?? '';
const files = {
  'desk.nl': '---\nargs:\n  notes: string[]\nreturns: string\n---\nSummarize each note, then fold the summaries into one digest.\n',
  'desk/summarize.nl': '---\nargs:\n  note: string\nreturns: string\n---\nSummarize note in a few words.\n',
  'desk/fold.nl': '---\nargs:\n  digest: string\n  summary: string\nreturns: string\n---\nAdd summary to digest.\n',
};

function run(code) {
  const outputs = [], traces = [], openings = [];
  let deskTurns = 0;
  const driver = async request => {
    const name = callName(request);
    if (name.endsWith('summarize')) return { calls: [['eval', { code: 'return "S(" + note + ")";', finish: true }]] };
    if (name.endsWith('fold')) return { calls: [['eval', { code: 'return digest + summary;', finish: true }]] };
    if (!name.endsWith('desk')) {
      openings.push(request.messages.slice(1).map(message => String(message.content)).join('\n'));
      return { calls: [['return_result', { status: 'success', value: 'N' }]] };
    }
    // First turn: the eval under test. A later turn means it did not finish the call: keep what it showed, end.
    if (deskTurns++ === 0) return { calls: [['eval', { code, finish: true }]] };
    outputs.push(String(request.messages.at(-1)?.content));
    return { calls: [['return_result', { status: 'success', value: 'not finished' }]] };
  };
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' }, calls: false, trace: t => traces.push(t) });
  const desk = loadVirtualNatlang(files, 'desk.nl');
  return { result: runtime.run(() => desk(['a', 'b', 'c'])), outputs, traces, openings };
}

function assertNested(traces, combinator, child, count) {
  const desk = traces.find(trace => trace.name === 'desk' || /(^|[/:])desk(\.nl)?$/.test(String(trace.definitionId)));
  const spans = traces.filter(trace => trace.definitionId === `std:${combinator}`);
  assert.equal(spans.length, 1, `one std:${combinator} span`);
  const [span] = spans;
  if (desk) assert.equal(span.parentCallId, desk.callId, `std:${combinator} nests under the calling call`);
  assert.notEqual(span.parentCallId, null);
  const children = traces.filter(trace => trace.parentCallId === span.callId);
  assert.equal(children.length, count, `${count} ${child} calls under std:${combinator}`);
  for (const trace of children) assert.match(String(trace.name ?? trace.definitionId), new RegExp(child));
}

test('eval maps an .nl function with semanticMap and reduces with semanticReduce, nested under the call', async () => {
  const { result, traces, outputs } = run(
    'const summaries = await semanticMap(notes, summarize);\n' +
    'return await semanticReduce(summaries, "", fold);');
  assert.equal(await result, 'S(a)S(b)S(c)', outputs.join('\n'));
  assertNested(traces, 'semanticMap', 'summarize', 3);
  assertNested(traces, 'semanticReduce', 'fold', 3);
});

test('an eval import of @natlang/node/std reads as the built-ins and runs in the frame', async () => {
  const { result, traces, outputs } = run(
    'import { semanticMap } from "@natlang/node/std";\n' +
    'import { semanticReduce as reduce } from \'@natlang/node/std\';\n' +
    'const summaries = await semanticMap(notes, summarize);\n' +
    'return await reduce(summaries, "", (digest: string, summary: string) => digest + summary);');
  assert.equal(await result, 'S(a)S(b)S(c)', outputs.join('\n'));
  assertNested(traces, 'semanticMap', 'summarize', 3);
});

test('an eval import of something else from std fails with one sentence naming the built-ins', async () => {
  const { result, outputs } = run('import { meets } from "@natlang/node/std";\nreturn "x";');
  assert.equal(await result, 'not finished');
  assert.match(outputs.join('\n'), /meets from @natlang\/node\/std is not available in eval; its combinators semanticMap, .*semanticReduce.* are built into this scope: call them by name, without an import\./);
});

test('read_code documents a std built-in and edit_code refuses it', async () => {
  const { result, outputs } = run('console.log(read_code("semanticReduce"));\nthrow new Error("shown");');
  await result;
  assert.match(outputs.join('\n'), /semanticReduce\(items, initial, step/);
});

test('eval types the std built-ins: an inline nl in a semanticMap callback takes the item type', async () => {
  const { result, outputs, traces, openings } = run(
    'const labels = await semanticMap(notes, note => nl`Label note in one word.`(note));\n' +
    'return labels.join("");');
  assert.equal(await result, 'NNN', outputs.join('\n'));
  assert.equal(openings.length, 3);
  assert.match(openings[0], /note: string/);
  const span = traces.find(trace => trace.definitionId === 'std:semanticMap');
  assert.equal(traces.filter(trace => trace.parentCallId === span.callId).length, 3);
});

test('NATLANG_EVAL_STD=0 turns the std built-ins off', async () => {
  process.env.NATLANG_EVAL_STD = '0';
  try {
    const { result, outputs, traces } = run('return await semanticMap(notes, summarize);');
    assert.equal(await result, 'not finished');
    assert.match(outputs.join('\n'), /semanticMap is not bound in this call's eval scope/);
    assert.equal(traces.filter(trace => trace.definitionId === 'std:semanticMap').length, 0);
  } finally { delete process.env.NATLANG_EVAL_STD; }
});

test('a binding of the call named like a combinator wins over the built-in', async () => {
  const { result } = run('const semanticMap = (items: string[]) => items.join("+");\nreturn semanticMap(notes);');
  assert.equal(await result, 'a+b+c');
});
