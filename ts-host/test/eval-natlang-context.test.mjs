// Eval code binds a function to another context with the `NatlangContext` built-in (the runtime package's `Context`;
// eval uses this name so a program type named Context does not collide): fn.in(NatlangContext.empty().with({ policy })).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';

const callName = request => /You are inside this call: ([\w@$.:-]+)\(/.exec(String(request.messages?.[1]?.content ?? ''))?.[1];
const files = {
  'desk.nl': '---\nargs:\n  message: string\nreturns: "\\"keep\\" | \\"drop\\""\n---\nLabel message under the strict policy.\n',
  'desk/read_policy.nl': '---\nargs:\n  text: string\nreturns: "\\"keep\\" | \\"drop\\""\n---\nLabel text under policy.\n',
};

function run(code, onChild = () => ({ calls: [['return_result', { status: 'success', value: 'drop' }]] })) {
  const openings = [], outputs = [];
  let deskTurns = 0;
  const driver = async request => {
    const name = callName(request);
    if (name === 'desk') {
      // First turn: the eval under test. A later turn means it did not finish the call: keep what it showed, end.
      if (deskTurns++ === 0) return { calls: [['eval', { code, finish: true }]] };
      outputs.push(String(request.messages.at(-1)?.content));
      return { calls: [['return_result', { status: 'success', value: 'keep' }]] };
    }
    openings.push(request.messages.slice(1).map(message => String(message.content)).join("\n"));
    return onChild(request);
  };
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' }, calls: false });
  const desk = loadVirtualNatlang(files, 'desk.nl');
  return { result: runtime.run(() => desk('Please call it off.')), openings, outputs };
}

test('eval binds a sibling function to a context built with NatlangContext', async () => {
  const { result, openings } = run(
    'const policy = { rule: "drop only when the writer calls their own order off" };\n' +
    'const strict = read_policy.in(NatlangContext.empty().with({ policy }));\n' +
    'policy.rule = "changed after binding";\n' +
    'return await strict(message);',
    // The child reads the bound entry in its own eval: the value as it was when the context was built.
    () => ({ calls: [['eval', { code: 'return policy.rule === "drop only when the writer calls their own order off" ? "drop" : "keep";', finish: true }]] }));
  assert.equal(await result, 'drop', 'the context holds a snapshot of the record built in eval');
  assert.equal(openings.length, 1);
  assert.match(openings[0], /In eval you can use text, policy\b/, 'the bound policy is in the call\'s scope');
});

test('the built-ins line and read_code document NatlangContext', async () => {
  const { result, outputs } = run('const docs = read_code("NatlangContext");\nconsole.log(docs);\nthrow new Error("shown");');
  await result;
  assert.match(outputs.join('\n'), /NatlangContext\.empty\(\)\.with\(\{ policy: strictPolicy \}\)/);
  const { BUILT_INS_LINE } = await import('../dist/native/agent.js');
  assert.match(BUILT_INS_LINE, /decide and NatlangContext/);
});

test('a program binding named NatlangContext wins over the built-in', async () => {
  const { result } = run('const NatlangContext = 3;\nreturn NatlangContext === 3 ? "drop" : "keep";');
  assert.equal(await result, 'drop');
});
