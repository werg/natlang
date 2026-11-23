// Callable-folder TypeScript imports the natlang surface as `@natlang/node`; `Context` is part of it, so a helper can bind
// a sibling function to another context (`fn.in(Context.empty().with({ policy }))`, composition.md "One function,
// another context") exactly as host code does.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';

test('a callable-folder helper binds a sibling .nl to a context built with Context from @natlang/node', async () => {
  const openings = [];
  const driver = async request => {
    const opening = String(request.messages[1]?.content ?? '');
    if (opening.includes('lens(')) {
      return { calls: [['eval', { code: 'const strict = under("strict");\nreturn await strict("Please call it off.");', finish: true }]] };
    }
    openings.push(request.messages.map(message => String(message.content)).join('\n'));
    return { calls: [['return_result', { status: 'success', value: 'drop' }]] };
  };
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' }, calls: false });
  const lens = loadVirtualNatlang({
    'lens.nl': '---\nargs: {}\nreturns: string\n---\nLabel the message under the strict policy.\n',
    'lens/under.ts': 'import { Context } from "@natlang/node";\nimport read_policy from "./read_policy.nl";\n' +
      'const POLICIES: Record<string, string> = { strict: "drop only when the writer calls their own order off" };\n' +
      'export default function under(name: string): (text: string) => Promise<\"keep\" | \"drop\"> {\n  return read_policy.in(Context.empty().with({ policy: POLICIES[name] }));\n}\n',
    'lens/read_policy.nl': '---\nargs:\n  text: string\nreturns: "\\"keep\\" | \\"drop\\""\n---\nLabel text under policy.\n',
  }, 'lens.nl');
  assert.equal(await runtime.run(() => lens()), 'drop');
  assert.equal(openings.length, 1);
  assert.match(openings[0], /In eval you can use text, policy\b/, 'the bound policy is in the call\'s scope');
});
