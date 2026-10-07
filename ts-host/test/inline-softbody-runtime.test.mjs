import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, defineNatlang } from '../dist/index.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';
import { neuraleseSentinel } from '../dist/native/neuralese.js';

const requestParts = messages => messages.flatMap(message => [message.content,
  ...(message.tool_calls ?? []).map(call => call.function.arguments)])
  .flatMap(content => Array.isArray(content) ? content : [{ type: 'text', text: String(content ?? '') }]);

async function exercise({ explicitCapture }) {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1');
  const body = await port.write('Classify the note using the supplied rule.');
  const note = 'The package arrived damaged.';
  const policy = 'Only use details stated in the note.';
  const prefix = 'const untouched = 17;\n';
  const child = explicitCapture
    ? `const child: Neuralese<(note: string) => Promise<boolean>> = nl.with({ policy })\`${neuraleseSentinel(body.id)}\`;`
    : `const child: Neuralese<(note: string) => Promise<boolean>> = nl\`${neuraleseSentinel(body.id)}\`;`;
  const source = `${prefix}${child}\nreturn await child(note);`;
  const root = defineNatlang('---\nargs:\n  note: string\n  policy: string\nreturns: boolean\n---\nCreate and ask the child checker.\n', { name: 'root' });
  const requests = [];
  const traces = [];
  const driver = Object.assign(async request => {
    requests.push(request.messages);
    if (requests.length === 1) return { calls: [['eval', { code: source }]] };
    // Stop at the child's assembled opening. No answer or semantic action is supplied.
    throw new Error('fixture stopped before child completion');
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port }, trace: trace => traces.push(trace) });
  await assert.rejects(() => runtime.run(() => root(note, policy)), /fixture stopped before child completion/);
  assert.ok(requests.length >= 2, 'the child invocation reached the model boundary');
  const childRequest = requests[1];
  const parts = requestParts(childRequest);
  assert.ok(parts.some(part => part.type === 'neuralese' && part.id === body.id), 'the child opening transports the exact body block');
  const openingText = parts.filter(part => part.type === 'text').map(part => part.text).join('\n');
  assert.match(openingText, /note:\s*string\s*=\s*"The package arrived damaged\."/);
  if (explicitCapture) {
    assert.ok(openingText.includes('const policy: string ='), 'the child scope declares the capture with its type');
    assert.ok(openingText.includes('Only use details stated in the note.'), 'the child scope receives the captured snapshot value');
  } else {
    assert.doesNotMatch(openingText, /policy:\s*string\s*=/, 'an unlisted parent binding is not captured by a soft body');
  }
  // The eval action retains all crisp code around the body marker.
  const parent = traces.find(trace => !trace.parentCallId);
  const action = parent.events.find(event => event.kind === 'action' && event.name === 'eval');
  assert.ok(action, 'the parent eval ran through the native action path');
  assert.match(action.arguments.code, /^const untouched = 17;\n/);
  assert.match(action.arguments.code, /return await child\(note\);$/);
  assert.ok(action.arguments.code.includes(body.id), 'the body marker remains in the eval source until compilation');
  return { body, source, requests };
}

test('whole-block soft body reaches child execution with no implicit parent captures', async () => {
  await exercise({ explicitCapture: false });
});

test('whole-block soft body preserves explicit snapshot capture values and types at child opening', async () => {
  await exercise({ explicitCapture: true });
});
