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

test('soft inline child carries eval-local Draft aliases through iterateOn and its captured snapshot', async () => {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1');
  const body = await port.write('Reconcile the scoped pass with the complete current Draft. Return the complete Draft.');
  const requests = [];
  const draft = { stop: 'WILLOW-9', route: 'R-16' };
  const code = `
type Draft = { stop: string; route: string };
type Progress = { pass: number; draft: Draft };
const revise = async (state: Progress): Promise<Progress> => {
  const currentDraft: string = JSON.stringify(state.draft);
  const update = nl.with<Draft>({ currentDraft })\`${neuraleseSentinel(body.id)}\`;
  const nextDraft = await update(state.draft);
  return { pass: state.pass + 1, draft: nextDraft };
};
const initial: Progress = { pass: 0, draft: ${JSON.stringify(draft)} };
const final = await iterateOn(revise, initial).checkProgress('off').withLimit({ maxSteps: 1 }).until(state => state.pass === 1);
return final.draft;
`;
  const driver = Object.assign(async request => {
    requests.push(request.messages);
    if (requests.length === 1) return { calls: [['eval', { code }]] };
    return { calls: [['return_result', { status: 'success', value: { stop: 'WILLOW-9', route: 'R-17' } }]] };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port } });
  const root = defineNatlang('---\nargs: {}\nreturns: "{ stop: string, route: string }"\n---\nRun one typed Draft revision through iterateOn.\n', { name: 'root' });

  assert.deepEqual(await runtime.run(() => root()), { stop: 'WILLOW-9', route: 'R-17' });
  assert.ok(requests.length >= 2, 'the root eval and child call both reached the model boundary');
  const childParts = requestParts(requests[1]);
  assert.ok(childParts.some(part => part.type === 'neuralese' && part.id === body.id), 'the child retains its exact soft body');
  const childOpening = childParts.filter(part => part.type === 'text').map(part => part.text).join('\n');
  assert.match(childOpening, /type Draft\s*=\s*\{[^}]*stop: string[^}]*route: string[^}]*\}/,
    'the alias used by the child signature is declared in the generated opening');
  assert.match(childOpening, /\(input: Draft\): Draft/,
    'the child keeps the named return and parameter type instead of degrading to any/unknown');
  const captureStart = childOpening.indexOf('const currentDraft: string =');
  assert.notEqual(captureStart, -1, 'the current Draft is explicitly captured as a string');
  const captureText = childOpening.slice(captureStart, captureStart + 300);
  assert.ok(captureText.includes('WILLOW-9') && captureText.includes('R-16'),
    'the captured snapshot contains both exact carried fields');
});
