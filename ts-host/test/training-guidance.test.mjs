import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { NativeToolAgent, modelTurnsSoFar } from '../dist/native/agent.js';
import { TrainingGuidance, requestText } from '../dist/native/training-guidance.js';
import { loadGuidanceWriter } from '../dist/teacher/guidance-writer.js';
import { chatCompletionModelTurn } from '../dist/model/chat-completion.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { ModelTurnRecorder, setRecorderSource } from '../dist/native/model-recording.js';
import { interpreter, lambda, nl } from './support/natlang.mjs';
import { nativeStateFixture } from './fixtures/native-state.mjs';

const DIR = fileURLToPath(new URL('../../applications/guidance-writer', import.meta.url));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const SELECT_SHA = sha(readFileSync(`${DIR}/selectGuidanceFacts.nl`));
const WRITE_SHA = sha(readFileSync(`${DIR}/writeGuidance.nl`));
const CRISP_SHA = sha('natlang.select-guidance-facts.crisp/1');
const pkg = loadGuidanceWriter(DIR);

const custody = { schema: 'natlang.jolt-privileged-custody/1', split: 'train', case_id: 'world-1' };
const FACTS = [
  { id: 'reference_solution:0', kind: 'reference_solution', text: 'A verified reference solution:\n1. eval:\nreturn await helper("child-question")', invocation: null, match: [] },
  { id: 'expected_result:0', kind: 'expected_result', text: 'The verified result of this program is "7".', invocation: null, match: [] },
  { id: 'child_value:0', kind: 'child_value', text: 'The verified answer of this call is "7".', invocation: null, match: ['child-question'] },
  { id: 'attempt_feedback:0', kind: 'attempt_feedback', text: 'Earlier attempt 1 received: the result was not a string.', invocation: 'run-1', match: [] },
  { id: 'attempt_feedback:1', kind: 'attempt_feedback', text: 'Earlier attempt 2 received: the result was late.', invocation: 'someone-else', match: [] },
];

test('the package loads with its exact .nl hashes and crisp identity', () => {
  assert.equal(pkg.selection.sha256, SELECT_SHA);
  assert.equal(pkg.writer.sha256, WRITE_SHA);
  assert.equal(pkg.crispSha256, CRISP_SHA);
  assert.equal(pkg.writer.definition.readout, 'template');
  assert.match(pkg.writer.definition.returns, /^Neuralese<string>$/);
  assert.equal(pkg.selection.definition.returns, 'string[]');
});

test('visibility mirrors rl/privileged_context.applicable', () => {
  const root = { name: 'run-1', role: 'root', text: 'opening' };
  const child = { name: 'run-1/1', role: 'child', text: 'q = "child-question"' };
  const other = { name: 'run-1/2', role: 'child', text: 'q = "other"' };
  const binding = { name: 'skill', role: 'binding', text: 'x' };
  const ids = consumer => pkg.visibleFacts(consumer, FACTS).map(fact => fact.id);
  assert.deepEqual(ids(root), ['reference_solution:0', 'expected_result:0', 'attempt_feedback:0']);
  assert.deepEqual(ids(child), ['child_value:0']);
  assert.deepEqual(ids(other), []);
  assert.deepEqual(ids(binding), ['reference_solution:0', 'expected_result:0']);
  const hindsight = { id: 'fit_hindsight:0', kind: 'fit_hindsight', text: 'h', invocation: null };
  assert.deepEqual(pkg.visibleFacts(other, [hindsight, { ...hindsight, id: 'h1', invocation: 'run-1/2' },
    { ...hindsight, id: 'h2', invocation: 'run-1/3' }]).map(fact => fact.id), ['fit_hindsight:0', 'h1']);
});

test('crisp selection orders by kind and drops an ambiguous child match', () => {
  const root = { name: 'run-1', role: 'root', text: '' };
  const facts = [...FACTS, { id: 'fit_hindsight:0', kind: 'fit_hindsight', text: 'h' },
    { id: 'decisive:0', kind: 'decisive', text: 'd' }, { id: 'expected_stop:0', kind: 'expected_stop', text: 's' }];
  assert.deepEqual(pkg.crispSelection(root, pkg.visibleFacts(root, facts), 512),
    ['expected_result:0', 'expected_stop:0', 'reference_solution:0', 'attempt_feedback:0', 'decisive:0', 'fit_hindsight:0']);
  const child = { name: 'c', role: 'child', text: 'child-question' };
  const second = { ...FACTS[2], id: 'child_value:1', text: 'The verified answer of this call is "8".' };
  const feedback = { ...FACTS[3], invocation: null };
  assert.deepEqual(pkg.crispSelection(child, [feedback, FACTS[2]], 512), ['child_value:0', 'attempt_feedback:0']);
  assert.deepEqual(pkg.crispSelection(child, [FACTS[2], second, feedback], 512), ['attempt_feedback:0']);
  assert.deepEqual(pkg.crispSelection(child, [FACTS[2], { ...FACTS[2], id: 'child_value:1' }], 512), ['child_value:0', 'child_value:1']);
});

test('requestText is the engine\'s visibility text: string contents joined by newlines', () => {
  assert.equal(requestText([{ role: 'system', content: 'a' }, { role: 'user', content: [{ type: 'text', text: 'x' }] },
    { role: 'tool', content: 'b' }, null]), 'a\nb');
});

test('training guidance travels on the wire only to a native server', async () => {
  const bodies = [];
  const transport = async body => { bodies.push(body); return { choices: [{ message: { role: 'assistant', content: 'ok' } }] }; };
  const request = { messages: [{ role: 'user', content: 'x' }], tools: [], seed: null, max_tokens: 8,
    privileged_context: { schema: 'natlang.guidance-request/1' }, guidance_policy: { function: 'writeGuidance', sha256: 'a' } };
  await chatCompletionModelTurn(transport, { adapters: true })(request);
  assert.deepEqual(bodies[0].x_natlang_privileged_context, { schema: 'natlang.guidance-request/1' });
  assert.deepEqual(bodies[0].x_natlang_guidance_policy, { function: 'writeGuidance', sha256: 'a' });
  await assert.rejects(chatCompletionModelTurn(transport, {})(request), /training-guidance-unavailable/);
});

/**
 * One episode: the root calls helper("child-question") in its first turn and returns in its second; helper answers in
 * one turn. The fake driver answers the guidance policy's calls by their `guidance_policy`.
 */
async function episode({ role = 'teacher', selection, select = ids => ids, writer = 'write', writerSetting } = {}) {
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture());
  const neuralese = { store, dialect: 'native-state/fixture' };
  const requests = [];
  const driver = Object.assign(async request => {
    requests.push(structuredClone(request));
    const policy = request.guidance_policy?.function;
    if (policy === 'selectGuidanceFacts') {
      const shown = requestText(request.messages);
      const visible = FACTS.map(fact => fact.id).filter(id => shown.includes(JSON.stringify(id)));
      return { calls: [['return_result', { status: 'success', value: select(visible) }]] };
    }
    if (policy === 'writeGuidance') {
      if (writer === 'fail') return { calls: [['return_result', { status: 'failed', reason: 'The writer could not write.' }]] };
      return { calls: [['return_result', { status: 'success', value: [{ type: 'native_state', id: artifact.id }] }]],
        raw_response: { native_state: { blocks: [{ ...artifact.metadata, id: artifact.id }] } } };
    }
    const opening = String(request.messages[1]?.content ?? '');
    if (opening.includes('Answer q.')) return { calls: [['return_result', { status: 'success', value: '7' }]] };
    if (modelTurnsSoFar(request.messages) === 0)
      return { calls: [['eval', { code: 'const answer = await helper("child-question");' }]] };
    return { calls: [['return_result', { status: 'success', value: '7' }]] };
  }, { neuralese: true });
  const run = { schema: 'natlang.guidance-run/1', role, custody, facts: FACTS, ...(selection ? { selection } : {}),
    ...(writerSetting ? { writer: writerSetting } : {}) };
  const guidance = new TrainingGuidance(run, pkg);
  const agent = new NativeToolAgent(driver, { maxTurns: 6, neuralese, trainingGuidance: guidance });
  const recorder = new ModelTurnRecorder();
  setRecorderSource(() => recorder);
  const traces = [];
  let result;
  try {
    const runtime = interpreter({ runId: 'run-1', calls: false, neuralese, agent: session => {
      traces.push(session.runtime.trace);
      return agent.run(session);
    } });
    result = await runtime.run(lambda({ type: '() => string', instructions: 'Ask the helper.',
      codebase: { helper: nl('helper', { args: { q: 'string' }, returns: 'string', instructions: 'Answer q.' }) } }));
  } finally { setRecorderSource(() => undefined); }
  return { result, requests, recorder, artifact, traces };
}

const consumerTurns = requests => requests.filter(request => !request.guidance_policy);
const policyTurns = (requests, name) => requests.filter(request => request.guidance_policy?.function === name);

test('each consumer\'s first turn runs selection then the writer; every consumer turn carries the exact request', async () => {
  const { result, requests, recorder, artifact } = await episode();
  assert.equal(result.outcome.kind, 'done', result.outcome.detail);
  const consumers = consumerTurns(requests);
  const root = consumers.filter(request => request.invocation_id === 'run-1');
  const child = consumers.filter(request => request.invocation_id !== 'run-1');
  assert.equal(root.length, 2);
  assert.equal(child.length, 1);
  // Two consumers, one selection and one writer each: later root turns reuse the cached state.
  assert.equal(policyTurns(requests, 'selectGuidanceFacts').length, 2);
  assert.equal(policyTurns(requests, 'writeGuidance').length, 2);
  // Order: selection, writer, then the consumer's first turn.
  const firstRoot = requests.indexOf(requests.find(request => request.invocation_id === 'run-1'));
  assert.deepEqual(requests.slice(0, 2).map(request => request.guidance_policy?.function), ['selectGuidanceFacts', 'writeGuidance']);
  assert.equal(firstRoot, 2);
  for (const request of requests.filter(request => request.guidance_policy)) {
    assert.equal(request.privileged_context, undefined, 'policy calls get no guidance');
    assert.equal(request.guidance_policy.sha256, request.guidance_policy.function === 'writeGuidance' ? WRITE_SHA : SELECT_SHA);
    assert.notEqual(request.invocation_id, 'run-1');
  }
  const writeTurn = policyTurns(requests, 'writeGuidance')[0];
  assert.equal(writeTurn.template.value, 'write', 'the writer is a template producer write');
  // The writer's own prompt carries every selected fact verbatim (rl/privileged_context.writer_saw).
  const writerText = requestText(writeTurn.messages);
  for (const fact of FACTS.slice(0, 2)) assert.ok(writerText.includes(fact.text), fact.id);
  const blockWrite = [...recorder.graphNodes.values()].find(node => node.kind === 'block_write' && node.block === artifact.id &&
    node.call_id === writeTurn.invocation_id);
  assert.ok(blockWrite, 'the recorder holds the writer\'s producer write');
  const rootRequest = { schema: 'natlang.guidance-request/1', invocation: { name: 'run-1', role: 'root' }, mode: 'import',
    custody, facts: [FACTS[0], FACTS[1], FACTS[3]], selection: { implementation: 'nl', sha256: SELECT_SHA },
    writer: { function: 'writeGuidance', sha256: WRITE_SHA }, state: { id: artifact.id, source_occurrence: blockWrite.node } };
  for (const request of root) assert.deepEqual(request.privileged_context, rootRequest);
  assert.deepEqual(child[0].privileged_context.invocation, { name: child[0].invocation_id, role: 'child' });
  assert.deepEqual(child[0].privileged_context.facts, [FACTS[2]]);
  assert.equal(child[0].privileged_context.mode, 'import');
  // The policy calls are graph nodes of the same episode: their model turns are recorded with the consumer's.
  const turnCalls = new Set([...recorder.graphNodes.values()].filter(node => node.kind === 'model_turn').map(node => node.call_id));
  for (const request of requests) assert.ok(turnCalls.has(request.invocation_id), `recorded turn of ${request.invocation_id}`);
});

test('a student writes guidance without importing it', async () => {
  const { requests } = await episode({ role: 'student' });
  for (const request of consumerTurns(requests)) assert.equal(request.privileged_context.mode, 'write-only');
  assert.equal(policyTurns(requests, 'writeGuidance').length, 2);
});

test('a failed writer gives a teacher the selected facts as text and a student nothing', async () => {
  const teacher = await episode({ writer: 'fail' });
  assert.equal(teacher.result.outcome.kind, 'done', teacher.result.outcome.detail);
  const root = consumerTurns(teacher.requests).find(request => request.invocation_id === 'run-1').privileged_context;
  assert.equal(root.mode, 'text');
  assert.match(root.fallback, /^writeGuidance failed: /);
  assert.equal(root.writer, undefined);
  assert.equal(root.state, undefined);
  assert.deepEqual(root.facts.map(fact => fact.id), ['reference_solution:0', 'expected_result:0', 'attempt_feedback:0']);
  const student = await episode({ role: 'student', writer: 'fail' });
  assert.equal(student.result.outcome.kind, 'done', student.result.outcome.detail);
  for (const request of consumerTurns(student.requests)) assert.equal(request.privileged_context, undefined);
  const events = student.traces.flatMap(trace => trace.events).filter(event => event.kind === 'training_guidance');
  assert.ok(events.some(event => event.phase === 'none' && /writeGuidance failed/.test(event.reason)));
});

test('an invalid natural-language selection falls back to crisp and records it', async () => {
  const { requests, traces } = await episode({ select: () => ['not-a-fact'] });
  const root = consumerTurns(requests).find(request => request.invocation_id === 'run-1').privileged_context;
  assert.deepEqual(root.selection, { implementation: 'crisp', sha256: CRISP_SHA });
  assert.deepEqual(root.facts.map(fact => fact.id), ['expected_result:0', 'reference_solution:0', 'attempt_feedback:0']);
  const events = traces.flatMap(trace => trace.events).filter(event => event.kind === 'training_guidance');
  assert.ok(events.some(event => event.phase === 'selection_fallback' && /distinct ids/.test(event.reason)));
  // A crisp setting runs no selection call at all.
  const crisp = await episode({ selection: 'crisp' });
  assert.equal(policyTurns(crisp.requests, 'selectGuidanceFacts').length, 0);
  assert.equal(consumerTurns(crisp.requests)[0].privileged_context.selection.implementation, 'crisp');
});

test('writer: off is the text baseline: no writer call, teacher text, student nothing', async () => {
  const teacher = await episode({ writerSetting: 'off', selection: 'crisp' });
  assert.equal(policyTurns(teacher.requests, 'writeGuidance').length, 0);
  assert.equal(policyTurns(teacher.requests, 'selectGuidanceFacts').length, 0);
  for (const request of consumerTurns(teacher.requests)) {
    assert.equal(request.privileged_context.mode, 'text');
    assert.equal(request.privileged_context.fallback, 'writer disabled by run setting');
  }
  const student = await episode({ role: 'student', writerSetting: 'off' });
  assert.equal(student.requests.filter(request => request.guidance_policy).length, 0);
  for (const request of consumerTurns(student.requests)) assert.equal(request.privileged_context, undefined);
});
