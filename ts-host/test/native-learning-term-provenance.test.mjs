import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createLearning, learningService } from '../dist/neuralese/learning.js';
import { activeRecorder } from '../dist/native/model-recording.js';
import { registerLocalEndpoint } from '../dist/model/chat-completion.js';

// Exercise the actual objective builders and nested recorder absorption. This
// fixture inspects request metadata; it does not qualify native model execution.
async function requestTerms(build) {
  const requests = [];
  const endpoint = 'http://native-term-provenance.invalid';
  const dispose = registerLocalEndpoint(endpoint, async (url, init) => {
    assert.equal(url, endpoint + '/v1/neuralese/grad');
    requests.push(JSON.parse(init.body));
    return Response.json({ loss: 0, gradients: {} });
  });
  try {
    const learning = createLearning(learningService({ endpoint }));
    await learning.valueAndGrad(() => build(learning), {});
    assert.equal(requests.length, 1);
    return requests[0];
  } finally { dispose(); }
}

function output(id, { decision = false, modes = undefined } = {}) {
  return async () => {
    activeRecorder().record({ messages: [{ role: 'user', content: id }],
      tools: [{ type: 'function', function: { name: 'tool-' + id, parameters: { type: 'object' } } }],
      reply: { role: 'assistant', content: 'answer' }, blocks: [],
      adapters: [{ id: 'adapter-' + id, scale: 0.25 }],
      ...(modes ? { native_state_modes: modes } : {}),
      ...(decision ? { decision: { options: ['"yes"', '"no"'] } } : {}),
      native: { schema: 'natlang.native-recorded-turn/1', id, retry: 0,
        ...(decision ? { kind: 'decision', decision: {} } : { kind: 'generation', sampling: {} }) } });
  };
}

test('generated distillation retains exact independent teacher occurrence and transport contract', async () => {
  for (const conditioned of [false, true]) {
    const request = await requestTerms(({ objectives }) => conditioned
      ? objectives.conditionedDistill(output('student'), output('teacher', {
          modes: { source: 'same-base-adapter-state' } }))
      : objectives.selfDistill(output('student'), output('teacher', {
          modes: { source: 'same-base-adapter-state' } })));
    const term = request.terms[0];
    assert.equal(term.native_turn_id, 'student');
    assert.equal(term.teacher_native_turn_id, 'teacher');
    assert.deepEqual(term.teacher_native_state_modes, { source: 'same-base-adapter-state' });
    assert.deepEqual(term.teacher_adapters, [{ id: 'adapter-teacher', scale: 0.25 }]);
    assert.deepEqual(request.native_turns.map(turn => turn.native.id), ['student', 'teacher']);
  }
});

test('decision distillation binds its actual teacher query without borrowing student modes', async () => {
  const request = await requestTerms(({ objectives }) => objectives.conditionedDistill(
    output('student', { decision: true }), output('teacher', { decision: true,
      modes: { source: 'same-base-adapter-state' } })));
  const term = request.terms[0];
  assert.equal(term.native_turn_id, 'student');
  assert.equal(term.teacher.native_turn_id, 'teacher');
  assert.deepEqual(term.teacher.native_state_modes, { source: 'same-base-adapter-state' });
  assert.deepEqual(term.teacher.adapters, [{ id: 'adapter-teacher', scale: 0.25 }]);
  assert.equal(term.tools[0].function.name, 'tool-student');
  assert.equal(term.teacher.tools[0].function.name, 'tool-teacher');
  assert.equal(term.native_state_modes, undefined);
});

test('REINFORCE and conditional likelihood retain distinct probability laws', async () => {
  const turn = { messages: [], reply: {}, blocks: [], native: {
    schema: 'natlang.native-recorded-turn/1', id: 'sample', kind: 'generation', retry: 0, sampling: {} } };
  const trajectory = { id: 'trajectory', turns: [turn] };
  const request = await requestTerms(async ({ objectives }) => objectives.sum(
    await objectives.logLikelihood(trajectory, 3),
    await objectives.policyGradient([{ trajectory, reward: 5 }], { baseline: 2 })));
  assert.deepEqual(request.terms.map(term => [term.likelihood_law, term.weight, term.native_turn_id]),
    [['model-conditional', 3, 'sample'], ['sampled-policy', 3, 'sample']]);
});
