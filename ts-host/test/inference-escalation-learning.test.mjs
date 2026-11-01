import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createLearning, learningService, Loss } from '../dist/neuralese/learning.js';
import { activeRecorder } from '../dist/native/model-recording.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';

const student = { id: 'student', revision: '1' }, expert = { id: 'expert', revision: '2' };
const turn = (actor, initial_actor = student) => ({ actor, initial_actor, messages: [], reply: { role: 'assistant', content: 'x' }, blocks: [] });
const trajectory = turns => ({ id: 'trajectory', turns });
const identity = { checkpoint_sha256: '1'.repeat(64), tokenizer_fingerprint: '2'.repeat(64),
  embedding_weight_sha256: '3'.repeat(64), architecture: 'fixture' };

test('likelihood objectives reject mixed, foreign and partially unknown actors', async () => {
  const learning = createLearning();
  for (const turns of [[turn(student), turn(expert)], [turn(expert)], [turn(student), { messages: [], reply: {}, blocks: [] }]]) {
    await assert.rejects(learning.objectives.logLikelihood(trajectory(turns)), /learning-mixed-actor/);
    await assert.rejects(learning.objectives.policyGradient([{ trajectory: trajectory(turns), reward: 1 }], { baseline: 0 }), /learning-mixed-actor/);
  }
  assert.equal((await learning.objectives.logLikelihood(trajectory([turn(student)]))).terms.length, 1);
  assert.equal((await learning.objectives.logLikelihood(trajectory([{ messages: [], reply: {}, blocks: [] }]))).terms.length, 1);
});

test('native replay rejects a recorded handoff before contacting a learner', async () => {
  const learning = createLearning(learningService({ endpoint: 'http://127.0.0.1:1', store: new MemoryNeuraleseStore() }));
  await assert.rejects(learning.nativeGraph.gradient(async () => {
    activeRecorder().recordGraphNode({ node: 'handoff', kind: 'combinator', operation: 'model-escalation' });
    return new Loss([{ kind: 'logLikelihood', native_turn_id: 'one' }]);
  }, { kind: 'model-parameters', names: ['x'] }, identity), /learning-mixed-actor/);
});

test('native replay rejects a recorded policy from a different checkpoint', async () => {
  const learning = createLearning(learningService({ endpoint: 'http://127.0.0.1:1', store: new MemoryNeuraleseStore() }));
  await assert.rejects(learning.nativeGraph.gradient(async () => {
    const rec = activeRecorder();
    rec.recordGraphNode({ node: 'turn', kind: 'model_turn', call_id: 'one', actor: student, initial_actor: student });
    rec.record({ ...turn(student), native: { schema: 'natlang.native-recorded-turn/1', id: 'one', invocation_id: 'one', retry: 0,
      kind: 'generation', sampling: { prompt: [], writes: [], policy: { ...identity, checkpoint_sha256: '4'.repeat(64) } } } });
    rec.linkNativeTurn('one', 'turn');
    return new Loss([{ kind: 'logLikelihood', native_turn_id: 'one' }]);
  }, { kind: 'model-parameters', names: ['x'] }, identity), /learning-foreign-actor/);
});
