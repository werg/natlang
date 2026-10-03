import assert from 'node:assert/strict';
import test from 'node:test';
import { decisionBatch } from './decisions.mjs';
import { clevr } from './scene.mjs';

const info = { path: '/pinned/source.jsonl', sha256: 'a'.repeat(64),
  source: 'test-source', license: 'CC-BY-4.0' };

test('decision batch v2 documents sparse nonzero-only label counts without changing gold', () => {
  const criteria = {
    '0': 'Authorized, and either not a heater or at least one person present',
    '1': 'Authorized heater request with zero people present',
    '2': 'Not authorized, regardless of device or occupancy',
  };
  const task = (id, device) => ({
    id, split: 'train', license: 'CC-BY-4.0', group_id: id,
    kind: 'categorical', labels: ['0', '1', '2'], criteria,
    state: `Request: off the ${device} in the kitchen. Authorization=no; people present=2; time=1:00.`,
    instruction: 'Classify the control request using these mutually exclusive rules.',
    gold: '2', gold_source: 'programmatic',
    source_meta: { family_id: 'smart_home_v2', question_key: 'risk' },
  });
  const row = decisionBatch([task('a', 'speaker'), task('b', 'heater')], info);
  assert.match(row.id, /batch-sparse-counts-v2/);
  assert.match(row.semantics.files['review_tasks.nl'], /omit labels whose count is zero/i);
  assert.deepEqual(row.semantics.expected.counts, { '2': 2 });
  assert.deepEqual(row.semantics.expected.labels, { a: '2', b: '2' });
});

test('scene contract v2 defines relation orientation, lexical mappings and value-only output', () => {
  const original = {
    id: 'scene-case', split: 'train', license: 'CC-BY-4.0', source_ids: ['scene:1'],
    source_groups: ['scene-group'],
    semantics: {
      question: 'What shape is the gray object?', answer: 'sphere',
      scene: { objects: [{ color: 'gray', material: 'metal', shape: 'sphere', size: 'large' }],
        relationships: { front: [[]], behind: [[]], left: [[]], right: [[]] } },
      nodes: [
        { function: 'scene', inputs: [], value_inputs: [] },
        { function: 'filter_color', inputs: [0], value_inputs: ['gray'] },
        { function: 'unique', inputs: [1], value_inputs: [] },
        { function: 'query_shape', inputs: [2], value_inputs: [] },
      ],
    },
  };
  const row = clevr(original, info);
  const contract = row.semantics.files['answer_scene_question.nl'];
  assert.match(row.id, /scene-contract-v2/);
  assert.match(contract, /relationships\[R\]\[i\].*indexes j/i);
  assert.match(contract, /matte or dull means material rubber/i);
  assert.match(contract, /shiny or metallic means material metal/i);
  assert.match(contract, /Return only the requested value/i);
  assert.equal(row.semantics.expected, 'sphere');
  assert.deepEqual(row.source_ids, original.source_ids);
  assert.deepEqual(row.source_groups, ['test-source:scene-group']);
});
