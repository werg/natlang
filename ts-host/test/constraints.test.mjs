import test from 'node:test';
import assert from 'node:assert/strict';
import { checkConstraints, describeConstraint } from '../dist/evaluation/constraints.js';
import { checkOracle } from '../dist/evaluation/oracles.js';
import { validateCurriculum } from '../dist/teacher/curriculum.js';
import { constrainedRewrite, constrainedWriting } from '../scripts/inline-curriculum/writing.mjs';

test('each constraint kind is checked by code', () => {
  const text = '<<Plan>>\n\nHello everyone. The review moves to Friday.\n\n- one item\n- two items\n\nIs there anything else I can help with?';
  const pass = [
    { kind: 'word_count', min: 10, max: 40 }, { kind: 'bullet_count', count: 2 }, { kind: 'title' },
    { kind: 'include_words', words: ['review'] }, { kind: 'exclude_words', words: ['very'] },
    { kind: 'ends_with', text: 'Is there anything else I can help with?' }, { kind: 'paragraph_count', count: 4 },
    { kind: 'word_frequency', word: 'items', min: 1 }, { kind: 'sentence_count', min: 3 },
  ];
  assert.deepEqual(checkConstraints(text, pass).failed, []);
  const fail = [{ kind: 'no_commas' }, { kind: 'all_lowercase' }, { kind: 'placeholders', min: 1 }, { kind: 'starts_with', text: 'Hi' },
    { kind: 'json_keys', keys: ['a'] }, { kind: 'include_words', words: ['budget'] }];
  const result = checkConstraints(text + ', x', fail);
  assert.equal(result.passed, false);
  assert.equal(result.failed.length, fail.length);
  assert.equal(checkConstraints('{"a": 1, "b": 2}', [{ kind: 'json_keys', keys: ['a', 'b'] }]).passed, true);
  assert.equal(checkConstraints(42, [{ kind: 'title' }]).passed, false);
  for (const c of [...pass, ...fail]) assert.match(describeConstraint(c), /\S/);
});

test('constraints oracle level accepts only when every constraint holds', async () => {
  const constraints = [{ kind: 'word_count', max: 5 }, { kind: 'no_commas' }];
  assert.equal((await checkOracle('short and plain', constraints, { level: 'constraints' })).accepted, true);
  const rejected = await checkOracle('short, but with a comma', constraints, { level: 'constraints' });
  assert.equal(rejected.accepted, false);
  assert.equal(rejected.score, 0.5);
  assert.match(rejected.verdict, /commas/);
});

test('constrained writing cases are valid and their references satisfy their constraints', async () => {
  for (let index = 0; index < 60; index++) {
    const [record] = constrainedWriting(11, index);
    validateCurriculum(record);
    assert.equal(record.semantics.oracle.level, 'constraints');
    const reference = record.curriculum.reference.root.at(-1)[1].value;
    assert.equal((await checkOracle(reference, record.semantics.expected, record.semantics.oracle)).accepted, true);
    assert.match(record.semantics.files[record.semantics.root], /Return only the text\./);
  }
  assert.deepEqual(constrainedWriting(11, 5), constrainedWriting(11, 5));
});

test('constrained rewrite cases keep their facts and pass their own checks', async () => {
  const variants = new Set();
  for (let index = 0; index < 60; index++) {
    const [record] = constrainedRewrite(3, index);
    validateCurriculum(record);
    variants.add(record.curriculum.variant);
    const reference = record.curriculum.reference.root.at(-1)[1].value;
    assert.equal((await checkOracle(reference, record.semantics.expected, record.semantics.oracle)).accepted, true);
    assert.equal(typeof record.semantics.inputs.passage, 'string');
  }
  assert.deepEqual([...variants].sort(), ['bullets', 'plain', 'summarize']);
});
