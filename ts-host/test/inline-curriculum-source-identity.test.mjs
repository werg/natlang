import assert from 'node:assert/strict';
import test from 'node:test';
import { identifySourceCase } from '../scripts/inline-curriculum/lib.mjs';
import { FAMILIES, buildRecords } from '../scripts/inline-curriculum/families.mjs';

function sample(index, text = 'story', source = 'row:1') {
  return { id: `sample:${index}`, curriculum: { family: 'story_choice', shape: `quality${index}`, variant: 'v0', split_group: 'article:1' },
    semantics: { root: 'choose.nl', files: { 'choose.nl': 'instructions' }, inputs: { question: 'why?' }, folder_files: { 'story.md': text },
      expected: 'A', expected_files: { 'story.md': text }, oracle: 'exact' },
    dataset: 'quality', dataset_records: [source], source_revisions: ['pinned-v1'], generation: { source_snapshot: 'export-v1' } };
}

test('source identity uses visible request and provenance, not sampling index', () => {
  const first = identifySourceCase(sample(0));
  assert.equal(first.id, identifySourceCase(sample(99)).id);
  assert.notEqual(first.id, identifySourceCase(sample(0, 'different story')).id);
  assert.notEqual(first.id, identifySourceCase(sample(0, 'story', 'row:2')).id);
  assert.equal(first.curriculum.split_group, 'article:1');
  assert.equal(first.generation.source_case_identity, 'visible-source-v1');
});

test('conflicting gold cannot disguise one visible task as a new identity', () => {
  const first = sample(0), other = sample(1);
  other.semantics.expected = 'D';
  other.semantics.expected_files = { 'story.md': 'wrong gold bytes' };
  other.semantics.oracle = { level: 'judged' };
  assert.equal(identifySourceCase(first).id, identifySourceCase(other).id);
});

test('object key order does not change source identity; changed snapshot does', () => {
  const first = sample(0), other = sample(1);
  other.semantics = Object.fromEntries(Object.entries(other.semantics).reverse());
  assert.equal(identifySourceCase(first).id, identifySourceCase(other).id);
  other.generation.source_snapshot = 'export-v2';
  assert.notEqual(first.id, identifySourceCase(other).id);
});

test('builder fails closed on conflicting labels rather than dropping a duplicate', () => {
  FAMILIES.identity_fixture = { source: 'fixture', contentIdentity: true, build: (_seed, index) => {
    const row = sample(index);
    if (index) row.semantics.expected = 'D';
    return [row];
  } };
  try {
    assert.throws(() => buildRecords({ seed: 1, shapes: 2, families: ['identity_fixture'], hints: false }), /conflicting source oracle/);
  } finally { delete FAMILIES.identity_fixture; }
});
