import test from 'node:test';
import assert from 'node:assert/strict';
import { makeGuidedSoftIterateCase } from '../scripts/inline-curriculum/semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { worlds } from '../scripts/inline-curriculum/semantic-iterate-worlds-v17-data.mjs';

test('guided intermediate and final enum types do not depend on revision spelling', () => {
  const world = worlds.find(candidate => candidate.field_enums?.decision?.intermediate?.includes('pending')
    && !candidate.field_enums.decision.final.includes('pending'));
  assert.ok(world, 'V17 has a decision placeholder that is excluded from its final enum');
  const first = makeGuidedSoftIterateCase(world, 0, { revision: 'arbitrary-provenance-alpha' });
  const second = makeGuidedSoftIterateCase(world, 0, { revision: 'revision/4-not-an-enum-suffix' });
  const firstCode = first.curriculum.reference.root[0][1].code;
  const secondCode = second.curriculum.reference.root[0][1].code;
  const firstContract = JSON.parse(first.semantics.folder_files['task.json']).output_contract;
  const secondContract = JSON.parse(second.semantics.folder_files['task.json']).output_contract;

  assert.equal(firstCode, secondCode);
  assert.deepEqual(firstContract.intermediate_field_enums, secondContract.intermediate_field_enums);
  assert.deepEqual(firstContract.final_field_enums, secondContract.final_field_enums);
  assert.match(firstCode, /type InitialDraft = \{[^\n]*decision: "pending" \|/);
  assert.match(firstCode, /initialDraft: InitialDraft/);
  assert.match(firstCode, /type Draft = \{[^\n]*decision: "approve" \| "hold"/);
  assert.match(firstCode, /Follow every declared field format exactly/);
  assert.match(firstCode, /one bare listed literal only/);
  assert.doesNotMatch(firstCode.match(/type Draft = \{[^\n]+/)[0], /"pending"/);
  assert.deepEqual(first.source_groups, second.source_groups);
  assert.equal(first.split, second.split);
  assert.deepEqual(first.semantics.expected, second.semantics.expected);
  assert.deepEqual(first.semantics.expected_files, second.semantics.expected_files);
});
