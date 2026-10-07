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

test('V17 authored reference notes carry verbatim evidence only as each pass observes it', () => {
  const world = worlds.find(candidate => candidate.source_logic?.candidates?.length === 4);
  const row = makeGuidedSoftIterateCase(world, 0, { revision: 'reference-evidence-provenance' });
  const children = row.curriculum.reference.children;
  const contract = JSON.parse(row.semantics.folder_files['task.json']).output_contract;
  assert.equal(contract.final_field_enums.eligibleIds, undefined);
  assert.match(contract.fields.eligibleIds, /exactly semicolon and one space \(; \)/);
  assert.match(contract.fields.eligibleIds, /descending priority score and then ascending complete ID/);

  for (let passIndex = 0; passIndex < world.passes.length; passIndex++) {
    const note = children[passIndex + 1].soft_output.text;
    for (let observedIndex = 0; observedIndex <= passIndex; observedIndex++) {
      const observed = world.passes[observedIndex];
      assert.ok(note.includes(`Observed source ${observed.evidence_path} (complete text): ${world.evidence[observed.evidence_path]}`));
    }
    for (let futureIndex = passIndex + 1; futureIndex < world.passes.length; futureIndex++) {
      const future = world.passes[futureIndex];
      assert.ok(!note.includes(world.evidence[future.evidence_path]));
    }
    assert.ok(!note.includes(JSON.stringify(world.passStates[passIndex])));
  }
  const priorityNote = children[2].soft_output.text;
  const auditNote = children[3].soft_output.text;
  assert.match(priorityNote, /Summary derived only from that file: Derived from this observed file, priority facts are/);
  assert.match(auditNote, /Summary derived only from that file: Derived from this observed file, candidate audit statements are/);
  const sourceScorePairs = [...world.evidence['pass-02-priority-register.md'].matchAll(/([A-Z]{3}-\d+[A-D]) has priority score (\d+)/g)];
  for (const [, itemId, score] of sourceScorePairs)
    assert.ok(priorityNote.includes(`${itemId}=${score}`));
});

test('all V17 authored reference notes preserve rank evidence without lookahead or rank enums', () => {
  for (const [worldIndex, world] of worlds.entries()) {
    const row = makeGuidedSoftIterateCase(world, worldIndex, { revision: 'arbitrary-reference-proof-revision' });
    const children = row.curriculum.reference.children;
    const contract = JSON.parse(row.semantics.folder_files['task.json']).output_contract;
    assert.equal(contract.final_field_enums.eligibleIds, undefined, world.slug);
    assert.equal(contract.intermediate_field_enums.eligibleIds, undefined, world.slug);
    assert.match(contract.fields.eligibleIds, /exactly semicolon and one space \(; \)/, world.slug);
    assert.match(contract.fields.eligibleIds, /descending priority score and then ascending complete ID/, world.slug);

    for (let passIndex = 0; passIndex < world.passes.length; passIndex++) {
      const note = children[passIndex + 1].soft_output.text;
      for (let seen = 0; seen <= passIndex; seen++) {
        const pass = world.passes[seen];
        assert.ok(note.includes(`Observed source ${pass.evidence_path} (complete text): ${world.evidence[pass.evidence_path]}`), world.slug);
      }
      for (let future = passIndex + 1; future < world.passes.length; future++)
        assert.ok(!note.includes(world.evidence[world.passes[future].evidence_path]), world.slug);
    }
    const priorityNote = children[2].soft_output.text;
    const auditNote = children[3].soft_output.text;
    const scorePairs = [...world.evidence['pass-02-priority-register.md'].matchAll(/([A-Z]{3}-\d+[A-D]) has priority score (\d+)/g)];
    const auditIds = [...world.evidence['pass-03-eligibility-audit.md'].matchAll(/([A-Z]{3}-\d+[A-D])\s+([^.]+)\./g)].map(([, id]) => id);
    assert.equal(scorePairs.length, 4, world.slug);
    assert.equal(auditIds.length, 4, world.slug);
    for (const [, id, score] of scorePairs) assert.ok(priorityNote.includes(`${id}=${score}`), world.slug);
    for (const id of auditIds) assert.ok(auditNote.includes(`${id}: `), world.slug);
  }
});
