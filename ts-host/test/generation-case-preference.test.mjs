import test from 'node:test';
import assert from 'node:assert/strict';
import {
  preferCurrentGenerationCase,
} from '../../scripts/prefer_current_generation_cases.mjs';
import {
  applyEvidenceScaleContract,
  applyTatqaNumericContract,
  applyReviewedTatqaUnitContract,
} from '../scripts/inline-curriculum/directory-sources.mjs';

const clone = value => structuredClone(value);
const generic = () => ({
  id: 'directory-case:one', source: 'commitpack', split: 'train', source_ids: ['source-one'],
  source_groups: ['repo:one', 'repository:one'], source_revisions: ['rev-one'], license: 'MIT',
  gold_sources: ['commitpack:original-gold'],
  external_source: { source: 'commitpack', source_id: 'source-one', original_split: 'train', revision: 'rev-one',
    snapshot_sha256: 'snapshot', license: 'MIT', files: [{ url: 'https://example.test/data', sha256: 'file-hash' }] },
  semantics: { root: 'process_workspace.nl', files: { 'process_workspace.nl': 'instructions' },
    folder_files: { 'input.txt': 'visible text' }, expected: '"gold"', expected_files: { 'answer.txt': 'gold\n' },
    inputs: undefined },
});

const tatqa = () => ({
  id: 'inline-curriculum:source_tatqa:shape:v1', source: 'tatqa', split: 'train', source_ids: ['source-tatqa-one'],
  source_groups: ['tatqa:context:one'], source_revisions: ['revision-one'], license: 'CC-BY-4.0',
  gold_sources: ['tatqa:original-gold', 'native-file-replay'],
  external_source: { source: 'tatqa', source_id: 'source-tatqa-one', original_split: 'train', revision: 'revision-one',
    snapshot_sha256: 'snapshot', license: 'CC-BY-4.0', files: [{ url: 'https://example.test/tatqa.json', sha256: 'file-hash' }] },
  generation: { generator: 'natlang.directory_source_adapter/1' },
  semantics: {
    root: 'process_workspace.nl',
    files: { 'process_workspace.nl': '---\nargs: {}\nreturns: "string"\nkind: directory-reducer\n---\nCompute the numeric answer. Use table.json and notes/. Preserve source files.\n' },
    folder_files: { 'table.json': '[["2019","2018"],["2.1","1.1"]]' },
    expected: '{"answer":"1","scale":""}',
    expected_files: { 'answer.json': '{"answer":"1","scale":""}\n' },
    oracle: { normalization: 'json-string-record' }, files_oracle: { compare: 'json-string-record' },
  },
});

function currentTatqa(old) {
  const current = clone(old);
  applyEvidenceScaleContract(current);
  applyTatqaNumericContract(current);
  applyReviewedTatqaUnitContract(current);
  return current;
}

test('keeps a non-singleton source-group lineage and accepts only identical source/gold candidate', () => {
  const old = generic(), current = clone(old);
  const decision = preferCurrentGenerationCase(old, [current]);
  assert.equal(decision.status, 'preferred_current');
  assert.deepEqual(decision.lineage.transformations, []);
  const changedGold = clone(current);
  changedGold.semantics.expected = 'different';
  assert.equal(preferCurrentGenerationCase(old, [changedGold]).reason, 'source_group_revision_files_or_gold_changed');
});

test('reconstructs the approved current TATQA contracts and blocks unregistered prompt changes', () => {
  const old = tatqa(), current = currentTatqa(old);
  const decision = preferCurrentGenerationCase(old, [current]);
  assert.equal(decision.status, 'preferred_current');
  assert.ok(decision.lineage.transformations.includes('tatqa-evidence-scale-and-numeric-contracts'));
  const unregistered = clone(current);
  unregistered.semantics.files[unregistered.semantics.root] += 'Unreviewed prompt clause.\n';
  assert.equal(preferCurrentGenerationCase(old, [unregistered]).reason, 'unregistered_or_unexpected_semantic_change');
});

test('rejects an unregistered TATQA display-prompt mutation', () => {
  const old = tatqa(), unregistered = currentTatqa(old);
  unregistered.id += ':unreviewed-proportion-display';
  unregistered.semantics.files[unregistered.semantics.root] += 'Use a dimensionless percentage.\n';
  assert.equal(preferCurrentGenerationCase(old, [unregistered]).reason, 'unregistered_or_unexpected_semantic_change');
});

test('marks source and generation holds, missing candidates, and duplicate candidates explicitly', () => {
  const old = generic(), current = clone(old);
  assert.equal(preferCurrentGenerationCase(old, []).status, 'unresolved');
  assert.equal(preferCurrentGenerationCase(old, [current, clone(current)]).reason, 'ambiguous_canonical_source_candidates');
  assert.equal(preferCurrentGenerationCase(old, [current], { sourceHeld: () => 'source_review_pending' }).status, 'held');
  assert.equal(preferCurrentGenerationCase(old, [current], { generationHeld: () => 'generation_held' }).status, 'held');
});
