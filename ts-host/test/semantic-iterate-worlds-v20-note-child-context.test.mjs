import test from 'node:test';
import assert from 'node:assert/strict';
import { makeGuidedSoftIterateCase } from '../scripts/inline-curriculum/semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { worlds } from '../scripts/inline-curriculum/semantic-iterate-reducers-v20-counterfactual-data.mjs';

test('guided note children receive decision semantics, not parent file-write instructions', () => {
  const world = worlds.find(candidate => candidate.slug.includes('px-authorized-direct-id-excluded'));
  assert.ok(world);
  const row = makeGuidedSoftIterateCase(world, 0, { revision: 'note-context-boundary-check' });
  const code = row.curriculum.reference.root[0][1].code;
  const stepStart = code.indexOf('const revise = async');
  const loopEnd = code.indexOf('const completed =');
  const stepCode = code.slice(stepStart, loopEnd);
  const finalStart = code.indexOf('const interpret:');
  const finalEnd = code.indexOf('const finalDraft =');
  const finalCode = code.slice(finalStart, finalEnd);

  assert.ok(stepStart >= 0 && loopEnd > stepStart);
  assert.match(stepCode, /decisionContext/);
  assert.match(stepCode, /passConstraint/);
  assert.match(stepCode, /allowedFields/);
  assert.match(stepCode, /This note child does not write files; the parent owns the final decision\.json write\./);
  assert.match(stepCode, /folder\.snapshot\(\)\.file\(current\.evidence_path\)/);
  assert.match(stepCode, /which is a read-only snapshot/);
  assert.match(stepCode, /For computed text, build an ordinary string in eval and return that value directly\. Marker bodies are literal text and do not interpolate JavaScript variables\./);
  assert.match(code, /Return ordinary prose as the Neuralese<string> result/);
  assert.doesNotMatch(stepCode, /task\.instruction|output_contract|outputPath|writable_paths|evidence_files/);

  assert.match(code, /format: task\.output_contract\.format/);
  assert.match(code, /fields: task\.output_contract\.fields/);
  assert.match(code, /decision_rule: task\.output_contract\.decision_rule/);
  assert.match(finalCode, /decisionContext/);
  assert.doesNotMatch(finalCode, /task\.instruction|outputPath|writable_paths|evidence_files/);
  assert.match(finalCode, /This interpreter returns a value to its caller; it does not write files\./);

  assert.match(code, /await folder\.file\(task\.output_path\)\.writeText\(JSON\.stringify\(finalDraft\)\)/);
  assert.match(code, /const saved = await folder\.file\(task\.output_path\)\.readJson\(\)/);
});

test('saved .with children rebind the same narrow note context', () => {
  const world = worlds.find(candidate => candidate.slug.includes('lab-authorized-permit-tie'));
  assert.ok(world);
  const row = makeGuidedSoftIterateCase(world, 0, { revision: 'saved-note-context-check', savedWith: true });
  const code = row.curriculum.reference.root[0][1].code;
  assert.match(code, /const stepTemplate = nl\.with<[\s\S]*?\{ current, decisionContext \}\)`/);
  assert.match(code, /const step = stepTemplate\.with\(\{ current, decisionContext \}\)/);
  const stepStart = code.indexOf('const revise = async');
  const finalStart = code.indexOf('const completed =');
  const stepCode = code.slice(stepStart, finalStart);
  assert.match(stepCode, /folder\.snapshot\(\)\.file\(current\.pass\.evidence_path\)/);
  assert.match(stepCode, /which is a read-only snapshot/);
  assert.match(stepCode, /For computed text, build an ordinary string in eval and return that value directly/);
  assert.doesNotMatch(stepCode, /task\.instruction|output_contract|outputPath|writable_paths|evidence_files/);
});
