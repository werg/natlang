import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildSkillAblations, summarizeSkillAblation } from '../dist/skills/index.js';

const baseline = {
  'main.nl': 'frozen target bytes',
  'skills/review/SKILL.md': '---\nname: review\ndescription: Baseline description.\nsummary: Baseline summary.\nmetadata:\n  keep: old\n---\nBaseline instructions.\n',
  'skills/review/references/rules.md': 'Baseline rules.\n',
  'skills/unchanged/SKILL.md': '---\nname: unchanged\ndescription: Same.\n---\nSame body.\n',
  'skills/format-only/SKILL.md': '---\nname: format-only\ndescription: Same description.\nmetadata:\n  keep: value\n---\nSame body.\n',
};
const selected = {
  'main.nl': 'frozen target bytes',
  'skills/review/SKILL.md': '---\nname: review\ndescription: Selected description.\nsummary: Selected summary.\nmetadata:\n  keep: new\n---\nSelected instructions.\n',
  'skills/review/references/rules.md': 'Selected rules.\n',
  'skills/unchanged/SKILL.md': '---\nname: unchanged\ndescription: Same.\n---\nSame body.\n',
  'skills/format-only/SKILL.md': '---\nname: format-only\ndescription: "Same description."\nmetadata: { keep: value }\n---\nSame body.\n',
};

test('skill ablations isolate skill removal, description metadata, and body changes', () => {
  const result = buildSkillAblations(baseline, selected);
  const removed = result.candidates.find(candidate => candidate.kind === 'leave_one_skill_out' && candidate.skillName === 'review');
  const description = result.candidates.find(candidate => candidate.kind === 'baseline_description' && candidate.skillName === 'review');
  const body = result.candidates.find(candidate => candidate.kind === 'baseline_body' && candidate.skillName === 'review');
  assert.ok(removed && description && body);
  assert.equal(Object.hasOwn(removed.files, 'skills/review/SKILL.md'), false);
  assert.equal(removed.files['main.nl'], selected['main.nl'], 'the frozen target executable is retained');
  assert.match(description.files['skills/review/SKILL.md'], /description: Baseline description/);
  assert.match(description.files['skills/review/SKILL.md'], /summary: Baseline summary/);
  assert.match(description.files['skills/review/SKILL.md'], /keep: new/, 'only discovery metadata is restored');
  assert.match(description.files['skills/review/SKILL.md'], /Selected instructions/);
  assert.match(body.files['skills/review/SKILL.md'], /Selected description/);
  assert.match(body.files['skills/review/SKILL.md'], /Baseline instructions/);
  assert.equal(body.files['skills/review/references/rules.md'], selected['skills/review/references/rules.md'],
    'support files are unchanged when only the body is restored');
  assert.ok(result.skipped.some(item => item.kind === 'baseline_body' && item.skillName === 'unchanged' && item.reason === 'unchanged'));
  assert.ok(result.skipped.some(item => item.kind === 'baseline_description' && item.skillName === 'format-only' && item.reason === 'unchanged'),
    'semantically identical YAML scalars do not create formatting-only variants');
  assert.ok(result.skipped.some(item => item.kind === 'baseline_body' && item.skillName === 'format-only' && item.reason === 'unchanged'));
  assert.doesNotMatch(JSON.stringify(result), /frozen target bytes|Selected instructions|Baseline rules/,
    'metadata and output descriptors do not leak snapshot content');
});

test('entry root must be relative and safe', () => {
  assert.throws(() => buildSkillAblations(baseline, selected, '/skills'), /safe relative path/);
  assert.throws(() => buildSkillAblations(baseline, selected, 'C:/skills'), /safe relative path/);
  assert.throws(() => buildSkillAblations(baseline, selected, 'skills/../outside'), /safe relative path/);
});

test('candidate snapshots detach binary inputs and returned file values', () => {
  const bytes = new Uint8Array([1, 2, 3]);
  const source = { ...selected, 'skills/review/blob.bin': bytes };
  const result = buildSkillAblations(baseline, source);
  bytes[0] = 9;
  const candidate = result.candidates.find(item => item.kind === 'leave_one_skill_out' && item.skillName === 'unchanged');
  assert.ok(candidate);
  const first = candidate.files['skills/review/blob.bin'];
  assert.ok(first instanceof Uint8Array);
  assert.deepEqual([...first], [1, 2, 3]);
  first[1] = 8;
  assert.deepEqual([...candidate.files['skills/review/blob.bin']], [1, 2, 3]);
});

test('effect summary distinguishes paired removal outcomes from observed host events', () => {
  const events = [
    { kind: 'skill_use', phase: 'offered', skill_name: 'review' },
    { kind: 'skill_use', phase: 'body_read', skill_name: 'review' },
    { kind: 'skill_use', phase: 'support_file_read', skill_name: 'review' },
    { kind: 'skill_use', phase: 'helper_invoked', skill_name: 'review' },
    { kind: 'skill_use', phase: 'body_read', skill_name: 'other' },
  ];
  const helpful = summarizeSkillAblation('review', [
    { baselinePassed: true, ablatedPassed: false }, { baselinePassed: true, ablatedPassed: true },
  ], events);
  assert.equal(helpful.classification, 'helpful-at-tested-context');
  assert.deepEqual(helpful.observedUse, { offered: 1, body_read: 1, support_file_read: 1, helper_invoked: 1 });
  assert.equal(helpful.interpretation, 'host_observations_only_not_cognitive_use');
  assert.equal(summarizeSkillAblation('review', [{ baselinePassed: true, ablatedPassed: true }]).classification,
    'nonessential-at-tested-context');
  assert.equal(summarizeSkillAblation('review', [{ baselinePassed: false, ablatedPassed: true }]).classification, 'unknown');
  assert.equal(summarizeSkillAblation('review', [{ baselinePassed: true }]).classification, 'unknown');
});
