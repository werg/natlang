import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as host from '../dist/index.js';
import { calibrateTeachers, frontierTeacher, routedTeacherProfile, routeTeacher, saveTeacherChoice } from '../dist/tune/teachers.js';
import { currentSplit } from '../dist/tune/pipeline.js';
import { scriptedModel } from './support/natlang.mjs';

const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
const calibration = (teacher, success, tokensPerRoot, extra = {}) => ({ schema: 'natlang.tune-teacher-calibration/v1', teacher, profile: `p-${teacher}`,
  examples: 'x', roots: 20, success, failures: 0, tokensPerRoot, msPerRoot: 100, ...extra });

test('the frontier rule compares success before cost: a cheap weak teacher never wins', () => {
  const strong = calibration('strong', 19, 4000), cheapWeak = calibration('weak', 12, 300), close = calibration('close', 18, 3000);
  assert.equal(frontierTeacher([strong, cheapWeak]).teacher, 'strong');
  assert.equal(frontierTeacher([strong, cheapWeak, close]).teacher, 'close', 'within 0.05 of the best rate, fewer tokens wins');
  const flaky = calibration('flaky', 19, 1000, { failures: 10 });
  assert.equal(frontierTeacher([strong, flaky]).teacher, 'strong', 'an unreliable teacher loses to a reliable one');
  assert.equal(frontierTeacher([]).teacher, null);
});

test('routing filters by the minimum crisply and accepts only an eligible natural-language answer', async () => {
  const items = [calibration('a', 18, 2000), calibration('b', 17, 1000), calibration('weak', 5, 10)];
  const crisp = await routeTeacher(items, { minSuccess: 0.5 });
  assert.equal(crisp.policy, 'crisp');
  assert.equal(crisp.teacher, 'b');
  let offered;
  const picked = await routeTeacher(items, { minSuccess: 0.5, mode: 'nl' }, candidates => { offered = JSON.parse(candidates); return { teacher: 'a', reason: 'best rate' }; });
  assert.deepEqual(offered.map(item => item.teacher), ['a', 'b'], 'teachers under the minimum are never offered');
  assert.equal(picked.teacher, 'a');
  assert.equal(picked.profile, 'p-a');
  const invented = await routeTeacher(items, { minSuccess: 0.5, mode: 'nl' }, () => ({ teacher: 'weak', reason: 'cheap' }));
  assert.equal(invented.teacher, 'b', 'an ineligible answer falls back to the crisp rule');
  assert.equal(invented.policy, 'crisp');
  assert.equal((await routeTeacher(items, { minSuccess: 0.99 })).teacher, null);
});

test('calibration measures each teacher once per identity and a changed actor needs recalibration', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-teachers-'));
  try {
    writeFileSync(join(root, 'shout.nl'), named('Return value in uppercase.'));
    writeFileSync(join(root, 'package.json'), '{"type":"module"}');
    const examples = Array.from({ length: 12 }, (_, index) => ({ function: 'shout.nl', input: { value: `v${index}` }, expected: `V${index}`, group: `g${index}` }));
    writeFileSync(join(root, 'tune.examples.jsonl'), examples.map(example => JSON.stringify(example)).join('\n') + '\n');
    const built = host.buildProject({ project: root, emit: false, write: false });
    assert.equal(built.ok, true, host.formatDiagnostics(built.diagnostics));
    const development = currentSplit(root).filter(example => example.partition === 'development');
    assert.ok(development.length > 0);
    let goodCalls = 0;
    const good = scriptedModel(() => { goodCalls++; return 'return value.toUpperCase()'; });
    const bad = scriptedModel(() => 'return value');
    const candidates = [{ profile: 'good', id: 'executor:good', driver: good.driver }, { profile: 'bad', id: 'executor:bad', driver: bad.driver }];
    const measured = await calibrateTeachers({ root, program: built.manifest.adaptation, candidates, examples: development });
    assert.deepEqual(measured.map(item => item.success), [development.length, 0]);
    const calls = goodCalls;
    await calibrateTeachers({ root, program: built.manifest.adaptation, candidates, examples: development });
    assert.equal(goodCalls, calls, 'a stored measurement for the same identity and roots is reused');
    const choice = await routeTeacher(measured, { minSuccess: 0.5 });
    saveTeacherChoice(root, choice);
    assert.equal(routedTeacherProfile(root, candidates), 'good');
    assert.throws(() => routedTeacherProfile(root, [{ profile: 'good', id: 'executor:good-v2' }, candidates[1]]), /changed or was added since calibration/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
