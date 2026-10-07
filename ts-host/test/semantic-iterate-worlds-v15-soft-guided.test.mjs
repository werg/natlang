import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const repo = resolve(import.meta.dirname, '../..');
const readableBuilder = join(repo, 'ts-host/scripts/inline-curriculum/build-semantic-iterate-worlds-v15-soft-readable.mjs');
const guidedBuilder = join(repo, 'ts-host/scripts/inline-curriculum/build-semantic-iterate-worlds-v15-soft-guided.mjs');
const readRows = async path => (await readFile(path, 'utf8')).trimEnd().split('\n').map(JSON.parse);

test('guided V15 source preserves authored data and declares actual Neuralese soft state', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'semantic-v15-soft-guided-'));
  const baseline = join(temp, 'baseline');
  const candidate = join(temp, 'candidate');
  try {
    execFileSync(process.execPath, [readableBuilder, '--out', baseline], { cwd: repo, stdio: 'pipe' });
    execFileSync(process.execPath, [guidedBuilder, '--out', candidate], { cwd: repo, stdio: 'pipe' });
    const [baseRows, rows] = await Promise.all([
      readRows(join(baseline, 'source.cases.jsonl')),
      readRows(join(candidate, 'source.cases.jsonl')),
    ]);
    assert.equal(rows.length, 12);
    assert.deepEqual(rows.map(row => [row.source_groups, row.split, row.semantics.expected]),
      baseRows.map(row => [row.source_groups, row.split, row.semantics.expected]));

    for (const row of rows) {
      const code = row.curriculum.reference.root[0][1].code;
      const rootFile = row.semantics.files['reconcile_scoped_evidence.nl'];
      const contract = JSON.parse(row.semantics.folder_files['task.json']).output_contract;
      assert.equal(row.curriculum.reference.children.length, 6);
      assert.equal(row.curriculum.reference.children.filter(child => child.expected_reads?.length).length, 4);
      assert.equal(row.curriculum.minimum_sequence.length, 6);
      assert.match(code, /type Progress = \{ pass: number; notes: Neuralese<string> \}/);
      assert.match(code, /nl\.with<Neuralese<string>>/);
      assert.match(code, /nl\.with<Draft>/);
      assert.match(code, /task\.passes\[progress\.pass\]/);
      assert.match(code, /folder\.file\(current\.evidence_path\)/);
      assert.doesNotMatch(code, /folder\.file\([^)]*pass-0[234]/);
      assert.doesNotMatch(code, /type Neuralese\s*=|type Neuralese</);
      assert.match(rootFile, /Suggested eval scaffold|Use this scaffold/);
      assert.match(rootFile, /type Progress = \{ pass: number; notes: Neuralese<string> \}/);
      assert.match(rootFile, /Do not read evidence until the current iterateOn step/);
      assert.doesNotMatch(rootFile, /No opaque|opaque-content/i);
      assert.equal(contract.carry_forward.includes('actual carried draft'), false);
      assert.doesNotMatch(`${code}\n${rootFile}`, /world\.passStates|noteFor\(|Starting state:.*(?:SV-|\$[0-9])/s);
      assert.doesNotMatch(`${code}\n${rootFile}`, /\b(?:SV-408|AM-408|FA-408|CA-73|AC-73A)\b/);
      if (!contract.final_field_enums) assert.equal(Object.hasOwn(contract, 'enum_contract'), false);
      assert.equal(row.generation.capture_contract.iterative_children.includes('nl.with<Neuralese<string>>'), true);
      assert.equal(row.generation.capture_contract.seed.includes('Neuralese<string>'), true);
      assert.equal(row.generation.teacher_observations, undefined);
    }
    const course = rows.find(row => row.curriculum.shape.includes('course_accommodation_version'));
    assert.match(course.semantics.folder_files['pass-04-coordinator.md'], /accommodation plan AC-73 and its signed addendum AC-73A/);
    const vaccine = rows.find(row => row.curriculum.shape.includes('vaccine_logger_correction'));
    const vaccineTask = JSON.parse(vaccine.semantics.folder_files['task.json']);
    assert.match(vaccineTask.instruction, /Only decision\.json is an allowed write target/);
    assert.deepEqual(vaccineTask.output_contract.writable_paths, ['decision.json']);
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
});
