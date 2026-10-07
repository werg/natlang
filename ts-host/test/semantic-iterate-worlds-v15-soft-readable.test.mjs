import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const repo = resolve(import.meta.dirname, '../..');
const baseBuilder = join(repo, 'ts-host/scripts/inline-curriculum/build-semantic-iterate-worlds-v15.mjs');
const softBuilder = join(repo, 'ts-host/scripts/inline-curriculum/build-semantic-iterate-worlds-v15-soft-readable.mjs');
const readRows = async path => (await readFile(path, 'utf8')).trimEnd().split('\n').map(JSON.parse);

test('readable soft V15 source preserves canonical worlds and six-child iterate topology', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'semantic-v15-soft-readable-'));
  const baseline = join(temp, 'baseline');
  const candidate = join(temp, 'candidate');
  try {
    execFileSync(process.execPath, [baseBuilder, '--out', baseline], { cwd: repo, stdio: 'pipe' });
    execFileSync(process.execPath, [softBuilder, '--out', candidate], { cwd: repo, stdio: 'pipe' });
    const [baseRows, rows] = await Promise.all([
      readRows(join(baseline, 'source.cases.jsonl')),
      readRows(join(candidate, 'source.cases.jsonl')),
    ]);
    assert.equal(rows.length, 12);
    assert.deepEqual(rows.map(row => [row.source_groups, row.split, row.semantics.expected]),
      baseRows.map(row => [row.source_groups, row.split, row.semantics.expected]));

    for (const row of rows) {
      assert.equal(row.curriculum.reference.children.length, 6);
      assert.equal(row.curriculum.reference.children.filter(child => child.expected_reads?.length).length, 4);
      assert.match(row.generation.capture_contract.final_interpreter, /readable prose/);
      assert.doesNotMatch(JSON.stringify(row.generation.capture_contract), /opaque/i);
      const code = row.curriculum.reference.root[0][1].code;
      assert.match(code, /PriorNotes is readable accumulated prose/);
      assert.match(code, /Read its actual text directly/);
      assert.doesNotMatch(code, /opaque/i);
      assert.doesNotMatch(code, /not a crisp draft/i);
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
