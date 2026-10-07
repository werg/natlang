import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const repo = resolve(import.meta.dirname, '../..');
const builder = join(repo, 'ts-host/scripts/inline-curriculum/build-semantic-iterate-worlds-v15-enums.mjs');
const v8 = join(repo, 'runs/neuralese-semantic-iterate-v15-source-candidate-20261007-v8/source.cases.jsonl');
const rowsFrom = async path => (await readFile(path, 'utf8')).trimEnd().split('\n').map(JSON.parse);

test('V15 enum variant preserves V8 groups, splits, gold and exposes placeholder-capable Draft enums', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'semantic-iterate-v15-enums-'));
  const out = join(temp, 'candidate');
  try {
    execFileSync(process.execPath, [builder, '--out', out], { cwd: repo, stdio: 'pipe' });
    const [rows, parent] = await Promise.all([rowsFrom(join(out, 'source.cases.jsonl')), rowsFrom(v8)]);
    assert.equal(rows.length, 12);
    assert.deepEqual(rows.map(row => [row.source_groups, row.split, row.semantics.expected]),
      parent.map(row => [row.source_groups, row.split, row.semantics.expected]));
    const byShape = slug => rows.find(row => row.curriculum.shape.includes(slug));
    const route = byShape('accessible_route_substitution');
    const routeTask = JSON.parse(route.semantics.folder_files['task.json']);
    assert.deepEqual(routeTask.output_contract.intermediate_field_enums.routeOperational, ['pending', 'yes', 'no']);
    assert.deepEqual(routeTask.output_contract.final_field_enums.routeOperational, ['yes', 'no']);
    assert.match(route.curriculum.reference.root[0][1].code, /routeOperational: "pending" \| "yes" \| "no"/);
    assert.equal(route.semantics.expected.routeOperational, 'yes');

    const archive = byShape('archive_channel_consent');
    const archiveTask = JSON.parse(archive.semantics.folder_files['task.json']);
    assert.deepEqual(archiveTask.output_contract.intermediate_field_enums.requestedChannel, ['unknown', 'public web', 'reading room']);
    assert.deepEqual(archiveTask.output_contract.final_field_enums.requestedChannel, ['public web', 'reading room']);
    const course = byShape('course_accommodation_version');
    assert.match(course.semantics.folder_files['pass-04-coordinator.md'], /AC-73A and BIO-42 version C8/);
    const vaccine = byShape('vaccine_logger_correction');
    const vaccineTask = JSON.parse(vaccine.semantics.folder_files['task.json']);
    assert.deepEqual(Object.keys(vaccineTask.output_contract.final_field_enums), ['decision']);
    assert.deepEqual(vaccineTask.output_contract.final_field_enums.decision, ['release', 'quarantine']);
    assert.deepEqual(vaccine.semantics.folder_files['task.json'] && JSON.parse(vaccine.semantics.folder_files['task.json']).output_contract.writable_paths, ['decision.json']);
    assert.match(vaccine.curriculum.reference.root[0][1].code, /decision: "pending" \| "release" \| "quarantine"/);
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
});
