import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const repo = resolve(import.meta.dirname, '../..');
const builder = join(repo, 'ts-host/scripts/inline-curriculum/build-semantic-source-worlds-v14.mjs');

test('V14 source contracts bind full file IDs, single unit values, and mediation consent', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'semantic-v14-'));
  const out = join(temp, 'candidate');
  try {
    execFileSync(process.execPath, [builder, '--out', out], { cwd: repo, stdio: 'pipe' });
    const sourceText = await readFile(join(out, 'source.cases.jsonl'), 'utf8');
    const cases = sourceText.trimEnd().split('\n').map(JSON.parse);
    const files = cases.filter((row) => row.curriculum.shape.endsWith('-nested-filehandle'));
    const iterates = cases.filter((row) => row.curriculum.shape.endsWith('-iterate'));

    assert.equal(files.length, 12);
    assert.equal(iterates.length, 12);

    for (const row of files) {
      const task = JSON.parse(row.semantics.folder_files['task.json']);
      const match = task.output_contract.item.match(/complete record ID at the start of each file text, before the colon/i);
      assert.ok(match, `${row.id}: item contract identifies the full ID in file text`);
      assert.match(task.output_contract.item, /record- filename prefix is not part of the ID/i);
      const idsFromText = Object.entries(row.semantics.folder_files)
        .filter(([path]) => /^record-.*\.md$/.test(path))
        .map(([path, text]) => {
          const expectedId = path.slice('record-'.length, -'.md'.length);
          assert.ok(text.startsWith(`${expectedId}: `), `${row.id}/${path}: text begins with its full ID`);
          return expectedId;
        })
        .sort();
      const expectedIds = row.semantics.expected;
      assert.deepEqual(JSON.parse(row.semantics.expected_files['selection.json']), expectedIds);
      assert.ok(expectedIds.every((id) => idsFromText.includes(id)));
      assert.ok(row.curriculum.reference.root[0][1].code.includes("replace(/^record-/,'')"));
    }

    const unitFields = [];
    for (const row of iterates) {
      const task = JSON.parse(row.semantics.folder_files['task.json']);
      const packet = row.semantics.folder_files['packet.md'];
      for (const [field, definition] of Object.entries(task.output_contract.fields)) {
        if (!/exactly once|single USD unit/i.test(definition)) continue;
        unitFields.push({ row, task, packet, field, definition });
      }
    }
    assert.equal(unitFields.length, 9);
    for (const { row, task, packet, field, definition } of unitFields) {
      assert.match(definition, /exactly once|single USD unit/i, `${row.id}/${field}`);
      const value = row.semantics.expected[field];
      assert.ok(packet.includes(value), `${row.id}/${field}: expected value is in packet`);
      assert.equal(row.semantics.expected_files['decision.json'], JSON.stringify(row.semantics.expected));
      const pass = task.passes.find((item) => item.allowed_fields.includes(field));
      assert.ok(pass, `${row.id}/${field}: one pass owns the field`);
      assert.match(pass.constraint, /do not append a second|single USD unit|exactly as written/i);
    }

    const mediation = iterates.find((row) => row.curriculum.shape.includes('community_mediation_referral'));
    assert.ok(mediation);
    const packet = mediation.semantics.folder_files['packet.md'];
    assert.match(packet, /Both parties signed consent form CF-117 explicitly consenting to mediation case MED-28-117 at North Quay Room 2 on meeting date 2028-11-19\./);
    const task = JSON.parse(mediation.semantics.folder_files['task.json']);
    assert.match(task.output_contract.status_policy, /same case, venue, and date/i);

    const meeting = iterates.find((row) => row.curriculum.shape.includes('public_meeting_quorum_certificate'));
    assert.ok(meeting);
    assert.match(meeting.semantics.folder_files['packet.md'], /district name "Westmere"/);
    assert.equal(meeting.semantics.expected.district, 'Westmere');
    assert.ok(meeting.source_revisions.includes('authored-semantic-source-worlds-v14/7'));
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
});
