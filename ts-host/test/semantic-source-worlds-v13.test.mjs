import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';

const repo = resolve(import.meta.dirname, '../..');
const builder = join(repo, 'ts-host/scripts/inline-curriculum/build-semantic-source-worlds-v13.mjs');

test('V13 builder emits balanced, independently grouped source worlds and CPU-only proofs', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'semantic-v13-'));
  const out = join(temp, 'candidate');
  try {
    execFileSync(process.execPath, [builder, '--out', out], { cwd: repo, stdio: 'pipe' });
    const [sourceText, qualityText, proofText, manifestText] = await Promise.all([
      readFile(join(out, 'source.cases.jsonl'), 'utf8'),
      readFile(join(out, 'source-quality-review.json'), 'utf8'),
      readFile(join(out, 'scripted-cpu-proof.json'), 'utf8'),
      readFile(join(out, 'source-manifest.json'), 'utf8'),
    ]);
    const cases = sourceText.trimEnd().split('\n').map(JSON.parse);
    const quality = JSON.parse(qualityText);
    const proof = JSON.parse(proofText);
    const manifest = JSON.parse(manifestText);
    const sha = createHash('sha256').update(sourceText).digest('hex');

    assert.equal(cases.length, 24);
    assert.equal(cases.filter((item) => item.split === 'train').length, 12);
    assert.equal(cases.filter((item) => item.split === 'test').length, 12);
    assert.equal(new Set(cases.map((item) => item.id)).size, 24);
    assert.equal(new Set(cases.map((item) => item.source_groups[0])).size, 24);
    assert.equal(cases.every((item) => item.source_groups.length === 1 && item.source_ids.length === 1), true);
    assert.equal(cases.filter((item) => item.curriculum.mode === 'single_call' && item.curriculum.iterate !== 'required').length, 12);
    assert.equal(cases.filter((item) => item.curriculum.iterate === 'required').length, 12);
    assert.equal(quality.source_cases_sha256, sha);
    assert.equal(proof.source_cases_sha256, sha);
    assert.equal(manifest.source_cases_sha256, sha);
    assert.equal(proof.model_calls, 0);
    assert.equal(proof.provider_calls, 0);
    assert.equal(proof.teacher_trajectories, 0);
    assert.equal(proof.admission_granted, false);

    const positiveRankCounts = { 1: 0, 2: 0, 3: 0, 4: 0 };
    const positiveRankCountsBySplit = {
      train: { 1: 0, 2: 0, 3: 0, 4: 0 },
      test: { 1: 0, 2: 0, 3: 0, 4: 0 },
    };
    for (const item of cases.filter((row) => row.curriculum.iterate !== 'required')) {
      const task = JSON.parse(item.semantics.folder_files['task.json']);
      const ids = item.semantics.expected;
      const records = Object.entries(item.semantics.folder_files)
        .filter(([path]) => path.startsWith('records/'));
      assert.equal(records.length, 4);
      assert.equal(ids.length, 1);
      assert.deepEqual(item.semantics.folder_files['selection.json'], '[]');
      assert.deepEqual(item.semantics.expected_files['selection.json'], JSON.stringify(ids));
      assert.equal(task.criterion, item.semantics.inputs.criterion);
      assert.equal(task.domain, item.semantics.inputs.domain);
      assert.equal(task.window, item.semantics.inputs.window);
      assert.equal(task.decision_rule, item.semantics.inputs.decisionRule);
      const children = item.curriculum.reference.children;
      assert.equal(children.length, 4);
      assert.equal(children.every((child) => child.calls[0][0] === 'eval' && child.calls[0][1].code === 'return await file.readText();'), true);
      assert.equal(children.filter((child) => child.calls[1][1].value === true).length, 1);
      assert.deepEqual(children.filter((child) => child.calls[1][1].value === true).map((child) => child.match.replace(/\.md$/, '')).sort(), ids);
      assert.match(task.window, /20\d\d/);
      const recordIds = records.map(([path]) => path.split('/').at(-1).replace(/\.md$/, '')).sort();
      const positiveRank = recordIds.indexOf(ids[0]) + 1;
      assert.ok(positiveRank >= 1 && positiveRank <= 4);
      positiveRankCounts[positiveRank]++;
      positiveRankCountsBySplit[item.split][positiveRank]++;
    }
    assert.deepEqual(positiveRankCounts, { 1: 3, 2: 3, 3: 3, 4: 3 });
    assert.deepEqual(positiveRankCountsBySplit, {
      train: { 1: 2, 2: 1, 3: 2, 4: 1 },
      test: { 1: 1, 2: 2, 3: 1, 4: 2 },
    });
    assert.deepEqual(quality.checks.positive_lexical_position_counts, positiveRankCounts);
    assert.deepEqual(quality.checks.positive_lexical_position_counts_by_split, positiveRankCountsBySplit);

    for (const item of cases.filter((row) => row.curriculum.iterate === 'required')) {
      const task = JSON.parse(item.semantics.folder_files['task.json']);
      const fields = Object.keys(task.output_contract.fields);
      assert.equal(fields.length, 4);
      assert.deepEqual(Object.keys(item.semantics.expected).sort(), [...fields].sort());
      assert.deepEqual(Object.keys(item.semantics.expected_files), Object.keys(item.semantics.folder_files));
      assert.equal(item.curriculum.reference.children.length, 3);
      assert.equal(item.curriculum.reference.children.every((child) => child.calls[0][0] === 'eval' && child.calls[0][1].code === 'return await packet.readText();'), true);
      assert.equal(item.curriculum.reference.children.every((child) => Object.keys(child.calls[1][1].value).length === 4), true);
      assert.deepEqual(JSON.parse(item.semantics.expected_files['selection.json']), item.semantics.expected);
      for (const value of Object.values(item.semantics.expected)) {
        assert.ok(item.semantics.folder_files['packet.md'].toLowerCase().includes(value.toLowerCase()));
      }
    }

    const fileCase = (slug) => cases.find((item) => item.curriculum.shape === `v13-${slug}-filehandle`);
    const record = (slug, id) => fileCase(slug).semantics.folder_files[`records/${id}.md`];
    const recordsText = (slug) => Object.entries(fileCase(slug).semantics.folder_files)
      .filter(([path]) => path.startsWith('records/')).map(([, text]) => text);
    const interpreter = fileCase('community_interpreter_roster');
    const interpreterTask = JSON.parse(interpreter.semantics.folder_files['task.json']);
    assert.match(interpreterTask.window, /assessment as of 11 January 2027/);
    assert.match(interpreterTask.window, /approval may precede this assessment date/);
    assert.doesNotMatch(interpreterTask.window, /session approval on/);
    const interpreterPositive = interpreter.semantics.folder_files[`records/${interpreter.semantics.expected[0]}.md`];
    assert.match(interpreterPositive, /approved the roster on 10 January 2027/);
    assert.match(interpreterPositive, /current through 30 June 2027, including 11 January/);
    const licenseTask = JSON.parse(fileCase('independent_publishing_license').semantics.folder_files['task.json']);
    assert.match(licenseTask.criterion, /North Coast region/);
    assert.ok(recordsText('independent_publishing_license').some(text => /South Islands, not North Coast/.test(text)));
    assert.ok(recordsText('oral_history_web_release').some(text => /Davi Sol signed consent for OHR-\d+ permitting digitization but expressly excluding public web access/.test(text)));
    const oralNegative = recordsText('oral_history_web_release').find(text => text.includes('Davi Sol'));
    assert.ok(oralNegative);
    assert.doesNotMatch(oralNegative, /permits digitization and public web access/);
    assert.match(record('nonprofit_invoice_release', 'NIR-410'), /invoice amount is \$39,000/);
    assert.match(record('nonprofit_invoice_release', 'NIR-411'), /invoice amount is \$36,000/);
    assert.match(record('nonprofit_invoice_release', 'NIR-413'), /invoice amount is \$15,500/);
    assert.ok(recordsText('accessible_shelter_opening').some(text => /route was operational; the lift alone failed/.test(text)));
    assert.ok(recordsText('public_library_translation').some(text => /Requested language Koro is included/.test(text)));
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
});
