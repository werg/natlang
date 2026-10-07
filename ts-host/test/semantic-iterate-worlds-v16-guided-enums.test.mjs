import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const repo = resolve(import.meta.dirname, '../..');
const guidedBuilder = join(repo, 'ts-host/scripts/inline-curriculum/build-semantic-iterate-worlds-v15-soft-guided.mjs');
const enumBuilder = join(repo, 'ts-host/scripts/inline-curriculum/build-semantic-iterate-worlds-v16-guided-enums.mjs');
const readRows = async path => (await readFile(path, 'utf8')).trimEnd().split('\n').map(JSON.parse);

test('V16 guided source reuses authored final enums as literal Draft types without changing worlds', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'semantic-v16-guided-enums-'));
  const baseline = join(temp, 'baseline');
  const candidate = join(temp, 'candidate');
  try {
    execFileSync(process.execPath, [guidedBuilder, '--out', baseline], { cwd: repo, stdio: 'pipe' });
    execFileSync(process.execPath, [enumBuilder, '--out', candidate], { cwd: repo, stdio: 'pipe' });
    const [baseRows, rows] = await Promise.all([
      readRows(join(baseline, 'source.cases.jsonl')),
      readRows(join(candidate, 'source.cases.jsonl')),
    ]);
    assert.equal(rows.length, 12);
    assert.deepEqual(rows.map(row => [row.source_groups, row.split, row.semantics.expected]),
      baseRows.map(row => [row.source_groups, row.split, row.semantics.expected]));

    for (const row of rows) {
      const code = row.curriculum.reference.root[0][1].code;
      const contract = JSON.parse(row.semantics.folder_files['task.json']).output_contract;
      const declarations = code.match(/type Draft = \{ ([^\n]+) \};/);
      const initialDeclarations = code.match(/type InitialDraft = \{ ([^\n]+) \};/);
      assert.ok(declarations, row.id);
      assert.ok(initialDeclarations, row.id);
      assert.ok(Object.keys(contract.final_field_enums).length > 0, row.id);
      assert.match(contract.enum_contract, /only the final Draft/);
      assert.match(code, /nl\.with<Draft>/);
      assert.match(code, /type InitialDraft = \{/);
      assert.match(code, /initialDraft: InitialDraft/);
      assert.match(code, /Follow every declared field format exactly/);
      assert.match(code, /one bare listed literal only/);
      assert.match(code, /do not add a prose explanation, unit label/);
      assert.match(row.semantics.files['reconcile_scoped_evidence.nl'], /returns: "\{/);
      for (const [field, values] of Object.entries(contract.final_field_enums)) {
        const union = values.map(value => JSON.stringify(value)).join(' | ');
        assert.ok(declarations[1].includes(`${field}: ${union}`), `${row.id}: ${field} literal union missing`);
      }
      for (const [field, values] of Object.entries(contract.intermediate_field_enums)) {
        const union = values.map(value => JSON.stringify(value)).join(' | ');
        assert.ok(initialDeclarations[1].includes(`${field}: ${union}`), `${row.id}: ${field} intermediate union missing`);
      }
      assert.doesNotMatch(code, /world\.passStates|noteFor\(/);
    }

    const settlement = rows.find(row => row.curriculum.shape.includes('demand_response_meter_settlement'));
    const settlementCode = settlement.curriculum.reference.root[0][1].code;
    assert.match(settlementCode, /decision: "pay" \| "hold"/);
    assert.match(settlementCode, /referenceReadsKwh: string/);
    assert.match(settlementCode, /baselineKwh: string/);
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
});
