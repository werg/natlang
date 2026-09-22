import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { auditProgramRows } from '../scripts/audit-program-ir.mjs';
import { combineSft } from '../scripts/combine-sft.mjs';

const program = (id, split = 'train') => ({ id, source: 'generated', split, kind: 'lambda_graph',
  source_groups: [id], semantics: { operations: [{ op: 'invoke' }, { op: 'branch', then: [{ op: 'assign' }], else: [] }], functions: {} } });

test('native Program IR audit checks identity, split isolation, and operation mix', () => {
  const report = auditProgramRows([{ path: 'one', rows: [program('a'), program('b', 'test')] }]);
  assert.equal(report.programs, 2); assert.equal(report.groups, 2);
  assert.deepEqual(report.operators, { assign: 2, branch: 2, invoke: 2 });
  assert.throws(() => auditProgramRows([{ path: 'bad', rows: [program('a'), program('a')] }]), /duplicate program/);
  const crossed = program('b', 'test'); crossed.source_groups = ['a'];
  assert.throws(() => auditProgramRows([{ path: 'bad', rows: [program('a'), crossed] }]), /crosses splits/);
});

test('native SFT bundler preserves bytes and rejects duplicate identities', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-combine-')), a = join(root, 'a.jsonl'), b = join(root, 'b.jsonl');
  await writeFile(a, JSON.stringify({ id: 'a', completion: '<think>x</think>' }) + '\n');
  await writeFile(b, JSON.stringify({ id: 'b', completion: 'y' }) + '\n');
  const output = join(root, 'all.jsonl'), result = await combineSft(output, [a, b]);
  assert.equal(result.pairs, 2); assert.equal(result.reasoning_pairs, 1);
  assert.equal((await readFile(output, 'utf8')).split('\n').filter(Boolean).length, 2);
  const duplicate = join(root, 'dup.jsonl');
  await writeFile(duplicate, JSON.stringify({ id: 'a', completion: 'z' }) + '\n');
  await assert.rejects(() => combineSft(join(root, 'bad.jsonl'), [a, duplicate]), /duplicate SFT id/);
});
