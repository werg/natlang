import test from 'node:test';
import assert from 'node:assert/strict';
import { Folder, folderToData } from '../dist/index.js';

test('record layouts round trip moves, edits, deletions and new outputs', async () => {
  const layout = { id: 'id', path: 'mail/{label}/{id}.md', body: 'text' };
  const folder = Folder.fromData([
    { id: 'a', label: 'inbox', subject: 'Hello', text: 'Body A' },
    { id: 'b', label: 'inbox', subject: 'Question', text: 'Body B' },
  ], layout);
  await folder.file('mail/inbox/a.md').moveTo('mail/archive/a.md');
  await folder.file('mail/archive/a.md').editText('Body A', 'Updated');
  await folder.file('mail/inbox/b.md').remove();
  folder.writeText('REPORT.md', 'One archived');
  const result = await folderToData(folder);
  assert.deepEqual(result.records, [{ id: 'a', label: 'archive', subject: 'Hello', text: 'Updated' }]);
  assert.deepEqual(result.deleted, ['b']);
  assert.deepEqual(result.unknown, ['REPORT.md']);
  assert.equal(result.outputs['REPORT.md'], 'One archived');
});

test('layout rejects duplicate identity and unsafe paths', () => {
  const layout = { id: 'id', path: 'items/{id}.md' };
  assert.throws(() => Folder.fromData([{ id: 'x' }, { id: 'x' }], layout), /duplicate record id/);
  assert.throws(() => Folder.fromData([{ id: '../escape' }], layout), /invalid path component/);
  const distinct = Folder.fromData([{ id: 'x', group: 'a' }, { id: 'x', group: 'b' }],
    { ...layout, path: 'items/{group}/{id}.md', disambiguation: 'group' });
  assert.equal(distinct.listFiles().length, 2);
});

test('CSV table is a regenerated read-only view of record files', async () => {
  const folder = Folder.fromData([{ id: 'a', label: 'inbox', body: 'text' }],
    { id: 'id', path: 'mail/{label}/{id}.md', table: 'all.csv' });
  assert.match(await folder.readText('all.csv'), /a,inbox,text/);
  await folder.file('mail/inbox/a.md').moveTo('mail/archive/a.md');
  assert.match(await folder.readText('all.csv'), /a,archive,text/);
  assert.throws(() => folder.writeText('all.csv', 'replacement'), /read-only/);
  assert.throws(() => folder.remove('all.csv'), /read-only/);
});

test('data source keeps a stable snapshot with accurate metadata', async () => {
  const records = [{ id: 'one', body: 'é' }];
  const folder = Folder.fromData(records, { id: 'id', path: 'items/{id}.md' });
  const size = (await folder.stat('items/one.md')).bytes;
  records[0].body = 'changed after import';
  const content = await folder.readText('items/one.md');
  assert.match(content, /é/);
  assert.equal(size, new TextEncoder().encode(content).length);
});

test('writable CSV layout round trips typed edits, quoted text and row deletion', async () => {
  const layout = { id: 'id', table: 'records.csv', writable: 'table' };
  const folder = Folder.fromData([
    { id: 'a', label: 'inbox', amount: 12, active: true, note: 'hello, world' },
    { id: 'b', label: 'inbox', amount: 5, active: false, note: 'line one\nline two' },
  ], layout);
  assert.deepEqual(folder.listFiles().map(entry => entry.path), ['records.csv']);
  assert.match(await folder.readText('records.csv'), /"hello, world"/);
  folder.writeText('records.csv', 'id,label,amount,active,note\na,archive,19,false,"revised, text"\n');
  folder.writeText('REPORT.md', 'One archived');
  assert.deepEqual(await folderToData(folder), {
    records: [{ id: 'a', label: 'archive', amount: 19, active: false, note: 'revised, text' }],
    deleted: ['b'], unknown: ['REPORT.md'], outputs: { 'REPORT.md': 'One archived' },
  });
  assert.throws(() => Folder.fromData([{ id: 'a' }], { id: 'id', writable: 'table' }), /writable table/);
});
