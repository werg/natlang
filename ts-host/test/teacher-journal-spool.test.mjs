import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { acquireJournalSpool, JOURNAL_SPOOL_CASE_LIMIT, treeBytes } from '../dist/teacher/journal-spool.js';

const digest = value => createHash('sha256').update(value).digest('hex');

test('journal spool resumes crash-retained case and publishes verified bytes before retirement', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-spool-'));
  try {
    const first = await acquireJournalSpool(root, '/external/jobs:case-a');
    const partial = join(first.directory, 'case.partial.json');
    await writeFile(partial, '{"turns":[1]}\n');
    const resumed = await acquireJournalSpool(root, '/external/jobs:case-a');
    assert.equal(resumed.directory, first.directory);
    assert.equal(await readFile(partial, 'utf8'), '{"turns":[1]}\n');
    const jobs = join(root, 'published');
    await mkdir(jobs);
    const bytes = await readFile(partial);
    await resumed.publish([{ source: partial, destination: join(jobs, 'case.partial.json'), sha256: digest(bytes) }]);
    assert.equal(await readFile(join(jobs, 'case.partial.json'), 'utf8'), '{"turns":[1]}\n');
    await assert.rejects(() => readFile(partial));
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('journal spool enforces per-case bytes and six active/retained case credits', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-spool-'));
  try {
    const first = await acquireJournalSpool(root, 'case-0');
    await assert.rejects(() => first.assertCapacity(JOURNAL_SPOOL_CASE_LIMIT + 1), /cap exceeded/);
    for (let i = 1; i < 6; i++) await acquireJournalSpool(root, `case-${i}`);
    await assert.rejects(() => acquireJournalSpool(root, 'case-6'), /6 active-case credits/);
    const { readdir } = await import('node:fs/promises');
    assert.equal((await readdir(join(root, 'cases'))).length, 6, 'rejected admission leaves no empty retained case directory');
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('journal spool fails closed on altered bytes and preserves interrupted data', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-spool-'));
  try {
    const spool = await acquireJournalSpool(root, 'case-tamper');
    const source = join(spool.directory, 'evidence.jsonl');
    await writeFile(source, 'actual bytes');
    const destination = join(root, 'published', 'evidence.jsonl');
    await assert.rejects(() => spool.publish([{ source, destination, sha256: digest('different bytes') }]), /changed before publication/);
    assert.equal(await readFile(source, 'utf8'), 'actual bytes');
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('aggregate spool scan tolerates a temp file atomically renamed by another case', async () => {
  const root = '/virtual-spool';
  const dirent = (name, kind) => ({ name, isDirectory: () => kind === 'directory', isFile: () => kind === 'file' });
  const vanished = Object.assign(new Error('renamed during scan'), { code: 'ENOENT' });
  let renamed = false, caseDirectoryScans = 0;
  const bytes = await treeBytes(root, {
    readdir: async path => path === root ? [dirent('cases', 'directory'), dirent('stable.json', 'file')] :
      path === `${root}/cases` ? [dirent('case-a', 'directory')] : (() => {
        caseDirectoryScans++;
        return [dirent(renamed ? 'published.partial.json' : 'temporary.partial.json.tmp-123', 'file')];
      })(),
    stat: async path => {
      if (path.endsWith('temporary.partial.json.tmp-123')) { renamed = true; throw vanished; }
      return { size: path.endsWith('stable.json') ? 17 : 7 };
    },
  });
  assert.equal(bytes, 24);
  assert.equal(caseDirectoryScans, 2, 'a missing temporary entry triggers one full aggregate rescan');
});
