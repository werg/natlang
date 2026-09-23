import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { acquireShards } from '../scripts/code-corpus/acquire-shards.mjs';

function mockFetch(rows) {
  const calls = [];
  const fetcher = async input => {
    const url = new URL(String(input)); calls.push(url);
    if (url.hostname === 'huggingface.co') return { ok: true, json: async () => ({ sha: 'f'.repeat(40) }) };
    const offset = Number(url.searchParams.get('offset'));
    const length = Number(url.searchParams.get('length'));
    return { ok: true, json: async () => ({ num_rows_total: rows.length, rows: rows.slice(offset, offset + length).map((row, i) => ({ row, row_idx: offset + i })) }) };
  };
  return { fetcher, calls };
}

test('acquisition commits bounded pages and resumes from the next offset with a larger limit', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'acquire-shards-')); t.after(() => rm(dir, { recursive: true, force: true }));
  const rows = Array.from({ length: 5 }, (_, i) => ({ instruction: `Do ${i}`, code: `const n${i} = ${i};`, language: 'JavaScript', id: i }));
  const first = mockFetch(rows);
  const stopped = await acquireShards({ out: dir, ids: ['tiny-codes'], limit: 2, pageSize: 2, fetcher: first.fetcher });
  assert.equal(stopped.sources['tiny-codes'].status, 'complete');
  assert.equal(stopped.sources['tiny-codes'].nextOffset, 2);
  const second = mockFetch(rows);
  const resumed = await acquireShards({ out: dir, ids: ['tiny-codes'], limit: 5, pageSize: 2, fetcher: second.fetcher });
  assert.equal(resumed.sources['tiny-codes'].nextOffset, 5);
  assert.equal(resumed.sources['tiny-codes'].totalRows, 5);
  assert.deepEqual(second.calls.filter(url => url.hostname === 'datasets-server.huggingface.co').map(url => Number(url.searchParams.get('offset'))), [2, 4]);
  const tasks = await readFile(join(dir, 'tiny-codes-000002.tasks.jsonl'), 'utf8');
  assert.equal(JSON.parse(tasks).language, 'javascript');
});

test('pause occurs after a committed shard and checksummed corruption blocks resume', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'acquire-shards-corrupt-')); t.after(() => rm(dir, { recursive: true, force: true }));
  const rows = Array.from({ length: 4 }, (_, i) => ({ instruction: `Do ${i}`, code: `const n${i} = ${i};`, language: 'TypeScript', id: i }));
  const { fetcher, calls } = mockFetch(rows);
  const paused = await acquireShards({ out: dir, ids: ['tiny-codes'], limit: 4, pageSize: 2, fetcher, shouldStop: () => calls.filter(url => url.hostname === 'datasets-server.huggingface.co').length >= 1 });
  assert.equal(paused.sources['tiny-codes'].status, 'paused');
  assert.equal(paused.sources['tiny-codes'].nextOffset, 2);
  const rawPath = join(dir, 'tiny-codes-000000.raw.jsonl');
  const original = await readFile(rawPath, 'utf8');
  const { writeFile } = await import('node:fs/promises');
  await writeFile(rawPath, original + ' ');
  await assert.rejects(acquireShards({ out: dir, ids: ['tiny-codes'], limit: 4, pageSize: 2, fetcher }), /hash mismatch/);
});

test('resume rejects config drift and limit reductions', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'acquire-shards-config-')); t.after(() => rm(dir, { recursive: true, force: true }));
  const { fetcher } = mockFetch([{ instruction: 'Do', code: 'const x=1', language: 'JS' }]);
  await acquireShards({ out: dir, ids: ['tiny-codes'], limit: 1, pageSize: 1, fetcher });
  await assert.rejects(acquireShards({ out: dir, ids: ['tiny-codes'], limit: 1, pageSize: 1, split: 'test', fetcher }), /config differs/);
  await assert.rejects(acquireShards({ out: dir, ids: ['tiny-codes'], limit: 0.5, pageSize: 1, fetcher }), /positive integer/);
});

test('source errors on unstable row totals and overlong pages', async t => {
  const unstableDir = await mkdtemp(join(tmpdir(), 'acquire-shards-total-')); t.after(() => rm(unstableDir, { recursive: true, force: true }));
  let rowCalls = 0;
  const unstable = async input => {
    const url = new URL(String(input));
    if (url.hostname === 'huggingface.co') return { ok: true, json: async () => ({ sha: 'f'.repeat(40) }) };
    rowCalls++;
    const offset = Number(url.searchParams.get('offset'));
    return { ok: true, json: async () => ({ num_rows_total: rowCalls === 1 ? 3 : 2,
      rows: [{ row: { instruction: `Do ${offset}`, code: 'const x=1', language: 'JavaScript' } }] }) };
  };
  const unstableResult = await acquireShards({ out: unstableDir, ids: ['tiny-codes'], limit: 2, pageSize: 1, fetcher: unstable });
  assert.equal(unstableResult.sources['tiny-codes'].status, 'error');
  assert.match(unstableResult.sources['tiny-codes'].error, /total row count changed/);

  const longDir = await mkdtemp(join(tmpdir(), 'acquire-shards-overlong-')); t.after(() => rm(longDir, { recursive: true, force: true }));
  const { fetcher: normal } = mockFetch([{ instruction: 'a', code: 'a', language: 'JS' }, { instruction: 'b', code: 'b', language: 'JS' }]);
  const overlong = async (input, options) => {
    const url = new URL(String(input));
    if (url.hostname === 'huggingface.co') return normal(input, options);
    return { ok: true, json: async () => ({ num_rows_total: 2, rows: [
      { row: { instruction: 'a', code: 'a', language: 'JS' } },
      { row: { instruction: 'b', code: 'b', language: 'JS' } },
    ] }) };
  };
  const overlongResult = await acquireShards({ out: longDir, ids: ['tiny-codes'], limit: 2, pageSize: 1, fetcher: overlong });
  assert.equal(overlongResult.sources['tiny-codes'].status, 'error');
  assert.match(overlongResult.sources['tiny-codes'].error, /exceeds requested page bounds/);
});

test('429 honors Retry-After and retries successfully without advancing past the page', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'acquire-shards-retry-')); t.after(() => rm(dir, { recursive: true, force: true }));
  const rows = [{ instruction: 'Do it', code: 'const x=1', language: 'JavaScript', id: 1 }];
  let rowsRequests = 0;
  const waits = [];
  const fetcher = async input => {
    const url = new URL(String(input));
    if (url.hostname === 'huggingface.co') return { ok: true, json: async () => ({ sha: 'f'.repeat(40) }) };
    rowsRequests++;
    if (rowsRequests === 1) return { ok: false, status: 429, headers: { get: key => key === 'retry-after' ? '2' : null } };
    return { ok: true, json: async () => ({ num_rows_total: 1, rows: [{ row: rows[0] }] }) };
  };
  const result = await acquireShards({ out: dir, ids: ['tiny-codes'], limit: 1, pageSize: 1, fetcher,
    minIntervalMs: 0, sleep: async ms => waits.push(ms) });
  assert.equal(rowsRequests, 2);
  assert.ok(waits.reduce((sum, ms) => sum + ms, 0) >= 2000, 'Retry-After delay was honored');
  assert.equal(result.sources['tiny-codes'].nextOffset, 1);
  assert.equal(result.sources['tiny-codes'].status, 'complete');
});

test('stop during Retry-After wait checkpoints as paused instead of error', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'acquire-shards-stop-retry-')); t.after(() => rm(dir, { recursive: true, force: true }));
  let stopping = false;
  const fetcher = async input => {
    const url = new URL(String(input));
    if (url.hostname === 'huggingface.co') return { ok: true, json: async () => ({ sha: 'f'.repeat(40) }) };
    return { ok: false, status: 429, headers: { get: key => key === 'retry-after' ? '60' : null } };
  };
  const result = await acquireShards({ out: dir, ids: ['tiny-codes'], limit: 1, pageSize: 1, fetcher,
    minIntervalMs: 0, maxRetries: 5, shouldStop: () => stopping,
    sleep: async () => { stopping = true; } });
  assert.equal(result.sources['tiny-codes'].status, 'paused');
  assert.equal(result.sources['tiny-codes'].nextOffset, 0);
  assert.equal(result.sources['tiny-codes'].error, undefined);
});
