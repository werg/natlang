import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { collectBatch, defaultSystemPrompt, defaultToolSurfaceHash, expectedProvenance, jobKey,
  loadRecords, nativeJobRunner, recordDigest } from '../dist/teacher/collector.js';

const record = id => ({ version: 'natlang.program/2', id, kind: 'lambda_source', source: 'fixture',
  split: 'test', source_ids: [id], source_groups: [id], license: 'test', semantics: {
    root: 'one.nl', files: { 'one.nl': '---\nargs: {}\nreturns: number\n---\nReturn one.\n' }, inputs: {}, expected: 1,
    operation: 'exact' } });
const config = (dir, surface = 'surface-a') => ({ jobs: join(dir, 'jobs'), output: join(dir, 'out.jsonl'),
  workers: 2, modelId: 'teacher', rootSeed: 7, systemPrompt: 'prompt', segmentTurns: 3,
  segmentMessages: 8, toolSurfaceSha256: surface });
const row = (item, provenance) => ({ version: 'test', task: { program_ir: item.record }, provenance,
  outcome: { status: 'done', accepted: true } });

test('focused loader keeps source indexes and selects an exact range', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-load-')), path = join(dir, 'ir.jsonl');
  await writeFile(path, Array.from({ length: 12 }, (_, index) => JSON.stringify(record(`r${index}`))).join('\n') + '\n');
  const loaded = await loadRecords(path, 1, 10);
  assert.deepEqual(loaded.map(item => item.index), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.match(jobKey(loaded[0]), /^000001-[0-9a-f]{16}$/);
  assert.equal(recordDigest(loaded[0].record).length, 64);
});

test('all-mode permits an empty hard-state queue without inventing a teacher job', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-empty-')), path = join(dir, 'empty.jsonl');
  await writeFile(path, '');
  assert.deepEqual(await loadRecords(path, 0, 0), []);
  const options = config(dir);
  const result = await collectBatch([], options, async () => { throw new Error('must not run'); });
  assert.deepEqual(result, { completed: 0, missing: [] });
  assert.equal(await readFile(options.output, 'utf8'), '');
});

test('parallel completion merges in source order and matching jobs resume without calls', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-resume-'));
  const records = [record('first'), record('second'), record('third')].map((value, index) => ({ index, record: value }));
  let calls = 0;
  await collectBatch(records, config(dir), async (item, provenance) => {
    calls++; await new Promise(resolve => setTimeout(resolve, (2 - item.index) * 8)); return row(item, provenance);
  });
  const merged = (await readFile(config(dir).output, 'utf8')).trim().split('\n').map(JSON.parse);
  assert.deepEqual(merged.map(item => item.task.program_ir.id), ['first', 'second', 'third']);
  await collectBatch(records, config(dir), async () => { throw new Error('resume called the model'); });
  assert.equal(calls, 3);
});

test('provenance invalidates old jobs and partial results remain resumable', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-provenance-'));
  const records = [record('first'), record('second')].map((value, index) => ({ index, record: value }));
  await collectBatch(records, config(dir), async (item, provenance) => {
    if (item.index === 1) throw new Error('fixture failure'); return row(item, provenance);
  });
  assert.equal((await readFile(config(dir).output, 'utf8')).trim().split('\n').length, 1);
  let resumed = 0;
  await collectBatch(records, config(dir), async (item, provenance) => { resumed++; return row(item, provenance); });
  assert.equal(resumed, 1);
  let invalidated = 0;
  await collectBatch(records, config(dir, 'surface-b'), async (item, provenance) => {
    invalidated++; return row(item, provenance);
  });
  assert.equal(invalidated, 2);
  const provenance = expectedProvenance(records[0].record, config(dir, 'surface-b'));
  assert.equal(provenance.tool_surface_sha256, 'surface-b');
  assert.equal(provenance.runtime, 'typescript-native');
});

test('finished rows of an earlier run stand in for the same programs at other indexes', async () => {
  const earlier = await mkdtemp(join(tmpdir(), 'teacher-reuse-a-')), later = await mkdtemp(join(tmpdir(), 'teacher-reuse-b-'));
  const old = [record('kept'), record('stale'), record('migrated')].map((value, index) => ({ index, record: value }));
  await collectBatch(old, config(earlier), async (item, provenance) => row(item, provenance));
  const rows = (await readFile(config(earlier).output, 'utf8')).trim().split('\n').map(JSON.parse);
  rows[1].provenance.model = 'other-teacher';
  rows[2].provenance.tool_surface_sha256 = 'surface-old'; rows[2].provenance.finish_surface_migration = 'return_result-status/1';
  const source = join(earlier, 'earlier.jsonl');
  await writeFile(source, rows.map(value => JSON.stringify(value)).join('\n') + '\n');
  // The same programs sit at new indexes, next to a new one, in a later shard.
  const shard = [record('new'), record('migrated'), record('stale'), record('kept')].map((value, index) => ({ index, record: value }));
  const ran = [];
  const options = { ...config(later), reuse: [source] };
  await collectBatch(shard, options, async (item, provenance) => { ran.push(item.record.id); return row(item, provenance); });
  assert.deepEqual(ran.sort(), ['new', 'stale'], 'a row from another model is collected again');
  const merged = (await readFile(options.output, 'utf8')).trim().split('\n').map(JSON.parse);
  assert.deepEqual(merged.map(value => value.task.program_ir.id), ['new', 'migrated', 'stale', 'kept']);
  assert.equal(merged[3].provenance.reused_from.path, source);
  assert.equal(merged[1].provenance.reused_from.provenance.tool_surface_sha256, 'surface-old');
  await collectBatch(shard, options, async () => { throw new Error('reused rows were collected again'); });
});

test('native collector journals model replies and replays them after an interrupted request', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-turn-resume-'));
  let requests = 0, interrupted = false;
  const replies = [
    ['eval', { code: 'const answer: number = 1; return answer' }],
    null,
  ];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8'); request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      requests++;
      if (!interrupted && requests === 2) { interrupted = true; request.socket.destroy(); return; }
      const logical = requests === 1 ? 0 : requests - 2;
      const reply = replies[logical];
      const message = reply ? { role: 'assistant', content: '', tool_calls: [{ id: `c${logical}`,
        type: 'function', function: { name: reply[0], arguments: JSON.stringify(reply[1]) } }] } :
        { role: 'assistant', content: '' };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ message }], usage: { prompt_tokens: 10, completion_tokens: 4 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const port = server.address().port, item = { index: 0, record: record('resumable-turns') };
    const options = { ...config(dir), workers: 1, endpoint: `http://127.0.0.1:${port}`,
      systemPrompt: defaultSystemPrompt, toolSurfaceSha256: await defaultToolSurfaceHash(),
      transportRetries: 1, retryDelayMs: 0 };
    const result = await collectBatch([item], options, nativeJobRunner(options));
    assert.equal(result.completed, 1);
    assert.equal(requests, 3, 'the first decoded response must be replayed locally after interruption');
    const output = JSON.parse((await readFile(options.output, 'utf8')).trim());
    assert.equal(output.outcome.accepted, true);
    assert.equal(output.trajectory.length, 2);
    await assert.rejects(readFile(join(options.jobs, `${jobKey(item)}.partial.json`)), /ENOENT/);
  } finally { await new Promise(resolve => server.close(resolve)); }
});
