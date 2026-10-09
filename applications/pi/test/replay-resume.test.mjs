/**
 * Replay resumes an interrupted output: trajectories it holds are skipped, a line cut off by a kill is dropped and
 * replayed again, and the rest are appended.
 * Run: node --test applications/pi/test/replay-resume.test.mjs
 */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { replayedIds, replayFile } from '../bench/replay.ts';

test('replay skips replayed trajectories and drops a truncated tail', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'pi-replay-resume-'));
  try {
    // A local repository stands in for the bare clone replay would fetch.
    const work = join(dir, 'work'), repos = join(dir, 'repos');
    mkdirSync(work);
    writeFileSync(join(work, 'a.txt'), 'hello\n');
    execFileSync('sh', ['-c', 'git init -q && git add -A && git -c user.name=t -c user.email=t@example.com commit -qm base'], { cwd: work });
    const commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: work, encoding: 'utf8' }).trim();
    mkdirSync(repos);
    execFileSync('git', ['clone', '--quiet', '--bare', work, join(repos, 'test__repo.git')]);

    const trajectory = id => ({ id, repo: 'test/repo', cwd: '/workspace/project', base_commit: commit, messages: [] });
    const input = join(dir, 'prepared.jsonl'), output = join(dir, 'replayed.jsonl');
    writeFileSync(input, ['t1', 't2', 't3'].map(id => JSON.stringify(trajectory(id))).join('\n') + '\n');
    // t1 was replayed before the kill (marked so a rewrite would show); t2 was being written when it came.
    const kept = JSON.stringify({ ...trajectory('t1'), replay: { kept: true } }) + '\n';
    writeFileSync(output, kept + JSON.stringify(trajectory('t2')).slice(0, 40));

    assert.deepEqual([...replayedIds(output)], ['t1']);
    assert.equal(readFileSync(output, 'utf8'), kept);

    const log = [];
    await replayFile(input, output, repos, line => log.push(line));
    const lines = readFileSync(output, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
    assert.deepEqual(lines.map(line => line.id), ['t1', 't2', 't3']);
    assert.deepEqual(lines[0].replay, { kept: true });
    assert.deepEqual(JSON.parse(log.at(-1)), { trajectories: 3, resumed: 1, fullyVerified: 2, diverged: 0, failed: 0 });

    // A finished output resumes to nothing.
    await replayFile(input, output, repos, () => {});
    assert.equal(readFileSync(output, 'utf8').split('\n').filter(Boolean).length, 3);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
