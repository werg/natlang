import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { startStudio } from '../scripts/serve-studio.mjs';
test('local companion runs real operations, verifies cache and journals cancellation', async () => {
    const root = await mkdtemp(join(tmpdir(), 'natlang-studio-test-'));
    const studio = await startStudio({ port: 0, dataRoot: root });
    const origin = new URL(studio.url).origin;
    try {
        const auth = await fetch(origin + '/api/studio/session').then(r => r.json());
        const headers = { 'x-studio-token': auth.token, 'x-studio-session': auth.session };
        const request = async (path, options = {}) => { const r = await fetch(origin + '/api/studio/' + path, { ...options, headers: { ...headers, ...options.headers } }); const value = await r.json(); assert.ok(r.ok, JSON.stringify(value)); return value; };
        const wait = async (id) => { for (;;) {
            const job = await request('jobs/' + id);
            if (job.status !== 'running')
                return job;
            await new Promise(resolve => setTimeout(resolve, 25));
        } };
        const run = async (operation, payload) => { const id = randomUUID(); await request('jobs', { method: 'POST', body: JSON.stringify({ id, operation, payload }) }); const job = await wait(id); assert.equal(job.status, 'complete', job.error); return job.result; };
        const sql = await run('notebook.query', { source: 'SELECT SUM(value) AS total FROM observations' });
        assert.equal(sql.rows[0].total, 46);
        const first = await run('build.run', { source: 'a careful build', operation: 'uppercase' });
        assert.equal(first.output, 'A CAREFUL BUILD');
        const second = await run('build.run', { source: 'a careful build', operation: 'uppercase' });
        assert.equal(second.detail, 'cache hit');
        const registry = await run('packages.resolve', { name: 'greetings', range: '^1.0.0' });
        assert.equal(registry.locks[0].packages[0].version, '1.1.0');
        const installed = await run('packages.install', { lock: registry.locks[0], target: 'test_greetings' });
        assert.equal(installed.status, 'installed');
        const before = 'export const greet = name => "Hello, " + name;\n';
        const good = await run('repository.check', { before, after: before });
        assert.equal(good.status, 'reviewable');
        assert.equal(good.checks.length, 2);
        const bad = await run('repository.check', { before, after: before.replace('Hello', 'Welcome') });
        assert.equal(bad.status, 'checks-failed');
        const term = await run('terminal.run', { command: 'printf studio-command' });
        assert.equal(term.output, 'studio-command');
        assert.equal(term.code, 0);
        const media = await run('media.sample', {});
        assert.equal(media.probe.width, 640);
        const rendered = await run('media.transform', { asset: media.asset, kind: 'trim', parameters: { start: 0, end: 1 } });
        assert.equal(rendered.receipt.status, 'ok');
        assert.ok(rendered.inspection.duration <= 1.1);
        const asset = await fetch(origin + `/studio-assets/${auth.session}/${rendered.asset}`, { headers: { Range: 'bytes=0-31' } });
        assert.equal(asset.status, 206);
        assert.equal((await asset.arrayBuffer()).byteLength, 32);
        const id = randomUUID(), config = { id, operation: 'terminal.run', payload: { command: 'sleep 20' } };
        await request('jobs', { method: 'POST', body: JSON.stringify(config) });
        await request(`jobs/${id}/cancel`, { method: 'POST' });
        const stopped = await wait(id);
        assert.equal(stopped.status, 'unknown');
        const retry = await request('jobs', { method: 'POST', body: JSON.stringify(config) });
        assert.equal(retry.status, 'unknown');
        const unauthorized = await fetch(origin + '/api/studio/jobs', { method: 'POST', body: '{}' });
        assert.equal(unauthorized.status, 403);
        const foreign = await fetch(origin + '/api/studio/session', { headers: { Origin: 'https://example.test' } });
        assert.equal(foreign.status, 403);
    }
    finally {
        await studio.close();
        await rm(root, { recursive: true, force: true });
    }
});
