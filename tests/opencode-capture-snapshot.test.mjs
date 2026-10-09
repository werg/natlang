import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { snapshotOpenCodeCliCaptures } from '../scripts/snapshot_opencode_cli_captures.mjs';

const required = [
  'cli-stdout.raw', 'cli-invocations.jsonl', 'action-mcp-calls.jsonl',
  'bootstrap-config.json', 'bootstrap-events.jsonl', 'mcp-handshake.jsonl',
  'lifecycle.json', 'opencode-home/data/opencode/log/opencode.log',
];

test('snapshots stopped bridge captures to a separate immutable tree with hashes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'opencode-capture-test-'));
  const source = join(root, 'bridge'), output = join(root, 'snapshot');
  try {
    for (const name of required) {
      const path = join(source, name);
      await mkdir(path.slice(0, path.lastIndexOf('/')), { recursive: true });
      await writeFile(path, name === 'lifecycle.json' ? '{"status":"stopped"}\n' : `capture:${name}\n`);
    }
    const result = await snapshotOpenCodeCliCaptures({ sourceDirectory: source, outputDirectory: output, scope: 'case-02' });
    const manifest = JSON.parse(await readFile(result.manifest_path, 'utf8'));
    assert.equal(manifest.scope, 'case-02');
    assert.equal(manifest.files.length, required.length);
    assert.ok(manifest.files.some(file => file.snapshot_path.endsWith('/opencode.log')));
    await assert.rejects(snapshotOpenCodeCliCaptures({ sourceDirectory: source, outputDirectory: output, scope: 'again' }));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('refuses running bridge and output inside source', async () => {
  const root = await mkdtemp(join(tmpdir(), 'opencode-capture-test-'));
  const source = join(root, 'bridge');
  try {
    await mkdir(source);
    await writeFile(join(source, 'lifecycle.json'), '{"status":"running"}\n');
    await assert.rejects(snapshotOpenCodeCliCaptures({ sourceDirectory: source, outputDirectory: join(root, 'out'), scope: 'x' }), /must be stopped/);
    await writeFile(join(source, 'lifecycle.json'), '{"status":"stopped"}\n');
    await assert.rejects(snapshotOpenCodeCliCaptures({ sourceDirectory: source, outputDirectory: join(source, 'out'), scope: 'x' }), /outside source/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
