#!/usr/bin/env node
/** Snapshot one stopped isolated OpenCode CLI run before publishing any review. */
import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, writeFile } from 'node:fs/promises';
import { isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REQUIRED = [
  'cli-invocations.jsonl', 'action-mcp-calls.jsonl',
  'bootstrap-config.json', 'bootstrap-events.jsonl', 'mcp-handshake.jsonl',
  'lifecycle.json', 'opencode-home/data/opencode/log/opencode.log',
];
const OPTIONAL = ['cli-stdout.raw', 'cli-stderr.raw'];
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

export async function snapshotOpenCodeCliCaptures({ sourceDirectory, outputDirectory, scope }) {
  const source = resolve(sourceDirectory), output = resolve(outputDirectory);
  if (!sourceDirectory || !outputDirectory || typeof scope !== 'string' || !scope.trim())
    throw new Error('source directory, output directory, and nonempty scope are required');
  const rel = relative(source, output);
  if (!rel.startsWith('..') && !isAbsolute(rel)) throw new Error('snapshot output must be outside source directory');
  const lifecyclePath = resolve(source, 'lifecycle.json');
  const lifecycle = JSON.parse(await readFile(lifecyclePath, 'utf8'));
  if (lifecycle.status !== 'stopped') throw new Error('OpenCode bridge must be stopped before capture snapshot');
  const sourceFiles = [];
  for (const name of REQUIRED) {
    const path = resolve(source, name);
    if (relative(source, path).startsWith('..')) throw new Error(`capture path escapes source directory: ${name}`);
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink()) throw new Error(`capture is not a regular file: ${name}`);
    sourceFiles.push({ name, path, bytes: await readFile(path) });
  }
  for (const name of OPTIONAL) {
    const path = resolve(source, name);
    try {
      const info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink()) throw new Error(`capture is not a regular file: ${name}`);
      sourceFiles.push({ name, path, bytes: await readFile(path) });
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
    }
  }
  await mkdir(output, { recursive: false, mode: 0o700 });
  const files = [];
  for (const entry of sourceFiles) {
    const destination = resolve(output, entry.name);
    await mkdir(resolve(destination, '..'), { recursive: true, mode: 0o700 });
    await writeFile(destination, entry.bytes, { flag: 'wx', mode: 0o600 });
    files.push({ source_path: entry.path, snapshot_path: destination,
      bytes: entry.bytes.byteLength, sha256: sha256(entry.bytes) });
  }
  const manifest = { schema: 'natlang.opencode_cli_capture_snapshot/1',
    captured_at_utc: new Date().toISOString(), source_directory: source,
    scope: scope.trim(), bridge_lifecycle: { status: lifecycle.status, at: lifecycle.at ?? null }, files,
    omissions: OPTIONAL.filter(name => !sourceFiles.some(file => file.name === name))
      .map(name => `Optional capture was not emitted: ${name}`) };
  const manifestPath = resolve(output, 'snapshot-manifest.json');
  const manifestBytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(manifestPath, manifestBytes, { flag: 'wx', mode: 0o600 });
  return { manifest_path: manifestPath, manifest_sha256: sha256(manifestBytes), files };
}

function parseArgs(argv) {
  const options = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index], value = argv[index + 1];
    if (!['--source-directory', '--output-directory', '--scope'].includes(key) || !value || options.has(key))
      throw new Error(`invalid or duplicate option ${key}`);
    options.set(key, value);
  }
  for (const key of ['--source-directory', '--output-directory', '--scope'])
    if (!options.has(key)) throw new Error(`${key} is required`);
  return { sourceDirectory: options.get('--source-directory'), outputDirectory: options.get('--output-directory'),
    scope: options.get('--scope') };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  snapshotOpenCodeCliCaptures(parseArgs(process.argv.slice(2)))
    .then(result => process.stdout.write(`${JSON.stringify(result)}\n`))
    .catch(error => { process.stderr.write(`OpenCode capture snapshot refused: ${error.message}\n`); process.exitCode = 1; });
}
