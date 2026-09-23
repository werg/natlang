#!/usr/bin/env node
import { readFile, mkdir, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { assemble } from './assemble.mjs';
import { readJsonl, writeJsonl, digest } from './common.mjs';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');

/** Read only committed acquisition shards, never discover orphan files by glob. */
export async function assembleShards(output, inputs) {
  const paths = [], identities = {};
  for (const input of inputs.map(p => resolve(p))) {
    if ((await stat(input)).isDirectory()) {
      const manifest = JSON.parse(await readFile(join(input, 'manifest.json'), 'utf8'));
      for (const source of Object.values(manifest.sources ?? {})) {
        if (source.status !== 'complete') throw new Error('Acquisition is not complete; resume it before assembly');
        for (const shard of source.shards) {
          const path = join(input, shard.tasks), bytes = await readFile(path);
          if (hash(bytes) !== shard.tasksSha256) throw new Error(`Shard changed: ${path}`);
          paths.push(path);
        }
      }
    } else paths.push(input);
  }
  if (!paths.length) throw new Error('No source task shards');
  for (const path of paths) identities[path] = hash(await readFile(path));
  const identity = { inputs: identities, builder: hash(await readFile(new URL('./assemble.mjs', import.meta.url))) };
  await mkdir(output, { recursive: true });
  const identityPath = join(output, 'inputs.jsonl');
  try {
    const [old] = await readJsonl(identityPath);
    if (digest(old) !== digest(identity)) throw new Error('Assembly inputs changed; choose a new bundle');
  } catch (error) { if (error.code !== 'ENOENT') throw error; await writeJsonl(identityPath, [identity]); }
  const commit = join(output, 'complete.jsonl');
  try {
    const [done] = await readJsonl(commit);
    for (const [name, expected] of Object.entries(done.outputs))
      if (hash(await readFile(join(output, name))) !== expected) throw new Error(`Assembly output changed: ${name}`);
    return done.report;
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const tasks = (await Promise.all(paths.map(path => readJsonl(path)))).flat();
  const result = assemble(tasks), files = {
    'train.jsonl': result.rows.filter(r => r.split === 'train'),
    'test.jsonl': result.rows.filter(r => r.split === 'test'),
    'rejected.jsonl': result.rejected,
    'manifest.jsonl': [{ ...result.report, version:'natlang.code_corpus_bundle/1', tasks_sha256:digest(tasks) }],
  };
  const outputs = {};
  for (const [name, rows] of Object.entries(files)) {
    await writeJsonl(join(output, name), rows, { replace:true });
    outputs[name] = hash(await readFile(join(output, name)));
  }
  await writeJsonl(commit, [{ identity_sha256:digest(identity), outputs, report:result.report }]);
  return result.report;
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const [output, ...inputs] = process.argv.slice(2);
  if (!output || !inputs.length) throw new Error('Usage: assemble-shards.mjs OUTPUT_DIR TASKS_OR_COMMITTED_SHARD_DIR ...');
  console.log(JSON.stringify(await assembleShards(resolve(output), inputs)));
}
