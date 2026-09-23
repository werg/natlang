#!/usr/bin/env node
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const RENDERER_FIELDS = ['template_sha256', 'end_token', 'terminal_tool_policy',
  'tokenizer_fingerprint_sha256', 'local_tokenizer_artifacts_sha256',
  'teacher_reasoning_policy', 'cache_stable_tools', 'native_roundtrip',
  'native_target_policy', 'invalid_action_policy'];
const sha256 = value => createHash('sha256').update(value).digest('hex');
const canonical = value => JSON.stringify(value, Object.keys(value ?? {}).sort());

async function optionalJson(path) { try { return JSON.parse(await readFile(path, 'utf8')); } catch (error) {
  if (error?.code === 'ENOENT') return null; throw error;
} }

export async function combineSft(destination, sources) {
  if (!sources.length || new Set(sources.map(source => resolve(source))).size !== sources.length)
    throw new Error('supply distinct source files');
  const identities = new Set(), inputs = [], chunks = []; let renderer = null, pairs = 0, reasoningPairs = 0;
  for (const source of sources) {
    const bytes = await readFile(source);
    if (bytes.length && bytes.at(-1) !== 10) throw new Error(`${source}: missing JSONL newline`);
    const manifestPath = `${source}.manifest.json`, manifest = await optionalJson(manifestPath);
    if (manifest?.renderer) {
      const identity = Object.fromEntries(RENDERER_FIELDS.map(key => [key, manifest.renderer[key] ?? null]));
      if (renderer && canonical(identity) !== canonical(renderer)) throw new Error(`incompatible SFT renderer: ${source}`);
      renderer = identity;
    }
    let count = 0;
    for (const [index, line] of bytes.toString('utf8').split('\n').entries()) {
      if (!line) continue;
      let row; try { row = JSON.parse(line); } catch (error) { throw new Error(`${source}:${index + 1}: ${error.message}`); }
      if (!row.id) throw new Error(`${source}:${index + 1}: missing id`);
      if (identities.has(row.id)) throw new Error(`duplicate SFT id: ${row.id}`);
      identities.add(row.id); count++; pairs++;
      if (String(row.completion ?? '').includes('<think>')) reasoningPairs++;
    }
    inputs.push({ path: source, sha256: sha256(bytes), pairs: count, manifest: manifest ? manifestPath : null });
    chunks.push(bytes);
  }
  const data = Buffer.concat(chunks), target = resolve(destination), staged = `${target}.building-${process.pid}-${randomUUID()}`;
  await mkdir(dirname(target), { recursive: true });
  await writeFile(staged, data, { flag: 'wx' });
  try { await rename(staged, target); } catch (error) { await unlink(staged).catch(() => {}); throw error; }
  const result = { schema: 'natlang.sft_bundle/1', sources: inputs, renderer,
    pairs, reasoning_pairs: reasoningPairs, sha256: sha256(data) };
  await writeFile(`${target}.manifest.json`, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  return result;
}

async function main() {
  const [destination, ...sources] = process.argv.slice(2);
  if (!destination || !sources.length) throw new Error('usage: combine-sft.mjs DESTINATION SOURCE...');
  const result = await combineSft(destination, sources);
  console.log(`${result.pairs} pairs, ${result.reasoning_pairs} with reasoning -> ${destination}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => { console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1; });
