#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { collectBatch, defaultSystemPrompt, defaultToolSurfaceHash, loadRecords, nativeJobRunner,
  sha256, writeAtomic, type CollectorConfig } from './collector.js';

function argumentsOf(argv: string[]): { positionals: string[]; flags: Map<string, string> } {
  const positionals: string[] = [], flags = new Map<string, string>();
  for (let index = 0; index < argv.length; index++) {
    const item = argv[index]!;
    if (!item.startsWith('--')) { positionals.push(item); continue; }
    const [name, inline] = item.split('=', 2);
    if (inline !== undefined) flags.set(name!, inline);
    else if (argv[index + 1] && !argv[index + 1]!.startsWith('--')) flags.set(name!, argv[++index]!);
    else flags.set(name!, 'true');
  }
  return { positionals, flags };
}
const integer = (flags: Map<string, string>, name: string, fallback: number): number => {
  const value = Number(flags.get(name) ?? fallback);
  if (!Number.isInteger(value)) throw new Error(`${name} must be an integer`);
  return value;
};

async function main(): Promise<void> {
  const { positionals, flags } = argumentsOf(process.argv.slice(2));
  if (flags.has('--help')) {
    process.stdout.write('usage: teacher-collector IR JOBS OUT --model-id ID --root-seed N [options]\n\n' +
      'Options: --server URL --start N --limit N --workers N --segment-turns N\n' +
      '         --segment-messages N --thinking-tokens N --reasoning-effort LEVEL\n' +
      '         --transport-retries N --retry-delay-ms N --system-file PATH\n' +
      '         --cache-stable-tools\n');
    return;
  }
  if (positionals.length !== 3 || !flags.get('--model-id') || !flags.get('--root-seed'))
    throw new Error('usage: collect-teacher-batch IR JOBS OUT --model-id ID --root-seed N [--limit 10 --workers 1]');
  const [ir, jobs, output] = positionals.map(value => resolve(value)) as [string, string, string];
  const systemPrompt = flags.has('--system-file') ? await readFile(resolve(flags.get('--system-file')!), 'utf8') : defaultSystemPrompt;
  const config: CollectorConfig = { jobs, output, modelId: flags.get('--model-id')!,
    rootSeed: integer(flags, '--root-seed', 0), workers: integer(flags, '--workers', 1),
    segmentTurns: integer(flags, '--segment-turns', 24), segmentMessages: integer(flags, '--segment-messages', 48),
    transportRetries: integer(flags, '--transport-retries', 8),
    retryDelayMs: Number(flags.get('--retry-delay-ms') ?? 5000), systemPrompt,
    cacheStableTools: flags.has('--cache-stable-tools'),
    toolSurfaceSha256: await defaultToolSurfaceHash(), endpoint: flags.get('--server') ?? 'http://127.0.0.1:8081',
    request: { thinking_budget_tokens: integer(flags, '--thinking-tokens', 256), top_p: 0.95, top_k: 20,
      chat_template_kwargs: { reasoning_effort: flags.get('--reasoning-effort') ?? 'low' } } };
  const records = await loadRecords(ir, integer(flags, '--start', 0), integer(flags, '--limit', 10));
  const controller = new AbortController();
  for (const signal of ['SIGINT', 'SIGTERM'] as const) process.on(signal, () => controller.abort());
  const result = await collectBatch(records, config, nativeJobRunner(config), controller.signal);
  const source = await readFile(ir);
  const merged = await readFile(output);
  await writeAtomic(`${output}.manifest.json`, JSON.stringify({
    version: 'natlang.teacher_batch.native/1', source: ir, source_sha256: sha256(source),
    range: { start: records[0]?.index ?? 0, count: records.length }, model: config.modelId,
    root_seed: config.rootSeed, tool_schema: 'scope-eval-v1', segment_turns: config.segmentTurns,
    segment_messages: config.segmentMessages, workers: config.workers, completed: result.completed,
    missing: result.missing, output_sha256: sha256(merged) }) + '\n');
  process.stdout.write(`final: ${result.completed}/${records.length} complete -> ${output}\n`);
  if (result.missing.length) process.exitCode = 2;
}

main().catch(error => { process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`); process.exitCode = 1; });
