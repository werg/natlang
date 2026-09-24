#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { collectBatch, defaultSystemPrompt, defaultToolSurfaceHash, loadRecords, nativeJobRunner,
  recordDigest, sha256, writeAtomic, type CollectorConfig, type HandoffRecord } from './collector.js';
import { Agent, setGlobalDispatcher } from 'undici';

// A request can wait behind other slots' requests and then reason for minutes before its first byte, so fetch's
// five-minute header and body timeouts are off; each episode has its own wall-clock budget.
setGlobalDispatcher(new Agent({ headersTimeout: 0, bodyTimeout: 0 }));

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
      'Options: --server URL --start N --limit N|--all --workers N --segment-turns N\n' +
      '         --segment-messages N --thinking-tokens N --reasoning-effort LEVEL\n' +
      '         --transport-retries N --retry-delay-ms N --system-file PATH\n' +
      '         --cache-stable-tools --handoff-queue PATH --collection-role student|teacher\n' +
      '         --reuse RESULTS.jsonl[,RESULTS.jsonl...]  (finished rows of earlier runs stand in for the same programs)\n');
    return;
  }
  if (positionals.length !== 3 || !flags.get('--model-id') || !flags.get('--root-seed'))
    throw new Error('usage: collect-teacher-batch IR JOBS OUT --model-id ID --root-seed N [--limit 10 --workers 1]');
  const [ir, jobs, output] = positionals.map(value => resolve(value)) as [string, string, string];
  let handoffs: Map<string, HandoffRecord> | undefined;
  if (flags.has('--handoff-queue')) {
    handoffs = new Map();
    for (const line of (await readFile(resolve(flags.get('--handoff-queue')!), 'utf8')).split(/\r?\n/)) {
      if (!line.trim()) continue;
      const handoff = JSON.parse(line) as HandoffRecord;
      if (handoff.version !== 'natlang.hard_state/1' || !handoff.id || handoffs.has(handoff.id) ||
          !Array.isArray(handoff.prefix) || handoff.handoff_at !== handoff.prefix.length)
        throw new Error('invalid or duplicate student handoff');
      handoffs.set(handoff.id, handoff);
    }
  }
  const systemPrompt = flags.has('--system-file') ? await readFile(resolve(flags.get('--system-file')!), 'utf8') : defaultSystemPrompt;
  const collectionRole = flags.get('--collection-role') ?? 'teacher';
  if (!['student', 'teacher'].includes(collectionRole)) throw new Error('invalid collection role');
  if (handoffs && collectionRole !== 'teacher') throw new Error('handoff collection must use the teacher role');
  const config: CollectorConfig = { jobs, output, modelId: flags.get('--model-id')!,
    rootSeed: integer(flags, '--root-seed', 0), workers: integer(flags, '--workers', 1),
    segmentTurns: integer(flags, '--segment-turns', 24), segmentMessages: integer(flags, '--segment-messages', 48),
    ...(flags.has('--max-turns') ? { maxTurns: integer(flags, '--max-turns', 0) } : {}),
    transportRetries: integer(flags, '--transport-retries', 8),
    retryDelayMs: Number(flags.get('--retry-delay-ms') ?? 5000), systemPrompt,
    cacheStableTools: flags.has('--cache-stable-tools'),
    ...(handoffs ? { handoffs } : {}),
    ...(flags.has('--reuse') ? { reuse: flags.get('--reuse')!.split(',').filter(Boolean).map(path => resolve(path)) } : {}),
    collectionRole: collectionRole as 'student' | 'teacher',
    toolSurfaceSha256: await defaultToolSurfaceHash(), endpoint: flags.get('--server') ?? 'http://127.0.0.1:8081',
    request: { thinking_budget_tokens: integer(flags, '--thinking-tokens', 256), top_p: 0.95, top_k: 20,
      chat_template_kwargs: { reasoning_effort: flags.get('--reasoning-effort') ?? 'low' } } };
  const records = await loadRecords(ir, integer(flags, '--start', 0), flags.has('--all') ? 0 : integer(flags, '--limit', 10));
  if (handoffs) for (const item of records) {
    const handoff = handoffs.get(item.record.id);
    if (!handoff || handoff.program_ir_sha256 !== recordDigest(item.record))
      throw new Error(`${item.record.id}: handoff does not match the selected program`);
  }
  const controller = new AbortController();
  for (const signal of ['SIGINT', 'SIGTERM'] as const) process.on(signal, () => controller.abort());
  const result = await collectBatch(records, config, nativeJobRunner(config), controller.signal);
  const source = await readFile(ir);
  const merged = await readFile(output);
  await writeAtomic(`${output}.manifest.json`, JSON.stringify({
    version: 'natlang.teacher_batch.native/1', source: ir, source_sha256: sha256(source),
    range: { start: records[0]?.index ?? 0, count: records.length }, model: config.modelId,
    root_seed: config.rootSeed, tool_schema: 'scope-eval-v1', segment_turns: config.segmentTurns,
    ...(handoffs ? { handoff_queue: resolve(flags.get('--handoff-queue')!),
      handoff_queue_sha256: sha256(await readFile(resolve(flags.get('--handoff-queue')!))) } : {}),
    segment_messages: config.segmentMessages, workers: config.workers, completed: result.completed,
    missing: result.missing, output_sha256: sha256(merged) }) + '\n');
  process.stdout.write(`final: ${result.completed}/${records.length} complete -> ${output}\n`);
  if (result.missing.length) process.exitCode = 2;
}

main().catch(error => { process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`); process.exitCode = 1; });
