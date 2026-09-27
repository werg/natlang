#!/usr/bin/env node
import { APPROACH_PROMPT, FILE_TOOL_SURFACES, type FileToolSurface } from '../native/prompt.js';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { collectBatch, defaultSystemPrompt, defaultToolSurfaceHash, loadRecords, nativeJobRunner,
  sha256, writeAtomic, type CollectorConfig } from './collector.js';

function fileToolSurface(value: string): FileToolSurface {
  if (!FILE_TOOL_SURFACES.includes(value as FileToolSurface)) throw new Error(`--file-tools must be one of ${FILE_TOOL_SURFACES.join(', ')}`);
  return value as FileToolSurface;
}

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
      'Options: --server URL | --provider PI_ID --start N --limit N|--all --workers N --context-tokens N\n' +
      '         --thinking-tokens N --reasoning-effort LEVEL --approach-guide --temperature T (default 0: greedy)\n' +
      '         --execution-plans [--execution-plan-tokens N]  plan before each action and retain it as reasoning\n' +
      '         --transport-retries N --retry-delay-ms N --worker-stagger SECONDS --system-file PATH\n' +
      '         --cache-stable-tools --collection-role student|teacher\n' +
      '         --file-tools all|editor|files  (the file tools directory reducers offer; default all)\n' +
      '         --judge-model-id ID (--judge-server URL | --judge-provider PI_ID) for judged oracles\n' +
      '         --reuse RESULTS.jsonl[,RESULTS.jsonl...]  (finished rows of earlier runs stand in for the same programs)\n' +
      '         --reuse-surfaces HASH[,HASH...]  (earlier tool surfaces declared equivalent for reuse)\n' +
      '         --kv-tokens N  (the server\'s shared KV buffer; requests wait to fit, default 53248, 0 = off)\n');
    return;
  }
  if (positionals.length !== 3 || !flags.get('--model-id') || !flags.get('--root-seed'))
    throw new Error('usage: collect-teacher-batch IR JOBS OUT --model-id ID --root-seed N [--limit 10 --workers 6]');
  const [ir, jobs, output] = positionals.map(value => resolve(value)) as [string, string, string];
  // --approach-guide appends the schematic of how calls usually go (native/prompt.ts APPROACH_PROMPT).
  const systemPrompt = (flags.has('--system-file') ? await readFile(resolve(flags.get('--system-file')!), 'utf8') : defaultSystemPrompt) +
    (flags.has('--approach-guide') ? APPROACH_PROMPT : '');
  const collectionRole = flags.get('--collection-role') ?? 'teacher';
  if (!['student', 'teacher'].includes(collectionRole)) throw new Error('invalid collection role');
  const provider = flags.get('--provider');
  if (provider && flags.has('--server')) throw new Error('--provider and --server cannot be used together');
  const judgeModelId = flags.get('--judge-model-id');
  if ([judgeModelId, flags.get('--judge-server'), flags.get('--judge-provider')].some(Boolean) &&
      (!judgeModelId || Number(flags.has('--judge-server')) + Number(flags.has('--judge-provider')) !== 1))
    throw new Error('--judge-model-id needs exactly one of --judge-server or --judge-provider');
  if (flags.has('--execution-plan-tokens') && !flags.has('--execution-plans'))
    throw new Error('--execution-plan-tokens requires --execution-plans');
  const config: CollectorConfig = { jobs, output, modelId: flags.get('--model-id')!,
    rootSeed: integer(flags, '--root-seed', 0), workers: integer(flags, '--workers', 6),
    contextTokens: integer(flags, '--context-tokens', 16384),
    ...(flags.has('--max-turns') ? { maxTurns: integer(flags, '--max-turns', 0) } : {}),
    ...(flags.has('--temperature') ? { temperature: Number(flags.get('--temperature')) } : {}),
    transportRetries: integer(flags, '--transport-retries', 8),
    ...(flags.has('--worker-stagger') ? { workerStaggerMs: integer(flags, '--worker-stagger', 0) * 1000 } : {}),
    retryDelayMs: Number(flags.get('--retry-delay-ms') ?? 5000), systemPrompt,
    cacheStableTools: flags.has('--cache-stable-tools'),
    executionPlans: flags.has('--execution-plans'),
    ...(flags.has('--file-tools') ? { fileTools: fileToolSurface(flags.get('--file-tools')!) } : {}),
    ...(flags.has('--execution-plan-tokens') ?
      { executionPlanTokens: integer(flags, '--execution-plan-tokens', 512) } : {}),
    ...(flags.has('--reuse') ? { reuse: flags.get('--reuse')!.split(',').filter(Boolean).map(path => resolve(path)) } : {}),
    // The Bonsai server's default buffer (serve_bonsai.sh: 53,248 tokens); 0 turns admission off.
    ...(!provider && integer(flags, '--kv-tokens', 53_248) > 0 ? { kvTokens: integer(flags, '--kv-tokens', 53_248) } : {}),
    ...(flags.has('--reuse-surfaces') ? { reuseSurfaces: flags.get('--reuse-surfaces')!.split(',').filter(Boolean) } : {}),
    collectionRole: collectionRole as 'student' | 'teacher',
    ...(judgeModelId ? { judgeModel: { modelId: judgeModelId,
      ...(flags.has('--judge-provider') ? { provider: flags.get('--judge-provider'),
        piOptions: { reasoningEffort: flags.get('--judge-reasoning-effort') ?? 'low' } } :
        { endpoint: flags.get('--judge-server') }) } } : {}),
    toolSurfaceSha256: await defaultToolSurfaceHash(),
    ...(provider ? { provider, piOptions: { reasoningEffort: flags.get('--reasoning-effort') ?? 'low' } } :
      { endpoint: flags.get('--server') ?? 'http://127.0.0.1:8081' }),
    // The thinking budget is the server's (serve_bonsai.sh --reasoning-budget) unless given: a server that enforces a
    // request budget ends a turn at it, so a small default cut every turn of a verbose reasoner short of its tool call.
    request: { ...(flags.has('--thinking-tokens') ? { thinking_budget_tokens: integer(flags, '--thinking-tokens', 0) } : {}),
      top_p: 0.95, top_k: 20,
      chat_template_kwargs: { reasoning_effort: flags.get('--reasoning-effort') ?? 'low' } } };
  // A call is compacted as it nears its context budget, so the server must accept a request of that size; a server
  // with a smaller context would reject the call's later requests. llama.cpp reports its per-request context in /props.
  if (config.endpoint) {
    const served = await fetch(new URL('/props', config.endpoint)).then(response => response.ok ? response.json() : undefined, () => undefined)
      .then(props => (props as { default_generation_settings?: { n_ctx?: number } } | undefined)?.default_generation_settings?.n_ctx);
    if (typeof served === 'number' && served < config.contextTokens)
      throw new Error(`the server accepts ${served} tokens per request, less than --context-tokens ${config.contextTokens}; ` +
        'serve with a larger context per slot or lower --context-tokens');
    if (served === undefined) process.stderr.write('note: the server does not report its context size; not checked against --context-tokens\n');
  }
  const records = await loadRecords(ir, integer(flags, '--start', 0), flags.has('--all') ? 0 : integer(flags, '--limit', 10));
  const controller = new AbortController();
  for (const signal of ['SIGINT', 'SIGTERM'] as const) process.on(signal, () => controller.abort());
  const result = await collectBatch(records, config, nativeJobRunner(config), controller.signal);
  const source = await readFile(ir);
  const merged = await readFile(output);
  await writeAtomic(`${output}.manifest.json`, JSON.stringify({
    version: 'natlang.teacher_batch.native/1', source: ir, source_sha256: sha256(source),
    range: { start: records[0]?.index ?? 0, count: records.length }, model: config.modelId,
    root_seed: config.rootSeed, tool_schema: 'scope-eval-v1', context_tokens: config.contextTokens,
    workers: config.workers, completed: result.completed,
    missing: result.missing, output_sha256: sha256(merged) }) + '\n');
  process.stdout.write(`final: ${result.completed}/${records.length} complete -> ${output}\n`);
  if (result.missing.length) process.exitCode = 2;
}

// Evals run model-written code in this process; a promise it leaves running can reject after its eval has ended.
// That is the model's error, already outside any row, so it is logged and collection goes on.
process.on('unhandledRejection', error => {
  process.stderr.write(`ignored a rejection left over from an eval: ${error instanceof Error ? error.message : String(error)}\n`);
});
main().catch(error => { process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`); process.exitCode = 1; });
