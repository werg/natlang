#!/usr/bin/env node
import { TEXT_NEURALESE_EMULATION_VERSION, TEXT_NEURALESE_PROMPT_REVISION } from '../model/text-neuralese-emulation.js';
import { chatRequestControls } from './chat-request-controls.js';
import { providerRequestControls } from './provider-request-controls.js';
import { APPROACH_PROMPT, FILE_TOOL_SURFACES, type FileToolSurface } from '../native/prompt.js';
import { readFile } from 'node:fs/promises';
import { createReadStream, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import {pathToFileURL} from 'node:url';
import { observeCollectionPromise, type CollectionLivenessSnapshot } from './collection-liveness.js';
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
      '         --chat-request-config FILE  JSON sampling/template controls retained in provenance\n' +
      '         --provider-request-config FILE  JSON transport controls retained in provenance\n' +
      '         --provider-request-timeout-ms N --provider-action-cycle-timeout-ms N\n' +
      '           (optional Pi provider collection resource controls; off by default)\n' +
      '         --model-concurrency N --max-model-requests N  (limits include all child calls)\n' +
      '         --cache-stable-tools --collection-role student|teacher\n' +
      '         --text-neuralese-emulation  explicit marker-and-literal text transport for teacher Neuralese calls only\n' +
      '         --execution-adapter MODULE  caller-supplied program execution fixture\n' +
      '         --file-tools all|editor|files  (the file tools directory reducers offer; default all)\n' +
      '         --judge-model-id ID (--judge-server URL | --judge-provider PI_ID) for judged oracles\n' +
      '         --reuse RESULTS.jsonl[,RESULTS.jsonl...]  (finished rows of earlier runs stand in for the same programs)\n' +
      '         --reuse-surfaces HASH[,HASH...]  (earlier tool surfaces declared equivalent for reuse)\n' +
      '         --drain-file PATH  (stop admitting new cases when present; let active cases finish)\n' +
      '         --case-events-file PATH  (append-only per-case lease and terminal journal)\n' +
      '         --final-export-only  (write merged output at final flush; per-case result files remain live)\n' +
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
  if (flags.has('--text-neuralese-emulation') && collectionRole !== 'teacher')
    throw new Error('--text-neuralese-emulation is available only for teacher collection');
  const provider = flags.get('--provider');
  if (provider && flags.has('--server')) throw new Error('--provider and --server cannot be used together');
  if (flags.has('--provider-request-config') && !provider) throw new Error('--provider-request-config requires --provider');
  const controls = flags.has('--provider-request-config') ? providerRequestControls(JSON.parse(
    await readFile(resolve(flags.get('--provider-request-config')!), 'utf8'))) : undefined;
  if (flags.has('--chat-request-config') && provider) throw new Error('--chat-request-config requires an OpenAI-compatible server');
  const chatControls = flags.has('--chat-request-config') ? chatRequestControls(JSON.parse(
    await readFile(resolve(flags.get('--chat-request-config')!), 'utf8'))) : undefined;
  const judgeModelId = flags.get('--judge-model-id');
  if ([judgeModelId, flags.get('--judge-server'), flags.get('--judge-provider')].some(Boolean) &&
      (!judgeModelId || Number(flags.has('--judge-server')) + Number(flags.has('--judge-provider')) !== 1))
    throw new Error('--judge-model-id needs exactly one of --judge-server or --judge-provider');
  if (flags.has('--execution-plan-tokens') && !flags.has('--execution-plans'))
    throw new Error('--execution-plan-tokens requires --execution-plans');
  const adapterPath=flags.get('--execution-adapter');
  let execution:CollectorConfig['execution'];
  if(adapterPath){const path=resolve(adapterPath),loaded=(await import(pathToFileURL(path).href)).default;
    if(!loaded||typeof loaded.identity!=='string'||typeof loaded.run!=='function')throw Error('execution adapter must export default {identity,run}');
    execution={identity:loaded.identity+':'+sha256(await readFile(path)),run:loaded.run};}
  const config: CollectorConfig = { jobs, output, ...(execution?{execution}:{}), modelId: flags.get('--model-id')!,
    rootSeed: integer(flags, '--root-seed', 0), workers: integer(flags, '--workers', 6),
    contextTokens: integer(flags, '--context-tokens', 16384),
    ...(flags.has('--max-turns') ? { maxTurns: integer(flags, '--max-turns', 0) } : {}),
    ...(flags.has('--temperature') ? { temperature: Number(flags.get('--temperature')) } : {}),
    ...(flags.has('--model-concurrency') ? { modelConcurrency: integer(flags, '--model-concurrency', 2) } : {}),
    ...(flags.has('--max-model-requests') ? { maxModelRequests: integer(flags, '--max-model-requests', 128) } : {}),
    ...(flags.has('--provider-request-timeout-ms') ?
      { providerRequestTimeoutMs: integer(flags, '--provider-request-timeout-ms', 0) } : {}),
    ...(flags.has('--provider-action-cycle-timeout-ms') ?
      { providerActionCycleTimeoutMs: integer(flags, '--provider-action-cycle-timeout-ms', 0) } : {}),
    ...(chatControls ? { chatRequestControls: chatControls } : {}),
    transportRetries: integer(flags, '--transport-retries', 8),
    ...(flags.has('--worker-stagger') ? { workerStaggerMs: integer(flags, '--worker-stagger', 0) * 1000 } : {}),
    retryDelayMs: Number(flags.get('--retry-delay-ms') ?? 5000), systemPrompt,
    cacheStableTools: flags.has('--cache-stable-tools'),
    executionPlans: flags.has('--execution-plans'),
    ...(flags.has('--file-tools') ? { fileTools: fileToolSurface(flags.get('--file-tools')!) } : {}),
    ...(flags.has('--execution-plan-tokens') ?
      { executionPlanTokens: integer(flags, '--execution-plan-tokens', 512) } : {}),
    ...(flags.has('--case-events-file') ? { caseEventsFile: resolve(flags.get('--case-events-file')!) } : {}),
    ...(flags.has('--final-export-only') ? { finalExportOnly: true } : {}),
    ...(flags.has('--reuse') ? { reuse: flags.get('--reuse')!.split(',').filter(Boolean).map(path => resolve(path)) } : {}),
    // The Bonsai server's default buffer (serve_bonsai.sh: 53,248 tokens); 0 turns admission off.
    ...(!provider && integer(flags, '--kv-tokens', 53_248) > 0 ? { kvTokens: integer(flags, '--kv-tokens', 53_248) } : {}),
    ...(flags.has('--reuse-surfaces') ? { reuseSurfaces: flags.get('--reuse-surfaces')!.split(',').filter(Boolean) } : {}),
    collectionRole: collectionRole as 'student' | 'teacher',
    ...(flags.has('--text-neuralese-emulation') ? { textNeuraleseEmulation: true } : {}),
    ...(judgeModelId ? { judgeModel: { modelId: judgeModelId,
      ...(flags.has('--judge-provider') ? { provider: flags.get('--judge-provider'),
        piOptions: { reasoningEffort: flags.get('--judge-reasoning-effort') ?? 'low' } } :
        { endpoint: flags.get('--judge-server') }) } } : {}),
    toolSurfaceSha256: await defaultToolSurfaceHash(),
    ...(provider ? { provider, ...(controls ? { providerRequestControls: controls } : {}), piOptions: { reasoningEffort: flags.get('--reasoning-effort') ?? 'low' } } :
      { endpoint: flags.get('--server') ?? 'http://127.0.0.1:8081' }),
    // The thinking budget is the server's (serve_bonsai.sh --reasoning-budget) unless given: a server that enforces a
    // request budget ends a turn at it, so a small default cut every turn of a verbose reasoner short of its tool call.
    request: { ...(flags.has('--thinking-tokens') ? { thinking_budget_tokens: integer(flags, '--thinking-tokens', 0) } : {}),
      top_p: 0.95, top_k: 20,
      chat_template_kwargs: { reasoning_effort: flags.get('--reasoning-effort') ?? 'low' }, ...chatControls } };
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
  const admissionController = new AbortController();
  const drainFile = flags.has('--drain-file') ? resolve(flags.get('--drain-file')!) : undefined;
  let drainPoll: NodeJS.Timeout | undefined;
  if (drainFile) {
    const pollDrain = () => { if (existsSync(drainFile)) admissionController.abort(); };
    pollDrain();
    drainPoll = setInterval(pollDrain, 250);
    drainPoll.unref();
  }
  let result: Awaited<ReturnType<typeof collectBatch>>;
  let collectionSnapshot: CollectionLivenessSnapshot = { stage: 'collectBatch', active_cases: [] };
  config.collectionState = snapshot => { collectionSnapshot = snapshot; };
  try { result = await observeCollectionPromise(
    collectBatch(records, config, nativeJobRunner(config), controller.signal, admissionController.signal),
    () => collectionSnapshot); }
  finally { if (drainPoll) clearInterval(drainPoll); }
  const source = await readFile(ir);
  const outputHash = createHash('sha256');
  for await (const chunk of createReadStream(output)) outputHash.update(chunk);
  await writeAtomic(`${output}.manifest.json`, JSON.stringify({
    version: 'natlang.teacher_batch.native/1', source: ir, source_sha256: sha256(source),
    range: { start: records[0]?.index ?? 0, count: records.length }, model: config.modelId,
    root_seed: config.rootSeed, tool_schema: 'scope-eval-v1', context_tokens: config.contextTokens,
    text_neuralese_transport: config.textNeuraleseEmulation ? TEXT_NEURALESE_EMULATION_VERSION : null,
    text_neuralese_prompt_revision: config.textNeuraleseEmulation ? TEXT_NEURALESE_PROMPT_REVISION : null,
    workers: config.workers, completed: result.completed,
    missing: result.missing, output_sha256: outputHash.digest('hex') }) + '\n');
  process.stdout.write(`final: ${result.completed}/${records.length} complete -> ${output}\n`);
  if (result.missing.length) process.exitCode = 2;
}

// Evals run model-written code in this process; a promise it leaves running can reject after its eval has ended.
// That is the model's error, already outside any row, so it is logged and collection goes on.
process.on('unhandledRejection', error => {
  process.stderr.write(`ignored a rejection left over from an eval: ${error instanceof Error ? error.message : String(error)}\n`);
});
main().catch(error => { process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`); process.exitCode = 1; });
