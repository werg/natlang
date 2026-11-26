#!/usr/bin/env node
import { chatRequestControls, wireControls } from './chat-request-controls.js';
import type { GuidanceRun } from '../native/training-guidance.js';
import { programGuidance as parseProgramGuidance } from '../guidance/program.js';
import { providerRequestControls } from './provider-request-controls.js';
import { APPROACH_PROMPT, FILE_TOOL_SURFACES, type FileToolSurface } from '../native/prompt.js';
import { NATIVE_TOOL_SCHEMA } from '../native/tool-surface.js';
import type { PromptRendering } from '../native/system-prompts.js';
import { readFile } from 'node:fs/promises';
import { createReadStream, existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import {pathToFileURL} from 'node:url';
import { observeCollectionPromise, type CollectionLivenessSnapshot } from './collection-liveness.js';
import { collectBatch, defaultSystemPrompt, defaultToolSurfaceHash, loadRecords, nativeJobRunner,
  sha256, writeAtomic, type CollectorConfig } from './collector.js';
import { CollectorRequestTelemetryAccumulator } from './request-telemetry.js';
import { teacherCollectorCapabilities } from './runtime-capabilities.js';

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

async function main(argv: string[] = process.argv.slice(2), stop?: AbortSignal): Promise<number> {
  const { positionals, flags } = argumentsOf(argv);
  if (flags.has('--capabilities')) {
    if (flags.get('--capabilities') !== 'json') throw new Error('--capabilities must be json');
    process.stdout.write(JSON.stringify(await teacherCollectorCapabilities()) + '\n');
    return 0;
  }
  if (flags.has('--help')) {
    process.stdout.write('usage: teacher-collector IR JOBS OUT --model-id ID --root-seed N [options]\n\n' +
      'Options: --server URL | --provider PI_ID --start N --limit N|--all --workers N --context-tokens N\n' +
      '         --chat-completions-url URL --api-key-env NAME  explicit external HTTP transport\n' +
      '         --thinking-tokens N --reasoning-effort LEVEL --approach-guide --temperature T (default 0: greedy)\n' +
      '         --execution-plans [--execution-plan-tokens N]  plan before each action and retain it as reasoning\n' +
      '         --action-max-tokens N  independent per-action output cap (planner cap remains separate)\n' +
      '         --request-retries N (default 1) repeats the same request; --transport-retries N (default 0) retries the whole case\n' +
      '         --retry-delay-ms N --worker-stagger SECONDS --system-file PATH\n' +
      '         --program-guidance-file PATH  (natlang.memetic-program-guidance/1 binding, or a list of scoped ones; needs --neuralese-server)\n' +
      '         --chat-request-config FILE  JSON sampling/template controls retained in provenance\n' +
      '         --provider-request-config FILE  JSON transport controls retained in provenance\n' +
      '         --provider-request-timeout-ms N --provider-action-cycle-timeout-ms N\n' +
      '           (optional Pi provider collection resource controls; off by default)\n' +
      '         --model-concurrency N --max-model-requests N  (limits include all child calls)\n' +
      '         --prompt-rendering full|program-header  native prompt and tool surface (default full)\n' +
      '         --cache-stable-tools --collection-role student|teacher\n' +
      '         --text-neuralese-emulation  explicit teacher-only literal-text Neuralese over provider/HTTP; no vectors or training qualification\n' +
      '         --in-context-neuralese  Neuralese as engines read it by default (each state as its text in place), no engine: authored replays\n' +
      '         --execution-adapter MODULE  caller-supplied program execution fixture\n' +
      '         --file-tools all|editor|files  (the file tools directory reducers offer; default all)\n' +
      '         --judge-model-id ID (--judge-server URL | --judge-provider PI_ID) for judged oracles\n' +
      '         --reuse RESULTS.jsonl[,RESULTS.jsonl...]  (finished rows of earlier runs stand in for the same programs)\n' +
      '         --reuse-surfaces HASH[,HASH...]  (earlier tool surfaces declared equivalent for reuse)\n' +
      '         --drain-file PATH  (stop admitting new cases when present; let active cases finish)\n' +
      '         --case-events-file PATH  (append-only per-case lease and terminal journal)\n' +
      '         --journal-spool-root PATH  (bounded local crash-recoverable journal; publish verified files to JOBS_DIR)\n' +
      '         --final-export-only  (write merged output at final flush; per-case result files remain live)\n' +
      '         --kv-tokens N  (the server\'s shared KV buffer; requests wait to fit, default 53248, 0 = off)\n' +
      '       teacher-collector --serve SOCKET [--capacity N]  one process runs up to N invocations sent over SOCKET\n');
    return 0;
  }
  if (positionals.length !== 3 || !flags.get('--model-id') || !flags.get('--root-seed'))
    throw new Error('usage: collect-teacher-batch IR JOBS OUT --model-id ID --root-seed N [--limit 10 --workers 6]');
  const [ir, jobs, output] = positionals.map(value => resolve(value)) as [string, string, string];
  if (flags.has('--journal-spool-root')) {
    const approved = process.env.NATLANG_TEACHER_JOURNAL_SPOOL_ROOT;
    if (!approved || resolve(flags.get('--journal-spool-root')!) !== resolve(approved))
      throw new Error('--journal-spool-root must match the caller-approved campaign storage binding');
  }
  // --approach-guide appends the schematic of how calls usually go (native/prompt.ts APPROACH_PROMPT).
  const systemPrompt = (flags.has('--system-file') ? await readFile(resolve(flags.get('--system-file')!), 'utf8') : defaultSystemPrompt) +
    (flags.has('--approach-guide') ? APPROACH_PROMPT : '');
  const collectionRole = flags.get('--collection-role') ?? 'teacher';
  if (!['student', 'teacher'].includes(collectionRole)) throw new Error('invalid collection role');
  const provider = flags.get('--provider');
  if (flags.has('--text-neuralese-emulation') && (collectionRole !== 'teacher' || !provider && !flags.has('--server') ||
      flags.has('--neuralese-server') || flags.has('--program-guidance-file')))
    // Sampling controls (--chat-request-config) are compatible; their x_natlang_guidance needs --neuralese-server (below).
    throw new Error('--text-neuralese-emulation requires teacher role and an explicit provider/server, and excludes native Neuralese/guidance');
  if (flags.has('--neuralese-server') && (!flags.has('--server') || provider))
    throw new Error('--neuralese-server requires --server and excludes Pi');
  if (provider && flags.has('--server')) throw new Error('--provider and --server cannot be used together');
  if (provider && (flags.has('--chat-completions-url') || flags.has('--api-key-env')))
    throw new Error('external HTTP controls cannot be used with --provider');
  if (flags.has('--provider-request-config') && !provider) throw new Error('--provider-request-config requires --provider');
  const controls = flags.has('--provider-request-config') ? providerRequestControls(JSON.parse(
    await readFile(resolve(flags.get('--provider-request-config')!), 'utf8'))) : undefined;
  if (flags.has('--chat-request-config') && provider) throw new Error('--chat-request-config requires an OpenAI-compatible server');
  const chatControls = flags.has('--chat-request-config') ? chatRequestControls(JSON.parse(
    await readFile(resolve(flags.get('--chat-request-config')!), 'utf8'))) : undefined;
  if (chatControls?.x_natlang_guidance && !flags.has('--neuralese-server'))
    throw new Error('training-guidance-unavailable: x_natlang_guidance runs on a native-state server; add --neuralese-server');
  if (flags.has('--program-guidance-file') && !flags.has('--neuralese-server'))
    throw new Error('--program-guidance-file needs the native reader of --neuralese-server');
  const programGuidance = flags.has('--program-guidance-file') ? parseProgramGuidance(JSON.parse(
    await readFile(resolve(flags.get('--program-guidance-file')!), 'utf8'))) : undefined;
  const judgeModelId = flags.get('--judge-model-id');
  if ([judgeModelId, flags.get('--judge-server'), flags.get('--judge-provider')].some(Boolean) &&
      (!judgeModelId || Number(flags.has('--judge-server')) + Number(flags.has('--judge-provider')) !== 1))
    throw new Error('--judge-model-id needs exactly one of --judge-server or --judge-provider');
  if (flags.has('--execution-plan-tokens') && !flags.has('--execution-plans'))
    throw new Error('--execution-plan-tokens requires --execution-plans');
  if (flags.has('--action-max-tokens') && integer(flags, '--action-max-tokens', 0) < 1)
    throw new Error('--action-max-tokens must be a positive integer');
  const promptRendering = flags.get('--prompt-rendering');
  if (promptRendering && promptRendering !== 'full' && promptRendering !== 'program-header')
    throw new Error('--prompt-rendering must be full or program-header');
  const adapterPath=flags.get('--execution-adapter');
  let execution:CollectorConfig['execution'];
  if(adapterPath){const path=resolve(adapterPath),loaded=(await import(pathToFileURL(path).href)).default;
    if(!loaded||typeof loaded.identity!=='string'||typeof loaded.run!=='function')throw Error('execution adapter must export default {identity,run}');
    execution={identity:loaded.identity+':'+sha256(await readFile(path)),run:loaded.run};}
  const config: CollectorConfig = { jobs, output, ...(execution?{execution}:{}),
    ...(flags.has('--journal-spool-root') ? { journalSpoolRoot: resolve(flags.get('--journal-spool-root')!) } : {}), modelId: flags.get('--model-id')!,
    rootSeed: integer(flags, '--root-seed', 0), workers: integer(flags, '--workers', 6),
    contextTokens: integer(flags, '--context-tokens', 16384),
    ...(promptRendering ? { promptRendering: promptRendering as PromptRendering } : {}),
    ...(flags.has('--max-turns') ? { maxTurns: integer(flags, '--max-turns', 0) } : {}),
    ...(flags.has('--temperature') ? { temperature: Number(flags.get('--temperature')) } : {}),
    ...(flags.has('--model-concurrency') ? { modelConcurrency: integer(flags, '--model-concurrency', 2) } : {}),
    ...(flags.has('--max-model-requests') ? { maxModelRequests: integer(flags, '--max-model-requests', 128) } : {}),
    ...(flags.has('--provider-request-timeout-ms') ?
      { providerRequestTimeoutMs: integer(flags, '--provider-request-timeout-ms', 0) } : {}),
    ...(flags.has('--provider-action-cycle-timeout-ms') ?
      { providerActionCycleTimeoutMs: integer(flags, '--provider-action-cycle-timeout-ms', 0) } : {}),
    ...(chatControls ? { chatRequestControls: chatControls } : {}),
    // Run-level training guidance goes to the agent, never on the wire (native/training-guidance.ts).
    ...(chatControls?.x_natlang_guidance ? { trainingGuidance: chatControls.x_natlang_guidance as GuidanceRun } : {}),
    ...(programGuidance ? { programGuidance } : {}),
    transportRetries: integer(flags, '--transport-retries', 0),
    requestRetries: integer(flags, '--request-retries', 1),
    ...(flags.has('--worker-stagger') ? { workerStaggerMs: integer(flags, '--worker-stagger', 0) * 1000 } : {}),
    retryDelayMs: Number(flags.get('--retry-delay-ms') ?? 5000), systemPrompt,
    cacheStableTools: flags.has('--cache-stable-tools'),
    executionPlans: flags.has('--execution-plans'),
    ...(flags.has('--action-max-tokens') ? { actionMaxTokens: integer(flags, '--action-max-tokens', 0) } : {}),
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
    ...(flags.has('--neuralese-server') ? { neuraleseServer: true } : {}),
    ...(flags.has('--text-neuralese-emulation') ? { textNeuraleseEmulation: true } : {}),
    ...(flags.has('--in-context-neuralese') ? { inContextNeuralese: true } : {}),
    ...(judgeModelId ? { judgeModel: { modelId: judgeModelId,
      ...(flags.has('--judge-provider') ? { provider: flags.get('--judge-provider'),
        piOptions: { reasoningEffort: flags.get('--judge-reasoning-effort') ?? 'low' } } :
        { endpoint: flags.get('--judge-server') }) } } : {}),
    toolSurfaceSha256: await defaultToolSurfaceHash(),
    ...(provider ? { provider, ...(controls ? { providerRequestControls: controls } : {}), piOptions: { reasoningEffort: flags.get('--reasoning-effort') ?? 'low' } } :
      { endpoint: flags.get('--server') ?? 'http://127.0.0.1:8081',
        chatCompletionsUrl: flags.get('--chat-completions-url'), apiKeyEnv: flags.get('--api-key-env') }),
    // The thinking budget is the server's (serve_bonsai.sh --reasoning-budget) unless given: a server that enforces a
    // request budget ends a turn at it, so a small default cut every turn of a verbose reasoner short of its tool call.
    request: { ...(flags.has('--thinking-tokens') ? { thinking_budget_tokens: integer(flags, '--thinking-tokens', 0) } : {}),
      ...(flags.has('--chat-completions-url') ? {} : { top_p: 0.95, top_k: 20,
        chat_template_kwargs: { reasoning_effort: flags.get('--reasoning-effort') ?? 'low' } }), ...(chatControls ? wireControls(chatControls) : {}) } };
  // A call is compacted as it nears its context budget, so the server must accept a request of that size; a server
  // with a smaller context would reject the call's later requests. llama.cpp reports its per-request context in /props.
  if (config.endpoint && !config.chatCompletionsUrl) {
    const served = await fetch(new URL('/props', config.endpoint)).then(response => response.ok ? response.json() : undefined, () => undefined)
      .then(props => (props as { default_generation_settings?: { n_ctx?: number } } | undefined)?.default_generation_settings?.n_ctx);
    if (typeof served === 'number' && served < config.contextTokens)
      throw new Error(`the server accepts ${served} tokens per request, less than --context-tokens ${config.contextTokens}; ` +
        'serve with a larger context per slot or lower --context-tokens');
    if (served === undefined) process.stderr.write('note: the server does not report its context size; not checked against --context-tokens\n');
  }
  // A Neuralese actor reports its window in /v1/neuralese/info: the call's budget follows it (never above
  // --context-tokens), so a harness default larger than an older actor's window compacts at the actor's size instead of
  // sending requests the actor rejects.
  if (config.endpoint && config.chatCompletionsUrl && flags.has('--neuralese-server')) {
    const served = await fetch(new URL('/v1/neuralese/info', config.endpoint)).then(response => response.ok ? response.json() : undefined,
      () => undefined).then(info => (info as { context?: unknown } | undefined)?.context);
    if (typeof served === 'number' && served > 0 && served < config.contextTokens) {
      process.stderr.write(`note: the actor serves ${served} tokens per request; the call budget follows it (--context-tokens ${config.contextTokens})\n`);
      config.contextTokens = served;
    }
  }
  const source = await readFile(ir);
  config.sourceSha256 = sha256(source);
  const records = await loadRecords(ir, integer(flags, '--start', 0), flags.has('--all') ? 0 : integer(flags, '--limit', 10), source.toString('utf8'));
  const controller = new AbortController();
  if (stop) stop.addEventListener('abort', () => controller.abort(), { once: true });
  else for (const signal of ['SIGINT', 'SIGTERM'] as const) process.on(signal, () => controller.abort());
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
  const outputHash = createHash('sha256');
  for await (const chunk of createReadStream(output)) outputHash.update(chunk);
  const requestTotals = new CollectorRequestTelemetryAccumulator();
  const requestStream = createInterface({ input: createReadStream(output), crlfDelay: Infinity });
  for await (const line of requestStream) {
    if (!line.trim()) continue;
    requestTotals.add(JSON.parse(line) as Parameters<CollectorRequestTelemetryAccumulator['add']>[0]);
  }
  await writeAtomic(`${output}.manifest.json`, JSON.stringify({
    version: 'natlang.teacher_batch.native/1', source: ir, source_sha256: sha256(source),
    range: { start: records[0]?.index ?? 0, count: records.length }, model: config.modelId,
    root_seed: config.rootSeed, tool_schema: NATIVE_TOOL_SCHEMA, context_tokens: config.contextTokens,
    neuralese_server: config.neuraleseServer ?? false,
    workers: config.workers, completed: result.completed,
    missing: result.missing, output_sha256: outputHash.digest('hex'),
    request_telemetry: { schema: 'natlang.collector_request_manifest/1',
      scope: 'rows in merged output (includes completed/reused rows); exact collector sender invocations and ChatTransport calls; provider SDK entries are turn starts, not physical network counts',
      ...requestTotals.totals } }) + '\n');
  process.stdout.write(`final: ${result.completed}/${records.length} complete -> ${output}\n`);
  return result.missing.length ? 2 : 0;
}

/** Shared collector process (scripts/run_bonsai_queue.py, NATLANG_COLLECTOR_SHARED): one node process serves up to
 * `capacity` concurrent invocations, each a JSON line {argv} on its own connection, answered by {exit}. Node, the
 * bundle and the TypeScript library cost ~150 MB per process; an episode's own heap is tens of MB, so sharing the
 * process is what lets a collection run 128+ slots in ~16 GB. Closing the connection aborts that invocation (the
 * supervisor's terminate). The server exits after `idle` seconds without an invocation.
 * Every connection is greeted with {ready, pid, active, capacity} before the client sends its request: an episode's
 * synchronous work blocks the whole event loop, and a blocked server still completes connects (the kernel's backlog), so
 * the greeting is what tells a live server from a stalled one. The client sends nothing to a server that does not
 * greet it, so a server that wakes up later finds only closed connections and never runs an invocation twice. */
/** run_bonsai_queue uses a runtime's shared server only at this protocol (/1 died with EPIPE when a peer left; /2 had no
 * greeting: a client waited 60 s on a stalled server and then failed its episode, grpo-v4-run1 2026-11-18). */
const SERVE_PROTOCOL = 'natlang-collector-serve/4';

async function serve(argv: string[]): Promise<void> {
  const { flags } = argumentsOf(argv);
  const socket = resolve(flags.get('--serve')!), capacity = integer(flags, '--capacity', 8);
  const idleMs = integer(flags, '--idle-seconds', 120) * 1000;
  const { createServer } = await import('node:net');
  const { rmSync } = await import('node:fs');
  // Backstop against retention across invocations: after --recycle-after invocations the server stops listening (a new
  // one takes the socket path) and exits once its running invocations finish.
  const recycleAfter = integer(flags, '--recycle-after', 64);
  // And against a process that grew: past --recycle-rss-mb of resident memory (default 1536; NATLANG_COLLECTOR_RECYCLE_RSS_MB,
  // 0 turns it off) it stops listening the same way, and its running invocations finish. Typical servers sit at 0.3-1 GB
  // with 8 episodes; a model-written program that grew external memory left one at 5.1 GB until it had served 64
  // invocations (grpo-v4-run2 2026-11-19: 2.3 GB external after one invocation, 5.2 GB of the unit's 40 GB host memory).
  const recycleRssMb = integer(flags, '--recycle-rss-mb', Number(process.env.NATLANG_COLLECTOR_RECYCLE_RSS_MB ?? 1536));
  let active = 0, served = 0, admitted = 0, listening = true, idle: NodeJS.Timeout | undefined;
  const release = () => { if (listening) { listening = false; server.close(); rmSync(socket, { force: true }); } };
  const arm = () => {
    if (idle) clearTimeout(idle);
    if (!active && !listening) { idle = setTimeout(() => process.exit(0), 1000); return; }   // replies flush first
    if (!active) idle = setTimeout(() => { release(); process.exit(0); }, idleMs);
  };
  // Event-loop stalls (an invocation's synchronous work, swap-in, GC) hold every invocation of this process and its
  // greetings: logged with their length so a slow collection can be traced to them.
  let tick = Date.now();
  // A stall that does not end takes every co-hosted invocation with it, and a blocked loop cannot end it: a watchdog
  // thread kills the process once the loop has not ticked for --stall-kill-seconds (default 300;
  // NATLANG_COLLECTOR_STALL_KILL_SECONDS, 0 turns it off). Its clients see the connection close without an exit and
  // resume their episodes from their journals in their own processes (run_bonsai_queue.SharedChild `lost`). Eval code
  // is bounded per synchronous stretch (native/eval-memory.ts syncGuard, 60 s); this covers what the guard cannot
  // interrupt. sft-v7 held candidate 2026-11-22: one eval's runaway recursion held s000 for 60 min, 8 episodes timed out.
  const stallKillSeconds = integer(flags, '--stall-kill-seconds', Number(process.env.NATLANG_COLLECTOR_STALL_KILL_SECONDS ?? 300));
  const heartbeat = new Int32Array(new SharedArrayBuffer(8));   // [seconds since start at the last tick, active]
  const startedAt = Date.now();
  if (stallKillSeconds > 0) {
    const { Worker } = await import('node:worker_threads');
    new Worker(`
      const { workerData } = require('node:worker_threads');
      const { writeSync } = require('node:fs');
      const { heartbeat, startedAt, limit, pid } = workerData;
      setInterval(() => {
        const now = (Date.now() - startedAt) / 1000, last = Atomics.load(heartbeat, 0);
        if (now - last < limit) return;
        try { writeSync(1, JSON.stringify({ event: 'loop_stall_kill', stall_seconds: Math.round(now - last), limit_seconds: limit,
          active: Atomics.load(heartbeat, 1), pid, time: new Date().toISOString() }) + '\\n'); } catch {}
        process.kill(pid, 'SIGKILL');
      }, 1000);`, { eval: true, workerData: { heartbeat, startedAt, limit: stallKillSeconds, pid: process.pid } })
      // A watchdog that fails is logged; it never ends the process serving the invocations.
      .on('error', error => process.stdout.write(JSON.stringify({ event: 'stall_watchdog_error', error: String(error) }) + '\n'))
      .unref();
  }
  setInterval(() => {
    const now = Date.now(), lag = now - tick - 1000;
    tick = now;
    Atomics.store(heartbeat, 0, Math.floor((now - startedAt) / 1000)); Atomics.store(heartbeat, 1, active);
    if (lag >= 5000) process.stdout.write(JSON.stringify({ event: 'loop_stall', stall_ms: lag, active,
      time: new Date(now).toISOString() }) + '\n');
    if (recycleRssMb > 0 && listening) {
      const rssMb = Math.round(process.memoryUsage.rss() / 2**20);
      if (rssMb > recycleRssMb) {
        process.stdout.write(JSON.stringify({ event: 'recycle_rss', rss_mb: rssMb, limit_mb: recycleRssMb, active, served,
          time: new Date(now).toISOString() }) + '\n');
        release(); arm();
      }
    }
  }, 1000).unref();
  const server = createServer(connection => {
    // A peer can go away at any point (a probe, a supervisor that timed out or was killed): a write to it then fails
    // with EPIPE, which must end that connection only, never the process serving the other invocations.
    connection.on('error', () => {});
    connection.write(JSON.stringify({ ready: true, protocol: SERVE_PROTOCOL, pid: process.pid, active, capacity, listening }) + '\n');
    let buffer = '', started = false, done = false, requestId: string | null = null;
    const stop = new AbortController();
    connection.on('close', () => stop.abort());
    const control = () => {
      for (let end; (end = buffer.indexOf('\n')) >= 0;) {
        const line = buffer.slice(0, end); buffer = buffer.slice(end + 1);
        try {
          const message = JSON.parse(line) as { cancel?: unknown; request_id?: unknown };
          if (!done && message.cancel === true && message.request_id === requestId) stop.abort();
        } catch { /* An invalid control frame never cancels another invocation. */ }
      }
    };
    connection.on('data', chunk => {
      buffer += chunk.toString('utf8');
      if (started) { control(); return; }
      const end = buffer.indexOf('\n');
      if (end < 0) return;
      const first = buffer.slice(0, end); buffer = buffer.slice(end + 1);
      started = true;
      // Capacity is taken by a request, not by a connection: a connection that closes without one holds nothing.
      if (active >= capacity || !listening) { connection.end(JSON.stringify({ busy: true }) + '\n'); return; }
      active++; admitted++; arm();
      if (admitted >= recycleAfter) release();
      const finish = (value: object) => {
        if (done) return;
        done = true; active--; served++; arm();
        if (!connection.destroyed) connection.end(JSON.stringify({ ...value, request_id: requestId }) + '\n');
        // Retention is visible here: the heap after a collection, per finished invocation.
        (globalThis as { gc?: () => void }).gc?.();
        const memory = process.memoryUsage();
        process.stdout.write(JSON.stringify({ event: 'invocation_end', time: new Date().toISOString(), served, active,
          heap_used_mb: Math.round(memory.heapUsed / 2**20),
          rss_mb: Math.round(memory.rss / 2**20), external_mb: Math.round(memory.external / 2**20) }) + '\n');
      };
      let request: { argv?: unknown; request_id?: unknown };
      try { request = JSON.parse(first) as { argv?: unknown; request_id?: unknown }; }
      catch (error) { finish({ exit: 1, error: String(error) }); return; }
      if (!Array.isArray(request.argv) || !request.argv.every(item => typeof item === 'string')) {
        finish({ exit: 1, error: 'argv must be a list of strings' }); return; }
      if (typeof request.request_id !== 'string' || !/^[a-f0-9]{32}$/u.test(request.request_id)) {
        finish({ exit: 1, error: 'request_id must name this exact invocation' }); return;
      }
      requestId = request.request_id;
      connection.write(JSON.stringify({ accepted: true, pid: process.pid, request_id: requestId }) + '\n');
      main(request.argv as string[], stop.signal).then(code => finish({ exit: code }),
        error => { process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`); finish({ exit: 1, error: String(error) }); });
      control();
    });
  });
  rmSync(socket, { force: true });
  server.listen(socket, () => { process.stdout.write(JSON.stringify({ event: 'collector_server', protocol: SERVE_PROTOCOL, socket, capacity, pid: process.pid }) + '\n'); arm(); });
}

// Evals run model-written code in this process; a promise it leaves running can reject after its eval has ended.
// That is the model's error, already outside any row, so it is logged and collection goes on.
process.on('unhandledRejection', error => {
  process.stderr.write(`ignored a rejection left over from an eval: ${error instanceof Error ? error.message : String(error)}\n`);
});
if (process.argv[2] === '--serve') serve(process.argv.slice(2)).catch(error => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`); process.exitCode = 1; });
else main().then(code => { process.exitCode = code; },
  error => { process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`); process.exitCode = 1; });
