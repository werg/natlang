import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readFile, readdir, rename, unlink } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { openAICompatibleModelTurn } from '../model/openai-compatible.js';
import { TypeScriptEnvironment } from '../environment.js';
import { NativeToolAgent } from '../native/agent.js';
import { TOOLS_PROMPT } from '../native/prompt.js';
import { NodeNativeRuntime } from '../node-runtime.js';
import { Folder } from '../native/scoped-fs.js';
import { dump } from '../native/values.js';
import { PROGRAM_VERSION, programNode, type ProgramRecord } from './program.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';

export const TEACHER_BATCH_VERSION = 'natlang.teacher_batch.native/1';
export const TEACHER_TRAJECTORY_VERSION = 'natlang.teacher_trajectory.native/1';
export const TEACHER_PARTIAL_VERSION = 'natlang.teacher_partial.native/1';
const TOOL_SCHEMA = 'scope-eval-v1';

export type { ProgramRecord };
export type IndexedRecord = { index: number; record: ProgramRecord };
export type ProvenanceOptions = { modelId: string; rootSeed: number; systemPrompt: string;
  segmentTurns: number; segmentMessages: number; toolSurfaceSha256: string;
  /** Model turns allowed per call; unlimited unless set. A collection run should set one. */
  maxTurns?: number;
  endpoint?: string; request?: Record<string, unknown>; cacheStableTools?: boolean;
  handoffs?: Map<string, HandoffRecord>; collectionRole?: 'student' | 'teacher' };
export type HandoffRecord = { version: 'natlang.hard_state/1'; id: string;
  program_ir_sha256: string; student_trajectory_id: string; student_trajectory_sha256: string;
  handoff_at: number; target_request_sha256: string;
  prefix: { request_sha256: string; response: ModelTurn }[]; failure: Record<string, unknown>;
  student_provenance: Record<string, unknown> };
export type CollectorConfig = ProvenanceOptions & { jobs: string; output: string; workers: number;
  transportRetries?: number; retryDelayMs?: number };
export type TeacherRow = Record<string, unknown> & { task: { program_ir: ProgramRecord };
  provenance: Record<string, unknown>; outcome?: Record<string, unknown> };
export type JobRunner = (item: IndexedRecord, expected: Record<string, unknown>, signal?: AbortSignal) => Promise<TeacherRow>;

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as Record<string, unknown>)
    .sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(',')}}`;
  return JSON.stringify(value);
}
export const sha256 = (value: string | Uint8Array): string => createHash('sha256').update(value).digest('hex');
export const recordDigest = (record: ProgramRecord): string => sha256(canonical(record));

export function validateFocusedRecord(record: ProgramRecord): void {
  if (record.version !== PROGRAM_VERSION || !['lambda_graph', 'lambda_source'].includes(record.kind))
    throw new Error(`unsupported focused program IR: ${record.version}/${record.kind}`);
  if (!record.id || !record.semantics || typeof record.semantics !== 'object' || typeof record.semantics.root !== 'string' ||
      !record.semantics.root.endsWith('.nl') || !record.semantics.files || typeof record.semantics.files !== 'object' ||
      typeof record.semantics.files[record.semantics.root] !== 'string' || !record.semantics.inputs ||
      !Object.hasOwn(record.semantics, 'expected'))
    throw new Error('invalid focused program IR record');
  if (record.semantics.folder_files &&
      (typeof record.semantics.folder_files !== 'object' || Array.isArray(record.semantics.folder_files) ||
       !Object.values(record.semantics.folder_files).every(value => typeof value === 'string')))
    throw new Error('folder_files must map relative paths to text');
  if (record.semantics.expected_files && !record.semantics.folder_files)
    throw new Error('expected_files requires folder_files');
  if (record.semantics.failure_seed &&
      (typeof record.semantics.failure_seed.code !== 'string' || !record.semantics.failure_seed.code.trim() ||
       (record.semantics.failure_seed.kind !== undefined &&
        !['compile', 'runtime', 'boundary'].includes(record.semantics.failure_seed.kind))))
    throw new Error('failure_seed must contain nonempty code and a supported failure kind');
}

export async function loadRecords(path: string, start = 0, limit = 10): Promise<IndexedRecord[]> {
  if (!Number.isInteger(start) || start < 0 || !Number.isInteger(limit) || limit < 0)
    throw new RangeError('start must be nonnegative and limit must be nonnegative (zero means all)');
  const lines = (await readFile(path, 'utf8')).split(/\r?\n/);
  const records: IndexedRecord[] = [];
  for (let index = start; index < lines.length && (limit === 0 || records.length < limit); index++) {
    if (!lines[index]!.trim()) continue;
    const record = JSON.parse(lines[index]!) as ProgramRecord;
    validateFocusedRecord(record);
    records.push({ index, record });
  }
  if (limit !== 0 && records.length !== limit) throw new Error('requested source range exceeds the frozen batch');
  return records;
}

export function jobKey({ index, record }: IndexedRecord): string {
  return `${String(index).padStart(6, '0')}-${recordDigest(record).slice(0, 16)}`;
}

export function expectedProvenance(record: ProgramRecord, options: ProvenanceOptions): Record<string, unknown> {
  const handoff = options.handoffs?.get(record.id);
  if (options.handoffs && !handoff) throw new Error(`${record.id}: missing student handoff`);
  return { program_ir_sha256: recordDigest(record), model: options.modelId, tool_schema: TOOL_SCHEMA,
    runtime: 'typescript-native', collector_version: TEACHER_BATCH_VERSION,
    tool_surface_sha256: options.toolSurfaceSha256, seed_policy: { mode: 'derived', root: options.rootSeed },
    system_prompt_sha256: sha256(options.systemPrompt), segment_turns: options.segmentTurns,
    segment_messages: options.segmentMessages, transport: 'openai-compatible',
    ...(options.maxTurns === undefined ? {} : { max_turns: options.maxTurns }),
    ...(options.cacheStableTools ? { cache_stable_tools: true } : {}),
    collection_role: options.collectionRole ?? 'teacher',
    ...(handoff ? { handoff_sha256: sha256(canonical(handoff)) } : {}) };
}

export function resultMatches(row: unknown, record: ProgramRecord, expected: Record<string, unknown>): row is TeacherRow {
  if (!row || typeof row !== 'object') return false;
  const value = row as TeacherRow, provenance = value.provenance;
  return value.task?.program_ir?.id === record.id && !!provenance &&
    Object.entries(expected).every(([key, wanted]) => canonical(provenance[key]) === canonical(wanted));
}

async function readMatching(path: string, record: ProgramRecord,
  expected: Record<string, unknown>): Promise<TeacherRow | undefined> {
  try {
    const row = JSON.parse(await readFile(path, 'utf8')) as unknown;
    return resultMatches(row, record, expected) ? row : undefined;
  } catch { return; }
}

export async function writeAtomic(path: string, data: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`;
  const handle = await open(temporary, 'wx');
  try { await handle.writeFile(data); await handle.sync(); } finally { await handle.close(); }
  await rename(temporary, path);
}

async function mergeCompleted(records: IndexedRecord[], config: CollectorConfig): Promise<{ completed: number; missing: number[] }> {
  const lines: string[] = [], missing: number[] = [];
  for (const item of records) {
    const expected = expectedProvenance(item.record, config);
    const path = join(config.jobs, `${jobKey(item)}.result.json`);
    const row = await readMatching(path, item.record, expected);
    if (row) lines.push(JSON.stringify(row) + '\n'); else missing.push(item.index);
  }
  await writeAtomic(config.output, lines.join(''));
  return { completed: lines.length, missing };
}

function transportFailure(error: unknown): boolean {
  const text = String(error instanceof Error ? `${error.name}: ${error.message}` : error).toLowerCase();
  return ['connection refused', 'connection reset', 'fetch failed', 'socket', 'timed out', 'econnreset',
    'econnrefused', 'remote end closed'].some(phrase => text.includes(phrase));
}
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/** Queue incomplete jobs, publish each result atomically, and rebuild the ordered merge after every job. */
export async function collectBatch(records: IndexedRecord[], config: CollectorConfig, runner: JobRunner,
  signal?: AbortSignal): Promise<{ completed: number; missing: number[] }> {
  if (!Number.isInteger(config.workers) || config.workers < 1) throw new RangeError('workers must be positive');
  await mkdir(config.jobs, { recursive: true });
  const pending: IndexedRecord[] = [];
  for (const item of records) {
    const expected = expectedProvenance(item.record, config);
    if (!await readMatching(join(config.jobs, `${jobKey(item)}.result.json`), item.record, expected)) pending.push(item);
  }
  await mergeCompleted(records, config);
  let cursor = 0;
  const worker = async () => {
    while (cursor < pending.length && !signal?.aborted) {
      const item = pending[cursor++]!, expected = expectedProvenance(item.record, config);
      let attempt = 0;
      while (true) try {
        const row = await runner(item, expected, signal);
        if (!resultMatches(row, item.record, expected)) throw new Error('job returned mismatched provenance');
        await writeAtomic(join(config.jobs, `${jobKey(item)}.result.json`), JSON.stringify(row) + '\n');
        await mergeCompleted(records, config);
        break;
      } catch (error) {
        if (signal?.aborted) return;
        if (!transportFailure(error) || attempt >= (config.transportRetries ?? 8)) {
          await writeAtomic(join(config.jobs, `${String(item.index).padStart(6, '0')}.error.json`),
            JSON.stringify({ index: item.index, program_id: item.record.id,
              error: `${error instanceof Error ? error.name : 'Error'}: ${error instanceof Error ? error.message : String(error)}` }) + '\n');
          break;
        }
        const wait = Math.min(30_000, (config.retryDelayMs ?? 5_000) * 2 ** Math.min(attempt++, 3));
        if (wait) await delay(wait);
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(config.workers, Math.max(1, pending.length)) }, worker));
  return mergeCompleted(records, config);
}

function same(a: unknown, b: unknown): boolean { return canonical(a) === canonical(b); }

function effectHarness(specs: Record<string, unknown>): {
  capabilities: Record<string, (args: unknown[]) => unknown>; observed: Record<string, unknown[]>;
  expected: Record<string, unknown[]>;
} {
  const observed: Record<string, unknown[]> = {}, expected: Record<string, unknown[]> = {}, capabilities: Record<string, (args: unknown[]) => unknown> = {};
  for (const [name, raw] of Object.entries(specs)) {
    observed[name] = [];
    if (Array.isArray(raw)) {
      expected[name] = raw;
      capabilities[name] = args => { observed[name]!.push(args[0]); return null; };
      continue;
    }
    const spec = raw as Record<string, unknown>;
    if (spec.kind === 'record_args') {
      expected[name] = spec.expected as unknown[];
      capabilities[name] = args => { observed[name]!.push(structuredClone(args)); return null; };
      continue;
    }
    if (spec.kind === 'deliver_once_ack_loss') {
      expected[name] = spec.expected_delivered as unknown[];
      const delivered = new Set<string>(), failed = new Set<string>(), failKey = String(spec.fail_key);
      capabilities[name] = args => {
        const command = args[0] as Record<string, unknown>, key = String(command.key);
        if (!delivered.has(key)) { delivered.add(key); observed[name]!.push(structuredClone(command)); }
        if (key === failKey && !failed.has(key)) {
          failed.add(key); throw new Error('delivery succeeded but its acknowledgement was lost');
        }
        return null;
      };
      continue;
    }
    throw new Error(`unsupported effect contract for ${name}`);
  }
  return { capabilities, observed, expected };
}

function trajectoryTurn(request: ModelTurnRequest, response: ModelTurn): Record<string, unknown> {
  const raw = response.raw_response as Record<string, unknown> | undefined;
  const message = ((raw?.choices as Record<string, unknown>[] | undefined)?.[0]?.message ?? {}) as Record<string, unknown>;
  return { phase: request.tools.length ? 'action' : 'checkpoint', context: structuredClone(request.messages),
    request_sha256: sha256(canonical(request)),
    model_response: { calls: structuredClone(response.calls ?? []), text: response.text ?? '',
      raw_calls: structuredClone(response.raw_calls ?? []),
      ...(response.completion_tokens === undefined ? {} : { completion_tokens: response.completion_tokens }),
      ...(response.prompt_tokens === undefined ? {} : { prompt_tokens: response.prompt_tokens }) },
    tools_offered: structuredClone(request.tools), assistant: { content: response.text ?? '',
      reasoning: message.reasoning_content ?? message.reasoning ?? message.thinking ?? null,
      calls: (response.calls ?? []).map(([tool, args]) => ({ tool, source_tool: tool, arguments: args, call_id: null })) },
    raw_response_sha256: raw ? sha256(canonical(raw)) : null };
}

type PartialTurn = { request_sha256: string; response: ModelTurn };
type PartialJob = { version: string; program_id: string; provenance: Record<string, unknown>; turns: PartialTurn[] };

async function loadPartial(path: string, item: IndexedRecord,
  expected: Record<string, unknown>): Promise<PartialJob | undefined> {
  try {
    const value = JSON.parse(await readFile(path, 'utf8')) as PartialJob;
    if (value.version !== TEACHER_PARTIAL_VERSION || value.program_id !== item.record.id ||
        canonical(value.provenance) !== canonical(expected) || !Array.isArray(value.turns)) return;
    for (const turn of value.turns) if (typeof turn.request_sha256 !== 'string' ||
        !turn.response || typeof turn.response !== 'object') return;
    return value;
  } catch { return; }
}

async function removeIfPresent(path: string): Promise<void> {
  try { await unlink(path); }
  catch (error) {
    if (!error || typeof error !== 'object' || !('code' in error) || error.code !== 'ENOENT') throw error;
  }
}

export function nativeJobRunner(config: CollectorConfig): JobRunner {
  if (!config.endpoint) throw new Error('endpoint is required for native teacher collection');
  return async (item, expected, signal) => {
    const handoff = config.handoffs?.get(item.record.id);
    if (config.handoffs && (!handoff || handoff.program_ir_sha256 !== recordDigest(item.record) ||
        handoff.handoff_at !== handoff.prefix.length || !handoff.target_request_sha256))
      throw new Error(`${item.record.id}: invalid or missing student handoff`);
    if (handoff && (!handoff.student_provenance ||
        handoff.student_provenance.tool_surface_sha256 !== config.toolSurfaceSha256 ||
        handoff.student_provenance.system_prompt_sha256 !== sha256(config.systemPrompt) ||
        handoff.student_provenance.collection_role !== 'student' ||
        handoff.student_provenance.runtime !== 'typescript-native' ||
        handoff.student_provenance.tool_schema !== TOOL_SCHEMA ||
        handoff.student_provenance.program_ir_sha256 !== recordDigest(item.record) ||
        (handoff.student_provenance.seed_policy as Record<string, unknown>)?.root !== config.rootSeed ||
        handoff.student_provenance.segment_turns !== config.segmentTurns ||
        handoff.student_provenance.segment_messages !== config.segmentMessages))
      throw new Error(`${item.record.id}: student handoff runtime/prompt/seed settings differ`);
    const transport = openAICompatibleModelTurn({ endpoint: config.endpoint!, model: config.modelId,
      request: config.request });
    const trajectory: Record<string, unknown>[] = [];
    const partialPath = join(config.jobs, `${jobKey(item)}.partial.json`);
    const saved = await loadPartial(partialPath, item, expected);
    const partial: PartialJob = saved ?? { version: TEACHER_PARTIAL_VERSION,
      program_id: item.record.id, provenance: structuredClone(expected), turns: [] };
    let replayIndex = 0;
    const driver = async (request: ModelTurnRequest): Promise<ModelTurn> => {
      const requestSha256 = sha256(canonical(request));
      const recorded = partial.turns[replayIndex];
      let response: ModelTurn;
      if (recorded) {
        if (recorded.request_sha256 !== requestSha256)
          throw new Error(`partial teacher replay diverged at model turn ${replayIndex}`);
        response = structuredClone(recorded.response);
      } else {
        if (handoff && replayIndex < handoff.prefix.length) {
          const prefix = handoff.prefix[replayIndex]!;
          if (prefix.request_sha256 !== requestSha256)
            throw new Error(`student prefix replay diverged at model turn ${replayIndex}`);
          response = structuredClone(prefix.response);
        } else if (handoff && replayIndex === handoff.prefix.length) {
          if (handoff.target_request_sha256 !== requestSha256)
            throw new Error(`teacher handoff request diverged at model turn ${replayIndex}`);
          response = await transport(request);
        } else response = replayIndex === 0 && item.record.semantics.failure_seed ?
          { calls: [['eval', { code: item.record.semantics.failure_seed.code }]], completion_tokens: 1 } :
          await transport(request);
        partial.turns.push({ request_sha256: requestSha256, response: structuredClone(response) });
        // The response is durable before its actions execute. A restart can
        // replay it into the deterministic frozen harness without another decode.
        await writeAtomic(partialPath, JSON.stringify(partial) + '\n');
      }
      replayIndex++;
      trajectory.push(trajectoryTurn(request, response));
      return response;
    };
    const root = programNode(item.record);
    const folderFiles = item.record.semantics.folder_files;
    if (folderFiles && (root.nodeKind !== 'lambda' || root.subtype !== 'directory-reducer'))
      throw new Error(`${item.record.id}: folder_files requires a directory reducer root`);
    const folder = folderFiles ? Folder.fromFiles(folderFiles) : undefined;
    if (folder) {
      root.projectTransaction = await folder.beginTransaction(false);
      root.reducerMode = 'apply';
    }
    const environment = new TypeScriptEnvironment({ mode: 'fresh' });
    const effects = effectHarness(item.record.semantics.effects ?? {});
    const agent = new NativeToolAgent(driver, { systemPrompt: config.systemPrompt, temperature: 0,
      segmentTurns: config.segmentTurns, segmentMessages: config.segmentMessages, maxTurns: config.maxTurns });
    // Seeds derive from the run ID, so it names the program and seed root only: a teacher
    // handed a student's failed state must reproduce the student's requests exactly.
    const runId = sha256(canonical({ batch: TEACHER_BATCH_VERSION, index: item.index,
      program_ir_sha256: expected.program_ir_sha256, seed_policy: expected.seed_policy })).slice(0, 32);
    // Recorded effects become host services: capability `svc.method` is method `method` of service `svc`.
    const services: Record<string, Record<string, (...args: unknown[]) => unknown>> = {};
    for (const [name, fn] of Object.entries(effects.capabilities)) {
      const [service, method] = name.split('.') as [string, string];
      (services[service] ??= {})[method] = (...args: unknown[]) => fn(args);
    }
    const runtime = new NodeNativeRuntime({ environment, agent: session => agent.run(session), services,
      seedPolicy: { mode: 'derived', root: config.rootSeed }, runId, signal });
    try {
      const result = await runtime.run(root), actual = dump(result.value);
      const actualFiles = folder ? Object.fromEntries(await Promise.all(folder.listFiles().map(async file =>
        [file.path, await folder.readText(file.path)] as const))) : undefined;
      const expectedKind = item.record.semantics.operation === 'blocked' ? 'quiesced' : 'done';
      const seededFailure = item.record.semantics.failure_seed;
      const failureSeen = !seededFailure || runtime.trace.events.some(event => event.kind === 'scope_failure' &&
        (!seededFailure.kind || event.failure_kind === seededFailure.kind));
      
      const effectsOk = same(effects.observed, effects.expected);
      const filesOk = !folder || same(actualFiles, item.record.semantics.expected_files ?? folderFiles);
      const accepted = failureSeen && result.outcome.kind === expectedKind && effectsOk && filesOk &&
        (expectedKind !== 'done' || same(actual, item.record.semantics.expected));
      const row: TeacherRow = { version: TEACHER_TRAJECTORY_VERSION,
        id: `teacher-program:${sha256(canonical([item.record.id, config.modelId, runId])).slice(0, 20)}`,
        task: { kind: 'whole_program', program_ir: item.record,
          source_program_ids: [item.record.id] } as TeacherRow['task'], provenance: { ...expected,
          trace_sha256: sha256(canonical(runtime.trace.events)) }, outcome: { status: result.outcome.kind,
          detail: result.outcome.detail, value: actual, effects: effects.observed,
          ...(actualFiles ? { files: actualFiles } : {}), accepted,
          action_ledger: runtime.trace.events.filter(event => event.kind === 'action'),
          scope_failures: runtime.trace.events.filter(event => event.kind === 'scope_failure'),
          host_events: runtime.trace.events.filter(event => event.kind === 'host') }, trajectory,
        ...(handoff ? { handoff: { student_trajectory_id: handoff.student_trajectory_id,
          student_trajectory_sha256: handoff.student_trajectory_sha256,
          handoff_at: handoff.handoff_at, failure: handoff.failure } } : {}),
        capture_limits: [] };
      await writeAtomic(join(config.jobs, `${jobKey(item)}.trace.jsonl`),
        runtime.trace.events.map(event => JSON.stringify(event)).join('\n') + '\n');
      await removeIfPresent(partialPath);
      return row;
    } finally { environment.close(); }
  };
}

export async function defaultToolSurfaceHash(root = fileURLToPath(new URL('../..', import.meta.url))): Promise<string> {
  // Resume only against the exact interpreter implementation. Compiler, type,
  // source-loading, and filesystem changes can alter an identical tool call
  // even when its public JSON schema is unchanged.
  const native = (await readdir(join(root, 'src/native'))).filter(name => name.endsWith('.ts'))
    .map(name => `src/native/${name}`);
  const files = [...native, 'src/scope-compiler.ts', 'src/environment.ts'].sort();
  const chunks = await Promise.all(files.map(path => readFile(join(root, path))));
  return sha256(Buffer.concat(chunks.flatMap((chunk, index) => index ? [Buffer.from([0]), chunk] : [chunk])));
}

export const defaultSystemPrompt = TOOLS_PROMPT;
