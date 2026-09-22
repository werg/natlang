import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readFile, rename } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { openAICompatibleModelTurn } from '../model/openai-compatible.js';
import { TypeScriptEnvironment } from '../environment.js';
import { NativeToolAgent } from '../native/agent.js';
import { EXPLICIT_TOOLS_PROMPT } from '../native/prompt.js';
import { NativeRuntime } from '../native/runtime.js';
import { TypeEnv } from '../native/types.js';
import { buildPending, coerce, dump, isPending } from '../native/values.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';

export const TEACHER_BATCH_VERSION = 'natlang.teacher_batch.native/1';
export const TEACHER_TRAJECTORY_VERSION = 'natlang.teacher_trajectory.native/1';
const PROGRAM_VERSION = 'natlang.program/1';
const TOOL_SCHEMA = 'scope-eval-v1';

export type ProgramRecord = { version: string; id: string; kind: string;
  semantics: { root: Record<string, unknown>; inputs: Record<string, unknown>; expected: unknown;
    operation?: string; effects?: Record<string, unknown> }; [key: string]: unknown };
export type IndexedRecord = { index: number; record: ProgramRecord };
export type ProvenanceOptions = { modelId: string; rootSeed: number; systemPrompt: string;
  segmentTurns: number; segmentMessages: number; toolSurfaceSha256: string;
  endpoint?: string; request?: Record<string, unknown>; cacheStableTools?: boolean };
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
  if (!record.id || !record.semantics || typeof record.semantics !== 'object' ||
      !record.semantics.root || !record.semantics.inputs || !Object.hasOwn(record.semantics, 'expected'))
    throw new Error('invalid focused program IR record');
}

export async function loadRecords(path: string, start = 0, limit = 10): Promise<IndexedRecord[]> {
  if (!Number.isInteger(start) || start < 0 || !Number.isInteger(limit) || limit < 1)
    throw new RangeError('start must be nonnegative and limit must be positive');
  const lines = (await readFile(path, 'utf8')).split(/\r?\n/);
  const records: IndexedRecord[] = [];
  for (let index = start; index < lines.length && records.length < limit; index++) {
    if (!lines[index]!.trim()) continue;
    const record = JSON.parse(lines[index]!) as ProgramRecord;
    validateFocusedRecord(record);
    records.push({ index, record });
  }
  if (records.length !== limit) throw new Error('requested source range exceeds the frozen batch');
  return records;
}

export function jobKey({ index, record }: IndexedRecord): string {
  return `${String(index).padStart(6, '0')}-${recordDigest(record).slice(0, 16)}`;
}

export function expectedProvenance(record: ProgramRecord, options: ProvenanceOptions): Record<string, unknown> {
  return { program_ir_sha256: recordDigest(record), model: options.modelId, tool_schema: TOOL_SCHEMA,
    runtime: 'typescript-native', collector_version: TEACHER_BATCH_VERSION,
    tool_surface_sha256: options.toolSurfaceSha256, seed_policy: { mode: 'derived', root: options.rootSeed },
    system_prompt_sha256: sha256(options.systemPrompt), segment_turns: options.segmentTurns,
    segment_messages: options.segmentMessages, transport: 'openai-compatible',
    ...(options.cacheStableTools ? { cache_stable_tools: true } : {}) };
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

function bindInputs(root: ReturnType<typeof buildPending>, inputs: Record<string, unknown>): void {
  if (!Object.keys(inputs).length) return;
  if (!isPending(root) || root.nodeKind !== 'lambda' || root.type.kind !== 'lambda')
    throw new TypeError('program inputs require a root Lambda');
  const env = new TypeEnv().child(root.types);
  for (const [name, value] of Object.entries(inputs)) {
    const field = root.type.params.fields.find(item => item.name === name);
    if (!field) throw new TypeError(`${name} is not a program parameter`);
    root.args[name] = coerce(value, field.type, env, `args/${name}`);
  }
}

function trajectoryTurn(request: ModelTurnRequest, response: ModelTurn): Record<string, unknown> {
  const raw = response.raw_response as Record<string, unknown> | undefined;
  const message = ((raw?.choices as Record<string, unknown>[] | undefined)?.[0]?.message ?? {}) as Record<string, unknown>;
  return { phase: request.tools.length ? 'action' : 'checkpoint', context: request.messages,
    tools_offered: request.tools, assistant: { content: response.text ?? '',
      reasoning: message.reasoning_content ?? message.reasoning ?? message.thinking ?? null,
      calls: (response.calls ?? []).map(([tool, args]) => ({ tool, source_tool: tool, arguments: args, call_id: null })) },
    raw_response_sha256: raw ? sha256(canonical(raw)) : null };
}

export function nativeJobRunner(config: CollectorConfig): JobRunner {
  if (!config.endpoint) throw new Error('endpoint is required for native teacher collection');
  return async (item, expected, signal) => {
    const transport = openAICompatibleModelTurn({ endpoint: config.endpoint!, model: config.modelId,
      request: config.request });
    const trajectory: Record<string, unknown>[] = [];
    const driver = async (request: ModelTurnRequest): Promise<ModelTurn> => {
      const response = await transport(request); trajectory.push(trajectoryTurn(request, response)); return response;
    };
    const root = buildPending(item.record.semantics.root);
    bindInputs(root, item.record.semantics.inputs);
    const environment = new TypeScriptEnvironment({ mode: 'fresh' });
    const capabilities = Object.fromEntries(Object.keys(item.record.semantics.effects ?? {}).map(name => [name, () => null]));
    const agent = new NativeToolAgent(driver, { systemPrompt: config.systemPrompt, temperature: 0,
      segmentTurns: config.segmentTurns, segmentMessages: config.segmentMessages, toolSchema: TOOL_SCHEMA,
      validationFeedback: 'caller' });
    const runId = sha256(canonical({ batch: TEACHER_BATCH_VERSION, index: item.index, ...expected })).slice(0, 32);
    const runtime = new NativeRuntime({ environment, agent: session => agent.run(session), capabilities,
      seedPolicy: { mode: 'derived', root: config.rootSeed }, runId, signal });
    try {
      const result = await runtime.runRoot(root), actual = dump(result.value);
      const expectedKind = item.record.semantics.operation === 'blocked' ? 'quiesced' : 'done';
      const accepted = result.outcome.kind === expectedKind &&
        (expectedKind !== 'done' || same(actual, item.record.semantics.expected));
      const row: TeacherRow = { version: TEACHER_TRAJECTORY_VERSION,
        id: `teacher-program:${sha256(canonical([item.record.id, config.modelId, runId])).slice(0, 20)}`,
        task: { kind: 'whole_program', program_ir: item.record,
          source_program_ids: [item.record.id] } as TeacherRow['task'], provenance: { ...expected,
          trace_sha256: sha256(canonical(runtime.trace.events)) }, outcome: { status: result.outcome.kind,
          detail: result.outcome.detail, value: actual, effects: result.emitted, accepted,
          action_ledger: runtime.trace.events.filter(event => event.kind === 'action') }, trajectory,
        capture_limits: [] };
      await writeAtomic(join(config.jobs, `${jobKey(item)}.trace.jsonl`),
        runtime.trace.events.map(event => JSON.stringify(event)).join('\n') + '\n');
      return row;
    } finally { runtime.close(); environment.close(); }
  };
}

export async function defaultToolSurfaceHash(root = fileURLToPath(new URL('../..', import.meta.url))): Promise<string> {
  const files = ['src/native/agent.ts', 'src/native/runtime.ts', 'src/native/values.ts',
    'src/native/prompt.ts', 'src/environment.ts'];
  const chunks = await Promise.all(files.map(path => readFile(join(root, path))));
  return sha256(Buffer.concat(chunks.flatMap((chunk, index) => index ? [Buffer.from([0]), chunk] : [chunk])));
}

export const defaultSystemPrompt = EXPLICIT_TOOLS_PROMPT;
