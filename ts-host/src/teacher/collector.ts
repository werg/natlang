import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readFile, readdir, rename, unlink } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { openAICompatibleModelTurn } from '../model/openai-compatible.js';
import { createManagedModelSession } from '../model/local-server.js';
import { TypeScriptEnvironment } from '../environment.js';
import { NativeToolAgent } from '../native/agent.js';
import { TOOLS_PROMPT } from '../native/prompt.js';
import { NodeNativeRuntime } from '../node-runtime.js';
import { Folder } from '../native/scoped-fs.js';
import { dump } from '../native/values.js';
import { externalModule } from '../native/external.js';
import { PROGRAM_VERSION, programNode, type ProgramRecord } from './program.js';
import { checkAuthoring, type AuthoringSpec } from './authoring.js';
import { WorldBridge, type WorldSpec } from './world-bridge.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';

export const TEACHER_BATCH_VERSION = 'natlang.teacher_batch.native/1';
export const TEACHER_TRAJECTORY_VERSION = 'natlang.teacher_trajectory.native/1';
export const TEACHER_PARTIAL_VERSION = 'natlang.teacher_partial.native/1';
const TOOL_SCHEMA = 'scope-eval-v1';
export const EXECUTION_PLAN_VERSION = 'execution-plan-tool/2';
// The plan is the turn's training reasoning: it says what decides the step (for a conclusion, the reasons that settle
// it) before the step, so a concluding turn does not train a conclusion without its reasons.
export const EXECUTION_PLAN_PROMPT = 'Before taking the next action, make a concise execution plan from the current ' +
  'instructions and evidence. First state briefly what the evidence so far shows that decides this step (for a ' +
  'conclusion, the reasons that settle it), then plan only the next useful step and any immediate checks it needs; ' +
  'do not execute the step yet. Call execution_plan exactly once with the plan.';
const EXECUTION_PLAN_TOOL = { type: 'function', function: { name: 'execution_plan',
  description: 'Record the concise plan that will guide the next action. This tool does not execute the plan.',
  parameters: { type: 'object', properties: { plan: { type: 'string',
    description: 'Briefly, what the evidence shows that decides the next step and why; then the step.' } },
    required: ['plan'], additionalProperties: false } } };

export type { ProgramRecord };
export type IndexedRecord = { index: number; record: ProgramRecord };
export type ProvenanceOptions = { modelId: string; rootSeed: number; systemPrompt: string;
  /** The agent's context budget in prompt tokens (see NativeToolAgent contextTokens). */
  contextTokens: number; toolSurfaceSha256: string;
  /** Model turns allowed per call; unlimited unless set. A collection run should set one. */
  maxTurns?: number;
  /** Sampling temperature; greedy unless set. Reasoning models are tuned for sampling (Ling: 1.0) and, decoded
   * greedily, can skip their thinking. */
  temperature?: number;
  endpoint?: string; provider?: string; piOptions?: Record<string, unknown>;
  request?: Record<string, unknown>; cacheStableTools?: boolean;
  /** Elicit one required execution_plan tool call before every model action and use it as the turn's reasoning. */
  executionPlans?: boolean; executionPlanTokens?: number;
  handoffs?: Map<string, HandoffRecord>;
  /** Who answered: a model being taught (student), a teacher, or a case's scripted reference solution. */
  collectionRole?: 'student' | 'teacher' | 'reference' };
export type HandoffRecord = { version: 'natlang.hard_state/1'; id: string;
  program_ir_sha256: string; student_trajectory_id: string; student_trajectory_sha256: string;
  handoff_at: number; target_request_sha256: string;
  prefix: { request_sha256: string; response: ModelTurn }[]; failure: Record<string, unknown>;
  student_provenance: Record<string, unknown> };
export type CollectorConfig = ProvenanceOptions & { jobs: string; output: string; workers: number;
  transportRetries?: number; retryDelayMs?: number;
  /** Worker n starts its first job n times this later, so a rate-limited provider does not see them all at once. */
  workerStaggerMs?: number;
  /** Result files of earlier runs whose finished rows stand in for jobs of the same program (see reusedRow). */
  reuse?: string[];
  /** The model server's shared KV buffer in tokens; requests wait to fit into it (see KvBudget). Unset: no limit. */
  kvTokens?: number;
  /** Earlier tool surfaces whose rows the operator declares equivalent to the current one for reuse. */
  reuseSurfaces?: string[] };
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
    system_prompt_sha256: sha256(options.systemPrompt), context_tokens: options.contextTokens,
    transport: options.provider ? 'pi-provider' : 'openai-compatible',
    ...(options.provider ? { provider: options.provider, pi_options: options.piOptions ?? {} } : {}),
    ...(options.maxTurns === undefined ? {} : { max_turns: options.maxTurns }),
    ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
    ...(options.cacheStableTools ? { cache_stable_tools: true } : {}),
    ...(options.executionPlans ? { execution_plans: { version: EXECUTION_PLAN_VERSION,
      max_tokens: options.executionPlanTokens ?? 512 } } : {}),
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

const errorText = (error: unknown) => String(error instanceof Error ? `${error.name}: ${error.message}` : error).toLowerCase();

/** A provider refusing for request rate or concurrency (a subscription's burst limit): wait longer, then go on. */
function rateLimited(error: unknown): boolean {
  return /rate limit|too many requests|\b429\b/.test(errorText(error));
}

function transportFailure(error: unknown): boolean {
  const text = errorText(error);
  return rateLimited(error) || ['connection refused', 'connection reset', 'fetch failed', 'socket', 'timed out', 'econnreset',
    'econnrefused', 'remote end closed', 'headerstimeout', 'bodytimeout'].some(phrase => text.includes(phrase)) ||
    // A restarting server answers 502/503 (llama.cpp: "Loading model") until it is ready.
    /\bmodel http (?:502|503|504)\b/.test(text) ||
    // Slots share one KV buffer: when the running sequences together fill it, the server fails them all, and
    // space frees as soon as any finishes. The job resumes from its journal.
    text.includes('context size has been exceeded');
}
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Finished rows of earlier runs, by program digest. A row stands in for a job when the program, model, turn budget,
 * collection role, and handoff match, and it would have been collected the same way now (see reusedRow).
 * The tool surface and system prompt may differ only when the row was migrated
 * to the current finishing surface (scripts/inline-curriculum/migrate-status.mjs), since the row keeps the context
 * it was collected with. Later files win over earlier ones.
 */
async function reusableRows(paths: string[]): Promise<Map<string, Array<{ row: TeacherRow; path: string }>>> {
  const rows = new Map<string, Array<{ row: TeacherRow; path: string }>>();
  for (const path of paths) for (const line of (await readFile(path, 'utf8')).split(/\r?\n/)) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as TeacherRow, digest = row.provenance?.program_ir_sha256;
    if (typeof digest === 'string') rows.set(digest, [...rows.get(digest) ?? [], { row, path }]);
  }
  return rows;
}
/** Truncation notes from before cutoff.ts: read_page page markers, CUT OFF previews, comment cut-offs, char counts. */
const RETIRED_CUT_OFFS = /shown; read_page\(|CUT OFF: only the beginning|\/\* cut off:|more \(read to see\)|\(\d+ chars\)|more fields \(read to see\)/;
const REUSE_KEYS = ['program_ir_sha256', 'model', 'collection_role', 'handoff_sha256'];
/** Turns before the limit at which the model is first told how many are left (native/agent.ts). */
const TURN_NOTICE = 4;
function reusedRow(found: { row: TeacherRow; path: string }, expected: Record<string, unknown>,
  surfaces: string[] = []): TeacherRow | undefined {
  const { row } = found;
  // A row reused before is judged by where it was actually collected, and keeps pointing there.
  const earlier = row.provenance.reused_from as { path: string; provenance: Record<string, unknown> } | undefined;
  const provenance = earlier?.provenance ?? row.provenance, path = earlier?.path ?? found.path;
  if (!REUSE_KEYS.every(key => canonical(provenance[key] ?? (key === 'collection_role' ? 'teacher' : undefined)) === canonical(expected[key])))
    return;
  // Under another turn limit a run sees the same requests as long as no call came near either limit: the model is
  // told how many turns are left from TURN_NOTICE turns before the end. All turns, child calls' too, are counted.
  if (canonical(provenance.max_turns) !== canonical(expected.max_turns)) {
    const limits = [provenance.max_turns, expected.max_turns].map(Number);
    if (!limits.every(Number.isFinite) || ((row.trajectory ?? []) as unknown[]).length > Math.min(...limits) - TURN_NOTICE) return;
  }
  if (provenance.tool_surface_sha256 !== expected.tool_surface_sha256 && provenance.finish_surface_migration === undefined &&
      !surfaces.includes(String(provenance.tool_surface_sha256))) return;
  // A row stands in for a new run only if that run would have seen the same requests: it never rolled over into a
  // checkpoint (retired), and none of its prompts was large enough for compaction to have elided outputs.
  const turns = (row.trajectory ?? []) as Array<{ phase?: string; model_response?: { prompt_tokens?: number } }>;
  const budget = Number(expected.context_tokens);
  if (turns.some(turn => turn.phase === 'checkpoint' ||
      (Number.isFinite(budget) && (turn.model_response?.prompt_tokens ?? 0) > budget * 0.75))) return;
  // Nor did anything get shortened the way the runtime no longer shortens it (see native/cutoff.ts), or run into a
  // runtime fault fixed since: an unparsable inferred type, a failed eval with services but no report of the calls
  // it had already made, or tool-call markup taken as the result.
  const text = JSON.stringify(row.trajectory ?? []), semantics = row.task.program_ir.semantics as Record<string, unknown>;
  if (RETIRED_CUT_OFFS.test(text) || text.includes('bad character at')) return;
  if ((semantics.effects || semantics.world) && text.includes('Nothing else from this eval was kept') &&
      !text.includes('Already performed before the failure')) return;
  if (/<\/?(?:tool_call|function|parameter)\b/.test(JSON.stringify(row.outcome?.value ?? null))) return;
  return { ...row, provenance: { ...expected, reused_from: { path, provenance } } };
}

/** Queue incomplete jobs, publish each result atomically, and rebuild the ordered merge after every job. */
export async function collectBatch(records: IndexedRecord[], config: CollectorConfig, runner: JobRunner,
  signal?: AbortSignal): Promise<{ completed: number; missing: number[] }> {
  if (!Number.isInteger(config.workers) || config.workers < 1) throw new RangeError('workers must be positive');
  await mkdir(config.jobs, { recursive: true });
  const pending: IndexedRecord[] = [];
  const reusable = config.reuse?.length ? await reusableRows(config.reuse) :
    new Map<string, Array<{ row: TeacherRow; path: string }>>();
  let reused = 0;
  for (const item of records) {
    const expected = expectedProvenance(item.record, config), path = join(config.jobs, `${jobKey(item)}.result.json`);
    if (await readMatching(path, item.record, expected)) continue;
    // The last listed file with a qualifying row wins; a row that does not qualify never hides an earlier one.
    const row = (reusable.get(expected.program_ir_sha256 as string) ?? []).map(found => reusedRow(found, expected, config.reuseSurfaces))
      .filter(Boolean).at(-1);
    if (row) { await writeAtomic(path, JSON.stringify(row) + '\n'); reused++; continue; }
    pending.push(item);
  }
  if (reused) process.stderr.write(`reused ${reused} finished rows from ${config.reuse!.length} earlier result files\n`);
  // Work is picked up in a fixed pseudo-random order rather than shard order: a shard keeps a case's variants and
  // hint twins together, and running several long cases of one family at once fills the server's shared KV buffer.
  // The order depends only on the job keys, so it is the same on every resume; the merged output stays in shard order.
  pending.sort((a, b) => sha256(jobKey(a)).localeCompare(sha256(jobKey(b))));
  await mergeCompleted(records, config);
  let cursor = 0;
  const worker = async (slot: number) => {
    if (slot && config.workerStaggerMs) await delay(slot * config.workerStaggerMs);
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
        const limited = rateLimited(error);
        if (!transportFailure(error) || attempt >= (config.transportRetries ?? 8) * (limited ? 3 : 1)) {
          await writeAtomic(join(config.jobs, `${String(item.index).padStart(6, '0')}.error.json`),
            JSON.stringify({ index: item.index, program_id: item.record.id,
              error: `${error instanceof Error ? error.name : 'Error'}: ${error instanceof Error ? error.message : String(error)}` }) + '\n');
          break;
        }
        // Jitter spreads out jobs that failed together, so they do not all return at once.
        // A rate limit lifts on the provider's clock, not ours: those waits start longer and grow further.
        const wait = limited ? Math.min(120_000, (config.retryDelayMs ?? 5_000) * 3 * 2 ** Math.min(attempt++, 3)) * (0.5 + Math.random()) :
          Math.min(30_000, (config.retryDelayMs ?? 5_000) * 2 ** Math.min(attempt++, 3)) * (0.5 + Math.random());
        if (wait) await delay(wait);
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(config.workers, Math.max(1, pending.length)) }, (_, slot) => worker(slot)));
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

/** The trajectory record of one model turn: the request's context and the response, as training reads them. */
export function trajectoryTurn(request: ModelTurnRequest, response: ModelTurn): Record<string, unknown> {
  const raw = response.raw_response as Record<string, unknown> | undefined;
  const message = ((raw?.choices as Record<string, unknown>[] | undefined)?.[0]?.message ?? {}) as Record<string, unknown>;
  const planned = Object.hasOwn(response, 'execution_plan');
  const retainedReasoning = planned ? response.execution_plan :
    response.reasoning ?? message.reasoning_content ?? message.reasoning ?? message.thinking ?? null;
  return { phase: 'action', context: structuredClone(request.messages),
    request_sha256: sha256(canonical(request)),
    model_response: { calls: structuredClone(response.calls ?? []), text: response.text ?? '',
      raw_calls: structuredClone(response.raw_calls ?? []),
      ...(planned ? { execution_plan: response.execution_plan } : {}),
      ...(response.completion_tokens === undefined ? {} : { completion_tokens: response.completion_tokens }),
      ...(response.prompt_tokens === undefined ? {} : { prompt_tokens: response.prompt_tokens }) },
    tools_offered: structuredClone(request.tools), assistant: { content: response.text ?? '',
      ...(planned ? { execution_plan: response.execution_plan } : {}),
      reasoning: retainedReasoning,
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

/**
 * Admission to the model server's shared KV buffer. Parallel slots share one buffer; when the sequences in flight
 * together outgrow it, the server fails all of them. A request waits until its estimated size (prompt, plus room to
 * generate) fits beside those in flight; one that is larger than the whole budget runs alone.
 */
export class KvBudget {
  private used = 0;
  private readonly waiting: { need: number; start: () => void }[] = [];
  constructor(readonly tokens: number) {}
  async acquire(need: number): Promise<void> {
    if (this.fits(need) && !this.waiting.length) { this.used += need; return; }
    await new Promise<void>(start => this.waiting.push({ need, start }));
  }
  release(need: number): void {
    this.used -= need;
    // First come, first served: a large request is not overtaken indefinitely by small ones.
    while (this.waiting.length && this.fits(this.waiting[0]!.need)) {
      const next = this.waiting.shift()!;
      this.used += next.need;
      next.start();
    }
  }
  private fits(need: number): boolean { return this.used === 0 || this.used + need <= this.tokens; }
}
/**
 * Tokens a request is expected to occupy: its prompt at 4 characters per token (measured on Bonsai collection: 4.06 at
 * the median) and a typical reply (p90 399 tokens). Deliberately not a worst case: an occasional overflow is retried,
 * while an overcautious estimate would idle slots.
 */
const requestTokens = (request: ModelTurnRequest) =>
  Math.ceil((JSON.stringify(request.messages).length + JSON.stringify(request.tools).length) / 4) + 512;

/**
 * Wrap an action driver with a separate planning turn. The action request itself remains the collector's canonical
 * request: the synthetic prompt/tool exchange is visible to the action model, but is not spliced into training IR.
 * Required tool choice is sent when the backend supports it, and the result is checked here for every backend.
 */
export function withExecutionPlans(send: (request: ModelTurnRequest) => Promise<ModelTurn>,
  options: { maxTokens?: number; attempts?: number } = {}): (request: ModelTurnRequest) => Promise<ModelTurn> {
  const planLimit = options.maxTokens ?? 512, attempts = options.attempts ?? 2;
  if (!Number.isInteger(planLimit) || planLimit < 1) throw new RangeError('execution plan token limit must be positive');
  if (!Number.isInteger(attempts) || attempts < 1) throw new RangeError('execution plan attempts must be positive');
  return async request => {
    const maxTokens = request.max_tokens === null ? planLimit : Math.min(planLimit, request.max_tokens);
    let plan = '', planningCompletionTokens = 0;
    let retryMessages: unknown[] = [...request.messages, { role: 'user', content: EXECUTION_PLAN_PROMPT }];
    for (let attempt = 0; attempt < attempts; attempt++) {
      let planned: ModelTurn;
      try {
        planned = await send({ ...request, messages: retryMessages, tools: [EXECUTION_PLAN_TOOL],
          tool_choice: 'required', max_tokens: maxTokens });
      } catch (error) {
        // An outage or rate limit fails the job, which resumes from its journal: skipping the plan would leave a
        // training turn without reasoning.
        if (transportFailure(error)) throw error;
        // Required tool selection is not universal. Otherwise planning is best-effort: the ordinary action still runs.
        break;
      }
      planningCompletionTokens += planned.completion_tokens ?? maxTokens;
      const calls = planned.calls ?? [], value = calls.length === 1 && calls[0]![0] === 'execution_plan' ?
        calls[0]![1].plan : undefined;
      if (typeof value === 'string' && value.trim()) { plan = value.trim(); break; }
      retryMessages = [...retryMessages,
        { role: 'assistant', content: planned.text ?? '' },
        { role: 'user', content: 'The plan was not recorded. Call execution_plan exactly once with a nonempty plan.' }];
    }
    if (!plan) {
      const action = await send(request);
      // Presence of null suppresses an opaque provider reasoning trace in captured IR: this mode retains plans only.
      return { ...action, execution_plan: null, reasoning: undefined,
        completion_tokens: planningCompletionTokens + (action.completion_tokens ?? 0) };
    }
    const planCall = { id: 'execution_plan_0', type: 'function',
      function: { name: 'execution_plan', arguments: JSON.stringify({ plan }) } };
    const actionLimit = request.max_tokens === null ? null : Math.max(1, request.max_tokens - planningCompletionTokens);
    const action = await send({ ...request, max_tokens: actionLimit, messages: [...request.messages,
      { role: 'assistant', content: '', tool_calls: [planCall] },
      { role: 'tool', tool_call_id: planCall.id, content: 'Plan recorded. Now take the planned next step.' }] });
    // The plan deliberately replaces provider reasoning: it is the reproducible thinking signal retained in IR and
    // threaded through later action history. Keep the action response otherwise intact for execution and accounting.
    return { ...action, execution_plan: plan, reasoning: plan,
      completion_tokens: planningCompletionTokens + (action.completion_tokens ?? actionLimit ?? 0) };
  };
}

export function nativeJobRunner(config: CollectorConfig): JobRunner {
  if (!config.endpoint && !config.provider) throw new Error('endpoint or Pi provider is required for native teacher collection');
  if (config.endpoint && config.provider) throw new Error('teacher collection cannot use both endpoint and Pi provider');
  const kv = config.kvTokens ? new KvBudget(config.kvTokens) : undefined;
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
        handoff.student_provenance.context_tokens !== config.contextTokens))
      throw new Error(`${item.record.id}: student handoff runtime/prompt/seed settings differ`);
    const session = config.provider ? createManagedModelSession({ provider: config.provider,
      model: config.modelId, piOptions: config.piOptions }) : undefined;
    let ready: Promise<unknown> | undefined;
    const send = session ? async (request: ModelTurnRequest) => {
      await (ready ??= session.prepare());
      return session.turn(request);
    } : openAICompatibleModelTurn({ endpoint: config.endpoint!, model: config.modelId,
      request: config.request });
    try {
    const admittedSend = kv ? async (request: ModelTurnRequest) => {
      const need = requestTokens(request);
      await kv.acquire(need);
      try { return await send(request); } finally { kv.release(need); }
    } : send;
    const transport = config.executionPlans ? withExecutionPlans(admittedSend,
      { maxTokens: config.executionPlanTokens }) : admittedSend;
    const trajectory: Record<string, unknown>[] = [];
    const partialPath = join(config.jobs, `${jobKey(item)}.partial.json`);
    const saved = await loadPartial(partialPath, item, expected);
    const partial: PartialJob = saved ?? { version: TEACHER_PARTIAL_VERSION,
      program_id: item.record.id, provenance: structuredClone(expected), turns: [] };
    let replayIndex = 0;
    // Journaled responses are replayed by their exact request, not by position: the child calls of one eval run
    // concurrently, so their requests can reach the model in a different order after a restart. A request with no
    // unused journal entry is decoded live.
    const unused = new Map<string, PartialJob['turns']>();
    for (const turn of partial.turns) unused.set(turn.request_sha256, [...unused.get(turn.request_sha256) ?? [], turn]);
    const driver = async (request: ModelTurnRequest): Promise<ModelTurn> => {
      const requestSha256 = sha256(canonical(request));
      const recorded = unused.get(requestSha256)?.shift();
      let response: ModelTurn;
      if (recorded) {
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
    const runId = programRunId(item.index, expected);
    const run = await executeProgram(item.record, driver, { ...config, runId, signal });
    const row = programRow(item.record, config.modelId, runId, expected, run, trajectory,
      handoff ? { handoff: { student_trajectory_id: handoff.student_trajectory_id,
        student_trajectory_sha256: handoff.student_trajectory_sha256,
        handoff_at: handoff.handoff_at, failure: handoff.failure } } : {});
    await writeAtomic(join(config.jobs, `${jobKey(item)}.trace.jsonl`),
      run.trace.map(event => JSON.stringify(event)).join('\n') + '\n');
    await removeIfPresent(partialPath);
    return row;
    } finally { await session?.close(); }
  };
}

/**
 * The ID a program's run derives its seeds from. It names the program and seed root only: a teacher handed a
 * student's failed state must reproduce the student's requests exactly.
 */
export function programRunId(index: number, expected: Record<string, unknown>): string {
  return sha256(canonical({ batch: TEACHER_BATCH_VERSION, index,
    program_ir_sha256: expected.program_ir_sha256, seed_policy: expected.seed_policy })).slice(0, 32);
}

/** The result row of one run of a program, whoever answered its turns. */
export function programRow(record: ProgramRecord, modelId: string, runId: string, expected: Record<string, unknown>,
  run: ProgramRun, trajectory: Record<string, unknown>[], extra: Record<string, unknown> = {}): TeacherRow {
  return { version: TEACHER_TRAJECTORY_VERSION,
    id: `teacher-program:${sha256(canonical([record.id, modelId, runId])).slice(0, 20)}`,
    task: { kind: 'whole_program', program_ir: record, source_program_ids: [record.id] } as TeacherRow['task'],
    provenance: { ...expected, trace_sha256: sha256(canonical(run.trace)) }, outcome: run.outcome, trajectory,
    ...extra, capture_limits: [] };
}

export type ExecuteOptions = { systemPrompt: string; contextTokens: number;
  maxTurns?: number; temperature?: number; rootSeed: number; runId: string; signal?: AbortSignal };
export type ProgramRun = { outcome: Record<string, unknown> & { accepted: boolean }; trace: Record<string, unknown>[] };

/**
 * Run a program's root invocation with a model driver in a fresh environment and check the result,
 * effects, and folder against its contract. The collector and reference replays share this path.
 */
export async function executeProgram(record: ProgramRecord, driver: (request: ModelTurnRequest) => Promise<ModelTurn>,
  options: ExecuteOptions): Promise<ProgramRun> {
  const root = programNode(record);
  const folderFiles = record.semantics.folder_files;
  if (folderFiles && (root.nodeKind !== 'lambda' || root.subtype !== 'directory-reducer'))
    throw new Error(`${record.id}: folder_files requires a directory reducer root`);
  const folder = folderFiles ? Folder.fromFiles(folderFiles) : undefined;
  if (folder) {
    root.projectTransaction = await folder.beginTransaction(false);
    root.reducerMode = 'apply';
  }
  const environment = new TypeScriptEnvironment({ mode: 'fresh' });
  const effects = effectHarness(record.semantics.effects ?? {});
  const agent = new NativeToolAgent(driver, { systemPrompt: options.systemPrompt, temperature: options.temperature ?? 0,
    contextTokens: options.contextTokens, maxTurns: options.maxTurns });
  // Recorded effects become host services: capability `svc.method` is method `method` of service `svc`.
  const services: Record<string, Record<string, (...args: unknown[]) => unknown>> = {};
  for (const [name, fn] of Object.entries(effects.capabilities)) {
    const [service, method] = name.split('.') as [string, string];
    (services[service] ??= {})[method] = (...args: unknown[]) => fn(args);
  }
  // An interactive world in its own process becomes the service `world`; the task is done when its score reaches 100.
  // External modules (semantics.services: name -> TypeScript source) run in the host as services; the model sees
  // their declarations only (native/external.ts).
  const declarations: Record<string, string> = {};
  for (const [name, source] of Object.entries((record.semantics as { services?: Record<string, string> }).services ?? {})) {
    const external = externalModule(name, source);
    services[name] = external.exports as Record<string, (...args: unknown[]) => unknown>;
    declarations[name] = external.declaration;
  }
  const worldSpec = (record.semantics as { world?: WorldSpec }).world;
  const world = worldSpec ? await WorldBridge.open(worldSpec) : undefined;
  if (world) { services.world = world.service(); declarations.world = WorldBridge.declaration(worldSpec!.kind); }
  const serviceScopes = (record.semantics as { service_scopes?: Record<string, string[]> }).service_scopes;
  const runtime = new NodeNativeRuntime({ environment, agent: session => agent.run(session), services, declarations,
    ...(serviceScopes ? { serviceScopes } : {}),
    seedPolicy: { mode: 'derived', root: options.rootSeed }, runId: options.runId, signal: options.signal });
  try {
    const result = await runtime.run(root), actual = dump(result.value);
    const actualFiles = folder ? Object.fromEntries(await Promise.all(folder.listFiles().map(async file =>
      [file.path, await folder.readText(file.path)] as const))) : undefined;
    const expectedKind = record.semantics.operation === 'blocked' ? 'quiesced' : 'done';
    const seededFailure = record.semantics.failure_seed;
    const failureSeen = !seededFailure || runtime.trace.events.some(event => event.kind === 'scope_failure' &&
      (!seededFailure.kind || event.failure_kind === seededFailure.kind));
    const effectsOk = same(effects.observed, effects.expected);
    // An authoring task is judged by running what was written, not by the files' exact text or the call's reply.
    const authoringSpec = (record.semantics as { authoring?: AuthoringSpec }).authoring;
    const authoring = authoringSpec && actualFiles ? await checkAuthoring(actualFiles, authoringSpec) : undefined;
    const filesOk = !folder || (authoring ? authoring.ok : same(actualFiles, record.semantics.expected_files ?? folderFiles));
    // A blocked case needs the model's own blocked or failed call; running out of turns also quiesces.
    const honestStop = expectedKind !== 'quiesced' || /^(?:blocked|error): /.test(String(result.outcome.detail ?? ''));
    const worldScore = world ? await world.request('score') as { score: number; done: boolean } : undefined;
    const worldOk = !worldScore || worldScore.score >= 100;
    const accepted = failureSeen && result.outcome.kind === expectedKind && honestStop && effectsOk && filesOk && worldOk &&
      (expectedKind !== 'done' || !!authoring || !!world || same(actual, record.semantics.expected));
    const trace = runtime.trace.events as unknown as Record<string, unknown>[];
    return { trace, outcome: { status: result.outcome.kind, detail: result.outcome.detail, value: actual,
      effects: effects.observed, ...(actualFiles ? { files: actualFiles } : {}), ...(authoring ? { authoring } : {}),
      ...(worldScore ? { world: worldScore } : {}), accepted,
      // Every call's actions, children included: a child nl call runs in its own runtime and reports its trace to
      // the task (call_id tells them apart), so its decisions can be linked to what they did.
      action_ledger: [...trace, ...(runtime.frame?.task.traces ?? []).flatMap(child => child.events)]
        .filter(event => event.kind === 'action'),
      scope_failures: trace.filter(event => event.kind === 'scope_failure'),
      host_events: trace.filter(event => event.kind === 'host') } };
  } finally { environment.close(); world?.close(); }
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
