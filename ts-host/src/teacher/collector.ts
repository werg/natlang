import { controlledProviderProfile, type ProviderRequestControls } from './provider-request-controls.js';
import { createHash, randomUUID } from 'node:crypto';
import { setTimeout as sleep } from 'node:timers/promises';
import { rateLimited, transportFailure, retryWaitMs, retryAfterMs, providerFinishReason } from './retry.js';
import { mkdir, open, readFile, readdir, rename, unlink } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { openAICompatibleModelTurn } from '../model/openai-compatible.js';
import { createManagedModelSession } from '../model/local-server.js';
import { TypeScriptEnvironment } from '../environment.js';
import { NativeToolAgent } from '../native/agent.js';
import { GENERATION_GUIDANCE, TOOLS_PROMPT } from '../native/prompt.js';
import { NodeNativeRuntime } from '../node-runtime.js';
import { Folder } from '../native/scoped-fs.js';
import { dump } from '../native/values.js';
import { externalModule } from '../native/external.js';
import { PROGRAM_VERSION, prepareProgramNode, type ProgramRecord } from './program.js';
import { fileReturnValue, checkFilesWithJudge, checkFileReturn, DATA_QUALITY_VERSION, FILE_CONTENT_COMPARISON_VERSION, checkOracle } from './oracle.js';
import { modelOracleJudge } from './model-judge.js';
import { callMatcher } from './replay.js';
import type { FileToolSurface } from '../native/prompt.js';
import type { Handoff } from './handoff.js';
import { replacesPlantedFailure } from './seeded-failure.js';
import { quarantineReason, retiredFamily, generationHoldReason } from './curriculum-policy.js';
import { checkAuthoring, type AuthoringSpec } from './authoring.js';
import { WorldBridge, type WorldSpec } from './world-bridge.js';
import { ANSWER_COMPARISON_VERSION } from '../evaluation/oracles.js';
import type { ModelStreamProgress, ModelTurn, ModelTurnRequest } from '../contracts.js';
import { closeProviderSession, ProviderActionCycleTimeoutError, ProviderRequestTimeoutError,
  withProviderActionCycle, withProviderRequestDeadline } from './provider-deadline.js';
import type { CollectionLivenessSnapshot } from './collection-liveness.js';

export const TEACHER_BATCH_VERSION = 'natlang.teacher_batch.native/1';
export const TEACHER_TRAJECTORY_VERSION = 'natlang.teacher_trajectory.native/1';
export const TEACHER_PARTIAL_VERSION = 'natlang.teacher_partial.native/1';
const TOOL_SCHEMA = 'scope-eval-v1';
export const EXECUTION_PLAN_VERSION = 'execution-plan-tool/2';
export const PROVIDER_REQUEST_TIMEOUT_POLICY_VERSION = 'pi-provider-request-timeout/1';
export const PROVIDER_ACTION_CYCLE_POLICY_VERSION = 'pi-provider-action-cycle/1';
const PROVIDER_CLEANUP_TIMEOUT_MS = 15_000;
const MAX_PROVIDER_TIMEOUT_MS = 2_147_483_647;
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
  /** Caller-supplied execution adapter for external fixtures; identity is pinned in provenance. */
  execution?: {identity:string;run:typeof executeProgram};
  maxTurns?: number;
  /** Sampling temperature; greedy unless set. Reasoning models are tuned for sampling (Ling: 1.0) and, decoded
   * greedily, can skip their thinking. */
  temperature?: number;
  endpoint?: string; provider?: string; piOptions?: Record<string, unknown>;
  providerRequestControls?: ProviderRequestControls;
  chatRequestControls?: Record<string, unknown>;
  /** Optional collection-specific maximum wall time for one Pi provider request. */
  providerRequestTimeoutMs?: number;
  /** Optional collection-specific maximum wall time for preparation plus provider turns in one action cycle. */
  providerActionCycleTimeoutMs?: number;
  request?: Record<string, unknown>; cacheStableTools?: boolean;
  /** Elicit one required execution_plan tool call before every model action and use it as the turn's reasoning. */
  executionPlans?: boolean; executionPlanTokens?: number;
  /** The file tools directory reducers offer (native/prompt.ts); all unless set. */
  fileTools?: FileToolSurface;
  /** Who answered: a model being taught (student), a teacher, or a case's scripted reference solution. */
  collectionRole?: 'student' | 'teacher' | 'reference';
  /** A separately identified model for rubric-backed `judged` oracles. */
  judgeModel?: { modelId: string; endpoint?: string; provider?: string; piOptions?: Record<string, unknown> } };
export type CollectorConfig = ProvenanceOptions & { jobs: string; output: string; workers: number;
  /** Optional low-sensitivity live state for diagnosing an idle, unresolved collection. */
  collectionState?: (snapshot: CollectionLivenessSnapshot) => void;
  transportRetries?: number; retryDelayMs?: number;
  modelConcurrency?: number; maxModelRequests?: number;
  /** Optional append-only per-case lease/terminal journal used by reviewed pool supervisors. */
  caseEventsFile?: string;
  /** For large pools, keep exact per-case result files live and write the merged export only at final flush. */
  finalExportOnly?: boolean;
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

function logProviderStreamProgress(role: 'teacher' | 'judge', provider: string, requestOrdinal: number,
  progress: ModelStreamProgress): void {
  try {
    process.stderr.write(`${JSON.stringify({ event: 'provider_stream_progress', role, provider,
      request_ordinal: requestOrdinal, ...progress })}\n`);
  } catch { /* progress logging must not affect collection */ }
}

export function expectedProvenance(record: ProgramRecord, options: ProvenanceOptions): Record<string, unknown> {
  return { program_ir_sha256: recordDigest(record), model: options.modelId, tool_schema: TOOL_SCHEMA,
    runtime: 'typescript-native', runtime_contract_version: 19, trajectory_link_version: 2, collector_version: TEACHER_BATCH_VERSION, execution_policy_version: 2, data_quality_version: DATA_QUALITY_VERSION,
    file_content_comparison_version: FILE_CONTENT_COMPARISON_VERSION,
    answer_comparison_version: ANSWER_COMPARISON_VERSION,
    counter_loop_policy_version: 2,
    tool_surface_sha256: options.toolSurfaceSha256, seed_policy: { mode: 'derived', root: options.rootSeed },
    system_prompt_sha256: sha256(options.systemPrompt), context_tokens: options.contextTokens,
    transport: options.provider ? 'pi-provider' : 'openai-compatible',
    ...(options.chatRequestControls ? { chat_request_controls: options.chatRequestControls } : {}),
    ...(options.provider ? { provider: options.provider, pi_options: options.piOptions ?? {},
      ...(options.providerRequestControls ? { provider_request_controls: options.providerRequestControls } : {}),
      stream_observation: { version: 'pi-stream-observation/1', detail: 'aggregate-delta-counts', watchdog_refresh: false },
      ...(options.providerRequestTimeoutMs === undefined ? {} : { provider_request_timeout: {
        version: PROVIDER_REQUEST_TIMEOUT_POLICY_VERSION, timeout_ms: options.providerRequestTimeoutMs,
        retry: 'no-case-retry' } }),
      ...(options.providerActionCycleTimeoutMs === undefined ? {} : { provider_action_cycle: {
        version: PROVIDER_ACTION_CYCLE_POLICY_VERSION, timeout_ms: options.providerActionCycleTimeoutMs,
        retry: 'no-case-retry' } }),
      provider_cleanup: { timeout_ms: PROVIDER_CLEANUP_TIMEOUT_MS, scope: 'session-close-best-effort' } } : {}),
    ...(options.execution ? {execution_adapter:options.execution.identity} : {}),
    ...(options.maxTurns === undefined ? {} : { max_turns: options.maxTurns }),
    ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
    ...(options.cacheStableTools ? { cache_stable_tools: true } : {}),
    ...(options.executionPlans ? { execution_plans: { version: EXECUTION_PLAN_VERSION,
      max_tokens: options.executionPlanTokens ?? 512 } } : {}),
    collection_role: options.collectionRole ?? 'teacher',
    ...(record.handoff && record.semantics.failure_seed ? { seeded_handoff_version: 2 } : {}),
    ...(options.fileTools && options.fileTools !== 'all' ? { file_tools: options.fileTools } : {}),
    ...(options.judgeModel ? { judge: { model: options.judgeModel.modelId,
      ...(options.judgeModel.endpoint ? { endpoint_sha256: sha256(options.judgeModel.endpoint) } : {}),
      transport: options.judgeModel.provider ? 'pi-provider' : 'openai-compatible',
    ...(options.judgeModel.provider ? { provider: options.judgeModel.provider,
        pi_options: options.judgeModel.piOptions ?? {},
        stream_observation: { version: 'pi-stream-observation/1', detail: 'aggregate-delta-counts', watchdog_refresh: false },
        ...(options.providerRequestTimeoutMs === undefined ? {} : { provider_request_timeout: {
          version: PROVIDER_REQUEST_TIMEOUT_POLICY_VERSION, timeout_ms: options.providerRequestTimeoutMs,
          retry: 'no-case-retry' } }),
        ...(options.providerActionCycleTimeoutMs === undefined ? {} : { provider_action_cycle: {
          version: PROVIDER_ACTION_CYCLE_POLICY_VERSION, timeout_ms: options.providerActionCycleTimeoutMs,
          retry: 'no-case-retry' } }),
        provider_cleanup: { timeout_ms: PROVIDER_CLEANUP_TIMEOUT_MS, scope: 'session-close-best-effort' } } : {}) } } : {}) };
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

export async function writeAtomic(path: string, data: string | AsyncIterable<string>): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`;
  const handle = await open(temporary, 'wx');
  try {
    try {
      if (typeof data === 'string') await handle.writeFile(data);
      else for await (const chunk of data) await handle.writeFile(chunk);
      await handle.sync();
    } finally { await handle.close(); }
    await rename(temporary, path);
  } catch (error) { await unlink(temporary).catch(() => {}); throw error; }
}

async function mergeCompleted(records: IndexedRecord[], config: CollectorConfig): Promise<{ completed: number; missing: number[] }> {
  const missing: number[] = [];
  let completed = 0;
  async function* lines() {
    for (const item of records) {
      const expected = expectedProvenance(item.record, config);
      const path = join(config.jobs, `${jobKey(item)}.result.json`);
      const row = await readMatching(path, item.record, expected);
      if (row) { completed++; yield JSON.stringify(row) + '\n'; } else missing.push(item.index);
    }
  }
  await writeAtomic(config.output, lines());
  return { completed, missing };
}

const delay = (ms: number) => sleep(ms);

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
const REUSE_KEYS = ['program_ir_sha256', 'model', 'collection_role', 'seeded_handoff_version', 'execution_policy_version', 'data_quality_version', 'file_content_comparison_version', 'answer_comparison_version', 'counter_loop_policy_version', 'runtime_contract_version', 'trajectory_link_version', 'judge'];
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

/** Queue incomplete jobs, publish each result atomically, and coalesce ordered merge exports. */
export async function collectBatch(records: IndexedRecord[], config: CollectorConfig, runner: JobRunner,
  signal?: AbortSignal, admissionSignal?: AbortSignal): Promise<{ completed: number; missing: number[] }> {
  if (!Number.isInteger(config.workers) || config.workers < 1) throw new RangeError('workers must be positive');
  await mkdir(config.jobs, { recursive: true });
  if (config.caseEventsFile) await mkdir(dirname(config.caseEventsFile), { recursive: true });
  let caseEventFailure: unknown;
  let caseEventQueue = Promise.resolve();
  const caseEvent = (item: IndexedRecord, event: string, details: Record<string, unknown> = {}) => {
    if (!config.caseEventsFile || caseEventFailure) return Promise.resolve();
    caseEventQueue = caseEventQueue.then(async () => {
      const handle = await open(config.caseEventsFile!, 'a');
      try {
        await handle.writeFile(JSON.stringify({ event, at: new Date().toISOString(), index: item.index,
          program_id: item.record.id, program_ir_sha256: recordDigest(item.record), key: jobKey(item), ...details }) + '\n');
        await handle.sync();
      } finally { await handle.close(); }
    }).catch(error => { caseEventFailure = error; });
    return caseEventQueue;
  };
  const pending: IndexedRecord[] = [];
  const reusable = config.reuse?.length ? await reusableRows(config.reuse) :
    new Map<string, Array<{ row: TeacherRow; path: string }>>();
  let reused = 0, resumed = 0, incompatible = 0;
  for (const item of records) {
    const expected = expectedProvenance(item.record, config), path = join(config.jobs, `${jobKey(item)}.result.json`);
    if (await readMatching(path, item.record, expected)) {
      resumed++; await caseEvent(item, 'case_checkpoint', { disposition: 'already_complete' }); continue;
    }
    const hold = config.collectionRole === 'reference' ? undefined : generationHoldReason(item.record);
    if (hold) {
      await writeAtomic(join(config.jobs, `${String(item.index).padStart(6, '0')}.error.json`),
        JSON.stringify({ index: item.index, program_id: item.record.id, generation_hold: hold, error: `generation held: ${hold}` }) + '\n');
      await caseEvent(item, 'case_held', { reason: hold });
      continue;
    }
    try { await readFile(path, 'utf8'); incompatible++; } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
    // The last listed file with a qualifying row wins; a row that does not qualify never hides an earlier one.
    const row = (reusable.get(expected.program_ir_sha256 as string) ?? []).map(found => reusedRow(found, expected, config.reuseSurfaces))
      .filter(Boolean).at(-1);
    if (row) {
      await writeAtomic(path, JSON.stringify(row) + '\n'); reused++;
      await caseEvent(item, 'case_checkpoint', { disposition: 'reused' }); continue;
    }
    pending.push(item);
  }
  process.stderr.write(`queue: ${records.length} selected, ${resumed} already complete, ${reused} reused, ${pending.length} pending; ${config.workers} workers\n`);
  if (incompatible) process.stderr.write(`${incompatible} existing results did not match current provenance; inspect --reuse/--reuse-surfaces before recollecting\n`);
  if (reused) process.stderr.write(`reused ${reused} finished rows from ${config.reuse!.length} earlier result files\n`);
  // Work is picked up in a fixed pseudo-random order rather than shard order: a shard keeps a case's variants and
  // hint twins together, and running several long cases of one family at once fills the server's shared KV buffer.
  // The order depends only on the job keys, so it is the same on every resume; the merged output stays in shard order.
  pending.sort((a, b) => sha256(jobKey(a)).localeCompare(sha256(jobKey(b))));
  // The final-only pool mode relies on per-case files during collection and deliberately
  // leaves the aggregate absent until the final flush. The prepass has already validated
  // these resumed/reused per-case rows, so it can initialize progress without an export.
  let merged = config.finalExportOnly ? { completed: resumed + reused, missing: [] } :
    await mergeCompleted(records, config);
  // Job results are the durable checkpoints. Rebuilding the merged export is O(records), so do not
  // make each worker wait for its own full scan/write or allow concurrent snapshots to race.
  // A short coalescing window lets cases completing together share one export. The final flush below
  // is awaited, so a successful collectBatch still returns only after the ordered merge is current.
  let mergeDirty = false;
  let mergeFailure: unknown;
  let mergeTask: Promise<void> | undefined;
  const scheduleMerge = () => {
    mergeDirty = true;
    if (mergeTask || mergeFailure) return;
    mergeTask = (async () => {
      while (mergeDirty && !mergeFailure) {
        await delay(100);
        mergeDirty = false;
        try { merged = await mergeCompleted(records, config); }
        catch (error) { mergeFailure = error; }
      }
    })().finally(() => {
      mergeTask = undefined;
      if (mergeDirty && !mergeFailure) scheduleMerge();
    });
  };
  const flushMerge = async () => {
    scheduleMerge();
    while (mergeTask) await mergeTask;
    if (mergeFailure) throw mergeFailure;
    if (mergeDirty) return flushMerge();
    return merged;
  };
  let durableCompleted = merged.completed;
  let cursor = 0;
  const activeCases = new Map<number, { index: number; program_id: string }>();
  const publishCollectionState = (stage: string) => config.collectionState?.({
    stage, active_cases: [...activeCases.values()],
  });
  const worker = async (slot: number) => {
    if (slot && config.workerStaggerMs) await delay(slot * config.workerStaggerMs);
    // Stop admitting cases if the background export fails. Already-running cases may still
    // write their exact result files; flushMerge below then surfaces the export failure.
    while (cursor < pending.length && !signal?.aborted && !admissionSignal?.aborted &&
        !mergeFailure && !caseEventFailure) {
      const item = pending[cursor++]!, expected = expectedProvenance(item.record, config);
      await caseEvent(item, 'case_start');
      if (caseEventFailure) break;
      activeCases.set(slot, { index: item.index, program_id: item.record.id });
      publishCollectionState('case_run');
      let attempt = 0;
      while (true) try {
        const row = await runner(item, expected, signal);
        if (!resultMatches(row, item.record, expected)) throw new Error('job returned mismatched provenance');
        await writeAtomic(join(config.jobs, `${jobKey(item)}.result.json`), JSON.stringify(row) + '\n');
        durableCompleted++;
        await caseEvent(item, 'case_finish', { status: 'result', accepted: row.outcome?.accepted ?? null });
        if (!config.finalExportOnly) scheduleMerge();
        process.stderr.write(`completed ${item.index} ${item.record.id}: accepted=${row.outcome?.accepted ?? false}; ${durableCompleted}/${records.length} durable\n`);
        break;
      } catch (error) {
        if (signal?.aborted) return;
        if (error instanceof ProviderRequestTimeoutError || error instanceof ProviderActionCycleTimeoutError) {
          await writeAtomic(join(config.jobs, `${String(item.index).padStart(6, '0')}.error.json`),
            JSON.stringify({ index: item.index, program_id: item.record.id, code: error.code,
              error: `${error.name}: ${error.message}`, timeout: error.metadata }) + '\n');
          await caseEvent(item, 'case_finish', { status: 'error', code: error.code });
          break;
        }
        const limited = rateLimited(error);
        if (!transportFailure(error) || attempt >= (config.transportRetries ?? 8) * (limited ? 3 : 1)) {
          await writeAtomic(join(config.jobs, `${String(item.index).padStart(6, '0')}.error.json`),
            JSON.stringify({ index: item.index, program_id: item.record.id,
              ...(providerFinishReason(error) ? { provider_finish_reason: providerFinishReason(error) } : {}),
              retry_not_before: Date.now() + retryAfterMs(error),
              error: `${error instanceof Error ? error.name : 'Error'}: ${error instanceof Error ? error.message : String(error)}` }) + '\n');
          await caseEvent(item, 'case_finish', { status: 'error', error_type: error instanceof Error ? error.name : 'Error' });
          break;
        }
        const wait = retryWaitMs(error, attempt++, config.retryDelayMs ?? 5_000);
        const retryPath = join(config.jobs, `${jobKey(item)}.retry.json`);
        const event = { index: item.index, program_id: item.record.id, attempt,
          reason: limited ? 'rate_limit' : 'transport_failure',
          ...(providerFinishReason(error) ? { provider_finish_reason: providerFinishReason(error) } : {}),
          wait_ms: wait, until: Date.now() + wait };
        await caseEvent(item, 'case_retry', { attempt, reason: event.reason,
          ...(providerFinishReason(error) ? { provider_finish_reason: providerFinishReason(error) } : {}),
          wait_ms: wait, until: event.until });
        await writeAtomic(retryPath, JSON.stringify(event) + '\n');
        process.stderr.write(`retry: ${JSON.stringify(event)}\n`);
        try {
          const until = event.until;
          while (Date.now() < until) await sleep(Math.min(60_000, until - Date.now()), undefined, { signal });
        } catch (error) {
          if (!signal?.aborted) throw error;
          return;
        } finally {
          await unlink(retryPath).catch(error => { if (error.code !== 'ENOENT') throw error; });
        }
      }
      activeCases.delete(slot);
      publishCollectionState('case_admission');
    }
  };
  await Promise.all(Array.from({ length: Math.min(config.workers, Math.max(1, pending.length)) }, (_, slot) => worker(slot)));
  await caseEventQueue;
  if (caseEventFailure) throw caseEventFailure;
  return flushMerge();
}

function same(a: unknown, b: unknown): boolean { return canonical(a) === canonical(b); }

function effectHarness(specs: Record<string, unknown>): {
  capabilities: Record<string, (args: unknown[]) => unknown>; observed: Record<string, unknown[]>;
  expected: Record<string, unknown[]>; declarations: Record<string, string>;
} {
  const observed: Record<string, unknown[]> = {}, expected: Record<string, unknown[]> = {}, capabilities: Record<string, (args: unknown[]) => unknown> = {};
  const declarations: Record<string, string> = {};
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
      const delivered = new Map<string, unknown>(), failed = new Set<string>(), failKey = String(spec.fail_key);
      const [service, method] = name.split('.');
      declarations[service!] = (declarations[service!] ?? '') +
        `/** Delivers each key at most once. A normal return (null) confirms delivery. If an acknowledgement is lost, retry the identical command with the same key. Reusing a key with different contents is rejected. */\n` +
        `export function ${method}(command: { key: string, to: string, text: string }): null;\n`;
      capabilities[name] = args => {
        const command = args[0] as Record<string, unknown> | undefined;
        if (!command || typeof command.key !== 'string' || !command.key ||
          typeof command.to !== 'string' || typeof command.text !== 'string')
          throw new Error('delivery requires a nonempty string key and string to and text');
        const key = command.key;
        if (delivered.has(key) && !same(delivered.get(key), command))
          throw new Error('delivery key already belongs to a different command; retry the original command unchanged');
        if (!delivered.has(key)) { delivered.set(key, structuredClone(command)); observed[name]!.push(structuredClone(command)); }
        if (key === failKey && !failed.has(key)) {
          failed.add(key); throw new Error('delivery succeeded but its acknowledgement was lost');
        }
        return null;
      };
      continue;
    }
    throw new Error(`unsupported effect contract for ${name}`);
  }
  return { capabilities, observed, expected, declarations };
}

/** The trajectory record of one model turn: the request's context and the response, as training reads them. */
export function trajectoryTurn(request: ModelTurnRequest, response: ModelTurn): Record<string, unknown> {
  const raw = response.raw_response as Record<string, unknown> | undefined;
  const message = ((raw?.choices as Record<string, unknown>[] | undefined)?.[0]?.message ?? {}) as Record<string, unknown>;
  const planned = Object.hasOwn(response, 'execution_plan');
  const retainedReasoning = planned ? response.execution_plan :
    response.reasoning ?? message.reasoning_content ?? message.reasoning ?? message.thinking ?? null;
  return { phase: 'action', ...(request.invocation_id ? { invocation_id: request.invocation_id } : {}), context: structuredClone(request.messages),
    request_sha256: sha256(canonical(Object.fromEntries(Object.entries(request).filter(([key]) => key !== "invocation_id")))),
    model_response: { calls: structuredClone(response.calls ?? []), text: response.text ?? '',
      raw_calls: structuredClone(response.raw_calls ?? []),
      ...(raw?.pi_reply_diagnostic && typeof raw.pi_reply_diagnostic === 'object' ?
        { provider_reply_diagnostic: structuredClone(raw.pi_reply_diagnostic) } : {}),
      ...(planned ? { execution_plan: response.execution_plan } : {}),
      ...(response.completion_tokens === undefined ? {} : { completion_tokens: response.completion_tokens }),
      ...(response.prompt_tokens === undefined ? {} : { prompt_tokens: response.prompt_tokens }) },
    tools_offered: structuredClone(request.tools), assistant: { content: response.text ?? '',
      ...(planned ? { execution_plan: response.execution_plan } : {}),
      reasoning: retainedReasoning,
      calls: (response.calls ?? []).map(([tool, args]) => ({ tool, source_tool: tool, arguments: args, call_id: null })) },
    raw_response_sha256: raw ? sha256(canonical(raw)) : null };
}

type PartialTurn = { request_sha256: string; response: ModelTurn; invocation_id?: string;
  requested_at?: string; observed_at?: string;
  last_tool_observation?: { content_preview: string; content_sha256: string; truncated: boolean } };
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
        if (error instanceof ProviderRequestTimeoutError || error instanceof ProviderActionCycleTimeoutError ||
            transportFailure(error)) throw error;
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
  if (config.judgeModel && (!config.judgeModel.modelId ||
      Number(!!config.judgeModel.endpoint) + Number(!!config.judgeModel.provider) !== 1))
    throw new Error('judge model needs an ID and exactly one endpoint or Pi provider');
  if (config.judgeModel?.modelId === config.modelId) throw new Error('teacher and judge must use distinct model IDs');
  for (const [name, value] of Object.entries({ modelConcurrency: config.modelConcurrency, maxModelRequests: config.maxModelRequests }))
    if (value !== undefined && (!Number.isInteger(value) || value < 1)) throw new RangeError(`${name} must be positive`);
  if (config.providerRequestTimeoutMs !== undefined &&
      (!Number.isSafeInteger(config.providerRequestTimeoutMs) || config.providerRequestTimeoutMs < 1 ||
       config.providerRequestTimeoutMs > MAX_PROVIDER_TIMEOUT_MS))
    throw new RangeError(`providerRequestTimeoutMs must be an integer from 1 to ${MAX_PROVIDER_TIMEOUT_MS}`);
  if (config.providerActionCycleTimeoutMs !== undefined &&
      (!Number.isSafeInteger(config.providerActionCycleTimeoutMs) || config.providerActionCycleTimeoutMs < 1 ||
       config.providerActionCycleTimeoutMs > MAX_PROVIDER_TIMEOUT_MS))
    throw new RangeError(`providerActionCycleTimeoutMs must be an integer from 1 to ${MAX_PROVIDER_TIMEOUT_MS}`);
  const slots = config.modelConcurrency ? new KvBudget(config.modelConcurrency) : undefined;
  const kv = config.kvTokens ? new KvBudget(config.kvTokens) : undefined;
  return async (item, expected, signal) => {
    const hold = config.collectionRole === 'reference' ? undefined : generationHoldReason(item.record);
    if (hold) throw new Error(`generation held: ${hold}`);
    const retired = retiredFamily(item.record);
    if (retired) throw new Error(`retired curriculum family: ${retired}`);
    const quarantine = quarantineReason(item.record);
    if (quarantine) throw new Error(`quarantined curriculum case: ${quarantine}`);
    // A handoff (teacher/handoff.ts) replays another model's turns, call by call, up to the turn handed over.
    const handoff = item.record.handoff as Handoff | undefined;
    const placeOf = callMatcher(handoff?.openings ?? []);
    const session = config.provider ? createManagedModelSession(controlledProviderProfile(config.provider,
      config.modelId, config.piOptions, config.providerRequestControls)) : undefined;
    const judgeConfig = config.judgeModel;
    const judgeSession = judgeConfig?.provider ? createManagedModelSession({ provider: judgeConfig.provider,
      model: judgeConfig.modelId, piOptions: judgeConfig.piOptions }) : undefined;
    let ready: Promise<unknown> | undefined;
    let judgeReady: Promise<unknown> | undefined;
    let judgeSent = 0;
    let sent = 0;
    let fatalProviderDeadline: ProviderRequestTimeoutError | ProviderActionCycleTimeoutError | undefined;
    const providerFatalAbort = new AbortController();
    const rememberProviderDeadline = (error: unknown) => {
      if (!fatalProviderDeadline && (error instanceof ProviderRequestTimeoutError || error instanceof ProviderActionCycleTimeoutError)) {
        fatalProviderDeadline = error;
        providerFatalAbort.abort(error);
      }
      return fatalProviderDeadline ?? error;
    };
    const providerParentSignal = () => signal ? AbortSignal.any([signal, providerFatalAbort.signal]) : providerFatalAbort.signal;
    const providerActionCycle = <T>(options: { role: 'teacher' | 'judge'; provider: string;
      parentSignal?: AbortSignal; call(signal: AbortSignal): Promise<T> }) =>
      withProviderActionCycle({ ...options, timeoutMs: config.providerActionCycleTimeoutMs })
        .catch(error => { throw rememberProviderDeadline(error); });
    const send = session ? async (request: ModelTurnRequest, parentSignal = signal) => {
      const requestOrdinal = sent;
      await (ready ??= withProviderRequestDeadline({ role: 'teacher', provider: config.provider!, phase: 'provider_prepare',
        requestOrdinal: null, timeoutMs: config.providerRequestTimeoutMs,
        parentSignal, call: () => session.prepare() }));
      return withProviderRequestDeadline({ role: 'teacher', provider: config.provider!, phase: 'provider_turn', requestOrdinal,
        timeoutMs: config.providerRequestTimeoutMs, parentSignal,
        call: requestSignal => session.turn(request, requestSignal,
          progress => logProviderStreamProgress('teacher', config.provider!, requestOrdinal, progress)) });
    } : openAICompatibleModelTurn({ endpoint: config.endpoint!, model: config.modelId,
      request: config.request });
    try {
    let exhausted = false;
    const interrupted = (actionSignal?: AbortSignal) => actionSignal?.reason instanceof Error ? actionSignal.reason :
      new Error('collection cancelled');
    const admittedSend = async (request: ModelTurnRequest, sender = send, ownsSlot = false,
      actionSignal = signal) => {
      if (fatalProviderDeadline) throw fatalProviderDeadline;
      if (actionSignal?.aborted) throw interrupted(actionSignal);
      if (slots && !ownsSlot) await slots.acquire(1);
      try {
        if (fatalProviderDeadline) throw fatalProviderDeadline;
        if (actionSignal?.aborted) throw interrupted(actionSignal);
        const need = requestTokens(request);
        if (kv) await kv.acquire(need);
        try {
          if (fatalProviderDeadline) throw fatalProviderDeadline;
          if (actionSignal?.aborted) throw interrupted(actionSignal);
          // Waiting work has not sent a request. Count only after both capacity gates admit it.
          if (config.maxModelRequests && sent >= config.maxModelRequests) {
            exhausted = true;
            throw Object.assign(new Error(`whole-case model request budget exceeded (${config.maxModelRequests})`), { code: 'NATLANG_MODEL_REQUEST_BUDGET' });
          }
          sent++;
          try { return await sender(request); }
          catch (error) { throw rememberProviderDeadline(error); }
        } finally { if (kv) kv.release(need); }
      } finally { if (slots && !ownsSlot) slots.release(1); }
    };
    const teacherSend = (request: ModelTurnRequest, actionSignal = signal) =>
      admittedSend(request, (value: ModelTurnRequest) => send(value, actionSignal), true, actionSignal);
    const teacherTurn = (request: ModelTurnRequest, actionSignal = signal) => {
      const action = config.executionPlans ? withExecutionPlans((value: ModelTurnRequest) => teacherSend(value, actionSignal),
        { maxTokens: config.executionPlanTokens }) : (value: ModelTurnRequest) => teacherSend(value, actionSignal);
      return action(request);
    };
    const transport = async (request: ModelTurnRequest, persist: (response: ModelTurn) => Promise<void>) => {
      // Finish a plan/action pair before admitting another sibling's turn. Otherwise a large
      // Promise.all can spend the entire budget on plans without saving any completed actions.
      if (fatalProviderDeadline) throw fatalProviderDeadline;
      if (slots) await slots.acquire(1);
      try {
        if (fatalProviderDeadline) throw fatalProviderDeadline;
        const response = await (config.provider ? providerActionCycle({ role: 'teacher', provider: config.provider,
          parentSignal: providerParentSignal(), call: actionSignal => teacherTurn(request, actionSignal) }) : teacherTurn(request));
        // Keep the pair's slot until its response is durable. Otherwise a waiting sibling can
        // exhaust the request budget and end the run before this completed action is saved.
        await persist(response);
        return response;
      } finally { if (slots) slots.release(1); }
    };
    const judgeTransport = judgeConfig ? judgeSession ? async (request: ModelTurnRequest, parentSignal = signal) => {
      const requestOrdinal = ++judgeSent;
      await (judgeReady ??= withProviderRequestDeadline({ role: 'judge', provider: judgeConfig.provider!, phase: 'provider_prepare',
        requestOrdinal: null, timeoutMs: config.providerRequestTimeoutMs,
        parentSignal, call: () => judgeSession.prepare() }));
      return withProviderRequestDeadline({ role: 'judge', provider: judgeConfig.provider!, phase: 'provider_turn', requestOrdinal,
        timeoutMs: config.providerRequestTimeoutMs, parentSignal,
        call: requestSignal => judgeSession.turn(request, requestSignal,
          progress => logProviderStreamProgress('judge', judgeConfig.provider!, requestOrdinal, progress)) });
    } : openAICompatibleModelTurn({ endpoint: judgeConfig.endpoint!, model: judgeConfig.modelId }) : undefined;
    const judge = judgeTransport ? async (input: Parameters<ReturnType<typeof modelOracleJudge>>[0]) => {
      const grade = (actionSignal: AbortSignal | undefined) => modelOracleJudge(request =>
        admittedSend(request, value => judgeTransport(value, actionSignal), false, actionSignal))(input);
      return judgeConfig!.provider ? providerActionCycle({ role: 'judge', provider: judgeConfig!.provider,
        parentSignal: providerParentSignal(), call: actionSignal => grade(actionSignal) }) : grade(signal);
    } : undefined;
    const trajectory: Record<string, unknown>[] = [];
    const partialPath = join(config.jobs, `${jobKey(item)}.partial.json`);
    const saved = await loadPartial(partialPath, item, expected);
    const partial: PartialJob = saved ?? { version: TEACHER_PARTIAL_VERSION,
      program_id: item.record.id, provenance: structuredClone(expected), turns: [] };
    // Journaled responses are replayed by their exact request, not by position: the child calls of one eval run
    // concurrently, so their requests can reach the model in a different order after a restart. A request with no
    // unused journal entry is decoded live.
    const unused = new Map<string, PartialJob['turns']>();
    let journalWrites = Promise.resolve();
    for (const turn of partial.turns) unused.set(turn.request_sha256, [...unused.get(turn.request_sha256) ?? [], turn]);
    const driver = async (request: ModelTurnRequest): Promise<ModelTurn> => {
      if (fatalProviderDeadline) throw fatalProviderDeadline;
      const requestedAt = new Date().toISOString();
      const requestSha256 = sha256(canonical(Object.fromEntries(Object.entries(request).filter(([key]) => key !== "invocation_id"))));
      const recorded = unused.get(requestSha256)?.shift();
      const place = placeOf(request), replayed = handoff?.prefix[place.call]?.[place.nth];
      let response: ModelTurn;
      if (recorded) {
        response = structuredClone(recorded.response);
      } else {
        const persist = async (turn: ModelTurn) => {
          const last = request.messages.at(-1) as { role?: unknown; content?: unknown } | undefined;
          const observation = last?.role === 'tool' && typeof last.content === 'string' ? {
            content_preview: last.content.slice(0, 2000), content_sha256: sha256(last.content),
            truncated: last.content.length > 2000 } : undefined;
          partial.turns.push({ request_sha256: requestSha256, response: structuredClone(turn),
            ...(request.invocation_id ? { invocation_id: request.invocation_id } : {}),
            requested_at: requestedAt, observed_at: new Date().toISOString(),
            ...(observation ? { last_tool_observation: observation } : {}) });
          // Serialize atomic snapshots; an older concurrent write must never finish last.
          journalWrites = journalWrites.then(() => writeAtomic(partialPath, JSON.stringify(partial) + '\n'));
          await journalWrites;
        };
        if (replayed || (trajectory.length === 0 && !handoff && item.record.semantics.failure_seed)) {
          response = replayed ? structuredClone(replayed) :
            { calls: [['eval', { code: item.record.semantics.failure_seed!.code }]], completion_tokens: 1 };
          await persist(response);
        } else response = await transport(request, persist);
      }
      trajectory.push(trajectoryTurn(request, response));
      return response;
    };
    const runId = programRunId(item.index, expected);
    let run: ProgramRun;
    try { run = await (config.execution?.run ?? executeProgram)(item.record, driver, { ...config, runId, signal, ...(judge ? { judge } : {}) }); }
    catch (error) { throw fatalProviderDeadline ?? error; }
    if (fatalProviderDeadline) throw fatalProviderDeadline;
    if (exhausted) throw Object.assign(new Error(`whole-case model request budget exceeded (${config.maxModelRequests})`), { code: 'NATLANG_MODEL_REQUEST_BUDGET' });
    const row = programRow(item.record, config.modelId, runId, expected, run, trajectory,
      handoff ? { handoff: { kind: handoff.kind, source: handoff.source, run_id: runId } } : {});
    await writeAtomic(join(config.jobs, `${jobKey(item)}.trace.jsonl`),
      run.trace.map(event => JSON.stringify(event)).join('\n') + '\n');
    await removeIfPresent(partialPath);
    return row;
    } finally {
      await Promise.all([
        session ? config.provider ? closeProviderSession(session, config.provider) : session.close() : undefined,
        judgeSession ? judgeConfig?.provider ? closeProviderSession(judgeSession, judgeConfig.provider) : judgeSession.close() : undefined,
      ]);
    }
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
  maxTurns?: number; temperature?: number; rootSeed: number; runId: string; signal?: AbortSignal; fileTools?: FileToolSurface;
  judge?: (input: { actual: unknown; expected: unknown; rubric: string }) => Promise<{ accepted: boolean; verdict: string; needs_review?: boolean }> };
export type ProgramRun = { outcome: Record<string, unknown> & { accepted: boolean }; trace: Record<string, unknown>[] };

/**
 * Run a program's root invocation with a model driver in a fresh environment and check the result,
 * effects, and folder against its contract. The collector and reference replays share this path.
 */
export async function executeProgram(record: ProgramRecord, driver: (request: ModelTurnRequest) => Promise<ModelTurn>,
  options: ExecuteOptions): Promise<ProgramRun> {
  const root = await prepareProgramNode(record);
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
    ...(options.fileTools ? { fileTools: options.fileTools } : {}),
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
  const declarations: Record<string, string> = { ...effects.declarations };
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
    // Host-only provenance: exact portable values, bounded and explicitly incomplete otherwise.
    // Never invent a model return action from an eval-computed value.
    exactHostTraceCapture: { definitionSources: [], inputArguments: [], captureOutput: true,
      captureAllOutputs: true, maxBytes: 1_048_576 },
    seedPolicy: { mode: 'derived', root: options.rootSeed }, runId: options.runId, signal: options.signal });
  try {
    const result = await runtime.run(root), actual = dump(result.value);
    const actualFiles = folder ? Object.fromEntries(await Promise.all(folder.listFiles().map(async file =>
      [file.path, await folder.readText(file.path)] as const))) : undefined;
    const expectedKind = record.semantics.operation === 'blocked' ? 'quiesced' : 'done';
    const seededFailure = record.semantics.failure_seed;
    const seededFailureObserved = runtime.trace.events.some(event => event.kind === 'scope_failure' &&
      (!seededFailure?.kind || event.failure_kind === seededFailure.kind));
    // A teacher replacing the planted failing action must be allowed to prevent that failure entirely.
    const replacesSeed = replacesPlantedFailure(record);
    const failureSeen = !seededFailure || seededFailureObserved || replacesSeed;
    const effectsOk = same(effects.observed, effects.expected);
    // An authoring task is judged by running what was written, not by the files' exact text or the call's reply.
    const authoringSpec = (record.semantics as { authoring?: AuthoringSpec }).authoring;
    const authoring = authoringSpec && actualFiles ? await checkAuthoring(actualFiles, authoringSpec) : undefined;
    // Default JSON file grading compares content, preserving every semantic field.
    // Explicit exact oracles still cover byte-sensitive fixture/edit contracts.
    const filesCheck = folder && !authoring && actualFiles ?
      await checkFilesWithJudge(actualFiles, record.semantics.expected_files ?? folderFiles!, folderFiles!,
        record.semantics.files_oracle ?? { compare: 'content', threshold: 1 }, options.judge) : undefined;
    const filesOk = !folder || (authoring ? authoring.ok : filesCheck ? filesCheck.accepted :
      same(actualFiles, record.semantics.expected_files ?? folderFiles));
    // A blocked case needs the model's own blocked or failed call; running out of turns also quiesces.
    const honestStop = expectedKind !== 'quiesced' || /^(?:blocked|error): /.test(String(result.outcome.detail ?? ''));
    const worldScore = world ? await world.request('score') as { score: number; done: boolean } : undefined;
    // Some source-backed worlds are in-process external modules rather than WorldBridge processes. Their
    // certificate is trusted host state: a returned literal alone must not satisfy the world objective.
    const hostWorld = services.world;
    const hostCertificateFn = hostWorld?.certificate;
    let hostCertificate: unknown;
    let hostCertificateReadOk = true;
    if (typeof hostCertificateFn === 'function') {
      try { hostCertificate = hostCertificateFn(); }
      catch { hostCertificateReadOk = false; }
    }
    const hostCertificateRequired = expectedKind === 'done' && typeof hostCertificateFn === 'function';
    const hostCertificateOk = !hostCertificateRequired ||
      (hostCertificateReadOk && hostCertificate !== null && hostCertificate !== undefined &&
        same(hostCertificate, record.semantics.expected) && same(actual, hostCertificate));
    const worldOk = (!worldScore || worldScore.score >= 100) && hostCertificateOk;
    const answerExpected = actualFiles && record.semantics.files_oracle?.return_count === 'changed' ?
      fileReturnValue(actualFiles, folderFiles!, record.semantics.files_oracle) : record.semantics.expected;
    const oracle = await checkOracle(actual, answerExpected, record.semantics.oracle, options.judge);
    const checks = { seeded_failure_requirement: failureSeen, expected_status: result.outcome.kind === expectedKind,
      honest_stop: honestStop, effects: effectsOk, files: filesOk, world: worldOk,
      file_return_consistency: !actualFiles || !record.semantics.files_oracle ||
        checkFileReturn(actual, actualFiles, folderFiles!, record.semantics.files_oracle),
      answer: expectedKind !== 'done' || !!authoring || !!world || oracle.accepted };
    const rejectionReasons = Object.entries(checks).filter(([, passed]) => !passed).map(([name]) => name);
    const accepted = rejectionReasons.length === 0;
    const trace = runtime.trace.events as unknown as Record<string, unknown>[];
    return { trace, outcome: { status: result.outcome.kind, detail: result.outcome.detail, value: actual,
      effects: effects.observed, ...(actualFiles ? { files: actualFiles } : {}), ...(authoring ? { authoring } : {}),
      ...(worldScore ? { world: worldScore } : {}),
      ...(hostCertificateRequired ? { host_completion_certificate: hostCertificateReadOk ? hostCertificate ?? null : { read_error: true } } : {}),
      ...(!filesCheck && oracle.needs_review ? { quality_pending: ['answer_needs_review'] } : {}), ...(filesCheck ? { files_check: filesCheck, quality_pending: [...filesCheck.pending, ...(oracle.needs_review ? ['answer_needs_review'] : [])] } : {}), oracle, accepted, checks, rejection_reasons: rejectionReasons,
      ...(seededFailure ? { seeded_failure: { observed: seededFailureObserved, replaced_by_handoff: replacesSeed } } : {}),
      // Every call's actions, children included: a child nl call runs in its own runtime and reports its trace to
      // the task (call_id tells them apart), so its decisions can be linked to what they did.
      // Preserve observed invocation parentage, rather than reconstructing it from equal returned text.
      invocation_ledger: [trace, ...(runtime.frame?.task.traces ?? []).map(child => child.events)]
        .flatMap(events => {
          const manifest = events.find(event => event.kind === 'manifest');
          if (!manifest || typeof manifest.run_id !== 'string') return [];
          const output = events.find(event => event.kind === 'host_capture' && event.capture_kind === 'invocation_output');
          return [{ invocation_id: manifest.run_id, parent_invocation_id: manifest.parent_call_id ?? null,
            ...(manifest.inline_instruction_site ? { inline_instruction_site: manifest.inline_instruction_site } : {}),
            completion_status: events.filter(event => event.kind === 'state' && event.phase === 'final').at(-1)?.outcome ?? null,
            ...(output ? { host_result: output } : {}) }];
        }),
      action_ledger: [...trace, ...(runtime.frame?.task.traces ?? []).flatMap(child => child.events)]
        .filter(event => event.kind === 'action'),
      ...(answerExpected !== record.semantics.expected ? { derived_expected: answerExpected } : {}),
      scope_failures: trace.filter(event => event.kind === 'scope_failure'),
      host_events: trace.filter(event => event.kind === 'host') } };
  } finally { environment.close(); world?.close(); }
}

export async function defaultToolSurfaceHash(root = fileURLToPath(new URL('../..', import.meta.url))): Promise<string> {
  // Resume only against the exact interpreter implementation. Compiler, type,
  // source-loading, and filesystem changes can alter an identical tool call
  // even when its public JSON schema is unchanged.
  // Hash the implementation that executes: the compiled dist/ when running from it (the usual case), so pulling
  // newer sources under a running campaign does not orphan results of an unchanged runtime; src/ for source runs.
  const compiled = import.meta.url.includes('/dist/');
  const dir = compiled ? 'dist' : 'src', ext = compiled ? '.js' : '.ts';
  const native = (await readdir(join(root, `${dir}/native`))).filter(name => name.endsWith(ext))
    .map(name => `${dir}/native/${name}`);
  const files = [...native, `${dir}/scope-compiler${ext}`, `${dir}/environment${ext}`].sort();
  const chunks = await Promise.all(files.map(path => readFile(join(root, path))));
  return sha256(Buffer.concat(chunks.flatMap((chunk, index) => index ? [Buffer.from([0]), chunk] : [chunk])));
}

export const defaultSystemPrompt = `${TOOLS_PROMPT}${GENERATION_GUIDANCE}`;
