import { tuneRuntime, rethrowInfrastructureFailure, type TuneExecution } from './execution.js';
/**
 * Whole-program data collection for `natlang tune` (plans/HIERARCHICAL_LORA_TUNING.md §4).
 *
 * Root examples are split by source/world component BEFORE anything is expanded; every derived row inherits its
 * root's partition. The program then runs on the real runtime with the authorized teacher as executor. Each model turn
 * is captured with the binding of the invocation that made it (resolved from the callee, exactly as at serving time),
 * its exact request and reply, and separate evidence fields. Protected examples never enter collection.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import { sealManifest, type WeightManifest } from '../adaptation/weights/manifest.js';
import { compileScopeTree, type ScopeOptions } from '../adaptation/weights/scopes.js';
import { WeightBinding } from '../adaptation/weights/binding.js';
import type { NativeModelIdentity } from '../native/neuralese-store.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import type { ModelDriver } from '../runtime/runtime.js';
import { createNatlangRuntime, type Services } from '../runtime/runtime.js';
import { journaledServices, type World } from './world.js';
import { currentFrame } from '../runtime/context.js';
import { loadNatlang } from '../runtime/node-files.js';
import { canonicalJson } from '../package/archive.js';

export type Partition = 'train' | 'development' | 'confirmation';
export type RootExample = { readonly function: string; readonly input: unknown; readonly expected?: unknown;
  /** Source/world component: related examples share one, so they land on one side of every split. */
  readonly group?: string; readonly protected?: boolean; readonly source?: 'real' | 'synthetic' };
export type SplitExample = RootExample & { readonly id: string; readonly component: string; readonly partition: Partition };

/** One captured model turn of one inner invocation. */
export type CollectedTurn = {
  readonly schema: 'natlang.tune-turn/v1';
  readonly root: string; readonly component: string; readonly partition: Partition;
  readonly binding: string | null; readonly definition: string | null; readonly call: string | null; readonly turn: number;
  readonly messages: unknown[]; readonly tools: unknown[]; readonly reply: Pick<ModelTurn, 'calls' | 'text' | 'reasoning' | 'truncated'>;
  readonly teacher: string; readonly source: 'real' | 'synthetic';
  /** The invocation's exact instruction text and disclosed skill documents, where training places soft states. */
  readonly texts: { readonly instructions?: string; readonly skills?: Readonly<Record<string, string>> };
  /** Separate evidence (§4.3); never collapsed into one flag. */
  readonly evidence: { readonly rootOracle: boolean | null; readonly enclosingSuccess: boolean; readonly callOutcome: string | null;
    readonly actionPolicy: 'natlang.tune-action-outcome/2'; readonly actionDisposition: 'admissible' | 'rejected' | 'unverified';
    readonly failedHistory: boolean; readonly actionEvents: { readonly toolCallId: string; readonly name: string;
      readonly outcome: string; readonly argumentsSha256: string }[] };
  /** When a tuned release served the turn: its requested/effective stack (on-policy provenance, §4.4). */
  readonly stack?: { readonly identity: string; readonly status: string; readonly requested: readonly string[]; readonly effective: readonly string[] };
  /** An on-policy correction row (tune/repairs.ts): its prompt holds the student's prefix and is never a target. */
  readonly correction?: import('./repairs.js').CorrectionProvenance;
};

const digest = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');

/** Read `tune.examples.jsonl` beside the program (or an explicit path). Each line is one root invocation. */
export function readRootExamples(root: string, path = join(root, 'tune.examples.jsonl')): RootExample[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, 'utf8').split('\n').filter(line => line.trim()).map((line, index) => {
    const value = JSON.parse(line) as RootExample;
    if (typeof value.function !== 'string' || !value.function) throw new Error(`${path}:${index + 1}: each example names its root function`);
    if (!('input' in value)) throw new Error(`${path}:${index + 1}: each example has an input`);
    return value;
  });
}

/** A persisted split receipt: component -> partition. Once assigned, a component never moves (plan §4.1). */
export type SplitReceipt = { readonly schema: 'natlang.tune-split/v1'; readonly components: Readonly<Record<string, Partition>> };

const componentOf = (example: RootExample) => example.group ?? digest([example.function, example.input]);
// Identical arguments in distinct captured worlds are distinct invocations. Their
// records, teacher custody and evaluation caches must not share a root key.
const rootExampleId = (example: RootExample) => digest(['natlang.tune-root/v2', componentOf(example),
  example.function, example.input, 'expected' in example, example.expected ?? null]).slice(0, 16);

function uniqueRoots(rows: readonly SplitExample[]): SplitExample[] {
  const roots = new Map<string, SplitExample>();
  for (const row of rows) {
    const previous = roots.get(row.id);
    if (previous) {
      const binding = (example: SplitExample) => JSON.stringify([example.component, example.function,
        example.input, 'expected' in example, example.expected ?? null, example.group ?? null,
        example.source ?? 'real', example.partition]);
      if (binding(previous) !== binding(row)) throw new Error(`conflicting tune root identity: ${row.id}`);
      continue; // Repeated identical invocations in one world are one root, not extra evidence.
    }
    roots.set(row.id, row);
  }
  return [...roots.values()];
}

/**
 * Partition by connected component before expansion: the component is the declared group, else the example itself.
 * Components already in `receipt` keep their partition, so adding examples never moves a development root into
 * training. New components get a stable hash draw (about 70/20/10); while no development component exists, the first
 * new one goes to development so selection never has to use training roots. Protected examples are excluded here
 * and stay out of every later stage. Returns the examples and the extended receipt to persist.
 */
export function splitExamples(examples: readonly RootExample[], receipt?: SplitReceipt): SplitExample[] & { receipt: SplitReceipt } {
  const usable = examples.filter(example => !example.protected);
  const assigned: Record<string, Partition> = { ...(receipt?.components ?? {}) };
  const fresh = [...new Set(usable.map(componentOf))].filter(component => !(component in assigned))
    .sort((a, b) => digest(['natlang.tune-split/v1', a]).localeCompare(digest(['natlang.tune-split/v1', b])));
  for (const component of fresh) {
    const draw = Number.parseInt(digest(['natlang.tune-split/v1', component]).slice(0, 8), 16) / 2 ** 32;
    const hasDevelopment = Object.values(assigned).includes('development');
    const anyOther = Object.keys(assigned).length > 0;
    assigned[component] = !hasDevelopment && anyOther ? 'development' : draw < 0.7 ? 'train' : draw < 0.9 ? 'development' : 'confirmation';
  }
  const split = usable.map(example => ({ ...example, id: rootExampleId(example),
    component: componentOf(example), partition: assigned[componentOf(example)]! }));
  return Object.assign(uniqueRoots(split), { receipt: { schema: 'natlang.tune-split/v1' as const, components: assigned } });
}

/** Protected roots for final reporting only (never collected, trained on or used for selection). */
export function protectedExamples(examples: readonly RootExample[]): SplitExample[] {
  return uniqueRoots(examples.filter(example => example.protected).map(example => ({ ...example, partition: 'confirmation' as const,
    component: componentOf(example), id: rootExampleId(example) })));
}

/** A manifest with the program's bindings and no artifacts: every call resolves its binding, all run the base. */
export function scaffoldManifest(program: ProgramDescriptor, base: NativeModelIdentity, scopes: ScopeOptions = {}): WeightManifest {
  const { tree, bindings } = compileScopeTree(program, scopes);
  return sealManifest({ schema: 'natlang.adapters/v1', program: { id: program.id, buildHash: program.buildHash }, base, tree, bindings,
    artifacts: [], families: [], memberships: [], softStates: [], scales: {}, validatedReductions: [], fallback: 'base', receipts: {} });
}

const equal = (left: unknown, right: unknown) => JSON.stringify(left) === JSON.stringify(right);
const canonicalActionArgs = (left: unknown, right: unknown) => canonicalJson(left) === canonicalJson(right);
function hasFailedAssistantHistory(messages: unknown[], failedIds: ReadonlySet<string>): boolean {
  return messages.some(message => {
    if (!message || typeof message !== 'object') return false;
    const item = message as { tool_call_id?: unknown; tool_calls?: { id?: unknown }[] };
    return (typeof item.tool_call_id === 'string' && failedIds.has(item.tool_call_id)) ||
      (Array.isArray(item.tool_calls) && item.tool_calls.some(call => typeof call.id === 'string' && failedIds.has(call.id)));
  });
}

/**
 * Run each example's root function with the teacher and capture every inner model turn. A failing root is still
 * recorded (its turns are not positive targets: `enclosingSuccess` and `rootOracle` say so).
 */
export async function collect(options: { root: string; program: ProgramDescriptor; scaffold: WeightManifest; examples: readonly SplitExample[];
  teacher: ModelDriver; teacherId: string; partitions?: readonly Partition[]; onTurn?: (turn: CollectedTurn) => void;
  /** Serve a tuned release while collecting (distillation): its strict binding labels and serves each call. */
  weights?: WeightBinding;
  /** Host services for one root, fresh per root (a test or sandbox world); every call is journaled (world.ts). */
  execution?: TuneExecution; services?: (example: SplitExample) => Services }): Promise<{
  turns: CollectedTurn[]; roots: { id: string; ok: boolean; oracle: boolean | null; error?: string; world?: World }[] }> {
  const turns: CollectedTurn[] = [], roots: { id: string; ok: boolean; oracle: boolean | null; error?: string; world?: World }[] = [];
  const binding = options.weights ?? new WeightBinding(options.scaffold);
  for (const example of options.examples) {
    if (options.partitions && !options.partitions.includes(example.partition)) continue;
    const pending: (Omit<CollectedTurn, 'evidence'> & { rawCallIds: (string | null)[] })[] = [];
    const outcomes = new Map<string, string>();
    const events = new Map<string, Record<string, unknown>[]>();
    const counters = new Map<string, number>();
    const driver: ModelDriver = Object.assign(async (request: ModelTurnRequest, signal?: AbortSignal, turnOptions?: unknown) => {
      const frame = currentFrame();
      const reply = await (options.teacher as (r: ModelTurnRequest, s?: AbortSignal, o?: unknown) => Promise<ModelTurn>)(request, signal, turnOptions);
      const call = frame?.parentCallId ?? null, turn = (counters.get(call ?? '') ?? 0) + 1;
      counters.set(call ?? '', turn);
      pending.push({ schema: 'natlang.tune-turn/v1', root: example.id, component: example.component, partition: example.partition,
        binding: frame?.weights?.binding ?? null, definition: frame?.chain.at(-1) ?? null, call, turn,
        messages: structuredClone(request.messages), tools: structuredClone(request.tools),
        reply: { ...(reply.calls ? { calls: reply.calls } : {}), ...(reply.text !== undefined ? { text: reply.text } : {}),
          ...(reply.reasoning ? { reasoning: reply.reasoning } : {}), ...(reply.truncated ? { truncated: true } : {}) },
        rawCallIds: (reply.raw_calls ?? []).map(item => item && typeof item === 'object' && typeof (item as { id?: unknown }).id === 'string'
          ? (item as { id: string }).id : null),
        teacher: options.teacherId, source: example.source ?? 'real', texts: structuredClone(frame?.invocationTexts ?? {}),
        ...(options.weights && frame?.weights ? { stack: { identity: frame.weights.identity, status: frame.weights.status,
          requested: frame.weights.requested.map(entry => entry.id), effective: frame.weights.effective.map(entry => entry.id) } } : {}) });
      return reply;
    }, { decide: (options.teacher as { decide?: unknown }).decide, contextWindow: (options.teacher as { contextWindow?: unknown }).contextWindow }) as ModelDriver;
    const world: World = { schema: 'natlang.tune-world/v1', calls: [] };
    const services = options.services ? journaledServices(options.services(example), world) : undefined;
    const runtime = createNatlangRuntime({ program: options.program, ...await tuneRuntime(driver, options.execution), weights: [binding], ...(services ? { services } : {}),
      trace: trace => { outcomes.set(trace.callId, trace.outcome); events.set(trace.callId, trace.events ?? []); } });
    let ok = false, oracle: boolean | null = null, error: string | undefined;
    try {
      const fn = loadNatlang(resolve(options.root, example.function), options.root);
      const params = (fn as unknown as Record<symbol, { definition: { params: { name: string }[] } }>)[Symbol.for('natlang.callable')]!.definition.params;
      const input = example.input;
      const args = input && typeof input === 'object' && !Array.isArray(input) ? params.map(p => (input as Record<string, unknown>)[p.name])
        : Array.isArray(input) ? input : [input];
      const value = await runtime.run(() => fn(...args));
      ok = true;
      if ('expected' in example) oracle = equal(value, example.expected);
    } catch (failure) { rethrowInfrastructureFailure(failure); error = failure instanceof Error ? failure.message : String(failure); }
    finally { runtime.close(); }
    roots.push({ id: example.id, ok, oracle, ...(error ? { error } : {}), ...(services ? { world } : {}) });
    const failedIds = new Set<string>();
    const failedInvocations = new Set<string>();
    const actionEvents = new Map<string, Record<string, unknown>[]>();
    for (const [invocation, items] of events) actionEvents.set(invocation, items.filter(item => item.kind === 'action'));
    const actionOffsets = new Map<string, number>();
    for (const turn of pending) {
      const calls = turn.reply.calls ?? [];
      const inv = turn.call ?? '';
      const available = actionEvents.get(inv) ?? [];
      let offset = actionOffsets.get(inv) ?? 0;
      let disposition: 'admissible' | 'rejected' | 'unverified' = 'admissible';
      const matchedIds: string[] = [];
      const matchedActions: { toolCallId: string; name: string; outcome: string; argumentsSha256: string }[] = [];
      for (let index = 0; index < calls.length; index++) {
        const [name, args] = calls[index]!;
        const rawId = turn.rawCallIds[index];
        const candidateIndexes = available.flatMap((item, at) => at >= offset && item.name === name &&
          canonicalActionArgs(item.arguments, args) ? [at] : []);
        let eventIndex = rawId ? candidateIndexes.find(at => available[at]!.tool_call_id === rawId) : undefined;
        if (eventIndex === undefined) {
          const withoutId = candidateIndexes.filter(at => available[at]!.tool_call_id === undefined);
          if (!rawId && candidateIndexes.length === 1) eventIndex = candidateIndexes[0];
          else if (withoutId.length === 1 && candidateIndexes[0] === withoutId[0]) eventIndex = withoutId[0];
        }
        if (eventIndex === undefined) { disposition = 'unverified'; break; }
        const event = available[eventIndex]!;
        offset = eventIndex + 1;
        if (typeof event.tool_call_id === 'string') matchedIds.push(event.tool_call_id);
        const boundToolCallId = typeof event.tool_call_id === 'string' ? event.tool_call_id : rawId;
        if (typeof boundToolCallId === 'string' && boundToolCallId) matchedActions.push({ toolCallId: boundToolCallId, name,
          outcome: String(event.outcome), argumentsSha256: digest(canonicalJson(args)) });
        else { disposition = 'unverified'; break; }
        if (['rejected', 'refused', 'error', 'failed'].includes(String(event.outcome))) disposition = 'rejected';
        else if (!['ok', 'completed', 'done'].includes(String(event.outcome))) disposition = 'unverified';
      }
      actionOffsets.set(inv, offset);
      const failedHistory = hasFailedAssistantHistory(turn.messages, failedIds) || failedInvocations.has(inv);
      if (disposition !== 'admissible') {
        failedInvocations.add(inv);
        for (const id of turn.rawCallIds) if (id) failedIds.add(id);
        for (const id of matchedIds) failedIds.add(id);
      }
      const { rawCallIds: _rawCallIds, ...turnRecord } = turn;
      const record: CollectedTurn = { ...turnRecord, evidence: { rootOracle: oracle, enclosingSuccess: ok && oracle !== false,
        callOutcome: turn.call ? outcomes.get(turn.call) ?? null : null, actionPolicy: 'natlang.tune-action-outcome/2',
        actionDisposition: disposition, failedHistory, actionEvents: matchedActions.length === calls.length ? matchedActions : [] } };
      turns.push(record); options.onTurn?.(record);
    }
  }
  return { turns, roots };
}
