/**
 * The execution graph (spec/NEURALESE_GRAPH.md, `natlang.exec-graph/1`): typed nodes and edges on top of the native
 * trace. A node is a trace event that carries a `node` ID and lists its `inputs`; soft values are referenced by block
 * ID and by the node that produced them, so a trainer can replay a record for gradients.
 *
 * Node IDs are unique across one task's traces: an invocation is `call:<call id>`, every other node is
 * `<call id>#<seq>` (its event's position in that call's trace).
 */
import {activeRecorder} from './model-recording.js';
import { isNeuraleseRef, isNativeStateId } from './neuralese.js';
import type { NativeTraceRecorder, TraceEvent } from './trace.js';

export const GRAPH_VERSION = 'natlang.exec-graph/1';
export const GRAPH_NODE_KINDS = ['invocation', 'context_bind', 'model_turn', 'block_write', 'block_read', 'readout',
  'combinator', 'rewrite', 'effect', 'iteration_step', 'grad', 'literal'] as const;
export type GraphNodeKind = typeof GRAPH_NODE_KINDS[number];
export type GraphInput = { node: string; port?: string; block?: string; source?: string; stop?: boolean };

/** Manifest additions of a graph record. */
export type GraphManifest = { version: typeof GRAPH_VERSION; model: { id: string | null; revision: string | null };
  dialect: string | null; rewrites: string[]; root_context: string; seeds: Record<string, unknown> };

export function graphManifest(options: { model?: { id?: string | null; revision?: string | null }; dialect?: string | null;
  rewrites?: readonly string[]; rootContext?: string; seeds?: Record<string, unknown> }): GraphManifest {
  return { version: GRAPH_VERSION, model: { id: options.model?.id ?? null, revision: options.model?.revision ?? null },
    dialect: options.dialect ?? null, rewrites: [...(options.rewrites ?? [])].sort(), root_context: options.rootContext ?? FILE_CONTEXT,
    seeds: options.seeds ?? {} };
}
/** The context of a definition used where it was loaded (its own files), when it was not rebound. */
export const FILE_CONTEXT = 'ctx:files';

export const invocationNodeId = (callId: string): string => `call:${callId}`;

// --- Live traces and block producers -----------------------------------------------------------------------------

/** Traces of running invocations by call ID, so host-side operations (iteration, rebinding) record into their caller. */
const TRACES = new Map<string, NativeTraceRecorder>();
const TRACE_SCOPES = new WeakMap<NativeTraceRecorder, object>();
export function bindTraceScope(trace: NativeTraceRecorder, scope: object): void { TRACE_SCOPES.set(trace, scope); }
export function registerTrace(callId: string, trace: NativeTraceRecorder, scope: object = trace): void {
  TRACES.set(callId, trace);
  bindTraceScope(trace, scope);
}
/** Scope used for block provenance emitted by this trace. Traces without an explicitly bound scope are isolated to themselves. */
export function producerScopeForTrace(trace: NativeTraceRecorder | undefined): object | undefined {
  return trace ? TRACE_SCOPES.get(trace) ?? trace : undefined;
}
export function releaseTrace(callId: string): void { TRACES.delete(callId); }
export const traceFor = (callId: string | undefined): NativeTraceRecorder | undefined => callId ? TRACES.get(callId) : undefined;

/** Exact block-to-writer occurrences, shared only by calls with the same task-owned scope object. */
type ProducerIndex = { byBlock: Map<string, Set<string>>; byNode: Map<string, string> };
const PRODUCERS = new WeakMap<object, ProducerIndex>();
const PRODUCER_LIMIT = 100_000;
function assertProducerOccurrenceAvailable(block: string, node: string, scope: object): void {
  const previousBlock = PRODUCERS.get(scope)?.byNode.get(node);
  if (previousBlock !== undefined && previousBlock !== block)
    throw new Error(`neuralese-native-grad-invalid: writer occurrence ${node} was reused for a different native block`);
}
function rememberProducer(block: string, node: string, scope: object): void {
  assertProducerOccurrenceAvailable(block, node, scope);
  const index = PRODUCERS.get(scope) ?? { byBlock: new Map<string, Set<string>>(), byNode: new Map<string, string>() };
  const previousBlock = index.byNode.get(node);
  if (previousBlock === block) return;
  const occurrences = index.byBlock.get(block) ?? new Set<string>();
  occurrences.add(node);
  index.byBlock.set(block, occurrences);
  index.byNode.set(node, block);
  PRODUCERS.set(scope, index);
  while (index.byNode.size > PRODUCER_LIMIT) {
    const oldest = index.byNode.keys().next().value!;
    const oldBlock = index.byNode.get(oldest)!;
    index.byNode.delete(oldest);
    const values = index.byBlock.get(oldBlock)!;
    values.delete(oldest);
    if (!values.size) index.byBlock.delete(oldBlock);
  }
}
/** Whether this exact occurrence was observed writing this block in the same task scope. */
export function hasProducerOccurrence(block: string, node: string, scope: object): boolean {
  const index = PRODUCERS.get(scope);
  return index?.byNode.get(node) === block && (index.byBlock.get(block)?.has(node) ?? false);
}
/** An input edge for a block: from its producing node, or from `external` (a file, another task, an older record). */
export function blockInput(block: string, port?: string, sourceOccurrence?: string, scope?: object): GraphInput {
  const occurrences = scope ? PRODUCERS.get(scope)?.byBlock.get(block) : undefined;
  const producer = occurrences?.size ? [...occurrences].at(-1) : undefined;
  // A recorded trajectory can contain separately opened task invocations. Its
  // authenticated writer ledger is the same authority used by modelTurnNode;
  // do not detach those real dependencies because their local budgets differ.
  const recorded = activeRecorder()?.nativeStateWriters?.(block) ?? [];
  const node = sourceOccurrence !== undefined ?
    ((scope && hasProducerOccurrence(block, sourceOccurrence, scope)) || recorded.includes(sourceOccurrence) ? sourceOccurrence : 'external') :
    (isNativeStateId(block) ? recorded.length === 1 ? recorded[0]! : 'external' : producer ?? 'external');
  return { node, block, ...(port ? { port } : {}) };
}
/** Block inputs for every soft value inside `value`, with ports naming where each one sits. */
export function valueInputs(value: unknown, port: string, scope?: object): GraphInput[] {
  const found: GraphInput[] = [];
  const visit = (item: unknown, path: string, depth: number) => {
    if (depth > 32 || item === null || typeof item !== 'object') return;
    if (isNeuraleseRef(item)) { found.push(blockInput(item.$neuralese.id, path, item.$neuralese.source_occurrence, scope)); return; }
    if (Array.isArray(item)) item.forEach((entry, index) => visit(entry, `${path}[${index}]`, depth + 1));
    else for (const [key, entry] of Object.entries(item)) visit(entry, `${path}.${key}`, depth + 1);
  };
  visit(value, port, 0);
  return found;
}

/**
 * Emit a node into `trace`. With `event`, the node is that trace kind (an existing event kind such as `invocation` or
 * `effect` becomes a node by carrying a node ID); the node's ID is returned.
 */
export function graphNode(trace: NativeTraceRecorder | undefined, kind: GraphNodeKind, data: Record<string, unknown>,
  inputs: readonly GraphInput[] = [], id?: string): string | undefined {
  if (!trace) return;
  const callId = String(trace.events[0]?.run_id ?? 'run');
  const node = id ?? `${callId}#${trace.events.length}`;
  const producerScope = producerScopeForTrace(trace)!;
  if (kind === 'block_write' && typeof data.block === 'string')
    assertProducerOccurrenceAvailable(data.block, node, producerScope);
  trace.emit(kind, { ...data, node, inputs: [...inputs] });
  activeRecorder()?.recordGraphNode?.(structuredClone(trace.events.at(-1)!) as Record<string,unknown>);
  if (kind === 'block_write' && typeof data.block === 'string')
    rememberProducer(data.block, node, producerScope);
  return node;
}

// --- Validation --------------------------------------------------------------------------------------------------

/** The subset of JSON Schema (2020-12) the graph schema uses. */
type Schema = { type?: string | string[]; required?: string[]; properties?: Record<string, Schema>; enum?: unknown[]; const?: unknown;
  items?: Schema; minItems?: number; maxItems?: number; minimum?: number; maximum?: number; pattern?: string;
  additionalProperties?: boolean | Schema; allOf?: Schema[]; if?: Schema; then?: Schema; $ref?: string; $defs?: Record<string, Schema> };

function typeMatches(type: string, value: unknown): boolean {
  switch (type) {
    case 'object': return !!value && typeof value === 'object' && !Array.isArray(value);
    case 'array': return Array.isArray(value);
    case 'integer': return Number.isInteger(value);
    case 'number': return typeof value === 'number' && Number.isFinite(value);
    case 'string': return typeof value === 'string';
    case 'boolean': return typeof value === 'boolean';
    case 'null': return value === null;
    default: return false;
  }
}

/** Validate `value` against `schema`; returns error messages with JSON-pointer-like paths. */
export function validateSchema(schema: Schema, value: unknown, root: Schema = schema, path = ''): string[] {
  if (schema.$ref) {
    const name = /^#\/\$defs\/(.+)$/.exec(schema.$ref)?.[1];
    const target = name ? root.$defs?.[name] : undefined;
    if (!target) return [`${path || '/'}: unresolved $ref ${schema.$ref}`];
    return validateSchema(target, value, root, path);
  }
  const errors: string[] = [];
  const at = path || '/';
  if (schema.type !== undefined && !(Array.isArray(schema.type) ? schema.type : [schema.type]).some(type => typeMatches(type, value)))
    return [`${at}: expected ${schema.type}`];
  if (schema.const !== undefined && JSON.stringify(schema.const) !== JSON.stringify(value)) errors.push(`${at}: expected ${JSON.stringify(schema.const)}`);
  if (schema.enum && !schema.enum.some(item => JSON.stringify(item) === JSON.stringify(value))) errors.push(`${at}: not one of ${JSON.stringify(schema.enum)}`);
  if (typeof value === 'number') {
    if (schema.minimum !== undefined && value < schema.minimum) errors.push(`${at}: below ${schema.minimum}`);
    if (schema.maximum !== undefined && value > schema.maximum) errors.push(`${at}: above ${schema.maximum}`);
  }
  if (typeof value === 'string' && schema.pattern && !new RegExp(schema.pattern).test(value)) errors.push(`${at}: does not match ${schema.pattern}`);
  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) errors.push(`${at}: fewer than ${schema.minItems} items`);
    if (schema.maxItems !== undefined && value.length > schema.maxItems) errors.push(`${at}: more than ${schema.maxItems} items`);
    if (schema.items) value.forEach((item, index) => errors.push(...validateSchema(schema.items!, item, root, `${path}/${index}`)));
  }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    for (const key of schema.required ?? []) if (!(key in record)) errors.push(`${at}: missing ${key}`);
    for (const [key, item] of Object.entries(record)) {
      const property = schema.properties?.[key];
      if (property) errors.push(...validateSchema(property, item, root, `${path}/${key}`));
      else if (schema.additionalProperties === false) errors.push(`${path}/${key}: not allowed`);
      else if (schema.additionalProperties && typeof schema.additionalProperties === 'object')
        errors.push(...validateSchema(schema.additionalProperties, item, root, `${path}/${key}`));
    }
  }
  for (const part of schema.allOf ?? []) {
    if (part.if && validateSchema(part.if, value, root, path).length) continue;
    errors.push(...validateSchema(part.if ? part.then ?? {} : part, value, root, path));
  }
  return errors;
}

/**
 * Check a record's node events: each validates against the graph schema, node IDs are unique, and every input refers
 * to a node seen earlier in the given events, to `external`, or to a node of another call (`call:` or `<call>#<seq>`).
 */
export function validateGraph(events: readonly (TraceEvent | Record<string, unknown>)[], schema: Schema): { node: string; errors: string[] }[] {
  const seen = new Set<string>();
  const problems: { node: string; errors: string[] }[] = [];
  for (const event of events) {
    if (typeof event.node !== 'string') continue;
    const errors = validateSchema(schema, event);
    if (seen.has(event.node)) errors.push(`duplicate node ${event.node}`);
    seen.add(event.node);
    for (const input of (Array.isArray(event.inputs) ? event.inputs : []) as GraphInput[])
      if (input && typeof input.node === 'string' && input.node !== 'external' && !seen.has(input.node) && !/^call:|#\d+$/.test(input.node))
        errors.push(`input ${input.node} is not a node`);
    if (errors.length) problems.push({ node: event.node, errors });
  }
  return problems;
}
