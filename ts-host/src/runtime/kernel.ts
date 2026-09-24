/**
 * The shared invocation kernel. Every natlang call (a named `.nl` function, an inline `nl`, an
 * `iterateOn` step or judge) comes through `invokeDefinition`: it builds one lambda node, runs it
 * with the interpreter in the caller's task, and returns the checked value or throws `NatlangCallError`.
 */
import { NativeToolAgent } from '../native/agent.js';
import { NativeRuntime } from '../native/runtime.js';
import { Folder, FolderHandle, FileHandle, type FolderTransaction } from '../native/scoped-fs.js';
import { TypeEnv } from '../native/types.js';
import { MISSING, buildPending, coerce, isLive, type CaptureCell, type LambdaNode, type Value } from '../native/values.js';
import { NatlangRecursionError, runInFrame, type Frame } from './context.js';
import { recordingServices } from './runtime.js';
import { kernelHooks } from './hooks.js';

export type { CaptureCell };

/** A natlang function definition as the kernel runs it. */
export type CallableDefinition = {
  id: string;
  name: string;
  body: string;
  params: { name: string; type: string; optional?: boolean }[];
  returns: string;
  types: Record<string, string>;
  /** Callable context: the record tree this definition (and its inline descendants) may call. */
  codebase: Record<string, unknown>;
  subtype: 'function' | 'directory-reducer';
  revision?: string;
  description?: string;
  /** Source path for named definitions. */
  source?: string;
};

export type InvokeOptions = {
  captures?: Record<string, CaptureCell>;
  /** Instructions after interpolation, for inline lambdas. */
  instructions?: string;
  folder?: { transaction: FolderTransaction; mode: 'apply' | 'direct' };
  /** Constructors for class-typed parameters and returns. */
  classes?: ReadonlyMap<string, Function>;
  manifest?: Record<string, unknown>;
};

export class NatlangCallError extends Error {
  constructor(readonly definition: string, readonly outcome: string, readonly detail: string,
    readonly callId: string, readonly trace: Record<string, unknown>[]) {
    super(`${definition}: ${outcome}: ${detail}`);
    this.name = 'NatlangCallError';
  }
}

/** Convert an interpreter value into an ordinary JavaScript value for host code. */
export function toHost(value: Value): unknown {
  if (value === MISSING) return undefined;
  if (value === null || typeof value !== 'object') return value;
  if (isLive(value) || value instanceof Folder || value instanceof FolderHandle || value instanceof FileHandle)
    return value;
  if (Array.isArray(value)) return value.map(toHost);
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, toHost(item as Value)]));
}

/** The pending lambda node for one invocation of `definition` with positional inputs (not yet run). */
export function definitionNode(definition: CallableDefinition, inputs: unknown[], options: InvokeOptions = {}): LambdaNode {
  const type = `(${definition.params.map(parameter => `${parameter.name}${parameter.optional ? '?' : ''}: ${parameter.type}`).join(', ')}) => ${definition.returns}`;
  const node = buildPending({ $lambda: { type, instructions: options.instructions ?? definition.body,
    types: definition.types, function: definition.name,
    ...(definition.subtype !== 'function' ? { subtype: definition.subtype } : {}) } }) as LambdaNode;
  node.codebase = definition.codebase;
  node.hostClasses = options.classes;
  const env = new TypeEnv(node.types);
  env.classes = options.classes;
  if (node.type.kind === 'lambda') node.type.params.fields.forEach((field, index) => {
    if (inputs[index] !== undefined) node.args[field.name] = coerce(inputs[index], field.type, env, `${definition.name}/${field.name}`);
  });
  if (options.captures && Object.keys(options.captures).length) node.captures = options.captures;
  return node;
}

/** Run one natlang definition in the given frame and return its checked value. */
export async function invokeDefinition(frame: Frame, definition: CallableDefinition, positional: unknown[],
  options: InvokeOptions = {}): Promise<unknown> {
  const task = frame.task;
  task.checkOpen();
  if (frame.chain.includes(definition.id)) throw new NatlangRecursionError(definition.id, frame.chain, definition.name);
  const limits = task.runtime.options.limits ?? {};
  if (limits.maxDepth !== undefined && frame.chain.length >= limits.maxDepth)
    throw new NatlangCallError(definition.name, 'quiesced', `natlang calls nested deeper than ${limits.maxDepth}`, '', []);
  let inputs = positional;
  let folder = options.folder;
  if (definition.subtype === 'directory-reducer') {
    const handle = inputs[0];
    if (!folder) {
      if (!(handle instanceof Folder) && !(handle instanceof FolderHandle))
        throw new TypeError(`${definition.name} is a directory reducer; pass a Folder as its first argument or use folder.apply(...)`);
      folder = { transaction: await handle.beginTransaction(true), mode: 'direct' };
    }
    if (handle instanceof Folder || handle instanceof FolderHandle) inputs = inputs.slice(1);
  }
  const required = definition.params.filter(parameter => !parameter.optional).length;
  if (inputs.length < required || inputs.length > definition.params.length) {
    if (folder?.transaction.open) folder.transaction.abort();
    throw new TypeError(`${definition.name} expects ${required === definition.params.length ? required :
      `${required} to ${definition.params.length}`} arguments, got ${inputs.length}`);
  }
  const node = definitionNode(definition, inputs, options);
  if (folder) { node.projectTransaction = folder.transaction; node.reducerMode = folder.mode; }

  const callId = task.nextCallId();
  const childFrame: Frame = { task, chain: [...frame.chain, definition.id], parentCallId: callId };
  const model = task.model();
  const environment = task.environment();
  let runtime: NativeRuntime | undefined;
  const services = recordingServices(task.services, event =>
    runtime?.trace.emit('effect', { call_id: callId, capability: `${event.service}.${event.method}`, ...event }));
  const agent = model ? new NativeToolAgent(model.driver, { systemPrompt: () => task.systemPrompt(),
    maxTurns: model.maxTurns, maxTokens: model.maxTokens, turnTokens: model.turnTokens, temperature: model.temperature,
    maxSeconds: model.maxSeconds, contextTokens: model.contextTokens,
    maxFailureRepairs: model.maxFailureRepairs, review: model.review }) : undefined;
  runtime = new NativeRuntime({ environment, hooks: kernelHooks,
    agent: task.runtime.options.agent ?? (agent ? session => agent.run(session) : undefined),
    maxActions: limits.maxActions, maxToolCalls: limits.maxToolCalls,
    sharedEpisodeBudget: task.episodeBudget, seedPolicy: task.runtime.options.seed, runId: callId,
    sourceRevision: definition.revision, parentCallId: frame.parentCallId, signal: task.signal,
    frame: childFrame, services,
    manifest: { definition_id: definition.id, definition_name: definition.name, task_id: task.id,
      ...(definition.source ? { definition_source: definition.source } : {}), ...(options.manifest ?? {}) } });
  let outcome = 'failed', detail = '';
  try {
    const result = await runInFrame(childFrame, () => runtime!.run(node));
    outcome = result.outcome.kind; detail = result.outcome.detail;
    if (outcome !== 'done') throw new NatlangCallError(definition.name, outcome, detail, callId, runtime.trace.events as Record<string, unknown>[]);
    return toHost(result.value);
  } catch (error) {
    if (!(error instanceof NatlangCallError)) detail = error instanceof Error ? error.message : String(error);
    if (folder?.transaction.open) folder.transaction.abort();
    throw error;
  } finally {
    task.record({ callId, parentCallId: frame.parentCallId ?? null, taskId: task.id, definitionId: definition.id,
      name: definition.name, outcome, detail, events: runtime.trace.events as Record<string, unknown>[] });
    environment.close();
  }
}
