/**
 * The shared invocation kernel. Every natlang call (a named `.nl` function, an inline `nl`, an
 * `iterateOn` step or judge) comes through `invokeDefinition`: it builds one lambda node, runs it
 * with the interpreter in the caller's task, and returns the checked value or throws `NatlangCallError`.
 */
import { DECISION_SYSTEM_PROMPT } from '../native/decision.js';
import { hexDigest } from '../native/hash.js';
import { NativeToolAgent } from '../native/agent.js';
import { NativeRuntime, inferValueType } from '../native/runtime.js';
import { Folder, FolderHandle, FileHandle, type FolderTransaction } from '../native/scoped-fs.js';
import { TypeEnv } from '../native/types.js';
import { MISSING, buildPending, coerce, isLive, type CaptureCell, type LambdaNode, type Value } from '../native/values.js';
import { MAX_AD_HOC_NL_DEPTH, NatlangRecursionError, runInFrame, type Frame } from './context.js';
import { recordingServices } from './runtime.js';
import { kernelHooks } from './hooks.js';
import { FILE_CONTEXT, graphManifest, graphNode, invocationNodeId, registerTrace, releaseTrace } from '../native/graph.js';
import { loadSkills, memorySkillSource } from '../skills/registry.js';
import { readSkillDocument, renderScopeDeclarations, renderSkillListing, scopeBindings } from '../skills/disclosure.js';

export type { CaptureCell };

/** A natlang function definition as the kernel runs it. */
export type CallableDefinition = {
  programId?: string;
  id: string;
  name: string;
  body: string;
  params: { name: string; type: string; optional?: boolean }[];
  /** Takes whatever each call passes (an inline function saved without parameters; see InlineLambdaPlan). */
  openParameters?: boolean;
  returns: string;
  types: Record<string, string>;
  /** Callable context: the record tree this definition (and its inline descendants) may call. */
  codebase: Record<string, unknown>;
  subtype: 'function' | 'directory-reducer';
  /** `decision`: the call scores its finite result values instead of running the tool loop (native/decision.ts).
   * `template`: the call's first reply is forced to `return_result`, its value written or decoded (template readout). */
  readout?: 'decision' | 'template';
  revision?: string;
  description?: string;
  /** Source path for named definitions. */
  source?: string;
  /** ID of the context the definition is bound to, when it was rebound (`fn.in(context)`) or defined in one. */
  contextId?: string;
};

export type InvokeOptions = {
  captures?: Record<string, CaptureCell>;
  /** Instructions after interpolation, for inline lambdas. */
  instructions?: string;
  folder?: { transaction: FolderTransaction; mode: 'apply' | 'direct' };
  /** Constructors for class-typed parameters and returns. */
  classes?: ReadonlyMap<string, Function>;
  manifest?: Record<string, unknown>;
  /** Files of the context's `skills/` data entries, by path (`skills/<name>/SKILL.md`, ...): the call's bound skills. */
  skillFiles?: Readonly<Record<string, string | Uint8Array>>;
};

type ScopedHandle = Folder | FolderHandle | FileHandle;
const isScopedHandle = (value: unknown): value is ScopedHandle =>
  value instanceof Folder || value instanceof FolderHandle || value instanceof FileHandle;
const isScopeRecord = (value: unknown): value is Record<string, unknown> => {
  if (value === null || typeof value !== 'object' || Array.isArray(value) || Object.prototype.toString.call(value) !== '[object Object]')
    return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === null || (Object.getPrototypeOf(prototype) === null &&
    Object.prototype.hasOwnProperty.call(prototype, 'constructor') && prototype.constructor?.name === 'Object');
};

/** Find folder capabilities nested in caller data; they still need the same copy-on-write boundary as top-level handles. */
function scopedHandles(value: unknown, found: ScopedHandle[] = [], seen = new Set<object>()): ScopedHandle[] {
  if (isScopedHandle(value)) { found.push(value); return found; }
  if (!value || typeof value !== 'object' || seen.has(value)) return found;
  seen.add(value);
  if (Array.isArray(value)) for (const item of value) scopedHandles(item, found, seen);
  else if (isScopeRecord(value)) for (const item of Object.values(value)) scopedHandles(item, found, seen);
  return found;
}

/** Rebuild traversed arrays/records while replacing capabilities; define keys safely (including `__proto__`). */
function rebaseScopedHandles(value: unknown, replacements: ReadonlyMap<ScopedHandle, ScopedHandle>, seen = new Map<object, unknown>()): unknown {
  if (isScopedHandle(value)) {
    let current = value;
    const visited = new Set<ScopedHandle>();
    while (replacements.has(current) && !visited.has(current)) {
      visited.add(current); current = replacements.get(current)!;
    }
    return current;
  }
  if (!value || typeof value !== 'object') return value;
  const previous = seen.get(value);
  if (previous !== undefined) return previous;
  if (Array.isArray(value)) {
    const copy: unknown[] = []; seen.set(value, copy);
    for (const item of value) copy.push(rebaseScopedHandles(item, replacements, seen));
    return copy;
  }
  if (isScopeRecord(value)) {
    const copy: Record<string, unknown> = Object.create(Object.getPrototypeOf(value)); seen.set(value, copy);
    for (const [key, item] of Object.entries(value)) Object.defineProperty(copy, key, {
      value: rebaseScopedHandles(item, replacements, seen), enumerable: true, configurable: true, writable: true,
    });
    return copy;
  }
  return value;
}

/** Captured handles share the caller's scope; rebase them through the same copy-on-write layers as arguments. */
function rebaseCaptureCells(captures: Record<string, CaptureCell> | undefined,
  replacements: ReadonlyMap<ScopedHandle, ScopedHandle>): Record<string, CaptureCell> | undefined {
  if (!captures || !replacements.size) return captures;
  const reverse = new Map<ScopedHandle, ScopedHandle>();
  for (const [from, to] of replacements) if (!reverse.has(to)) reverse.set(to, from);
  let changed = false;
  const result: Record<string, CaptureCell> = {};
  for (const [name, cell] of Object.entries(captures)) {
    const value = cell.get();
    const hasMappedHandle = scopedHandles(value).some(handle => replacements.has(handle));
    if (!hasMappedHandle) { result[name] = cell; continue; }
    changed = true;
    result[name] = { ...cell,
      get: () => rebaseScopedHandles(cell.get(), replacements),
      ...(cell.set ? { set: (next: unknown) => cell.set!(rebaseScopedHandles(next, reverse)) } : {}),
    };
  }
  return changed ? result : captures;
}

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
  if (definition.readout) node.readout = definition.readout;
  const env = new TypeEnv(node.types);
  env.classes = options.classes;
  if (node.type.kind === 'lambda') node.type.params.fields.forEach((field, index) => {
    if (inputs[index] !== undefined) node.args[field.name] = coerce(inputs[index], field.type, env, `${definition.name}/${field.name}`);
  });
  if (options.captures && Object.keys(options.captures).length) node.captures = options.captures;
  return node;
}

/**
 * Bind a context's skills to a call (S2 §2.2): the opening lists them, `read_code("skills.<name>")` discloses them, and
 * their declared scope bindings join the call's scope. A binding whose name is taken (a parameter, a capture) or that
 * fails its type is left out; invalid skills are left out by the loader.
 */
async function bindContextSkills(node: LambdaNode, files: Readonly<Record<string, string | Uint8Array>>): Promise<void> {
  const loaded = await loadSkills(memorySkillSource(files));
  const { set } = loaded;
  if (!set.size && !loaded.diagnostics.length) return;
  const reserved = [...(node.type.kind === 'lambda' ? node.type.params.fields.map(field => field.name) : []), ...Object.keys(node.captures ?? {})];
  const resolved = await scopeBindings(set, { env: new TypeEnv(node.types), reserved });
  const bindings = resolved.bindings;
  const documents: Record<string, string> = {};
  for (const skill of set.list()) for (const target of [`skills.${skill.name}`, ...skill.files.map(file => `skills.${skill.name}/${file}`)]) {
    const document = await readSkillDocument(set, target).catch(() => undefined);
    if (document?.kind === 'text') documents[target] = document.text;
  }
  node.skills = { listing: renderSkillListing(set, [...loaded.diagnostics, ...resolved.diagnostics]), documents,
    declarations: renderScopeDeclarations(bindings), inventory: set.list().map(skill => ({ name: skill.name, revision: skill.revision })) };
  if (bindings.length) node.captures = { ...node.captures, ...Object.fromEntries(bindings.map(binding => [binding.name,
    { name: binding.name, type: binding.typeText, mutable: false, get: () => binding.value, skill: binding.skill }])) };
}

/** Prepare the same context-bound node for public calls and direct host execution. */
export async function prepareDefinitionNode(definition: CallableDefinition, inputs: unknown[], options: InvokeOptions = {}): Promise<LambdaNode> {
  const node = definitionNode(definition, inputs, options);
  if (options.skillFiles && Object.keys(options.skillFiles).length) await bindContextSkills(node, options.skillFiles);
  return node;
}

/**
 * Run one natlang definition in the given frame and return its checked value. Model code can start a call and never
 * await it (an eval that fails first, a promise left in a variable): its failure is then no one's to handle, and it must
 * not end the host process as an unhandled rejection. Whoever awaits the call still gets the failure.
 */
export function invokeDefinition(frame: Frame, definition: CallableDefinition, positional: unknown[],
  options: InvokeOptions = {}): Promise<unknown> {
  const call = runDefinition(frame, definition, positional, options);
  call.catch(() => {});
  return frame.task.track(call, frame.parentCallId);
}

async function runDefinition(frame: Frame, definition: CallableDefinition, positional: unknown[],
  options: InvokeOptions): Promise<unknown> {
  // Folder transactions can be acquired by the caller before invokeDefinition (for example,
  // eval's delegate and folder.apply). Most validation happens before the runtime try/catch
  // below, so keep ownership here as well and release every supplied/acquired lease on any
  // preflight or setup error.
  const transactions = new Set<FolderTransaction>();
  if (options.folder) transactions.add(options.folder.transaction);
  try {
    return await runDefinitionBody(frame, definition, positional, options, transactions);
  } catch (error) {
    for (const transaction of transactions) if (transaction.open) transaction.abort();
    throw error;
  }
}

async function runDefinitionBody(frame: Frame, definition: CallableDefinition, positional: unknown[],
  options: InvokeOptions, transactions: Set<FolderTransaction>): Promise<unknown> {
  const task = frame.task;
  task.checkOpen();
  const view = task.programView;
  const replacement = view.value(definition.id, definition.programId);
  const descriptor = (!definition.programId || definition.programId === view.program?.id) ? view.component(definition.id) : undefined;
  if (view.binding && descriptor?.origin === 'named' && descriptor.source && !view.patched(descriptor.source.path)) {
    const original = view.program!.sources[descriptor.source.path]!;
    if (definition.revision !== hexDigest(original).slice(0, 16)) throw new Error('callable revision differs from adaptation build: ' + definition.name);
  }
  const patched = (!definition.programId || definition.programId === view.program?.id) && definition.source?.endsWith('.nl') && view.patched(definition.source);
  if (patched) {
    // Existing callable references resolve the committed task revision at invocation.
    const record = view.record({ kind: 'natlang', id: definition.id, source: definition.source!, name: definition.name,
      instructions: definition.body, text: '', revision: definition.revision ?? '', args: {}, returns: definition.returns,
      types: definition.types, subtype: definition.subtype, description: '', codebase: {} } as import('./loader.js').NatlangRecord);
    definition = { ...definition, body: record.instructions, revision: record.revision,
      params: Object.entries(record.args).map(([name, type]) => ({ name: name.replace(/\?$/, ''), type, optional: name.endsWith('?') })),
      returns: record.returns, types: record.types, codebase: record.codebase };
  } else if (replacement?.kind === 'lambda.instructions' && options.instructions === undefined)
    definition = { ...definition, body: replacement.template.segments[0]! };
  // Capture this call's source revision; later edits affect future calls only.
  definition = { ...definition, codebase: view.tree(definition.codebase as Record<string, import('./loader.js').ItemRecord>) };
  const owner = definition.programId ?? (descriptor ? view.program?.id : frame.programOwner);
  const eligibleGuidance = !!owner && (owner === view.program?.id || !!view.program?.guidanceScope.importedPrograms.includes(owner));
  const adaptationProvenance = view.program ? { ...view.provenance(definition.id, owner),
    effectiveInstructionHash: hexDigest(options.instructions ?? definition.body), originalInstructionHash: descriptor?.baselineHash ?? null,
    guidanceComponent: eligibleGuidance ? view.program.components.find(component => component.kind === 'program.guidance')?.key ?? null : null,
    guidanceApplied: eligibleGuidance, evaluation: task.runtime.options.evaluation ?? null } : undefined;
  const adHoc = !!(options.manifest?.inline || options.manifest?.delegate);
  const fileRoot = !adHoc && definition.source?.endsWith('.nl');
  const adHocDepth = fileRoot ? 0 : (frame.adHocDepth ?? 0) + (adHoc ? 1 : 0);
  if (adHocDepth > MAX_AD_HOC_NL_DEPTH)
    throw new NatlangCallError(definition.name, 'quiesced',
      `ad hoc nl calls are limited to ${MAX_AD_HOC_NL_DEPTH} nested layers; solve this part here or call an existing named function`, '', []);
  // Definitions from separate programs may share the same relative source ID.
  // Ownership scopes recursion without changing persisted component identities.
  // A definition bound to another context is another function: calls go down the context graph, so the same
  // definition rebound elsewhere may legitimately run below itself. The guard remains the runtime backstop for host
  // callbacks, where the structural rule cannot see the chain.
  const baseIdentity = definition.programId ? JSON.stringify([definition.programId, definition.id]) : definition.id;
  const callIdentity = definition.contextId ? `${baseIdentity}#${definition.contextId}` : baseIdentity;
  if (frame.chain.includes(callIdentity)) throw new NatlangRecursionError(definition.id, frame.chain, definition.name);
  const limits = task.runtime.options.limits ?? {};
  if (limits.maxDepth !== undefined && frame.chain.length >= limits.maxDepth)
    throw new NatlangCallError(definition.name, 'quiesced', `natlang calls nested deeper than ${limits.maxDepth}`, '', []);
  let inputs = positional;
  let folder = options.folder;
  const extraTransactions: FolderTransaction[] = [];
  if (definition.subtype === 'directory-reducer') {
    const inheritedReplacements = frame.scopedHandleReplacements ?? new Map<ScopedHandle, ScopedHandle>();
    inputs = inputs.map(input => rebaseScopedHandles(input, inheritedReplacements));
    const inheritedCaptures = rebaseCaptureCells(options.captures, inheritedReplacements);
    options = { ...options, captures: inheritedCaptures };
    const handle = inputs[0];
    if (!folder) {
      if (!(handle instanceof Folder) && !(handle instanceof FolderHandle))
        throw new TypeError(`${definition.name} is a directory reducer; pass a Folder as its first argument or use folder.apply(...)`);
      folder = { transaction: await handle.beginTransaction(true), mode: 'direct' };
      transactions.add(folder.transaction);
    }
    if (handle instanceof Folder || handle instanceof FolderHandle) {
      inputs = inputs.slice(1);
      const replacements = new Map<ScopedHandle, ScopedHandle>([[handle, folder.transaction.folder.root()]]);
      inputs = inputs.map(input => rebaseScopedHandles(input, replacements));
      const combined = new Map<ScopedHandle, ScopedHandle>();
      for (const [ancestor, parent] of inheritedReplacements)
        combined.set(ancestor, rebaseScopedHandles(parent, replacements) as ScopedHandle);
      for (const [source, target] of replacements) combined.set(source, target);
      options = { ...options, captures: rebaseCaptureCells(inheritedCaptures, replacements) };
      frame = { ...frame, scopedHandleReplacements: combined };
    }
  } else {
    // A handle is a capability, not a reference to its caller's whole backing folder.
    // Give the child its own copy and merge its changes only when it completes.
    const inheritedReplacements = frame.scopedHandleReplacements ?? new Map<ScopedHandle, ScopedHandle>();
    inputs = inputs.map(input => rebaseScopedHandles(input, inheritedReplacements));
    const inheritedCaptures = rebaseCaptureCells(options.captures, inheritedReplacements);
    const capturedValues = Object.values(inheritedCaptures ?? {}).map(cell => cell.get());
    const handles = [...new Set([...inputs, ...capturedValues].flatMap(input => scopedHandles(input)))];
    if (handles.length) {
      const roots = handles.map(value => ({ value, backing: value instanceof Folder ? value :
        (value as FolderHandle | FileHandle).folder, path: value instanceof Folder ? '' :
        (value as FolderHandle | FileHandle).path }));
      for (let left = 0; left < roots.length; left++) for (let right = left + 1; right < roots.length; right++) {
        const a = roots[left]!, b = roots[right]!;
        if (a.backing === b.backing && a.backing.access !== 'read' &&
          (!a.path || !b.path || a.path === b.path || a.path.startsWith(`${b.path}/`) || b.path.startsWith(`${a.path}/`)))
          throw new TypeError('overlapping writable handles in one child call; pass disjoint roots');
      }
      const replacements = new Map<ScopedHandle, ScopedHandle>();
      try {
        for (const { value } of roots) {
          const transaction = value instanceof FileHandle ? await value.folder.beginFileTransaction(value.path) :
            await (value as Folder | FolderHandle).beginTransaction(true);
          transactions.add(transaction);
          if (!folder) folder = { transaction, mode: 'apply' };
          else extraTransactions.push(transaction);
          replacements.set(value, value instanceof FileHandle ? transaction.folder.file(value.name) : transaction.folder.root());
        }
        inputs = inputs.map(input => rebaseScopedHandles(input, replacements));
        const combined = new Map<ScopedHandle, ScopedHandle>();
        for (const [ancestor, parent] of inheritedReplacements)
          combined.set(ancestor, rebaseScopedHandles(parent, replacements) as ScopedHandle);
        for (const [handle, replacement] of replacements) combined.set(handle, replacement);
        options = { ...options, captures: rebaseCaptureCells(inheritedCaptures, replacements) };
        frame = { ...frame, scopedHandleReplacements: combined };
      } catch (error) {
        if (folder?.transaction.open) folder.transaction.abort();
        for (const transaction of extraTransactions) if (transaction.open) transaction.abort();
        throw error;
      }
    }
  }
  // An open inline function takes whatever each call passes, typed from its values (a splat).
  if (definition.openParameters) definition = { ...definition, params: inputs.map((input, index) =>
    ({ name: index ? `input${index + 1}` : 'input', type: inferValueType(input) })) };
  // An inline function called with more values than its signature names takes the rest too, as open parameters
  // (input2, …): the values are what it was asked about. Too few values cannot be made up, and stay an error.
  if (definition.name.startsWith('nl@') && inputs.length > definition.params.length) {
    const taken = new Set(definition.params.map(parameter => parameter.name));
    const extra = inputs.slice(definition.params.length).map((input, offset) => {
      let index = definition.params.length + offset, name = index ? `input${index + 1}` : 'input';
      while (taken.has(name)) name = `input${++index + 1}`;
      taken.add(name);
      return { name, type: inferValueType(input) };
    });
    definition = { ...definition, params: [...definition.params, ...extra] };
  }
  const required = definition.params.filter(parameter => !parameter.optional).length;
  if (inputs.length < required || inputs.length > definition.params.length) {
    if (folder?.transaction.open) folder.transaction.abort();
    for (const transaction of extraTransactions) if (transaction.open) transaction.abort();
    throw new TypeError(`${definition.name} expects ${required === definition.params.length ? required :
      `${required} to ${definition.params.length}`} arguments, got ${inputs.length}`);
  }
  const node = await prepareDefinitionNode(definition, inputs, options);
  if (folder) { node.projectTransaction = folder.transaction; node.reducerMode = folder.mode; }
  if (extraTransactions.length) node.extraTransactions = extraTransactions;

  const callId = task.nextCallId();
  const childFrame: Frame = { task, chain: [...frame.chain, callIdentity], parentCallId: callId, adHocDepth, programOwner: owner,
    ...(frame.scopedHandleReplacements ? { scopedHandleReplacements: frame.scopedHandleReplacements } : {}),
    ...(options.manifest?.inline ? { inline: true } : {}) };
  const model = task.model();
  const environment = task.environment();
  let runtime: NativeRuntime | undefined;
  const services = recordingServices(task.services, event => event.phase === 'requested' ?
    runtime?.trace.emit('effect', { call_id: callId, capability: `${event.service}.${event.method}`, ...event }) :
    graphNode(runtime?.trace, 'effect', { call_id: callId, capability: `${event.service}.${event.method}`, ...event },
      [{ node: invocationNodeId(callId), port: 'caller' }]));
  // A stopping predicate of iterateOn runs under its own addition to the system prompt (runtime/iterate.ts).
  const addendum = frame.systemAddendum;
  const agent = model ? new NativeToolAgent(model.driver, {
    systemPrompt: () => task.systemPrompt() + (addendum ? `\n\n${addendum}` : ''),
    neuralese: task.runtime.options.neuralese,
    programGuidance: eligibleGuidance && (view.binding || view.guidance()) ? view.guidance() : undefined,
    maxTurns: model.maxTurns, maxTokens: model.maxTokens, turnTokens: model.turnTokens, temperature: model.temperature,
    maxSeconds: model.maxSeconds, contextTokens: model.contextTokens,
    maxFailureRepairs: model.maxFailureRepairs, review: model.review, decisionReadout: model.decisionReadout,
    guidance: model.guidance,
    decisionSystemPrompt: () => DECISION_SYSTEM_PROMPT + (addendum ? `\n\n${addendum}` : '') }) : undefined;
  runtime = new NativeRuntime({ environment, hooks: kernelHooks,
    agent: task.runtime.options.agent ?? (agent ? session => agent.run(session) : undefined),
    maxActions: limits.maxActions, maxToolCalls: limits.maxToolCalls,
    sharedEpisodeBudget: task.episodeBudget, seedPolicy: task.runtime.options.seed, runId: callId,
    exactHostTraceCapture: task.runtime.options.exactHostTraceCapture,
    seedId: task.definitionSeedId(descriptor?.key ?? (owner ?? '') + ':' + definition.id),
    sourceRevision: definition.revision, parentCallId: frame.parentCallId, signal: task.signal,
    frame: childFrame, services, declarations: task.serviceDeclarations, serviceScopes: task.serviceScopes,
    manifest: { definition_id: definition.id, definition_name: definition.name, task_id: task.id,
      context_id: definition.contextId ?? FILE_CONTEXT,
      graph: graphManifest({ model: model ? { id: model.id ?? (model.driver as { model?: string }).model ?? (model.driver.name || null),
        revision: model.revision ?? null } : undefined, dialect: task.runtime.options.neuralese?.port?.dialect ?? null,
        rewrites: task.runtime.options.rewrites?.enabledRules() ?? [], rootContext: frame.parentCallId ? undefined : definition.contextId,
        seeds: { policy: task.runtime.options.seed ?? null } }),
      ...(definition.source ? { definition_source: definition.source } : {}), ...(options.manifest ?? {}),
      ...(options.manifest?.inline_instruction_site ? { inline_instruction_site: {
        ...(options.manifest.inline_instruction_site as Record<string, unknown>),
        realized_instruction: options.instructions ?? definition.body } } : {}) } });
  let outcome = 'failed', detail = '';
  registerTrace(callId, runtime.trace);
  try {
    const result = await runInFrame(childFrame, () => runtime!.run(node));
    outcome = result.outcome.kind; detail = result.outcome.detail;
    if (outcome !== 'done') throw new NatlangCallError(definition.name, outcome, detail, callId, runtime.trace.events as Record<string, unknown>[]);
    return toHost(result.value);
  } catch (error) {
    if (!(error instanceof NatlangCallError)) detail = error instanceof Error ? error.message : String(error);
    if (folder?.transaction.open) folder.transaction.abort();
    for (const transaction of extraTransactions) if (transaction.open) transaction.abort();
    throw error;
  } finally {
    await task.drainChildren(callId);
    releaseTrace(callId);
    task.record({ callId, parentCallId: frame.parentCallId ?? null, taskId: task.id, definitionId: definition.id,
      name: definition.name, outcome, detail, adaptation: adaptationProvenance, events: runtime.trace.events as Record<string, unknown>[] });
    environment.close();
  }
}
