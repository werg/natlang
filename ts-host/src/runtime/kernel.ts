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
import { DEFAULT_DIALECT, TypeEnv, containsRefinement, parseType, type DialectBinding, type Type } from '../native/types.js';
import { NeuraleseDialectError, NeuraleseUnsupportedError, dialectBinding, readerDialect } from '../native/neuralese.js';
import { CRISP, representationType, type GenericResult, type Representation } from '../native/representation.js';
import { containsUntrusted, scopedUntrusted } from '../native/untrusted.js';
import { RefinementError, checkServiceResults, failureError, refinementCodeOf, type RefinementCode } from '../native/refinement.js';
import { MISSING, buildPending, coerce, isLive, type CaptureCell, type LambdaNode, type Value } from '../native/values.js';
import { MAX_AD_HOC_NL_DEPTH, NatlangRecursionError, runInFrame, type Frame } from './context.js';
import { recordingServices } from './runtime.js';
import { kernelHooks } from './hooks.js';
import { engageFusion } from './fusion.js';
import { FILE_CONTEXT, graphManifest, graphNode, invocationNodeId, registerTrace, releaseTrace, traceFor } from '../native/graph.js';
import { CallCapture, definitionKey, interfaceHash, type CallStoreLike } from '../calls/recorder.js';
import { Deopt, admit, handoffNote, isDeopt } from '../calls/dispatch.js';
import type { LoadedCase } from '../calls/compilations.js';
import { runTiered, type TierAttempt } from '../calls/tiers.js';
import { specializerOutput } from '../calls/compilations.js';
import type { DefinitionIdentity } from '../calls/types.js';
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
  /** The runtime's model this definition runs on (`models`); the default model when absent or not configured. */
  model?: string;
  revision?: string;
  description?: string;
  /** Source path for named definitions. */
  source?: string;
  /** ID of the context the definition is bound to, when it was rebound (`fn.in(context)`) or defined in one. */
  contextId?: string;
  /** A representation-generic result (`returns` is then its crisp instance until a call instantiates it). */
  generic?: GenericResult;
  /** The instance this definition runs, once a call has instantiated its generic result (`instantiateDefinition`). */
  representation?: Representation;
};

/** The representation a call asks for: from the compiler's annotation of its call site, or a host's `invokeAt`. */
export type Instantiation = { representation: Representation; /** The call site, for messages. */ site?: string };

export type InvokeOptions = {
  captures?: Record<string, CaptureCell>;
  /** Instructions after interpolation, for inline lambdas. */
  instructions?: string;
  folder?: { transaction: FolderTransaction; mode: 'apply' | 'direct' };
  /** Constructors for class-typed parameters and returns. */
  classes?: ReadonlyMap<string, Function>;
  /** The runtime's dialect binding for checking arguments and results (the kernel sets it from its Neuralese options). */
  dialects?: DialectBinding;
  manifest?: Record<string, unknown>;
  /** Files of the context's `skills/` data entries, by path (`skills/<name>/SKILL.md`, ...): the call's bound skills. */
  skillFiles?: Readonly<Record<string, string | Uint8Array>>;
  /** Set by the tiered engine (calls/tiers.ts): this call is one attempt at one tier. */
  tierAttempt?: TierAttempt;
  /** The representation this call runs the definition's result at (a generic result's crisp instance by default). */
  at?: Instantiation;
};

/**
 * The definition a call runs at a representation (DECISIONS.md 2026-10-09). A generic result's crisp instance runs as
 * written; its Neuralese instance returns the soft form, written by template readout (DECISIONS.md 44), and needs a
 * runtime with a reader dialect. A definition without a generic result runs only at the representation it declares.
 */
export function instantiateDefinition(frame: Frame, definition: CallableDefinition, at: Instantiation): CallableDefinition {
  const { representation, site } = at;
  const where = site ? ` at ${site}` : '';
  const generic = definition.generic;
  if (!generic) {
    let declared: Type | undefined;
    try { declared = parseType(definition.returns); } catch { declared = undefined; }
    if ((representation.kind === 'neuralese') === (declared?.kind === 'neuralese')) return definition;
    throw new TypeError(`${definition.name} returns ${definition.returns}, so it cannot run for ` +
      `${representationType(definition.returns, representation)}${where}. A function makes both forms when its result is ` +
      'representation-generic: declare `generic: { R: string | Neuralese<string> }` with `returns: R` (its crisp type in ' +
      'place of string), and each call runs the form its result is used as.');
  }
  if (representation.kind === 'crisp') return { ...definition, returns: generic.crisp, representation: CRISP };
  const reader = readerDialect(frame.task.runtime.options.neuralese);
  if (reader === null)
    throw new NeuraleseUnsupportedError(`${definition.name} is called for its ${generic.neuralese} result${where}, but this runtime ` +
      'has no Neuralese reader (configure `neuralese` with a port or a dialect), or use the result as ' + `${generic.crisp} there`);
  const declared = generic.dialect === DEFAULT_DIALECT ? reader : generic.dialect;
  const wanted = representation.dialect === undefined || representation.dialect === DEFAULT_DIALECT ? reader : representation.dialect;
  if (wanted !== declared)
    throw new NeuraleseDialectError(`${definition.name}${where} is asked for Neuralese in dialect ${JSON.stringify(wanted)}, but its ` +
      `generic result writes ${JSON.stringify(declared)}; use the result in dialect ${JSON.stringify(declared)}, or ` +
      `\`convert(value, ${JSON.stringify(wanted)})\` from natlang:neuralese`);
  return { ...definition, returns: generic.neuralese, readout: 'template',
    representation: { kind: 'neuralese', ...(representation.dialect !== undefined ? { dialect: representation.dialect } : {}) } };
}

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
    // FileHandle/FolderHandle values are often derived from an ancestor Folder inside an
    // eval. Replacing only the Folder identity leaves those derived handles pointing at the
    // pre-transaction snapshot. Rebase them through the most specific mapped folder while
    // retaining their path and handle kind, so a file capability never becomes a folder.
    if (current === value && (value instanceof FileHandle || value instanceof FolderHandle)) {
      const ancestor = [...replacements.entries()]
        .filter(([source, target]) => (source instanceof Folder || source instanceof FolderHandle) &&
          (target instanceof Folder || target instanceof FolderHandle) &&
          value.folder === (source instanceof Folder ? source : source.folder) &&
          (source instanceof Folder || value.path === source.path || value.path.startsWith(`${source.path}/`)))
        .sort(([a], [b]) => {
          const pathA = a instanceof Folder ? '' : a.path;
          const pathB = b instanceof Folder ? '' : b.path;
          return pathB.length - pathA.length;
        })[0];
      if (ancestor) {
        const [source, target] = ancestor;
        const sourcePath = source instanceof Folder ? '' : source.path;
        const targetFolder = target instanceof Folder ? target : target.folder;
        const targetPath = target instanceof Folder ? '' : target.path;
        const suffix = value.path.slice(sourcePath.length).replace(/^\//, '');
        const path = [targetPath, suffix].filter(Boolean).join('/');
        current = value instanceof FileHandle ? new FileHandle(targetFolder, path) : new FolderHandle(targetFolder, path);
        const chained = new Set<ScopedHandle>();
        while (replacements.has(current) && !chained.has(current)) {
          chained.add(current); current = replacements.get(current)!;
        }
      }
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

/** A call that ended because its result did not satisfy a refined type (`Is<T, P>`): the code is the one in `detail`. */
export class RefinementCallError extends NatlangCallError {
  constructor(readonly code: Exclude<RefinementCode, 'refinement-predicate-invalid'>, definition: string, outcome: string, detail: string,
    callId: string, trace: Record<string, unknown>[]) {
    super(definition, outcome, detail, callId, trace);
    this.name = 'RefinementCallError';
  }
}

/**
 * The parameter and result types of a definition when any of them is (or names) a refinement; undefined otherwise, which
 * is nearly always, so ordinary calls pay one regular-expression test. Unparseable types are left to the usual errors.
 */
function refinedSignature(definition: CallableDefinition): { env: TypeEnv; params: Type[]; returns: Type; refinedReturns: boolean } | undefined {
  const texts = [definition.returns, ...definition.params.map(parameter => parameter.type), ...Object.values(definition.types)];
  if (!texts.some(text => /\bIs\s*</.test(text))) return undefined;
  try {
    const env = new TypeEnv(Object.fromEntries(Object.entries(definition.types).map(([name, text]) => [name, parseType(text)])));
    const params = definition.params.map(parameter => parseType(parameter.type)), returns = parseType(definition.returns);
    if (![...params, returns].some(type => containsRefinement(type, env))) return undefined;
    return { env, params, returns, refinedReturns: containsRefinement(returns, env) };
  } catch { return undefined; }
}

/**
 * The task's services with the declared refined results (`refinements.services`) checked: a service result is judged like
 * an nl return, and a failure throws to whoever called the service.
 */
function checkedServices(task: Frame['task'], frame: Frame, services: Readonly<Record<string, object>>, model: ReturnType<Frame['task']['model']>, callId: string,
  emit: (kind: string, data: Record<string, unknown>) => void, signal?: AbortSignal): Readonly<Record<string, object>> {
  const declared = task.runtime.options.refinements?.services;
  if (!declared || !Object.keys(declared).length) return services;
  const env = new TypeEnv();
  const types: Record<string, Type> = {};
  for (const [name, text] of Object.entries(declared)) {
    try { types[name] = parseType(text); } catch (error) { throw new TypeError(`refinements.services.${name}: ${(error as Error).message}`); }
  }
  const checker = task.refinementChecker(), judges = task.refinementJudges(model, frame);
  const registry = scopedUntrusted(task);
  return checkServiceResults(services as Record<string, unknown>, types, async (value, type, label) => {
    // Outside text enters here: its source is the service method, and the renderer shows it as data from then on.
    if (containsUntrusted(type, env)) registry.markTyped(value, type, env, label);
    if (!containsRefinement(type, env)) return;
    const failures = await checker.checkValue(value, type, env, { phase: 'service', ...judges, callId, signal, emit }, label);
    if (failures.length) throw failureError(failures[0]!);
  }) as Readonly<Record<string, object>>;
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
  env.dialects = options.dialects;
  if (node.type.kind === 'lambda') node.type.params.fields.forEach((field, index) => {
    if (inputs[index] !== undefined) node.args[field.name] = coerce(inputs[index], field.type, env,
      `${definition.name}/${field.name}`, { preserveRecordExtras: true });
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
    files: Object.fromEntries(Object.entries(files).map(([path, content]) =>
      [path, typeof content === 'string' ? content : Uint8Array.from(content)])),
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
  // Each call can be stopped on its own (its eval failed, or finished without awaiting it) and stops with its caller.
  const abort = new AbortController();
  const signal = AbortSignal.any([frame.signal ?? frame.task.signal, abort.signal]);
  const call = runDefinition({ ...frame, signal, abort }, definition, positional, options);
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
  // A generic result runs at the representation its call site chose; a plain call runs the crisp instance.
  if (!definition.representation && (definition.generic || options.at)) {
    try { definition = instantiateDefinition(frame, definition, options.at ?? { representation: CRISP }); }
    catch (error) { if (options.folder?.transaction.open) options.folder.transaction.abort(); throw error; }
  }
  const tiered = tieredCall(frame, definition, positional, options);
  if (tiered) return tiered;
  try {
    return await runDefinitionBody(frame, definition, positional, options, transactions);
  } catch (error) {
    for (const transaction of transactions) if (transaction.open) transaction.abort();
    throw error;
  }
}

/** The tiered engine (calls/tiers.ts) serves a configured function: one attempt per tier through this same dispatch. */
function tieredCall(frame: Frame, definition: CallableDefinition, positional: unknown[], options: InvokeOptions): Promise<unknown> | undefined {
  const task = frame.task, engine = task.runtime.options.tiers;
  if (!engine || options.tierAttempt || options.folder || options.manifest?.internal || task.auditOf || definition.subtype !== 'function') return;
  const settings = engine.settingsFor(definition.name);
  if (!settings) return;
  const args = Object.fromEntries(definition.params.map((parameter, index) => [parameter.name, positional[index]]).filter(([, value]) => value !== undefined));
  let recorded = 0;
  const store = task.runtime.callStore();
  return runTiered(engine, settings, { name: definition.name, args, store,
    ...(store ? { specialized: () => specializerOutput(store, definitionKey({ ...definition, body: options.instructions ?? definition.body }),
      interfaceHash(definition.codebase)) } : {}),
    hasModel: name => !!task.runtime.options.models?.[name],
    run: plan => runDefinition(frame, plan.model ? { ...definition, model: plan.model } : definition, positional, { ...options, tierAttempt: plan }),
    emit: events => {
      const id = `${task.id}/tiers-${definition.name}-${++recorded}`;
      task.record({ callId: id, parentCallId: frame.parentCallId ?? null, taskId: task.id, definitionId: `tiers:${definition.name}`,
        name: `tiers ${definition.name}`, outcome: 'done', detail: '', events: events.map(event => ({ call_id: null, ...event })) } as never);
    } });
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
    // An instantiated generic result keeps the instance its call site chose.
    definition = { ...definition, body: record.instructions, revision: record.revision,
      params: Object.entries(record.args).map(([name, type]) => ({ name: name.replace(/\?$/, ''), type, optional: name.endsWith('?') })),
      returns: definition.representation ? definition.returns : record.returns, types: record.types, codebase: record.codebase };
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
  const adHoc = !!(options.manifest?.inline || options.manifest?.delegate) && options.manifest?.adHoc !== false;
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
  // Fused hand-offs (runtime/fusion.ts): a planned edge retypes this call's result or one parameter to Neuralese<T>.
  const fusion = engageFusion(frame, definition, inputs, (callFrame, fused, args) => invokeDefinition(callFrame, fused, args, options));
  if (fusion) definition = fusion.definition;
  const signature = refinedSignature(definition);
  // An argument into a refined parameter is checked at the caller; a failure leaves the callee unstarted.
  if (signature?.params.some(type => containsRefinement(type, signature.env))) {
    try {
      const checker = task.refinementChecker();
      const judges = task.refinementJudges(task.model(definition.model), frame);
      const failures = (await Promise.all(signature.params.map((type, index) => inputs[index] === undefined ? [] :
        checker.checkValue(inputs[index], type, signature.env, { phase: 'argument', ...judges, callId: frame.parentCallId ?? null, signal: frame.signal ?? task.signal,
          emit: (kind, data) => traceFor(frame.parentCallId)?.emit(kind, { ...data, callee: definition.name }) },
          definition.params[index]!.name)))).flat();
      if (failures.length) throw failureError(failures[0]!);
    } catch (error) {
      if (folder?.transaction.open) folder.transaction.abort();
      for (const transaction of extraTransactions) if (transaction.open) transaction.abort();
      throw error;
    }
  }
  const callId = task.nextCallId();
  // A file-backed call owns the files from its own companion folder. Inline calls inherit their
  // caller's bound skill set unless one is explicitly supplied for that inline definition.
  const invocationSkillFiles = options.manifest?.inline ? (options.skillFiles ?? frame.skillFiles) : options.skillFiles;
  const childFrame: Frame = { task, chain: [...frame.chain, callIdentity], parentCallId: callId, adHocDepth, programOwner: owner,
    signal: frame.signal, abort: frame.abort,
    ...(invocationSkillFiles ? { skillFiles: invocationSkillFiles } : {}),
    ...(frame.scopedHandleReplacements ? { scopedHandleReplacements: frame.scopedHandleReplacements } : {}),
    ...(options.manifest?.inline ? { inline: true } : {}) };
  const model = task.model(definition.model);
  const store = task.runtime.callStore();
  const capture = store ? openCapture(store, task, frame, callId, definition, options, inputs, folder, model) : undefined;
  let handoff: string | undefined;
  let shadows: LoadedCase[] = [];
  const attempt = options.tierAttempt;
  if (capture && attempt) attempt.capture = capture;
  const mode = capture && !options.manifest?.internal && !task.auditOf && attempt?.compiled !== false ? task.specializationMode() : 'off';
  // A tier's own crisp case (calls/tiers.ts) was admitted by the tier's guard; it runs whatever the specialization mode.
  if (capture && attempt?.cases?.[0]) {
    const crisp = await runCrispCase({ task, frame, childFrame, callId, definition, options, inputs, folder, extraTransactions,
      capture, store: store!, item: attempt.cases[0] });
    if (crisp.served) return crisp.value;
    handoff = crisp.note;
  }
  if (capture && mode !== 'off') {
    const compilation = task.runtime.compilations()?.get(capture.base.definition.key, capture.base.definition.interface,
      definition.codebase, definition.types);
    if (compilation) {
      const available = frame.services ?? task.services;
      const admitted = admit(compilation, caseArguments(capture, folder), mode, name => Object.hasOwn(available, name) && !!available[name]);
      shadows = admitted.shadows;
      if (admitted.active) {
        const crisp = await runCrispCase({ task, frame, childFrame, callId, definition, options, inputs, folder, extraTransactions,
          capture, store: store!, item: admitted.active });
        if (crisp.served) return crisp.value;
        handoff = crisp.note;
      }
    }
  }
  // A strict tier attempt serves from crisp cases only: with none, the engine goes on to the next tier and no model runs.
  if (attempt?.strict && !handoff) { capture?.discard(); throw new Deopt(attempt.declined ?? 'no crisp case of this tier served the call'); }
  const environment = task.environment();
  let runtime: NativeRuntime | undefined;
  const services = recordingServices(checkedServices(task, frame, task.services, model, callId, (kind, data) => { runtime?.trace.emit(kind, data); }, frame.signal ?? task.signal) as Record<string, object>, ({ exact, ...event }) => (capture?.effect({ ...event, exact }, 'agent'), event.phase === 'requested' ?
    runtime?.trace.emit('effect', { call_id: callId, capability: `${event.service}.${event.method}`, ...event }) :
    graphNode(runtime?.trace, 'effect', { call_id: callId, capability: `${event.service}.${event.method}`, ...event },
      [{ node: invocationNodeId(callId), port: 'caller' }])));
  // A stopping predicate of iterateOn runs under its own addition to the system prompt (runtime/iterate.ts).
  const addendum = [frame.systemAddendum, attempt?.guidance].filter(Boolean).join('\n\n') || undefined;
  const agent = model ? new NativeToolAgent(model.driver, {
    systemPrompt: () => task.systemPrompt() + (addendum ? `\n\n${addendum}` : ''),
    neuralese: task.runtime.options.neuralese,
    programGuidance: eligibleGuidance && (view.binding || view.guidance()) ? view.guidance() : undefined,
    maxTurns: model.maxTurns, maxTokens: model.maxTokens, turnTokens: model.turnTokens, temperature: model.temperature,
    maxSeconds: model.maxSeconds, contextTokens: model.contextTokens,
    maxFailureRepairs: model.maxFailureRepairs, review: model.review, decisionReadout: model.decisionReadout,
    guidance: model.guidance,
    ...(signature?.refinedReturns ? { refinement: { checker: task.refinementChecker(), ...task.refinementJudges(model, frame) } } : {}),
    decisionSystemPrompt: () => DECISION_SYSTEM_PROMPT + (addendum ? `\n\n${addendum}` : '') }) : undefined;
  runtime = new NativeRuntime({ environment, hooks: kernelHooks,
    agent: task.runtime.options.agent ?? (agent ? session => agent.run(session) : undefined),
    maxActions: limits.maxActions, maxToolCalls: limits.maxToolCalls,
    sharedEpisodeBudget: task.episodeBudget, seedPolicy: task.runtime.options.seed, runId: callId,
    exactHostTraceCapture: task.runtime.options.exactHostTraceCapture,
    seedId: task.definitionSeedId(descriptor?.key ?? (owner ?? '') + ':' + definition.id),
    sourceRevision: definition.revision, parentCallId: frame.parentCallId, signal: frame.signal ?? task.signal,
    frame: childFrame, services, declarations: task.serviceDeclarations, serviceScopes: task.serviceScopes,
    neuralese: task.runtime.options.neuralese,
    manifest: { definition_id: definition.id, definition_name: definition.name, task_id: task.id,
      ...(definition.representation ? { representation: definition.representation } : {}),
      context_id: definition.contextId ?? FILE_CONTEXT,
      graph: graphManifest({ model: model ? { id: model.id ?? (model.driver as { model?: string }).model ?? (model.driver.name || null),
        revision: model.revision ?? null } : undefined, dialect: readerDialect(task.runtime.options.neuralese),
        rewrites: task.runtime.options.rewrites?.enabledRules() ?? [], rootContext: frame.parentCallId ? undefined : definition.contextId,
        seeds: { policy: task.runtime.options.seed ?? null } }),
      ...(definition.source ? { definition_source: definition.source } : {}), ...(options.manifest ?? {}),
      ...(options.manifest?.inline_instruction_site ? { inline_instruction_site: {
        ...(options.manifest.inline_instruction_site as Record<string, unknown>),
        realized_instruction: options.instructions ?? definition.body } } : {}) } });
  let outcome = 'failed', detail = '';
  let hostValue: unknown, hasValue = false;
  registerTrace(callId, runtime.trace);
  capture?.watch(() => runtime!.trace.events as Record<string, unknown>[]);
  try {
    const normalizedInputs = await runtime.materializeSoftStringArguments(definition, inputs, frame.parentCallId);
    const node = await prepareDefinitionNode(definition, normalizedInputs,
      { ...options, dialects: dialectBinding(task.runtime.options.neuralese) });
    if (handoff) node.handoff = handoff;
    if (folder) { node.projectTransaction = folder.transaction; node.reducerMode = folder.mode; }
    if (extraTransactions.length) node.extraTransactions = extraTransactions;
    const result = await runInFrame(childFrame, () => runtime!.run(node));
    outcome = result.outcome.kind; detail = result.outcome.detail;
    const scored = frame.readout && runtime.trace.events.find(item => item.kind === 'decision_readout' && item.phase === 'scored');
    if (scored) frame.readout!({ options: scored.options as string[], probabilities: scored.probabilities as number[] });
    if (outcome !== 'done') {
      const code = refinementCodeOf(detail);
      if (code) throw new RefinementCallError(code, definition.name, outcome, detail, callId, runtime.trace.events as Record<string, unknown>[]);
      throw new NatlangCallError(definition.name, outcome, detail, callId, runtime.trace.events as Record<string, unknown>[]);
    }
    // Whatever path produced the value (an eval that finished it, a scripted agent), a refined result is judged here.
    // The agent's own check has cached its verdicts, so this costs no further scoring pass.
    if (signature?.refinedReturns) {
      const failures = await task.refinementChecker().checkValue(result.value, signature.returns, signature.env,
        { phase: 'return', ...task.refinementJudges(model, frame), callId, signal: frame.signal ?? task.signal,
          emit: (kind, data) => { runtime!.trace.emit(kind, data); } });
      if (failures.length) {
        outcome = 'failed'; detail = `${failures[0]!.code}: ${failures[0]!.message}`;
        throw new RefinementCallError(failures[0]!.code, definition.name, outcome, detail, callId, runtime.trace.events as Record<string, unknown>[]);
      }
    }
    hostValue = toHost(result.value); hasValue = true;
    fusion?.produced?.(hostValue);
    return hostValue;
  } catch (error) {
    if (!(error instanceof NatlangCallError)) detail = error instanceof Error ? error.message : String(error);
    // A failed call stops the calls it started that are still running.
    frame.abort?.abort(new Error(`${definition.name} ended without a result`));
    if (folder?.transaction.open) folder.transaction.abort();
    for (const transaction of extraTransactions) if (transaction.open) transaction.abort();
    throw error;
  } finally {
    await task.drainChildren(callId);
    releaseTrace(callId);
    task.record({ callId, parentCallId: frame.parentCallId ?? null, taskId: task.id, definitionId: definition.id,
      name: definition.name, outcome, detail, adaptation: adaptationProvenance, events: runtime.trace.events as Record<string, unknown>[] });
    environment.close();
    if (capture) {
      capture.finish({ outcome, detail, output: hostValue, hasOutput: hasValue, events: runtime.trace.events as Record<string, unknown>[] });
      if (outcome !== 'done') task.failedCalls.set(`${frame.parentCallId ?? ''}|${capture.base.definition.key}`, callId);
      if (outcome === 'done' && capture.executor.kind === 'agent')
        for (const item of shadows) { try { store!.enqueue('shadow', item.hash, callId); } catch { /* recording never fails a call */ } }
    }
  }
}

/** Open the record of a call (§3.1); recording never fails the call. */
function openCapture(store: CallStoreLike, task: Frame['task'], frame: Frame, callId: string, definition: CallableDefinition,
  options: InvokeOptions, inputs: unknown[], folder: InvokeOptions['folder'], model: ReturnType<Frame['task']['model']>): CallCapture | undefined {
  try {
    const settings = store.settings();
    const site = options.manifest?.internal ? 'internal' : options.manifest?.inline || options.manifest?.delegate ? 'inline' :
      frame.systemAddendum !== undefined ? 'iterate' : 'named';
    const inlineSite = options.manifest?.inline_instruction_site as { template_segments?: unknown } | undefined;
    const body = options.instructions ?? definition.body;
    const identity: DefinitionIdentity = { id: definition.id, name: definition.name, source: definition.source ?? null,
      key: definitionKey({ ...definition, body }), interface: interfaceHash(definition.codebase), site,
      ...(inlineSite?.template_segments ? { template: hexDigest(JSON.stringify(inlineSite.template_segments)).slice(0, 24) } : {}),
      subtype: definition.subtype, params: definition.params, returns: definition.returns,
      instructions: { complete: false, reason: 'excluded' }, types: definition.types, ...(definition.readout ? { readout: definition.readout } : {}),
      ...(definition.generic && definition.representation ? { generic: { name: definition.generic.name, constraint: definition.generic.constraint },
        representation: definition.representation } : {}) };
    const parentTrace = traceFor(frame.parentCallId);
    const view = task.programView;
    const capture = new CallCapture(store, settings, { callId, parentCallId: frame.parentCallId ?? null,
      parentActionIndex: parentTrace ? parentTrace.events.filter(event => event.kind === 'action').length : null,
      taskId: task.id, programId: view.program?.id ?? null, buildHash: view.program?.buildHash ?? null,
      programRoot: task.runtime.options.programRoot ?? null, definition: identity,
      model: { id: model ? model.id ?? (model.driver as { model?: string }).model ?? executorModel(task) ?? undeclaredExecutor(model.driver.name) : null,
        revision: model?.revision ?? null },
      exclude: task.runtime.options.recording?.exclude });
    identity.instructions = capture.ref(body);
    if (task.auditOf && !frame.parentCallId) capture.auditOf = task.auditOf;
    // A caller that runs the same function again after a failure is evidence about the failed call (§3.6).
    const retryKey = `${frame.parentCallId ?? ''}|${identity.key}`;
    const failed = task.failedCalls.get(retryKey);
    if (failed) { task.failedCalls.delete(retryKey); store.annotate?.(failed, 'retried', { by: callId }, 'runtime', false); }
    const named = Object.fromEntries(definition.params.map((parameter, index) => [parameter.name, inputs[index]])
      .filter(([, value]) => value !== undefined));
    const captures = Object.fromEntries(Object.values(options.captures ?? {}).filter(cell => !cell.skill)
      .map(cell => { try { return [cell.name, cell.get()]; } catch { return [cell.name, undefined]; } })
      .filter(([, value]) => typeof value !== 'function'));
    capture.setInputs(named, captures);
    if (folder) capture.setFolderInput(folder.transaction.folder);
    capture.announce();
    return capture;
  } catch (error) {
    console.warn(`natlang: call recording failed for ${definition.name}: ${error instanceof Error ? error.message : String(error)}`);
    return undefined;
  }
}

/** The model named by the runtime's executor identity (the CLI's profile), when there is one. */
/**
 * A driver that declares no model (a scripted test agent, an ad-hoc function) is recorded by its function name under
 * `undeclared:`, so mining never learns from it as if a model had run (calls/types.ts isModelExecutor).
 */
function undeclaredExecutor(name: string): string {
  return `undeclared:${name || 'anonymous'}`;
}

function executorModel(task: Frame['task']): string | undefined {
  const identity = task.runtime.options.executorIdentity;
  const configured = identity?.configuration?.model;
  return typeof configured === 'string' && configured ? configured : identity?.id || undefined;
}

/** The one argument a case's guard and body receive: the call's parameters and captures by name (and `folder`). */
function caseArguments(capture: CallCapture, folder: InvokeOptions['folder']): Record<string, unknown> {
  return { ...capture.hostInputs, ...(folder ? { folder: folder.transaction.folder.root() } : {}) };
}

/**
 * Run an admitted case (§6.4). Served: its value, after the type check, folder commit and capture writes. Otherwise
 * (it threw, or returned a value of the wrong type) the call goes to the agent with a note of what already happened (§6.5).
 */
async function runCrispCase(input: { task: Frame['task']; frame: Frame; childFrame: Frame; callId: string; definition: CallableDefinition;
  options: InvokeOptions; inputs: unknown[]; folder: InvokeOptions['folder']; extraTransactions: FolderTransaction[];
  capture: CallCapture; store: CallStoreLike; item: LoadedCase }): Promise<{ served: true; value: unknown } | { served: false; note?: string }> {
  const { task, childFrame, callId, definition, options, inputs, folder, capture, store, item } = input;
  const args = caseArguments(capture, folder);
  const original = { ...args };
  const services = recordingServices(task.services, event => capture.effect(event, 'crisp'));
  let error: unknown;
  try {
    const raw = await runInFrame({ ...childFrame, services }, () => item.run(args));
    await task.drainChildren(callId);
    const dialects = dialectBinding(task.runtime.options.neuralese);
    const node = definitionNode(definition, inputs, { ...options, dialects });
    if (node.type.kind !== 'lambda') throw new Error('not a function');
    const env = new TypeEnv(node.types);
    env.classes = options.classes;
    env.dialects = dialects;
    const structural = coerce(raw as Value, node.type.returns, env, 'return');
    // A compiled case's result must satisfy refinements too; if not, the case fails and the agent takes the call.
    if (containsRefinement(node.type.returns, env)) {
      const failures = await task.refinementChecker().checkValue(structural, node.type.returns, env,
        { phase: 'return', ...task.refinementJudges(task.model(definition.model), input.frame), callId, signal: input.frame.signal ?? task.signal,
          emit: (kind, data) => { traceFor(callId)?.emit(kind, data); } });
      if (failures.length) throw failureError(failures[0]!);
    }
    const value = toHost(structural);
    if (folder?.transaction.open) {
      const changes = folder.transaction.folder.diffSync().changes;
      if (folder.mode === 'apply') {
        folder.transaction.validateSync();
        for (const transaction of input.extraTransactions) transaction.validateSync();
        folder.transaction.commitSync();
        for (const transaction of input.extraTransactions) if (transaction.open) transaction.commitSync();
      } else {
        folder.transaction.abort();
        for (const transaction of input.extraTransactions) if (transaction.open) transaction.abort();
      }
      capture.folder = { mode: folder.mode, changes: changes.map(change => ({ path: change.path, kind: change.kind,
        ...(change.after ? { after: capture.ref(new TextDecoder().decode(change.after)) } : {}) })) };
    }
    for (const cell of Object.values(options.captures ?? {})) {
      if (!cell.mutable || !cell.set || cell.skill || !(cell.name in args) || Object.is(args[cell.name], original[cell.name])) continue;
      cell.set(args[cell.name]);
      capture.captureWrites.push({ name: cell.name, after: capture.ref(args[cell.name]) });
    }
    capture.executor = { ...capture.executor, kind: 'crisp', case_hash: item.hash };
    task.record({ callId, parentCallId: input.frame.parentCallId ?? null, taskId: task.id, definitionId: definition.id, name: definition.name,
      outcome: 'done', detail: `served by compiled case ${item.hash}`, events: [] });
    capture.finish({ outcome: 'done', detail: `served by compiled case ${item.hash}`, output: value, hasOutput: true, events: [] });
    try {
      store.caseServed(item.hash, callId, false);
      if (!item.hash.startsWith('tier:') && Math.random() < capture.settings.auditRate) store.enqueue('audit', item.hash, callId);
    } catch { /* recording never fails a call */ }
    return { served: true, value };
  } catch (caught) { error = caught; }
  await task.drainChildren(callId);
  const effects = capture.completedEffects('crisp');
  const files = folder?.transaction.open ? folder.transaction.folder.diffSync().changes.map(change => ({ path: change.path, kind: change.kind })) : [];
  const calls = task.traces.filter(trace => trace.parentCallId === callId).map(trace => ({ name: trace.name, outcome: trace.outcome }));
  // A case that declines before doing anything is a guard that did not apply: the agent runs as if it had not admitted.
  if (isDeopt(error) && !effects.length && !files.length && !calls.length) return { served: false };
  const message = error instanceof Error ? `${error.name === 'Error' ? '' : `${error.name}: `}${error.message}` : String(error);
  capture.executor = { ...capture.executor, kind: 'crisp-agent', case_hash: item.hash, case_error: message.slice(0, 2000) };
  try { store.caseServed(item.hash, callId, true); } catch { /* recording never fails a call */ }
  const valueOf = (ref: { complete: boolean; hash?: string }) => ref.complete && ref.hash ? JSON.parse(capture.blobs.get(ref.hash) ?? 'null') : undefined;
  if (!effects.length && !files.length && !calls.length)
    return { served: false, note: handoffNote({ error: message, effects: [], files: [], calls: [] }) };
  return { served: false, note: handoffNote({ error: message, files, calls,
    effects: effects.map(effect => ({ effect, args: valueOf(effect.args), result: effect.result ? valueOf(effect.result) : undefined })) }) };
}
