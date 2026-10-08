/**
 * The natural-language function interpreter: runs one invocation of a `.nl` or inline `nl`
 * function as a model-driven session over a persistent TypeScript scope.
 *
 * Portable data reaches eval as a frozen snapshot; live values (host objects, functions, folder
 * handles, captured bindings, callables, services) arrive by reference through `__live`.
 */
import { COMPACTED_RESULT } from './prompt.js';
import { arrayToStringNeuralese, concatNeuralese, invokeWithReceiver, joinNeuralese, mapNeuraleseReadout, readNeuraleseIfReference, rebindInlineCallable, type InlineInstructionOrigin } from '../runtime/lowered.js';
import { EvalFailure, type EvalEnvironment, type HostEvent } from './evaluator.js';
import { PageStore } from './pages.js';
import { isRecording, recordingServices, type EffectEvent } from './effects.js';
import { TypeEnv, formatType, parseType, type Type } from './types.js';
import { evalTypeDeclarations, inlineDeclaredTypes } from './eval-types.js';
import { MISSING, Reject, coerce, hostCopy, dump, isLive, isPending, liveLabel, problems, unboundParts, createLiveIdentity, scopedLiveIdentity,
  isPlainRecord, type LambdaNode, type Value } from './values.js';
import { changes, NativeTraceRecorder } from './trace.js';
import { FileHandle, Folder, FolderHandle, editTextContent, fileListingText, type EntryStat } from './scoped-fs.js';
import { fileDiffPreview } from './file-diff-preview.js';
import type { PythonHost } from './folder-python.js';
import { compileScopeSnippet, SCOPE_RUNTIME_PRELUDE, type PersistentScopeHelper } from '../scope-compiler.js';
import { livePreview, renderValue } from './agent.js';
import type { InlineLambdaPlan, NatlangDiagnostic } from '../compiler/inline.js';
import { NEURALESE_TYPE_DOCUMENTATION } from '../compiler/intrinsics.js';
import { desugarNlCalls } from '../compiler/nl-call.js';
import { isNeuraleseRef, neuraleseRef, NeuraleseUnsupportedError, sourceWithLiteralCalls,
  type NeuraleseRuntimeOptions } from './neuralese.js';
import { blockInput, FILE_CONTEXT, graphNode, invocationNodeId, valueInputs } from './graph.js';
import { canGenerateNl, currentFrame, racedCalls, runInFrame, type Frame } from '../runtime/context.js';
import { PATH_ONLY, parseModule, parseNatlang, type ItemRecord } from '../runtime/loader.js';
import { compileModule } from '../runtime/modules.js';
import { readNeuraleseForCurrentTask } from '../neuralese/combinators.js';
import { undeclaredServiceType } from './introspection.js';

/** Services the invocation kernel provides to an interpreter run. */
export type NativeRuntimeHooks = {
  /** Callable tree for a codebase record tree (functions with child attributes and `iterateOn`). */
  callables(codebase: Record<string, unknown>, session: NativeSession): Record<string, unknown>;
  /** Create an inline natlang callable from an eval plan. */
  inline(session: NativeSession, plan: InlineLambdaPlan, values: unknown[], accessors: Record<string, unknown>, origin?: InlineInstructionOrigin): unknown;
  iterateOn(session: NativeSession, step: unknown, initial: unknown, ...args: unknown[]): unknown;
  finite(source: unknown, label?: string): unknown;
  finiteArrayIterator(source: unknown, method: 'entries' | 'keys' | 'values', label?: string): unknown;
  finiteAsync(source: unknown, label?: string): unknown;
  guard(id: string, fn: () => unknown, args?: readonly unknown[]): unknown;
  /** Type-checked analysis of `nl` in eval snippets. */
  analyze(session: NativeSession, source: string): { plans: InlineLambdaPlan[]; diagnostics: NatlangDiagnostic[];
    neuralese?: import('../compiler/neuralese.js').NeuraleseLiteral[]; readouts?: import('../compiler/neuralese.js').NeuraleseReadout[] };
};
export type NativeOutcome = { kind: 'done' | 'quiesced'; detail: string; value?: Value };
/** A tool call's result. `entry` is its index in the session's transcript. */
export type NativeResult = { kind: string; text: string; value?: Value; codes?: string[]; entry?: number };
export type NativeAgent = (session: NativeSession) => Promise<string | void> | string | void;
export type NativeRuntimeOptions = { environment: EvalEnvironment; hooks: NativeRuntimeHooks; agent?: NativeAgent;
  /** Sees each service call this runtime wraps, with its exact arguments and result (a standalone run's call record). */
  observeEffect?: (event: EffectEvent) => void;
  maxActions?: number; maxToolCalls?: number;
  runId?: string; seedId?: string; signal?: AbortSignal; timeoutMs?: number;
  exactHostTraceCapture?: { definitionSources: string[]; inputArguments: string[]; captureOutput?: boolean; captureAllOutputs?: boolean; maxBytes: number };
  sourceRevision?: string; parentCallId?: string;
  /** Task frame of this invocation (task, caller chain, parent call). */
  frame?: Frame;
  /** Extra manifest fields recorded for this invocation. */
  manifest?: Record<string, unknown>;
  /** Host services, injected into eval as named bindings; their calls are recorded as effects (see effects.ts). */
  services?: Record<string, object>;
  /** Configured soft-value writer for exact typed result materialization. */
  neuralese?: NeuraleseRuntimeOptions;
  /**
   * Declarations of external services by name (see external.ts): what the model is shown of them, and what
   * read_code returns. Their implementations are not part of the program and cannot be read or edited.
   */
  declarations?: Record<string, string>;
  /**
   * Services only some functions may use, by the source path of each function (`answer_question/table_expert.nl`):
   * such a service is in scope in that function's calls and the calls they make, and nowhere else. Its declaration
   * stays readable everywhere; only its use is limited, as a specialist can reach systems its caller cannot.
   */
  serviceScopes?: Record<string, string[]>;
  sharedEpisodeBudget?: { limit?: number; used: number };
  seedPolicy?: { mode: 'derived' | 'backend'; root?: number } };

type Ref = { path: string; type?: Type; env: TypeEnv; deny?: string;
  get(): Value; set(value: Value): void; del(): void };

/** A one-line summary of a value for status lines; `holder` names where all of it is (see renderValue). */
const oneLine = (value: unknown, holder?: string, liveIdentity?: (value: object) => number) => renderValue(value, { holder, budget: 80, liveIdentity });
function diagnosticValue(value: Value, holder?: string, liveIdentity?: (value: object) => number): unknown {
  if (portableSizeAtMost(value, TRANSCRIPT_VALUE_CHARS) && !containsLive(value)) return dump(value);
  return { $diagnostic_preview: oneLine(value, holder, liveIdentity), complete: false, holder };
}
function diagnosticArgument(value: unknown, holder: string, liveIdentity?: (value: object) => number): unknown {
  if (portableSizeAtMost(value, TRANSCRIPT_VALUE_CHARS) && !containsLive(value)) return value;
  return { $diagnostic_preview: oneLine(value, holder, liveIdentity), complete: false, holder };
}

const DIAGNOSTIC_HINTS: Record<string, string> = {
  'type-mismatch': 'Pass the value itself with the type shown as expected, not wrapped in another object: for boolean use `true`, for number use `42.5`, for string use text, and for a record use an object with exactly its fields.',
  'unknown-field': 'Use one of the fields listed as expected.',
  'no-such-path': 'Use a variable or field that exists in the scope.',
  'capture-conflict': 'Another caller changed that captured variable; run the eval again with its current value.',
};
/** Every tool the runtime can apply; a call's offer is a subset. */
const NATIVE_TOOLS = ['eval', 'read_page', 'compact_history', 'return_result', 'blocked', 'failed',
  'read_code', 'edit_code', 'diff_code',
  'list_files', 'search_files', 'read_file', 'write_file', 'edit_file', 'diff_files', 'bash', 'python', 'delegate', 'editor'];
const CODE_TOOLS = ['read_code', 'edit_code', 'diff_code'];
/** What read_code shows for the built-ins of eval, which have no source in the program. */
const ITERATION_STATE_GUIDANCE = 'Extra arguments are fixed: iterateOn(step, initialState, 0) passes 0 to every step; it does not supply an iteration index. ' +
  'For ordered edits, put nextPass in the state alongside the current draft, apply passes[state.nextPass], return the revised draft with nextPass increased by one, and stop when nextPass equals passes.length. ' +
  'The stopping check also runs on the initial state before any step; existing fields or a complete object shape do not establish that all edits have been applied. ' +
  'maxSteps limits work; reaching it does not declare success.\n' +
  'For a typed draft revision, capture fixed context in nl.with<Draft>({ policy }), call that child with state.draft, ' +
  'and wrap its returned Draft only in the code step: return { pass: state.pass + 1, draft: nextDraft }. ' +
  'The child takes the Draft argument, not the capture object or the outer progress state; its result is the complete Draft. ' +
  'Capture the applicable output contract, including exact-copy and field-format rules, and pass the current full draft as the child argument. ' +
  'After each pass, build from the returned draft; do not reconstruct later state from an outer initial draft. ' +
  'nl.with accepts one type argument (result type or full callable signature), or two as <CaptureRecord, Result>; the first checks listed captures and does not type the child input. For example:\n' +
  '  type Progress = { pass: number; draft: Draft };\n' +
  '  const step = async (state: Progress): Promise<Progress> => {\n' +
  '    const nextDraft = await nl.with<Draft>({ policy, pass: passes[state.pass] })`Apply this pass to the current draft.`(state.draft);\n' +
  '    return { pass: state.pass + 1, draft: nextDraft };\n' +
  '  };\n' +
  'The child returns Draft; the TypeScript step returns Progress.\n';

export const BUILT_IN_DOCS: Record<string, string> = {
  Neuralese: NEURALESE_TYPE_DOCUMENTATION,
  read: 'Read an opaque Neuralese<T> value using the configured typed reader. In eval, call `await read(value)`; it returns Promise<T> and records the ordinary typed readout in the execution graph. The value must be a genuine Neuralese reference; use FileHandle.readText() or FileHandle.readJson() for files.',
  'neuralese.read': 'Compatibility access through the configured neuralese service. Prefer the shared eval builtin `await read(value)` for Neuralese<T>; it returns Promise<T> and records the ordinary typed readout in the execution graph. `neuralese.bodies.read` is only the configured reader body ID.',
  'neuralese.textReadSource': `neuralese.textReadSource is read-only standard-library metadata, not a callable API or a value to invoke.
  read_code("neuralese") shows the available host service declaration; read_code("Neuralese") shows the opaque compile-time type.
The descriptor identifies the configured read instruction body (export: "read") and its provenance. Do not call
neuralese.textReadSource.read or inspect/copy its fields to read a value. Use the supported typed operation
read(value), or a supported JavaScript text-conversion context such as String(value) for
Neuralese<string>. The runtime performs the configured typed readout; the descriptor itself does not perform it.`,
  decide: `decide(fn, ...args): call a function in this eval with those arguments and inspect its decision result.
  const decision = await decide(verdict, statement);
  decision.value                    the function's answer
  decision.probabilities            { value, probability }[] for its finite result choices
  decision.confidence               probability of the returned answer
  decision.scored                   whether the model supplied a scored distribution
The first argument must be a callable function in scope, such as a natural-language function with a finite result
type; the remaining arguments are passed to it in order. When the model driver has no decision scorer, the function
still runs normally and the result has one probability-1 answer with scored=false.`,
  nl: `nl: create a natural-language function inside eval code. Calling it runs another call like this one, with its own
instructions, on the arguments you pass; await the call.
  nl\`instructions\`(arg, ...)        a one-off judgment, extraction or transformation on these arguments
  nl<T>\`instructions\`               the same with its result type T written out
  nl.with<T>({ policy })\`instructions\`(item)   explicit context snapshots, result type T
  nl.with<C, T>({ policy })\`instructions\`(item)   C checks the capture record; T is the result type
  nl<T>\`instructions\`.with({ policy })(item)   the same capture binding on this inline tag
  const f: (item: Item) => Promise<T> = nl\`instructions\`   a named function with a signature, to call many times
  await nl(\`instructions with \${values}\`)   a one-shot question: the same as nl\`...\`(), its result is the answer
Parameters take their names from the call (nl\`Is item urgent?\`(item) names it item) or the signature; a saved nl
without either receives input, input2, ...: give it a signature so its instructions and arguments agree. The
instructions also see variables in scope that they mention by exact name. The result type comes from how the result
is used (an annotation, a comparison, a field read); write nl<T> when nothing says it.
In nl.with<T>, T describes the result, not the capture object. In nl.with<C, T>, C is a finite capture-record schema
and T is the result; the callable's input is passed separately and inferred from the call (or declared with one full
callable signature). A suffix .with(...) can apply directly to an inline
nl template, or to a saved inline nl function created earlier in the same eval. On a saved function, .with(record)
creates a fresh callable with the same instructions and already-evaluated interpolations, taking a snapshot of the
same named captures from the new finite record; a const alias of that function can be rebound too. For example,
nl<string>\`Write the complete explanation using authorization and authText2.\`.with({ authorization, authText2 })(item)
returns a string while snapshotting those two captures. Pass changing iteration state as an argument when each call needs a
different value. If a capture and input parameter share a name, the parameter shadows the capture inside the child; the
capture initializer is still evaluated when the callable is created, but the captured value is not exposed to the child.
Ordinary functions and opaque soft function values cannot be rebound this way.
Examples:
  const verdicts = await Promise.all(items.map(item => nl.with<boolean>({ policy })\`Decide whether item meets policy.\`(item)));
  const kept = items.filter((item, i) => verdicts[i] === true);
  const risk: 'low' | 'high' = await nl\`Rate the risk in note.\`(note);
For an eligibility judgment, declare boolean as the result even when its captures are an object. An object or
string result is not a Boolean verdict; do not select items by its truthiness.
When a child writes a structured answer, capture the applicable output contract (including exact-copy and formatting
rules). For a state revision, pass the full current state as its argument, return the declared child result type,
and preserve fields outside the current pass.
An nl function also has .iterateOn(initial).until(check); see iterateOn.`,
  iterateOn: `iterateOn: repeat a step an open-ended number of times (eval has no while).
  const final = await iterateOn(step, initialState, ...otherArgs).withLimit({ maxSteps: 20 }).until(state => isFinished(state));
step(state, ...otherArgs) returns the next state and may be async or an nl function; until's check receives each
state and says when to stop. Each step receives the preceding step's returned state (initialState only on the first
step). Build revisions from that current argument and preserve earlier supported edits; a captured outer initial
draft stays the original value. ${ITERATION_STATE_GUIDANCE}The state keeps the type of the initial value. An nl function has it as a method:
  const plan2 = await nl\`Make plan more concrete.\`.iterateOn(plan).until(nl\`plan names an owner for every task.\`);
A natural-language check (until(nl\`…\`)) needs no bound: it is told it decides when the loop stops, and a progress
review stops a loop that is stuck. A TypeScript check can loop forever, so it needs a bound:
  const n = await iterateOn(grow, 1).withLimit({ maxSteps: 20 }).until(value => value > 1000);
Configure the bound before .until(...), which starts execution. For a known list of edits, maxSteps is its length.
withMeasure(state => remainingWork) is the other bound: a count that must fall at every step. .checkProgress('off')
turns the progress review off and then also needs a bound.`,
  transcript: `transcript: this call's earlier tool calls, with their full outputs, in eval's scope.
  transcript.search(textOrRegex, { in, status, tool })   matching lines of your earlier reasoning, code and outputs
  transcript.entry(n)   call n: { turn, reasoning, tool, code, arguments, status, value, console, output }
status is ok, rejected or error; value is what an eval returned, as data. Search for something specific rather than
reading it through, and rather than repeating work.`,
};

function rejected(error: Reject): NativeResult {
  const softMismatch = error.diagnostics.some(diagnostic => diagnostic.code === 'type-mismatch' &&
    /\bNeuralese\s*</.test(diagnostic.expected ?? ''));
  const hint = softMismatch ? 'At a declared Neuralese<string> argument or final-result slot, an ordinary string is written as that typed value through the configured port; argument conversion applies only to the direct parameter, not nested fields. For a final Neuralese<T> result with a concrete JSON-serializable T, return a value of T or its JSON text; the runtime validates it and writes its canonical JSON through the configured port. These conversions do not apply to locals or other payload types: preserve typed values or use one exact block marker in an explicitly typed position. A displayed [[Neuralese text block ...]] label is a preview, not a JavaScript string value or a reference to copy; do not turn it into a string or object.' :
    error.diagnostics.map(diagnostic => DIAGNOSTIC_HINTS[diagnostic.code]).find(Boolean);
  return { kind: 'rejected', text: `rejected\n${error.message}${hint ? `\nhint: ${hint}` : ''}`,
    codes: error.diagnostics.map(diagnostic => diagnostic.code) };
}

/** A type whose values have an unambiguous JSON representation (excluding opaque and recursively aliased shapes). */
function jsonSerializableType(type: Type, env: TypeEnv, active = new Set<Type>()): boolean {
  const resolved = env.resolve(type);
  if (active.has(resolved)) return false;
  if (resolved.kind === 'prim') return ['string', 'number', 'boolean', 'null'].includes(resolved.name);
  if (resolved.kind === 'lit') return typeof resolved.value === 'string' ||
    (typeof resolved.value === 'number' && Number.isFinite(resolved.value));
  active.add(resolved);
  try {
    if (resolved.kind === 'union') return resolved.members.length > 0 && resolved.members.every(member => jsonSerializableType(member, env, active));
    if (resolved.kind === 'list' || resolved.kind === 'dict') return jsonSerializableType(resolved.element, env, active);
    if (resolved.kind === 'record') return resolved.fields.every(field => jsonSerializableType(field.type, env, active));
    return false;
  } finally { active.delete(resolved); }
}

function coerceCompleteJson(value: unknown, type: Type, env: TypeEnv, path: string): Value {
  const resolved = env.resolve(type);
  if (resolved.kind === 'union') {
    let first: Reject | undefined;
    for (const member of resolved.members) try {
      const candidate = coerce(value, member, env, path);
      return coerceCompleteJson(candidate, member, env, path);
    } catch (error) {
      if (!(error instanceof Reject)) throw error;
      first ??= error;
    }
    if (first) throw first;
  }
  const normalized = coerce(value, resolved, env, path);
  if (resolved.kind === 'record') {
    const record = normalized as Record<string, Value>;
    for (const field of resolved.fields) if (!Object.hasOwn(record, field.name) && !field.optional)
      throw new Reject([{ path: `${path}/${field.name}`, code: 'hole', expected: field.name }]);
    return Object.fromEntries(resolved.fields.filter(field => Object.hasOwn(record, field.name)).map(field =>
      [field.name, coerceCompleteJson(record[field.name], field.type, env, `${path}/${field.name}`)]));
  }
  if (resolved.kind === 'list') return (normalized as Value[]).map((item, index) =>
    coerceCompleteJson(item, resolved.element, env, `${path}/${index}`));
  if (resolved.kind === 'dict') return Object.fromEntries(Object.entries(normalized as Record<string, Value>).map(([key, item]) =>
    [key, coerceCompleteJson(item, resolved.element, env, `${path}/${key}`)]));
  return normalized;
}

type PlannedTypedText = { path: (string | number)[]; type: Type; text: string };
const JSON_STRING_TYPE: Type = { kind: 'prim', name: 'string' };

/** Build a temporary validation type that treats only plain text at declared Neuralese<string> leaves as strings. */
function typedTextPreflightType(value: unknown, type: Type, env: TypeEnv, path: (string | number)[], plans: PlannedTypedText[]): Type {
  const resolved = env.resolve(type), at = ['return', ...path].join('/');
  if (resolved.kind === 'neuralese' && env.resolve(resolved.element).kind === 'prim' &&
      (env.resolve(resolved.element) as Extract<Type, { kind: 'prim' }>).name === 'string') {
    try { coerce(value, resolved, env, at); return resolved; }
    catch (error) { if (!(error instanceof Reject)) throw error; }
    if (typeof value === 'string') {
      plans.push({ path: [...path], type: resolved, text: value });
      return JSON_STRING_TYPE;
    }
    return resolved;
  }
  if (resolved.kind === 'union') {
    // Preserve the existing first-fit branch whenever the original union already accepts the value.
    try { coerce(value, resolved, env, at); return resolved; }
    catch (error) { if (!(error instanceof Reject)) throw error; }
    for (const member of resolved.members) {
      const branchPlans: PlannedTypedText[] = [];
      const candidate = typedTextPreflightType(value, member, env, path, branchPlans);
      try { coerce(value, candidate, env, at); plans.push(...branchPlans); return candidate; }
      catch (error) { if (!(error instanceof Reject)) throw error; }
    }
    return resolved;
  }
  if (resolved.kind === 'record' && isPlainRecord(value)) return { ...resolved, fields: resolved.fields.map(field =>
    Object.hasOwn(value, field.name) ? { ...field, type: typedTextPreflightType(value[field.name], field.type, env,
      [...path, field.name], plans) } : field) };
  if (resolved.kind === 'list' && Array.isArray(value)) {
    const candidates = value.map((item, index) => typedTextPreflightType(item, resolved.element, env,
      [...path, index], plans));
    const unique = [...new Map(candidates.map(candidate => [formatType(candidate), candidate])).values()];
    return candidates.length ? { ...resolved, element: unique.length === 1 ? unique[0]! : { kind: 'union', members: unique } } : resolved;
  }
  if (resolved.kind === 'dict' && isPlainRecord(value)) {
    const candidates = Object.entries(value).map(([key, item]) => typedTextPreflightType(item, resolved.element, env,
      [...path, key], plans));
    const unique = [...new Map(candidates.map(candidate => [formatType(candidate), candidate])).values()];
    return candidates.length ? { ...resolved, element: unique.length === 1 ? unique[0]! : { kind: 'union', members: unique } } : resolved;
  }
  return resolved;
}

/** Replace only planned leaves in an already-coerced preflight value, retaining every other value by identity. */
function applyTypedTextRefs(value: Value, plans: PlannedTypedText[], refs: Value[]): Value {
  type Branch = { replacement?: Value; children: Map<string | number, Branch> };
  const root: Branch = { children: new Map() };
  plans.forEach((plan, index) => {
    let branch = root;
    for (const segment of plan.path) {
      let next = branch.children.get(segment);
      if (!next) { next = { children: new Map() }; branch.children.set(segment, next); }
      branch = next;
    }
    branch.replacement = refs[index]!;
  });
  const rewrite = (item: Value, branch: Branch): Value => {
    if (branch.replacement !== undefined) return branch.replacement;
    if (Array.isArray(item)) return item.map((child, index) => {
      const next = branch.children.get(index);
      return next ? rewrite(child, next) : child;
    });
    if (item && typeof item === 'object' && isPlainRecord(item)) return Object.fromEntries(Object.entries(item).map(([key, child]) => {
      const next = branch.children.get(key);
      return [key, next ? rewrite(child as Value, next) : child];
    }));
    return item;
  };
  return rewrite(value, root);
}

/** Stable JSON source for a validated value; preserve finite -0, which JSON.stringify would normalize to 0. */
function typedJsonSource(value: Value): string | undefined {
  const active = new Set<object>();
  const encode = (item: unknown): string => {
    if (item === null) return 'null';
    if (typeof item === 'string') return JSON.stringify(item);
    if (typeof item === 'boolean') return item ? 'true' : 'false';
    if (typeof item === 'number') {
      if (!Number.isFinite(item)) throw new TypeError('non-finite number is not JSON-representable');
      return Object.is(item, -0) ? '-0' : String(item);
    }
    if (Array.isArray(item)) {
      if (active.has(item)) throw new TypeError('cyclic JSON value');
      active.add(item);
      try { return `[${item.map(encode).join(',')}]`; } finally { active.delete(item); }
    }
    if (item && typeof item === 'object') {
      if (active.has(item)) throw new TypeError('cyclic JSON value');
      active.add(item);
      try {
        const record = item as Record<string, unknown>;
        return `{${Object.keys(record).sort().map(key => `${JSON.stringify(key)}:${encode(record[key])}`).join(',')}}`;
      } finally { active.delete(item); }
    }
    throw new TypeError('value is not JSON-representable');
  };
  try { return encode(value); } catch { return undefined; }
}

/** What the model is told when a value is staged as the call's result. */
const stagedMessage = (value: Value, liveIdentity?: (value: object) => number) => `\nStaged ${stagedText(value, liveIdentity)} as the result. If this is the result of the task you were given and ` +
  'you are satisfied with it, reply done to return exactly this value without a tool call, or call return_result with status "success" and omit value to finish using this exact stored result. You can keep working and return a different value later.';
/** A staged value, in full when it is small, so it can be checked (and never needs retyping); long ones are cut by structure. */
function stagedText(value: Value, liveIdentity?: (value: object) => number): string {
  return renderValue(value, { budget: 1500, liveIdentity });
}
/** A deep-frozen copy of portable data; live values and handles are kept by reference. */
function frozenCopy(value: Record<string, Value>): Record<string, unknown> {
  const copy = (item: unknown): unknown => Array.isArray(item) ? Object.freeze(item.map(copy)) :
    item && typeof item === 'object' && !isLive(item) && !isHandle(item) && Object.getPrototypeOf(item) === Object.prototype ?
      Object.freeze(Object.fromEntries(Object.entries(item).map(([key, child]) => [key, copy(child)]))) : item;
  return copy(value) as Record<string, unknown>;
}
const isHandle = (value: unknown) => value instanceof Folder || value instanceof FolderHandle || value instanceof FileHandle;
/** True when a value is, or contains, something that must be passed by reference. Iterative to avoid deep-scope recursion. */
function containsLive(value: unknown): boolean {
  const seen = new Set<object>();
  const children = function* (object: object): Generator<unknown> {
    if (Array.isArray(object)) { yield* object; return; }
    for (const key in object) if (Object.hasOwn(object, key)) yield (object as Record<string, unknown>)[key];
  };
  const stack: Generator<unknown>[] = [children({ root: value })];
  // The root wrapper lets primitives use the same iterator path without an extra branch.
  while (stack.length) {
    const next = stack.at(-1)!.next();
    if (next.done) { stack.pop(); continue; }
    const item = next.value;
    if (isLive(item) || isHandle(item)) return true;
    if (!item || typeof item !== 'object' || seen.has(item)) continue;
    seen.add(item); stack.push(children(item));
  }
  return false;
}
/** Split scope values into portable snapshot data and live references. */
function splitScope(values: Record<string, Value>): { portable: Record<string, unknown>; live: Record<string, unknown> } {
  const portable: Record<string, unknown> = {}, live: Record<string, unknown> = {};
  for (const [name, value] of Object.entries(values)) {
    if (value === MISSING || isPending(value)) continue;
    if (containsLive(value)) live[name] = value;
    else portable[name] = dump(value);
  }
  return { portable, live };
}
/** Natlang type text for a live value whose type was not declared. */
function liveTypeText(value: object): string {
  if (typeof value === 'function') return `Live<${JSON.stringify(`function ${(value as Function).name || ''}`.trim())}, "function", "">`;
  const tag = Object.prototype.toString.call(value).slice(8, -1), label = liveLabel(value);
  return tag !== 'Object' ? `Live<${JSON.stringify(tag)}, "tag", ${JSON.stringify(tag)}>` :
    `Live<${JSON.stringify(label)}, "class", ${JSON.stringify(label)}>`;
}
/** The scope type of a portable value, as a local holding it is declared. */
export function inferValueType(value: unknown): string {
  if (isLive(value)) return liveTypeText(value as object);
  if (value instanceof Folder || value instanceof FolderHandle) return 'Folder';
  if (value instanceof FileHandle) return 'FileHandle';
  if (value === null) return 'null';
  if (typeof value === 'boolean') return 'boolean';
  if (typeof value === 'number' && Number.isFinite(value)) return 'number';
  if (typeof value === 'string') return 'string';
  // A soft value carries its type in its reference.
  if (isNeuraleseRef(value)) return value.$neuralese.type;
  if (Array.isArray(value)) {
    // An empty list may still be filled with anything, and a mixed list holds the union of its item types.
    if (!value.length) return 'unknown[]';
    if (containsLive(value)) return 'Live<"array", "tag", "Array">';
    const types = [...new Set(value.map(item => inferValueType(item)))];
    return types.length === 1 ? `(${types[0]})[]` : `(${types.join(' | ')})[]`;
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value);
    if (!entries.length) return 'Record<string, unknown>';
    if (containsLive(value)) return 'Live<"object", "any", "">';
    // An object keyed by data (ids, emails, names) is a dictionary: Record<string, T>, so later evals can add keys.
    // Only a few identifier keys read as a record with fixed fields.
    if (entries.length > 12 || entries.some(([key]) => !/^[A-Za-z_$][\w$]*$/.test(key))) {
      const types = [...new Set(entries.map(([, item]) => inferValueType(item)))];
      return `Record<string, ${types.length === 1 ? types[0] : types.join(' | ')}>`;
    }
    return `{ ${entries.map(([key, item]) => item === undefined ? `${key}?: unknown` : `${key}: ${inferValueType(item)}`).join(', ')} }`;
  }
  throw new Reject([{ path: 'value', code: 'type-mismatch', expected: 'a portable value' }]);
}

const itemRef = (container: Record<string, Value>, key: string, type: Type, env: TypeEnv, path: string, deny = ''): Ref =>
  ({ path, type, env, deny,
    get: () => Object.hasOwn(container, key) ? container[key]! : MISSING,
    set: value => { container[key] = value; },
    del: () => { delete container[key]; } });

/** Find a codebase item by dotted path; module exports resolve to their module's source. */
export function findCodebaseItem(codebase: Record<string, unknown>, name: string):
  { path: string[]; record: ItemRecord } | undefined {
  const parts = name.trim().replace(/\//g, '.').split('.').filter(Boolean);
  const walk = (level: Record<string, ItemRecord>, index: number, trail: string[]): { path: string[]; record: ItemRecord } | undefined => {
    const item = level[parts[index]!];
    if (!item) return;
    const path = [...trail, parts[index]!];
    if (index === parts.length - 1) return { path, record: item };
    if (item.kind === 'module' && Object.hasOwn(item.exports, parts[index + 1]!) && index + 1 === parts.length - 1) return { path, record: item };
    return walk(item.codebase, index + 1, path);
  };
  const direct = parts.length ? walk(codebase as Record<string, ItemRecord>, 0, []) : undefined;
  if (direct || parts.length !== 1) return direct;
  const matches: { path: string[]; record: ItemRecord }[] = [];
  const search = (level: Record<string, ItemRecord>, trail: string[]) => {
    for (const [key, item] of Object.entries(level)) {
      if (key === parts[0]) matches.push({ path: [...trail, key], record: item });
      search(item.codebase, [...trail, key]);
    }
  };
  search(codebase as Record<string, ItemRecord>, []);
  return matches.length === 1 ? matches[0] : undefined;
}
function listCodebase(codebase: Record<string, unknown>, prefix = ''): string[] {
  return Object.entries(codebase as Record<string, ItemRecord>).flatMap(([name, item]) =>
    [`${prefix}${name}`, ...listCodebase(item.codebase, `${prefix}${name}.`)]);
}
function replaceCodebaseItem(codebase: Record<string, unknown>, path: string[], record: ItemRecord): Record<string, unknown> {
  const [head, ...rest] = path;
  const current = (codebase as Record<string, ItemRecord>)[head!]!;
  return { ...codebase, [head!]: rest.length ? { ...current, codebase: replaceCodebaseItem(current.codebase, rest, record) } : record };
}
/** Declared return type of a callable path in a codebase, for typing eval locals. */
function returnTypeOf(codebase: Record<string, unknown>, name: string): string | undefined {
  const found = findCodebaseItem(codebase, name);
  if (!found || found.path.join('.') !== name.replace(/\//g, '.') && found.record.kind !== 'module') return;
  const record = found.record;
  const expanded=(text:string)=>{
    const aliases:Record<string,Type>={};
    for(const [name,source]of Object.entries('types' in record ? record.types : {})){try{aliases[name]=parseType(source);}catch{/* TypeScript-only types stay outside portable checking. */}}
    const shared=callableTypes(codebase);
    for(const name of Object.keys(aliases))if(shared[name]&&formatType(shared[name]!)===formatType(aliases[name]!))delete aliases[name];
    return formatType(inlineDeclaredTypes(parseType(text),aliases));
  };
  if (record.kind === 'natlang') return expanded(record.returns);
  if (record.kind === 'module') {
    const exportName = name.split('.').at(-1)!;
    const spec = record.exports[exportName === record.name ? 'default' : exportName] ?? record.exports.default;
    return spec?.kind === 'function' ? expanded(spec.returns) : undefined;
  }
  return;
}

/** Settle with `work`, or reject with the signal's reason once it aborts. */
function untilAborted<T>(work: Promise<T>, signal: AbortSignal | undefined): Promise<T> {
  if (!signal) return work;
  if (signal.aborted) { work.catch(() => {}); return Promise.reject(signal.reason); }
  return new Promise<T>((resolve, reject) => {
    const stop = () => { work.catch(() => {}); reject(signal.reason); };
    signal.addEventListener('abort', stop, { once: true });
    work.then(value => { signal.removeEventListener('abort', stop); resolve(value); },
      error => { signal.removeEventListener('abort', stop); reject(error); });
  });
}

export class NativeRuntime {
  readonly events: HostEvent[] = [];
  readonly trace: NativeTraceRecorder;
  readonly options: { maxActions?: number; maxToolCalls?: number; runId: string; seedId?: string };
  readonly seedPolicy: { mode: 'derived' | 'backend'; root?: number };
  readonly environment: EvalEnvironment;
  readonly agent?: NativeAgent;
  readonly episodeBudget: { limit?: number; used: number };
  readonly frame?: Frame;
  readonly displayLiveId: (value: object) => number;
  readonly hooks: NativeRuntimeHooks;
  readonly services: Record<string, object>;
  readonly neuralese?: NeuraleseRuntimeOptions;
  readonly declarations: Record<string, string>;
  readonly serviceScopes: Record<string, string[]>;
  currentCallId?: string;
  private root?: LambdaNode;
  private lastObserved?: unknown;
  readonly signal?: AbortSignal;
  private readonly deadline?: number;
  private readonly exactHostTraceCapture?: NativeRuntimeOptions['exactHostTraceCapture'];

  constructor(options: NativeRuntimeOptions) {
    this.displayLiveId = options.frame ? scopedLiveIdentity(options.frame.task) : createLiveIdentity();
    this.options = { maxActions: options.maxActions, maxToolCalls: options.maxToolCalls, runId: options.runId ?? 'native-run', seedId: options.seedId };
    for (const [name, value] of Object.entries({ maxActions: this.options.maxActions, maxToolCalls: this.options.maxToolCalls }))
      if (value !== undefined && (!Number.isInteger(value) || value < 1)) throw new RangeError(`${name} must be a positive integer`);
    this.episodeBudget = options.sharedEpisodeBudget ?? { used: 0 };
    this.seedPolicy = options.seedPolicy??{mode:'derived',root:0};
    if(this.seedPolicy.mode==='derived'){this.seedPolicy={...this.seedPolicy,root:this.seedPolicy.root??0};if(!Number.isSafeInteger(this.seedPolicy.root))throw new RangeError('seed root must be a safe integer');}
    if (this.seedPolicy.mode === 'derived' && !Number.isInteger(this.seedPolicy.root))
      throw new TypeError('derived seed policy requires an integer root');
    this.trace = new NativeTraceRecorder({ run_id: this.options.runId, tool_schema: 'scope-eval-v2',
      ...(options.sourceRevision ? { source_revision: options.sourceRevision } : {}),
      ...(options.parentCallId ? { parent_call_id: options.parentCallId } : {}),
      environment: { mode: options.environment.mode, authority: options.environment.authority, native_state_replayable: false },
      seed_policy: this.seedPolicy, coverage: 'natlang-state-and-observed-host-effects', ...(options.manifest ?? {}) });
    this.exactHostTraceCapture = options.exactHostTraceCapture;
    const capture = this.exactHostTraceCapture;
    if (capture && (!Number.isSafeInteger(capture.maxBytes) || capture.maxBytes < 1 || capture.maxBytes > 8_000_000 ||
        !Array.isArray(capture.definitionSources) || (!capture.definitionSources.length && capture.captureAllOutputs !== true) ||
        (capture.captureAllOutputs !== undefined && typeof capture.captureAllOutputs !== 'boolean') ||
        capture.definitionSources.some(source => typeof source !== 'string' || !source.endsWith('.nl')) ||
        !Array.isArray(capture.inputArguments) || capture.inputArguments.some(name => typeof name !== 'string' || !name)))
      throw new RangeError('exact host trace capture requires source allowlist, argument names, and a bounded positive byte budget');
    // `offered` is emitted by NativeToolAgent only when it actually builds the opening shown to the model.
    this.frame = options.frame;
    this.hooks = options.hooks;
    // Every service call is recorded as an effect, so a failed eval can say what already happened. Services a caller
    // has already wrapped (the kernel records with its own call IDs) are used as given.
    const services = options.services ?? {};
    const observe = options.observeEffect;
    this.services = isRecording(services) ? services : recordingServices(services, ({ exact, ...event }) => (observe?.({ ...event, exact }), event.phase === 'requested') ?
      this.trace.emit('effect', { call_id: this.currentCallId ?? null, capability: `${event.service}.${event.method}`, ...event }) :
      graphNode(this.trace, 'effect', { call_id: this.currentCallId ?? null, capability: `${event.service}.${event.method}`, ...event },
        [{ node: invocationNodeId(this.options.runId), port: 'caller' }]));
    this.declarations = options.declarations ?? {};
    this.neuralese = options.neuralese;
    this.serviceScopes = options.serviceScopes ?? {};
    this.agent = options.agent;
    this.environment = options.environment;
    this.signal = options.signal;
    if (options.timeoutMs !== undefined) {
      if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 1) throw new RangeError('timeoutMs must be positive');
      this.deadline = Date.now() + options.timeoutMs;
    }
  }

  checkInterruption(): void {
    if (this.signal?.aborted) {
      const reason = this.signal.reason instanceof Error ? this.signal.reason.message : undefined;
      throw new Error(`natlang run ${reason ? `stopped: ${reason}` : 'aborted'}; external effects may have occurred`);
    }
    if (this.deadline !== undefined && Date.now() >= this.deadline)
      throw new Error('natlang run timed out; external effects may have occurred');
  }

  /** Lift only an ordinary string supplied directly to a declared Neuralese<string> parameter. */
  async materializeSoftStringArguments(definition: { params: { name: string; type: string }[]; types: Record<string, string> },
    inputs: readonly unknown[], parentCallId?: string): Promise<unknown[]> {
    const aliases: Record<string, Type> = {};
    for (const [name, source] of Object.entries(definition.types)) {
      try { aliases[name] = parseType(source); } catch { /* Keep TypeScript-only aliases outside portable checks. */ }
    }
    const env = new TypeEnv(aliases);
    const normalized = [...inputs];
    for (let index = 0; index < Math.min(definition.params.length, normalized.length); index++) {
      const input = normalized[index];
      if (typeof input !== 'string') continue;
      const parameter = definition.params[index]!;
      let wanted: Type;
      try { wanted = env.resolve(parseType(parameter.type)); }
      catch { continue; }
      const element = wanted.kind === 'neuralese' ? env.resolve(wanted.element) : undefined;
      if (wanted.kind !== 'neuralese' || element?.kind !== 'prim' || element.name !== 'string') continue;

      const callerInput = { node: parentCallId ? invocationNodeId(parentCallId) : 'external', port: `argument:${parameter.name}` };
      normalized[index] = await this.writeNeuraleseText(input, wanted, {
        sourceKind: 'typed-text-argument', source: 'call-argument', markerContext: 'typed-argument',
        argumentName: parameter.name, inputs: [callerInput],
      });
    }
    return normalized;
  }

  /** Shared, graph-recorded text block writer for typed result and typed argument boundaries. */
  async writeNeuraleseText(text: string, wanted: Type, metadata: {
    sourceKind: 'typed-text-result' | 'typed-json-result' | 'typed-text-result-field' | 'typed-text-argument';
    source: string;
    markerContext: string;
    inputs: readonly { node: string; port?: string }[];
    resultPath?: (string | number)[];
    argumentName?: string;
  }): Promise<Value> {
    const port = this.neuralese?.port;
    const type = formatType(wanted), bodySha = hexDigest(text);
    if (!port) throw new NeuraleseUnsupportedError(`${metadata.sourceKind === 'typed-json-result' ? 'plain JSON' : 'plain-text'} ` +
      `${type} needs a configured Neuralese write port`);
    const callId = this.currentCallId ?? this.options.runId;
    const producer = { source_kind: metadata.sourceKind, source: metadata.source, marker_context: metadata.markerContext,
      call_id: callId, ...(metadata.argumentName ? { argument_type: type } : { result_type: type }), text_body_sha256: bodySha,
      ...(metadata.resultPath ? { result_path: metadata.resultPath } : {}),
      ...(metadata.argumentName ? { argument_name: metadata.argumentName } : {}) };
    const block = await port.write(text, { type, producer });
    const sourceTurn = metadata.inputs.find(input => /#turn\d+$/.test(input.node))?.node ?? '';
    graphNode(this.trace, 'block_write', { call_id: callId, block: block.id, length: block.length, turn: sourceTurn, stops: [],
      truncated: !!block.truncated, producer: block.producer?.kind ?? null, source_kind: producer.source_kind,
      source: metadata.source, marker_context: metadata.markerContext,
      ...(metadata.argumentName ? { argument_type: type } : { result_type: type }), text_body_sha256: bodySha,
      ...(producer.result_path ? { result_path: producer.result_path } : {}),
      ...(producer.argument_name ? { argument_name: producer.argument_name } : {}) }, metadata.inputs);
    return neuraleseRef(type, block.id);
  }

  async evaluate(node: LambdaNode, code: string, scope: Record<string, unknown>, live: Record<string, unknown>, timeoutMs?: number,
    signal?: AbortSignal) {
    this.checkInterruption();
    try {
      const request = { code, body: true, path: 'eval', scope, live, ...(timeoutMs === undefined ? {} : { timeoutMs }) };
      // Calls the eval starts derive their signal from the eval's, so stopping the eval stops what it started.
      const frame = this.frame && signal ? { ...this.frame, signal } : this.frame;
      // A stopped call stops waiting for its eval at once; the eval's timers are cleared when the call ends.
      const result = await untilAborted(frame ? runInFrame(frame, () => this.environment.executeAsync(request)) :
        this.environment.executeAsync(request), this.signal);
      this.recordHostEvents(result.events);
      this.checkInterruption();
      return result;
    } catch (error) {
      if (error instanceof EvalFailure) this.recordHostEvents(error.events);
      throw error;
    }
    void node;
  }

  private recordHostEvents(events: HostEvent[]): void {
    this.events.push(...events);
    for (const event of events) if (event.operation !== 'typescript.eval') this.trace.emit('host', { event });
  }

  /**
   * The graph fields of an invocation's start event (spec/NEURALESE_GRAPH.md): definition and revision, context,
   * signature, capture bindings, and argument references (soft arguments by block and producer).
   */
  private invocationNode(node: LambdaNode, callId: string): Record<string, unknown> {
    const manifest = (this.trace.events[0] ?? {}) as Record<string, unknown>;
    const parent = typeof manifest.parent_call_id === "string" ? manifest.parent_call_id : undefined;
    const inputs = [...(parent ? [{ node: invocationNodeId(parent), port: 'caller' }] : []),
      ...Object.entries(node.args).flatMap(([name, value]) => valueInputs(value, `arg:${name}`)),
      ...Object.values(node.captures ?? {}).flatMap(cell => { try { return valueInputs(cell.get(), `capture:${cell.name}`); } catch { return []; } })];
    return { node: invocationNodeId(callId), inputs,
      definition: { id: String(manifest.definition_id ?? node.functionName), revision: String(manifest.source_revision ?? '') },
      context: String(manifest.context_id ?? FILE_CONTEXT), signature: formatType(node.type),
      captures: Object.fromEntries(Object.values(node.captures ?? {}).map(cell => [cell.name,
        { mode: cell.mutable ? 'live' : 'snapshot', type: cell.type, ...(cell.skill ? { skill: cell.skill } : {}) }])) };
  }

  /** Run one invocation to completion or quiescence. */
  async run(node: LambdaNode): Promise<{ outcome: NativeOutcome; value: Value }> {
    this.checkInterruption();
    this.root = node;
    const before = this.stateSummary(node);
    this.trace.emit('state', { phase: 'initial', value: before });
    this.captureExactInputs(node);
    this.lastObserved = before;
    const env = new TypeEnv(callableTypes(node.codebase)).child(node.types);
    env.classes = node.hostClasses;
    const outcome = await this.episode(node, env);
    const source = this.trace.events[0]?.definition_source;
    if (outcome.kind === 'done' && this.exactHostTraceCapture?.captureOutput &&
        (this.exactHostTraceCapture.captureAllOutputs === true ||
         (typeof source === 'string' && this.exactHostTraceCapture.definitionSources.includes(source))))
      this.captureExactValue('invocation_output', 'return', node.return, typeof source === 'string' ? source : undefined);
    this.observeState('final', outcome.kind);
    return { outcome, value: outcome.kind === 'done' ? node.return : MISSING };
  }

  private captureExactInputs(node: LambdaNode): void {
    const capture = this.exactHostTraceCapture;
    const source = this.trace.events[0]?.definition_source;
    if (!capture || typeof source !== 'string' || !capture.definitionSources.includes(source)) return;
    for (const name of capture.inputArguments) {
      if (!Object.hasOwn(node.args, name)) {
        this.trace.emit('host_capture', { capture_kind: 'invocation_input', call_id: this.options.runId,
          parent_call_id: this.trace.events[0]?.parent_call_id ?? null, definition_source: source,
          name, complete: false, reason: 'missing' });
        continue;
      }
      this.captureExactValue('invocation_input', name, node.args[name], source);
    }
  }

  private captureExactValue(captureKind: 'invocation_input' | 'invocation_output', name: string,
      value: unknown, source?: string): void {
    const capture = this.exactHostTraceCapture;
    if (!capture) return;
    const snapshot = exactPortableSnapshot(value, capture.maxBytes);
    this.trace.emit('host_capture', { capture_kind: captureKind, call_id: this.options.runId,
      parent_call_id: this.trace.events[0]?.parent_call_id ?? null,
      definition_source: source ?? this.trace.events[0]?.definition_source ?? null,
      name, ...snapshot,
      ...(captureKind === 'invocation_output' ? {
        result_type: this.root?.type.kind === 'lambda' ? formatType(this.root.type.returns) : null,
        terminal_action_seq: this.trace.events.filter(event => event.kind === 'action').at(-1)?.seq ?? null,
        origin: 'observed-host-result; not a model-generated writer target' } : {}),
      ...(snapshot.complete ? { value_sha256: hexDigest(JSON.stringify(snapshot.value)) } : {}) });
  }

  observeState(phase: string, outcome?: string): void {
    if (!this.root) return;
    const value = this.stateSummary(this.root);
    const delta = changes(this.lastObserved, value);
    if (delta.length) this.trace.emit('reduction', { phase, changes: delta });
    this.trace.emit('state', { phase, value, ...(outcome ? { outcome } : {}) });
    this.lastObserved = value;
  }

  private stateSummary(node: LambdaNode): Record<string, unknown> {
    // State events are durable diagnostics, not the execution store. Keep useful names and
    // bounded previews while leaving all live bindings untouched in the LambdaNode.
    const preview = (values: Record<string, Value>) => Object.fromEntries(Object.entries(values).slice(0, 128)
      .map(([name, item]) => [name, diagnosticValue(item, name, this.displayLiveId)]));
    return { $lambda: { type: formatType(node.type), instructions: node.body.slice(0, 8000),
      status: node.status, attempts: node.attempts, subtype: node.subtype,
      args: preview(node.args), let: preview(node.let),
      ...(node.return !== MISSING ? { return: diagnosticValue(node.return, 'result', this.displayLiveId) } : {}) } };
  }

  private quiesce(node: LambdaNode, detail: string): NativeOutcome {
    node.status = 'quiesced'; node.note = detail;
    this.trace.emit('node', { transition: 'quiesced', detail });
    if (node.projectTransaction?.open) {
      node.projectTransaction.abort();
      this.trace.emit('folder', { call_id: this.currentCallId ?? null, phase: 'discarded', mode: node.reducerMode, reason: detail });
    }
    for (const transaction of node.extraTransactions ?? []) if (transaction.open) transaction.abort();
    return { kind: 'quiesced', detail };
  }

  private async episode(node: LambdaNode, env: TypeEnv): Promise<NativeOutcome> {
    const unbound = unboundParts(node, env, '');
    if (unbound.length) return this.quiesce(node,
      `unbound: ${unbound.map(d => `${d.path} (expected ${d.expected})`).join(', ')}. ` +
      'Required argument values are missing or undefined. Inspect the passed values and bind every required parameter before retrying.');
    if (this.episodeBudget.limit !== undefined && this.episodeBudget.used >= this.episodeBudget.limit)
      return this.quiesce(node, `run budget: more than ${this.episodeBudget.limit} episodes`);
    if (!this.agent) return this.quiesce(node, 'no model or agent driver supplied');
    node.status = 'running'; node.note = ''; node.attempts++;
    node.originalBody ??= node.body;
    this.episodeBudget.used++;
    const callId = this.options.runId;
    this.currentCallId = callId;
    this.trace.emit('invocation', { phase: 'start', call_id: callId, attempt: node.attempts,
      ...(node.attempts === 1 ? this.invocationNode(node, callId) : {}) });
    const session = new NativeSession(this, node, env);
    let note: string | void;
    try {
      note = await this.agent(session);
      this.checkInterruption();
    } catch (error) {
      this.quiesce(node, `interpreter exception: ${error instanceof Error ? error.message : String(error)}`);
      throw error;
    } finally { this.trace.emit('invocation', { phase: 'end', call_id: callId }); }
    // The model ending its turn is the completion signal: the runtime requires a returned value of the
    // declared type, then commits any directory-reducer transaction.
    // An agent that stopped with a reason (a budget, a blocker) does not return a merely staged value.
    if (session.completed || (!note && session.finish())) {
      node.status = 'done';
      this.trace.emit('node', { transition: 'done' });
      return { kind: 'done', detail: oneLine(dump(node.return), undefined, this.displayLiveId), value: node.return };
    }
    return this.quiesce(node, String(note || 'budget exhausted'));
  }
}

export type ScopeFailureDebug = {
  version: 'natlang.scope_failure/1'; serial: number; kind: 'compile' | 'runtime' | 'boundary';
  message: string; code: string; scope: Record<string, unknown>; trace: Record<string, unknown>[];
  diagnostics: Record<string, unknown>[]; logs: string[]; stack?: string;
};

/** Longest note compact_history accepts. */
export const COMPACTION_NOTE_CHARS = 2000;

/**
 * One earlier tool call of this call: the reasoning that led to it, its arguments, its outcome (`status`, the result kind: ok, rejected, error,
 * completed, ...), for eval the returned value as data (`value`, when it is portable data of modest size) and what it
 * printed (`console`), and the full text the model was shown (`output`, nothing cut off).
 */
export type TranscriptEntry = { turn: number; tool: string; code?: string; arguments: Record<string, unknown>;
  status: string; value?: unknown; console?: string; output: string;
  /** The model's reasoning in the turn that made this call (on the turn's first call only). */
  reasoning?: string };

/** Largest returned value, as JSON characters, a transcript entry keeps as data. */
const TRANSCRIPT_VALUE_CHARS = 100_000;
/** A bounded estimate used only to decide whether a value is small enough for diagnostic transcript storage. */
function portableSizeAtMost(value: unknown, limit: number): boolean {
  let remaining = limit, nodes = 0;
  const seen = new Set<object>();
  const visit = (item: unknown, depth: number): boolean => {
    if (++nodes > 4096 || depth > 32) return false;
    if (typeof item === 'string') { remaining -= item.length + 2; return remaining >= 0; }
    if (item === null || typeof item !== 'object') { remaining -= 16; return remaining >= 0; }
    if (seen.has(item)) return false;
    seen.add(item);
    if (Array.isArray(item)) {
      remaining -= 2;
      if (remaining < 0) return false;
      for (const child of item) if (!visit(child, depth + 1)) return false;
      return true;
    }
    for (const key in item) if (Object.hasOwn(item, key)) {
      remaining -= key.length + 4;
      if (remaining < 0 || !visit((item as Record<string, unknown>)[key], depth + 1)) return false;
    }
    return true;
  };
  return visit(value, 0);
}
function exactPortableSnapshot(value: unknown, maxBytes: number): { complete: true; value: unknown; bytes: number } |
  { complete: false; reason: 'nonportable' | 'oversize' } {
  const jsonSafe = (item: unknown, seen = new Set<object>(), depth = 0): boolean => {
    if (depth > 32) return false;
    if (item === null || typeof item === 'string' || typeof item === 'boolean') return true;
    if (typeof item === 'number') return Number.isFinite(item);
    if (typeof item !== 'object' || seen.has(item)) return false;
    if (Array.isArray(item)) {
      seen.add(item); const valid = item.every(child => jsonSafe(child, seen, depth + 1)); seen.delete(item); return valid;
    }
    if (Object.getPrototypeOf(item) !== Object.prototype) return false;
    seen.add(item);
    const valid = Object.entries(item).every(([key, child]) => !['$host', '$live', '$lambda'].includes(key) &&
      child !== undefined && jsonSafe(child, seen, depth + 1));
    seen.delete(item); return valid;
  };
  if (containsLive(value)) return { complete: false, reason: 'nonportable' };
  try {
    // Capture is deliberately opt-in: declared rewrite I/O or collector invocation outputs.
    // The snapshot is validated as portable JSON above; the cast supplies dump's runtime value type.
    const snapshot = dump(value as Value);
    if (!jsonSafe(snapshot)) return { complete: false, reason: 'nonportable' };
    const text = JSON.stringify(snapshot);
    if (typeof text !== 'string') return { complete: false, reason: 'nonportable' };
    const bytes = new TextEncoder().encode(text).byteLength;
    if (bytes > maxBytes) return { complete: false, reason: 'oversize' };
    return { complete: true, value: JSON.parse(text), bytes };
  } catch { return { complete: false, reason: 'nonportable' }; }
}
/** Bounded iterative equality for diagnostic "Stored local" notices; never serializes the values. */
function sameValueWithin(left: unknown, right: unknown, maxNodes = 4096): boolean | undefined {
  const pending: [unknown, unknown][] = [[left, right]], paired = new Map<object, WeakSet<object>>();
  let visited = 0;
  while (pending.length) {
    if (++visited > maxNodes) return undefined;
    const [a, b] = pending.pop()!;
    if (Object.is(a, b)) continue;
    if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
    let rights = paired.get(a);
    if (rights?.has(b)) continue;
    if (!rights) { rights = new WeakSet(); paired.set(a, rights); }
    rights.add(b);
    if (Array.isArray(a) !== Array.isArray(b)) return false;
    if (Array.isArray(a)) {
      if (a.length !== (b as unknown[]).length) return false;
      if (a.length > maxNodes - visited) return undefined;
      for (let i = 0; i < a.length; i++) pending.push([a[i], (b as unknown[])[i]]);
      continue;
    }
    const collect = (value: object): string[] | undefined => {
      const keys: string[] = [];
      for (const key in value) if (Object.hasOwn(value, key)) {
        if (keys.length >= maxNodes - visited) return undefined;
        keys.push(key);
      }
      return keys;
    };
    const ak = collect(a), bk = collect(b);
    if (!ak || !bk) return undefined;
    if (ak.length !== bk.length) return false;
    for (let i = 0; i < ak.length; i++) {
      if (ak[i] !== bk[i]) return false;
      pending.push([(a as Record<string, unknown>)[ak[i]!], (b as Record<string, unknown>)[bk[i]!]]);
    }
  }
  return true;
}
const deepFreeze = <T,>(value: T): T => {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value)) deepFreeze(child);
  }
  return value;
};

/**
 * What evals see as `transcript`: the call's history, searched rather than read through. There is no array access
 * or iteration, and printing it shows a summary; `search` finds lines, `entry(n)` gives one call in full.
 */
export class TranscriptView {
  readonly #entries: readonly TranscriptEntry[];
  constructor(entries: readonly TranscriptEntry[]) { this.#entries = entries; }
  get length(): number { return this.#entries.length; }
  /**
   * Lines of earlier reasoning, code, and outputs that match: a case-insensitive text, or a regular expression (an empty text
   * matches every line). Each match is { entry, turn, tool, in, line }, the line cut to 200 characters around the
   * match; at most `limit` (default 20). `status` and `tool` restrict the calls searched, e.g. { status: 'error' }.
   */
  search(query: string | RegExp, options: { in?: 'output' | 'code' | 'reasoning' | 'all'; limit?: number; status?: string; tool?: string } = {}) {
    const where = options.in ?? 'all', limit = Math.max(1, Math.min(options.limit ?? 20, 100));
    const test = (line: string): number => {
      if (typeof query === 'string') return line.toLowerCase().indexOf(query.toLowerCase());
      const found = new RegExp(query.source, query.flags.replace('g', '')).exec(line);
      return found ? found.index : -1;
    };
    const matches: { entry: number; turn: number; tool: string; in: 'output' | 'code' | 'reasoning'; line: string }[] = [];
    for (const [index, item] of this.#entries.entries()) {
      if ((options.status !== undefined && item.status !== options.status) || (options.tool !== undefined && item.tool !== options.tool)) continue;
      for (const field of ['reasoning', 'code', 'output'] as const) {
        if (where !== 'all' && where !== field) continue;
        for (const line of (item[field] ?? '').split('\n')) {
          const at = test(line);
          if (at < 0) continue;
          const start = Math.max(0, at - 80);
          matches.push({ entry: index, turn: item.turn, tool: item.tool, in: field,
            line: (start ? '…' : '') + line.slice(start, start + 200) + (line.length > start + 200 ? '…' : '') });
          if (matches.length >= limit) return Object.freeze(matches);
        }
      }
    }
    return Object.freeze(matches);
  }
  /** One earlier call in full; negative numbers count from the end (entry(-1) is the latest). */
  entry(n: number): TranscriptEntry {
    const index = n < 0 ? this.#entries.length + n : n;
    const item = this.#entries[index];
    if (!Number.isInteger(n) || !item) throw new RangeError(`transcript has entries 0 to ${this.#entries.length - 1}`);
    return Object.freeze({ ...item, arguments: deepFreeze(structuredClone(item.arguments)) });
  }
  toString(): string { return `transcript: ${this.#entries.length} earlier calls; use transcript.search(query) or transcript.entry(n)`; }
  toJSON(): string { return this.toString(); }
}

export class NativeSession {
  completed = false;
  actions = 0;
  toolCalls = 0;
  readonly surfaceName = 'scope-eval-v2';
  failureSerial = 0;
  failureDebug?: ScopeFailureDebug;
  private readonly scopeLocalMutability = new Map<string, boolean>();
  private readonly explainedNaturalFunctions = new Set<string>();
  private readonly originalSources = new Map<string, string>();
  private readonly skillReads = new Map<string, { document: string; count: number }>();
  private callableCache?: { codebase: Record<string, unknown>; tree: Record<string, unknown> };
  /** Output cut off in this call's tool results, readable with read_page. */
  readonly pages = new PageStore();
  /** This call's tool calls with their full outputs, in order; evals see it through a TranscriptView. */
  readonly transcript: TranscriptEntry[] = [];
  /** The model turn the next recorded call belongs to (set by the agent). */
  turn = 0;
  /** The model's reasoning in that turn, recorded with the turn's first call (set by the agent). */
  turnReasoning?: string;
  /** What the eval being recorded returned and printed, for its transcript entry. */
  private evalDetail?: { value?: unknown; console?: string };
  /** Texts the current call's result shows cut off, with their full versions for its transcript entry. */
  private cuts: { shown: string; full: string }[] = [];
  /** Types this call's evals declared (`type X = …`, `interface X { … }`), by name. */
  private localTypes: Record<string, Type> = {};
  /** Authenticated source for top-level helpers from successful evals; never stores closures or host handles. */
  private readonly persistentScopeHelpers = new Map<string, PersistentScopeHelper>();
  /** Exact native tool schemas offered by NativeToolAgent for this call's current turn. */
  private readonly offeredToolDocs = new Map<string, { description: string; parameters: unknown }>();
  private activeScopeLocals?: Map<string, [() => unknown, ((value: unknown) => void)?]>;
  constructor(readonly runtime: NativeRuntime, readonly lam: LambdaNode, readonly env: TypeEnv) {}

  rememberOfferedTools(tools: readonly unknown[]): void {
    this.offeredToolDocs.clear();
    for (const candidate of tools) {
      if (!isPlainRecord(candidate) || !isPlainRecord(candidate.function)) continue;
      const fn = candidate.function;
      if (typeof fn.name !== 'string' || typeof fn.description !== 'string' || !Object.hasOwn(fn, 'parameters')) continue;
      this.offeredToolDocs.set(fn.name, { description: fn.description, parameters: fn.parameters });
    }
  }

  private toolDocumentation(name: string): string | undefined {
    const tool = this.offeredToolDocs.get(name);
    if (!tool) return;
    return `Native tool definition (not program function source): ${name}\n${tool.description}\n\nArgument schema (JSON Schema):\n` +
      `${JSON.stringify(tool.parameters, null, 2)}\n\nInvoke this as a separate tool action; it is not callable from eval.`;
  }

  /** Whether the model declared this persistent local with let (true) or const. */
  localMutable(name: string): boolean { return this.scopeLocalMutability.get(name) ?? true; }

  /** Type aliases from prior evals, made visible to the next eval's inline signature analysis. */
  analysisTypeAliases(): Record<string, string> {
    return Object.fromEntries(Object.entries(this.localTypes).map(([name, type]) => [name, formatType(type)]));
  }

  private captureScopeFailure(kind: ScopeFailureDebug['kind'], code: string, scope: Record<string, unknown>,
    message: string, diagnostics: Record<string, unknown>[] = [], error?: unknown, traceMark?: number): string {
    const trace = this.runtime.trace.events.filter(event =>
      ['action', 'eval', 'effect', 'host', 'node', 'invocation', 'folder'].includes(event.kind)).slice(-24)
      .map(event => {
        const fields: [string, unknown][] = [];
        for (const key of ['seq', 'kind', 'phase', 'transition', 'operation', 'capability', 'status', 'outcome',
          'error', 'result_text', 'call_id']) {
          const value = event[key];
          if (typeof value === 'string') fields.push([key, value.slice(0, 1200)]);
          else if (typeof value === 'number' || typeof value === 'boolean') fields.push([key, value]);
        }
        return Object.fromEntries(fields);
      });
    const sourceStack = error instanceof EvalFailure ? error.debug.sourceStack : error instanceof Error ? error.stack : undefined;
    const logs = error instanceof EvalFailure ? error.debug.logs ?? [] : [];
    this.failureDebug = { version: 'natlang.scope_failure/1', serial: ++this.failureSerial,
      kind, message, code: code.slice(0, 16000), scope, trace, diagnostics,
      logs: logs.slice(0, 32).map(line => line.slice(0, 2000)),
      ...(sourceStack ? { stack: sourceStack.split('\n').slice(0, 12).join('\n').slice(0, 4000) } : {}) };
    this.runtime.trace.emit('scope_failure', { failure_kind: kind, serial: this.failureSerial,
      message: message.slice(0, 1200), diagnostic_count: diagnostics.length });
    // Report what the model needs to repair the eval: its console output and any effect that already happened.
    const effectEvents = traceMark === undefined ? [] : this.runtime.trace.events.slice(traceMark)
      .filter(event => event.kind === 'effect' || event.kind === 'host');
    const effects = effectEvents
      .map(event => String(event.capability ?? event.operation ?? event.kind));
    const receipts = effectEvents.filter(event => event.kind === 'effect' && event.phase === 'completed').slice(-8)
      .map(event => `${String(event.capability ?? event.operation ?? 'service call')} completed: ${this.show(typeof event.result === 'string' ? event.result : JSON.stringify(event.result) ?? 'null')}`);
    this.evalDetail = { console: logs.join('\n') };
    // A model reaching for something the scope does not have (a name, a Node module, eval) is shown what it does have,
    // or it goes on probing the environment.
    const reaching = /is not defined|unavailable in eval|is not a function|Cannot find name/.test(message +
      diagnostics.map(item => String(item.message ?? '')).join('\n'));
    // A tool named in eval code: the tools are called as tools, next to eval, not from inside it.
    const tool = /\b(\w+) is not defined/.exec(message)?.[1];
    const toolNote = tool && NATIVE_TOOLS.includes(tool) && tool !== 'return_result' ?
      `\n${tool} is one of your tools: call it as a tool, not from eval code.` : '';
    return toolNote + (reaching ? `\n${this.scopeGuide()}` : '') + (logs.length ? `\nconsole:\n${this.show(logs.join('\n'))}` : '') +
      (effects.length ? `\nAlready performed before the failure (not undone): ${[...new Set(effects)].join(', ')}.` : '') +
      (receipts.length ? `\nCompleted service calls succeeded:\n${receipts.join('\n')}\nRepair subsequent work without repeating an already completed write.` : '') +
      '\nNothing else from this eval was kept. This refers to new eval bindings and the staged result; ' +
      'state changes made through live values, callable modules, files or external services are not undone. ' +
      'Inspect the current state before retrying a write.';
  }

  /**
   * A final reply's text as the result, for a call whose return type accepts that text (a string, or one of
   * a union's string members). A bare "done" is never a result: it asks for the staged value.
   */
  acceptTextResult(text: string): boolean {
    const answer = text.trim();
    if (this.lam.type.kind !== 'lambda' || this.lam.return !== MISSING || !answer || /^done[.!]?$/i.test(answer)) return false;
    // Tool-call markup is a failed tool call, never the answer.
    if (/<tool_call>|<\/tool_call>|<function=|<\/function>|<parameter=|<\/parameter>/.test(answer)) return false;
    let value: Value;
    try { value = coerce(answer, this.lam.type.returns, this.env, 'return'); } catch { value = MISSING; }
    if (typeof value === 'string') { this.lam.return = value; return true; }
    // Any other result may be written as JSON: a reply that is nothing but a value of the declared type (optionally
    // in a code fence) is that value. Prose around it is a reply, not a result.
    const body = answer.replace(/^```(?:json)?\s*\n([\s\S]*?)\n?```$/, '$1').trim();
    let parsed: unknown;
    try { parsed = JSON.parse(body); } catch { return false; }
    try { value = coerce(parsed as Value, this.lam.type.returns, this.env, 'return'); } catch { return false; }
    this.lam.return = value;
    return true;
  }

  /** Complete the invocation if an eval has returned a value of the declared type. */
  finish(): boolean {
    if (this.lam.return === MISSING || this.lam.type.kind !== 'lambda') return false;
    if (problems(this.lam.return, this.lam.type.returns, this.env, 'return').holes.length) return false;
    const tx = this.lam.projectTransaction;
    if (tx?.open) {
      if (this.lam.reducerMode === 'apply') {
        tx.validateSync();
        for (const transaction of this.lam.extraTransactions ?? []) transaction.validateSync();
      }
      const delta = tx.folder.diffSync();
      const selected = this.lam.reducerMode === 'apply' ? tx.commitSync() : (tx.abort(), delta);
      for (const transaction of this.lam.extraTransactions ?? []) {
        if (this.lam.reducerMode === 'apply') transaction.commitSync(); else transaction.abort();
      }
      this.runtime.trace.emit('folder', { call_id: this.runtime.currentCallId ?? null,
        phase: this.lam.reducerMode === 'apply' ? 'installed' : 'discarded', mode: this.lam.reducerMode,
        changes: selected.changes.map(change => ({ path: change.path, kind: change.kind })) });
    }
    this.completed = true;
    return true;
  }

  /** A text of the current call's result as it fits in a message; all of it goes to the call's transcript entry. */
  private show(text: string): string {
    const shown = this.pages.show(text, `transcript.entry(${this.transcript.length}).output`);
    if (shown !== text) this.cuts.push({ shown, full: text });
    return shown;
  }
  /** A value of the current call's result, cut by structure; all of it goes to the call's transcript entry. */
  private showValue(value: unknown): string {
    const root = this.lam.projectTransaction?.folder;
    const holder = `transcript.entry(${this.transcript.length}).output`;
    const shown = renderValue(value, { root, holder, liveIdentity: this.runtime.displayLiveId });
    // Preserve the existing page route for modest portable values, proven small by a bounded
    // estimator. Large/cyclic values remain intact in eval state but are not fully serialized.
    if (!portableSizeAtMost(value, TRANSCRIPT_VALUE_CHARS) || containsLive(value)) return shown;
    const full = renderValue(value, { root, budget: Infinity, liveIdentity: this.runtime.displayLiveId });
    if (shown === full) return shown;
    const { id, count } = this.pages.add(full);
    const paged = shown + `\n<<full value: ${count} pages; read_page("${id}", 1) shows the first page>>`;
    this.cuts.push({ shown: paged, full });
    return paged;
  }
  private record(name: string, args: Record<string, unknown>, result: NativeResult, toolCallId?: string): NativeResult {
    const capturedArgs = Object.fromEntries(Object.entries(args).map(([key, value]) =>
      [key, diagnosticArgument(value, `action.${name}.${key}`, this.runtime.displayLiveId)]));
    this.runtime.trace.emit('action', { call_id: this.runtime.currentCallId ?? null, surface: this.surfaceName, name, ...(toolCallId ? { tool_call_id: toolCallId } : {}),
      arguments: capturedArgs, outcome: result.kind, result_text: result.text, diagnostics: result.codes ?? [] });
    const output = this.cuts.reduce((text, cut) => text.replace(cut.shown, cut.full), result.text);
    this.cuts = [];
    const entry = this.transcript.length, detail = name === 'eval' ? this.evalDetail : undefined;
    this.evalDetail = undefined;
    // The returned value is kept as data when it is portable and of modest size; its text is always in output.
    let value: unknown;
    if (detail && Object.hasOwn(detail, 'value') && portableSizeAtMost(detail.value, TRANSCRIPT_VALUE_CHARS) &&
        !containsLive(detail.value) && !isHandle(detail.value)) {
      try {
        value = deepFreeze(structuredClone(detail.value));
      } catch { /* not portable after all */ }
    }
    const reasoning = this.turnReasoning;
    this.turnReasoning = undefined;
    this.transcript.push({ turn: this.turn, ...(reasoning ? { reasoning } : {}), tool: name, ...(name === 'eval' && typeof args.code === 'string' ? { code: args.code } : {}),
      arguments: structuredClone(capturedArgs), status: result.kind, ...(value !== undefined ? { value } : {}),
      ...(detail?.console ? { console: detail.console } : {}), output });
    this.runtime.observeState('after-action');
    return { ...result, entry };
  }
  private actionLimitReached(): boolean {
    return (this.runtime.options.maxActions !== undefined && this.actions >= this.runtime.options.maxActions) ||
      (this.runtime.options.maxToolCalls !== undefined && this.toolCalls >= this.runtime.options.maxToolCalls);
  }

  /** The session's callable tree, rebuilt when a codebase edit publishes a new revision. */
  callables(): Record<string, unknown> {
    if (this.callableCache?.codebase !== this.lam.codebase)
      this.callableCache = { codebase: this.lam.codebase, tree: this.runtime.hooks.callables(this.lam.codebase, this) };
    return this.callableCache.tree;
  }

  private async writeNeuraleseResult(text: string, wanted: Type, source: 'return_result' | 'eval-return' | 'eval-finish',
    sourceKind: 'typed-text-result' | 'typed-json-result' | 'typed-text-result-field', resultPath?: (string | number)[]): Promise<Value> {
    const callId = this.runtime.currentCallId ?? this.runtime.options.runId;
    const sourceEvent = [...this.runtime.trace.events].reverse().find(event =>
      event.kind === 'model_turn' && event.call_id === callId && typeof event.node === 'string');
    const input = sourceEvent ? { node: sourceEvent.node as string, port: 'result-source' } :
      { node: invocationNodeId(this.runtime.options.runId), port: 'typed-result-source' };
    const ref = await this.runtime.writeNeuraleseText(text, wanted, { sourceKind, source, markerContext: 'return-result',
      inputs: [input], ...(resultPath ? { resultPath: ['return', ...resultPath] } : {}) });
    return coerce(ref, wanted, this.env, 'return');
  }

  /** Materialize plain text or validated JSON only when the declared Neuralese element type makes that representation exact. */
  private async coerceReturn(value: unknown, source: 'return_result' | 'eval-return' | 'eval-finish',
    parseJsonFallback = false): Promise<Value> {
    const wanted = this.env.resolve(this.lam.type.kind === 'lambda' ? this.lam.type.returns : parseType('null'));
    const element = wanted.kind === 'neuralese' ? this.env.resolve(wanted.element) : undefined;
    if (typeof value === 'string' && wanted.kind === 'neuralese' && element?.kind === 'prim' && element.name === 'string') {
      try { return coerce(value, wanted, this.env, 'return'); }
      catch (error) { if (!(error instanceof Reject)) throw error; }
      if (parseJsonFallback) try { return coerce(JSON.parse(value), wanted, this.env, 'return'); }
      catch (error) { if (!(error instanceof Reject) && !(error instanceof SyntaxError)) throw error; }
      return this.writeNeuraleseResult(value, wanted, source, 'typed-text-result');
    }
    const resultType = this.lam.type.kind === 'lambda' ? this.lam.type.returns : parseType('null');
    // A value eval computed (a service's or a callee's result) is assignable as in TypeScript: fields the declared
    // record does not list are kept, not refused, so returning it never asks for a retyped copy. A literal written
    // into return_result is still checked exactly.
    try { return coerce(value, resultType, this.env, 'return', source === 'return_result' ? {} : { preserveRecordExtras: true }); }
    catch (first) {
      if (!(first instanceof Reject)) throw first;
      if (wanted.kind === 'neuralese' && element && !(element.kind === 'prim' && element.name === 'string') &&
          jsonSerializableType(element, this.env)) {
        let typedValue: Value;
        try { typedValue = coerceCompleteJson(value, element, this.env, 'return'); }
        catch (directError) {
          if (!(directError instanceof Reject)) throw directError;
          if (!parseJsonFallback || typeof value !== 'string') throw directError;
          let parsed: unknown;
          try { parsed = JSON.parse(value); } catch { throw directError; }
          typedValue = coerceCompleteJson(parsed, element, this.env, 'return');
        }
        const text = typedJsonSource(typedValue);
        if (text === undefined) throw first;
        return this.writeNeuraleseResult(text, wanted, source, 'typed-json-result');
      }
      let candidate = value;
      if (parseJsonFallback && typeof value === 'string') {
        try { candidate = JSON.parse(value); } catch { candidate = value; }
      }
      try { return coerce(candidate, resultType, this.env, 'return'); }
      catch {
        const plans: PlannedTypedText[] = [];
        const preflightType = typedTextPreflightType(candidate, resultType, this.env, [], plans);
        if (plans.length) {
          const preflightValue = coerce(candidate, preflightType, this.env, 'return');
          const holes = problems(preflightValue, preflightType, this.env, 'return').holes;
          if (holes.length) throw new Reject(holes);
          // Validate the full result first. A writer failure partway through this sequence can leave earlier
          // blocks stored; those real per-leaf effects are retained in the trace and are not rolled back.
          const refs: Value[] = [];
          for (const plan of plans) refs.push(await this.writeNeuraleseResult(plan.text, plan.type, source,
            'typed-text-result-field', plan.path));
          return applyTypedTextRefs(preflightValue, plans, refs);
        }
      }
      throw first;
    }
  }

  /** Apply one tool call. */
  async applyAsync(name: string, args: Record<string, unknown>, toolCallId?: string): Promise<NativeResult> {
    this.runtime.checkInterruption();
    if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
    if (!NATIVE_TOOLS.includes(name))
      return this.record(name, args, rejected(new Reject([{ path: name, code: 'bad-action', expected: 'a scope-eval tool' }])));
    if (this.actionLimitReached()) return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
    this.toolCalls++;
    this.actions++; this.lam.steps++;
    try {
      if (name === 'eval') {
        const timeout = args.timeout_ms;
        if (timeout !== undefined && (!Number.isInteger(timeout) || (timeout as number) < 1))
          throw new Reject([{ path: 'timeout_ms', code: 'bad-action', expected: 'a positive whole number of milliseconds' }]);
        if (args.finish !== undefined && typeof args.finish !== 'boolean')
          throw new Reject([{ path: 'finish', code: 'bad-action', expected: "a boolean: true completes this eval's fresh typed result" }]);
        return this.record(name, args, await this.evaluate(String(args.code ?? ''), timeout as number | undefined, args.finish === true, toolCallId), toolCallId);
      }
      if (name === 'return_result' && (args.status ?? 'success') === 'success' && Object.hasOwn(args, 'value') &&
          this.lam.type.kind === 'lambda') {
        const value = await this.coerceReturn(args.value, 'return_result', true);
        return this.record(name, args, this.scopeTool(name, { ...args, value }), toolCallId);
      }
      if (CODE_TOOLS.includes(name)) return this.record(name, args, this.functionTool(name, args));
      if (name === 'bash') {
        const folder = this.lam.projectTransaction?.folder;
        if (!folder) throw new Reject([{ path: name, code: 'bad-action', expected: 'a folder call' }]);
        const { runFolderBash } = await import('./folder-shell.js');
        const { invokeDefinition } = await import('../runtime/kernel.js');
        const value = await runFolderBash(folder, String(args.command ?? ''),
          { network: this.runtime.environment.scopeCapabilities?.allowNetwork !== false,
            python: await this.pythonHost(),
            ask: (instructions, returns, input) => invokeDefinition(this.runtime.frame!, {
              id: `bash:ask:${instructions}`, name: 'natlang-ask', body: instructions,
              params: [{ name: 'input', type: inferValueType(input) }], returns,
              types: this.lam.typesSrc, codebase: this.lam.codebase, subtype: 'function',
            }, [input]),
            call: async (name, input, mode) => {
              const { callableMeta } = await import('../runtime/callable.js');
              const callable = await this.folderCommandCallable(name, folder), meta = callableMeta(callable);
              if (!meta || meta.definition.subtype === 'directory-reducer')
                throw new TypeError(`${name} is not a callable function`);
              const params = meta.definition.params;
              if (mode === 'line' && params.length !== 1) throw new TypeError('--lines requires a function with one parameter');
              const values = input && typeof input === 'object' && !Array.isArray(input) ?
                params.map(param => (input as Record<string, unknown>)[param.name]) :
                params.length === 1 ? [input] : undefined;
              if (!values) throw new TypeError('call input must be a JSON object keyed by parameter name');
              return meta.invoke(values, this.runtime.frame!);
            },
            apply: async (name, path) => {
              const { callableMeta } = await import('../runtime/callable.js');
              const callable = await this.folderCommandCallable(name, folder), meta = callableMeta(callable);
              if (!meta || meta.definition.subtype !== 'directory-reducer')
                throw new TypeError(`${name} is not a directory reducer`);
              if (meta.definition.params.length) throw new TypeError(`${name} needs parameters; natlang apply accepts a folder-only reducer`);
              const handle = folder.dir(path), transaction = await handle.beginTransaction(true);
              return invokeDefinition(this.runtime.frame!, meta.definition, [handle],
                { folder: { transaction, mode: 'apply' } });
            } });
        return this.record(name, args, { kind: value.exitCode === 0 ? 'ok' : 'error',
          text: this.show(JSON.stringify(value)), value: value as unknown as Value });
      }
      if (name === 'python') {
        const folder = this.lam.projectTransaction?.folder;
        if (!folder) throw new Reject([{ path: name, code: 'bad-action', expected: 'a folder call' }]);
        const timeout = args.timeout_ms === undefined ? undefined : Number(args.timeout_ms);
        if (timeout !== undefined && (!Number.isSafeInteger(timeout) || timeout < 1))
          throw new Reject([{ path: 'timeout_ms', code: 'bad-action', expected: 'a positive timeout in milliseconds' }]);
        const { runFolderPython } = await import('./folder-python.js');
        const value = await runFolderPython(folder, String(args.code ?? ''), await this.pythonHost(), timeout);
        return this.record(name, args, { kind: 'ok', text: this.show(JSON.stringify(value)), value: value as unknown as Value });
      }
      if (name === 'delegate') {
        const folder = this.lam.projectTransaction?.folder;
        if (!folder) throw new Reject([{ path: name, code: 'bad-action', expected: 'a folder call' }]);
        const path = String(args.path ?? ''), instructions = String(args.instructions ?? '');
        if (!instructions.trim()) throw new Reject([{ path: 'instructions', code: 'bad-action', expected: 'nonempty child instructions' }]);
        const handle = folder.dir(path), transaction = await handle.beginTransaction(true);
        const { invokeDefinition } = await import('../runtime/kernel.js');
        const value = await invokeDefinition(this.runtime.frame!, {
          id: `delegate:${this.runtime.options.runId}:${this.lam.steps}`, name: 'delegate',
          body: instructions, params: [], returns: String(args.returns ?? 'unknown'),
          types: this.lam.typesSrc, codebase: this.lam.codebase, subtype: 'directory-reducer',
        }, [handle], { folder: { transaction, mode: 'apply' }, manifest: { delegate: true, path } });
        return this.record(name, args, { kind: 'ok', text: this.show(JSON.stringify(value) ?? 'null'), value: value as Value });
      }
      if (name === 'editor') return this.record(name, args, await this.editorTool(args));
      if (['list_files', 'search_files', 'read_file', 'write_file', 'edit_file', 'diff_files'].includes(name))
        return this.record(name, args, await this.fileTool(name, args));
      return this.record(name, args, this.scopeTool(name, args));
    } catch (error) {
      if (error instanceof Reject) return this.record(name, args, rejected(error));
      return this.record(name, args, { kind: 'error', text: error instanceof Error ? error.message : String(error) });
    }
  }
  private async folderCommandCallable(name: string, folder: Folder): Promise<unknown> {
    if (name.endsWith('.nl')) {
      const { namedCallable } = await import('../runtime/callable.js');
      const source = await folder.readText(name);
      const record = parseNatlang(name, source, this.lam.typesSrc, PATH_ONLY);
      return namedCallable(name.split('/').at(-1)!.slice(0, -3), { ...record,
        codebase: this.lam.codebase as Record<string, ItemRecord> });
    }
    return name.split('.').reduce<unknown>((part, key) =>
      part && (typeof part === 'object' || typeof part === 'function') ?
        (part as Record<string, unknown>)[key] : undefined, this.callables());
  }
  private async pythonHost(): Promise<PythonHost> {
    const { invokeDefinition } = await import('../runtime/kernel.js');
    const { iterateOn } = await import('../runtime/iterate.js');
    return {
      nl: (instructions, returns) => (...inputs) => invokeDefinition(this.runtime.frame!, {
        id: `python:${instructions}`, name: 'nl@python', body: instructions, params: [], openParameters: true,
        returns, types: this.lam.typesSrc, codebase: this.lam.codebase, subtype: 'function',
      }, inputs, { manifest: { inline: true, python: true } }),
      iterateOn: (step, initial, ...fixed) => iterateOn(step as never, initial, ...fixed).inFrame(this.runtime.frame),
    };
  }
  private async editorTool(args: Record<string, unknown>): Promise<NativeResult> {
    const folder = this.lam.projectTransaction?.folder;
    if (!folder) throw new Reject([{ path: 'editor', code: 'bad-action', expected: 'a folder call' }]);
    const command = String(args.command ?? ''), path = String(args.path ?? '');
    if (!path) throw new Reject([{ path: 'path', code: 'bad-action', expected: 'a file path' }]);
    let value: unknown;
    if (command === 'view') {
      const start = args.start_line === undefined ? 1 : Number(args.start_line);
      const end = args.end_line === undefined ? undefined : Number(args.end_line);
      const content = await folder.readText(path);
      const lines = content.split('\n');
      if (lines.at(-1) === '') lines.pop();
      if (!Number.isInteger(start) || start < 1 || end !== undefined && (!Number.isInteger(end) || end < start))
        throw new Reject([{ path: 'start_line', code: 'bad-action', expected: 'a valid line range' }]);
      value = lines.slice(start - 1, end).map((line, i) => `${start + i}\t${line}`).join('\n');
    } else if (command === 'create') {
      if (folder.isFile(path)) throw new Error(`file already exists: ${path}`);
      folder.writeText(path, String(args.file_text ?? ''));
      value = { path, changed: true };
    } else if (command === 'str_replace') {
      value = await folder.editText(path, String(args.old_str ?? ''), String(args.new_str ?? ''));
    } else if (command === 'insert') {
      const line = Number(args.insert_line ?? 0), content = await folder.readText(path);
      const lines = content.split('\n');
      if (!Number.isInteger(line) || line < 0 || line > lines.length)
        throw new Reject([{ path: 'insert_line', code: 'bad-action', expected: 'a line number in the file' }]);
      lines.splice(line, 0, String(args.new_str ?? ''));
      folder.writeText(path, lines.join('\n'));
      value = { path, changed: true };
    } else throw new Reject([{ path: 'command', code: 'bad-action', expected: 'view, create, str_replace or insert' }]);
    return { kind: 'ok', text: this.show(typeof value === 'string' ? value : JSON.stringify(value)), value: value as Value };
  }
  /** Synchronous tools (no eval). */
  apply(name: string, args: Record<string, unknown>): NativeResult {
    this.runtime.checkInterruption();
    if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
    if (!['read_page', 'compact_history', 'return_result', 'blocked', 'failed', 'read_code', 'edit_code', 'diff_code'].includes(name))
      return this.record(name, args, rejected(new Reject([{ path: name, code: 'bad-action', expected: 'a synchronous scope-eval tool' }])));
    if (this.actionLimitReached()) return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
    this.toolCalls++;
    this.actions++; this.lam.steps++;
    try {
      return this.record(name, args, CODE_TOOLS.includes(name) ? this.functionTool(name, args) : this.scopeTool(name, args));
    } catch (error) {
      if (error instanceof Reject) return this.record(name, args, rejected(error));
      return this.record(name, args, { kind: 'error', text: error instanceof Error ? error.message : String(error) });
    }
  }

  private scopeTool(name: string, args: Record<string, unknown>): NativeResult {
    if (name === 'read_page') {
      // A value in scope that pages itself (a store with page(n)) is read by calling it in eval, not with read_page.
      const id = String(args.id ?? ''), root = id.split('.')[0]!;
      const inScope = Object.hasOwn(this.runtime.services, root) || Object.hasOwn(this.lam.let, root) ||
        Object.hasOwn(this.lam.args, root) || !!findCodebaseItem(this.lam.codebase, root);
      if (root === 'transcript' && !this.pages.has(id)) throw new Reject([{ path: id, code: 'no-such-page', expected:
        'an ID from a cut-off message; transcript is in eval\'s scope: search it in eval with transcript.search("…"), ' +
        'then transcript.entry(n) for a match' }]);
      if (inScope && !this.pages.has(id)) throw new Reject([{ path: id, code: 'no-such-page', expected:
        `an ID from a cut-off message; ${root} is a name in eval's scope, so use it in eval (for example ${root}.page(${Number(args.page ?? 1)}) if it has pages)` }]);
      return { kind: 'ok', text: this.pages.read(id, Number(args.page ?? 1)) };
    }
    // The agent carries out the compaction itself once this call is accepted (it owns the conversation).
    if (name === 'compact_history') {
      const note = typeof args.note === 'string' ? args.note.trim() : '';
      if (!note || note.length > COMPACTION_NOTE_CHARS) throw new Reject([{ path: 'note', code: 'bad-action',
        expected: `a note of 1 to ${COMPACTION_NOTE_CHARS} characters: what you are doing, what you have found, what is left` }]);
      return { kind: 'ok', text: COMPACTED_RESULT };
    }
    // The retired blocked and failed tools are statuses of return_result now.
    if (name === 'blocked' || name === 'failed') throw new Reject([{ path: name, code: 'bad-action',
      expected: `return_result with status "${name}" and a reason` }]);
    if (name === 'return_result') {
      const status = args.status ?? 'success';
      if (status === 'blocked' || status === 'failed') {
        const reason = String(args.reason ?? '').trim();
        // Repair the invalid stop shape without encouraging a success claim.
        if (reason.length < 8) throw new Reject([{ path: 'reason', code: 'bad-action',
          expected: (status === 'blocked' ? 'a sentence saying what is missing' : 'a sentence explaining why the instructions cannot be carried out') +
            (Object.hasOwn(args, 'value') ? `; keep status "${status}", omit value, and explain this in reason. Use status "success" only if the task is actually complete and you have its result` : '') }]);
        return { kind: 'blocked', text: `${status === 'blocked' ? 'blocked' : 'error'}: ${reason}` };
      }
      if (status !== 'success') throw new Reject([{ path: 'status', code: 'bad-action', expected: '"success", "blocked", or "failed"' }]);
      if (this.lam.type.kind !== 'lambda') throw new Reject([{ path: 'value', code: 'bad-action', expected: 'a typed call' }]);
      let value: Value;
      if (!Object.hasOwn(args, 'value')) {
        if (this.lam.return === MISSING) throw new Reject([{ path: 'value', code: 'bad-action',
          expected: `a ${formatType(this.lam.type.returns)} or a complete typed result already staged in this call` }]);
        // Reuse the staged object as-is. finish() below still checks completeness, live holes,
        // pending transactions, and the normal terminal validation path.
        value = this.lam.return;
      } else {
        try { value = coerce(args.value, this.lam.type.returns, this.env, 'return'); }
        catch (first) {
          // Some models send structured values as JSON text.
          if (typeof args.value !== 'string') throw first;
          try { value = coerce(JSON.parse(args.value), this.lam.type.returns, this.env, 'return'); } catch { throw first; }
        }
      }
      this.lam.return = value;
      if (!this.finish()) throw new Reject([{ path: 'value', code: 'bad-action', expected: `a complete ${formatType(this.lam.type.returns)}` }]);
      return { kind: 'completed', text: `Returned ${oneLine(value, undefined, this.runtime.displayLiveId)}.`, value };
    }
    throw new Reject([{ path: name, code: 'bad-action', expected: 'a scope-eval tool' }]);
  }

  /** The services this call may use: all but those scoped to functions it is not running within. */
  /** What eval code in this call can use, for a model looking for something it does not have. */
  private scopeGuide(): string {
    const inputs = this.lam.type.kind === 'lambda' ? this.lam.type.params.fields.map(field => field.name) : [];
    const captures = Object.keys(this.lam.captures ?? {});
    const locals = Object.keys(this.lam.let).filter(name => !isPending(this.lam.let[name]!));
    const helperNames = [...this.persistentScopeHelpers.keys()];
    const names = [...new Set([...inputs, ...captures, ...locals, ...helperNames, ...Object.keys(this.lam.codebase),
      ...Object.keys(this.availableServices())])];
    return `This call's eval scope has ${names.length ? names.join(', ') : 'no names of its own'}, the built-ins ${canGenerateNl(this.runtime.frame) ? 'nl, ' : ''}read, iterateOn ` +
      'and transcript (read_code shows how to use them), and standard JavaScript; nothing else (no Node modules, no require).';
  }

  availableServices(): Record<string, object> {
    const chain = currentFrame()?.chain ?? [];
    return Object.fromEntries(Object.entries(this.runtime.services).filter(([name]) => {
      const scope = this.runtime.serviceScopes[name];
      return !scope || scope.some(path => path.endsWith('/**') ? chain.includes(`nl:${path.slice(0,-3)}`) : chain.at(-1)===`nl:${path}`);
    }).map(([name, service]) => [name, name === 'neuralese' && service && typeof service === 'object' ?
      Object.create(Object.getPrototypeOf(service), {
        // Descriptors, not a spread: a getter stays lazy (it is never read for a declaration) and prototype methods stay.
        ...Object.getOwnPropertyDescriptors(service),
        read: { value: (value: unknown) => readNeuraleseForCurrentTask(value), enumerable: true, configurable: true, writable: false } }) : service]));
  }

  /** read_code, edit_code, diff_code over the codebase record tree. */
  private functionTool(name: string, args: Record<string, unknown>): NativeResult {
    if (name === 'diff_code') {
      const changed = [...this.originalSources.keys()].map(source => ({ function: source, kind: 'modified' }));
      return { kind: 'ok', text: JSON.stringify(changed, null, 2), value: changed as Value };
    }
    const requested = String(args.name ?? '');
    // A bound skill is disclosed on request (S2 §2.2): its instructions, or one supporting file.
    if (requested.startsWith('skills.') && this.lam.skills) {
      // Skill instructions have an exact virtual path that can share a prefix with the executable helper
      // namespace (for example skills["exact-bookkeeping"].helpers). Resolve the document before that namespace.
      const document = this.lam.skills.documents[requested];
      if (document !== undefined) {
        if (name === 'edit_code') throw new Reject([{ path: requested, code: 'external', expected: 'a function of this program; skills are read, not edited, in a call' }]);
        const [, skillName, suffix] = /^skills\.([a-z0-9]+(?:-[a-z0-9]+)*)(?:\/(.*))?$/.exec(requested) ?? [];
        const skill = this.lam.skills.inventory?.find(item => item.name === skillName);
        const previous = this.skillReads.get(requested);
        const count = previous?.document === document ? previous.count + 1 : 1;
        this.skillReads.set(requested, { document, count });
        if (skill) this.runtime.trace.emit('skill_use', { phase: suffix ? 'support_file_read' : 'body_read',
          skill_name: skill.name, skill_revision: skill.revision, invocation_id: this.runtime.options.runId,
          path: suffix ?? 'SKILL.md', unchanged_read_count: count });
        const reminder = !suffix && count > 1 ?
          'These skill instructions are unchanged since your earlier read. Apply them to a task action; retrieving them again is not task progress.\n\n' : '';
        return { kind: 'ok', text: reminder + document, value: document,
          ...(reminder ? { codes: ['skill-unchanged-read'] } : {}) };
      }
      if (!findCodebaseItem(this.lam.codebase, requested)) throw new Reject([{ path: requested, code: 'no-such-function', expected:
        `a bound skill or one of its files: ${Object.keys(this.lam.skills.documents).join(', ')}` }]);
    }
    // An external service is shown by its declaration, and is not the program's to change.
    const service = requested.split('.')[0]!;
    if (Object.hasOwn(this.runtime.declarations, service) && !findCodebaseItem(this.lam.codebase, requested) &&
        !Object.hasOwn(BUILT_IN_DOCS, requested)) {
      if (name === 'edit_code') throw new Reject([{ path: requested, code: 'external', expected:
        `a function of this program; ${service} is an external service: its declaration can be read, but it runs outside this program and cannot be changed` }]);
      const declaration = this.runtime.declarations[service]!;
      return { kind: 'ok', text: `${declaration}\n// ${service} is an external service: this declaration is all of it there is to read.`, value: declaration };
    }
    const found = findCodebaseItem(this.lam.codebase, requested);
    // A built-in of eval is documented, not defined, here: read shows its documentation; it cannot be edited.
    if (!found && Object.hasOwn(BUILT_IN_DOCS, requested)) {
      if (name === 'edit_code') throw new Reject([{ path: requested, code: 'built-in', expected:
        `a function of this program's codebase; ${requested} is built into eval and cannot be changed` }]);
      const docs = canGenerateNl(this.runtime.frame) ? BUILT_IN_DOCS[requested]! : requested === 'nl' ?
        'Ad hoc nl calls are unavailable at this fifth layer. Make the judgment here or call an existing named function from a file.' :
        requested === 'iterateOn' ? BUILT_IN_DOCS.iterateOn!.split('step(state, ...otherArgs)')[0] +
          'step(state, ...otherArgs) returns the next state and may be async; the stopping check receives that state. ' + ITERATION_STATE_GUIDANCE : BUILT_IN_DOCS[requested]!;
      return { kind: 'ok', text: docs, value: docs };
    }
    // An importable package is read by its type declarations; it is not part of this program either.
    const packaged = found ? undefined : this.runtime.environment.declarationOf?.(requested);
    if (packaged !== undefined) {
      if (name === 'edit_code') throw new Reject([{ path: requested, code: 'external', expected:
        `a function of this program; ${requested} belongs to an imported package, which can be read but not changed` }]);
      return { kind: 'ok', text: this.show(packaged), value: packaged };
    }
    if (!found) {
      const toolDocumentation = name === 'read_code' ? this.toolDocumentation(requested) : undefined;
      if (toolDocumentation !== undefined) return { kind: 'ok', text: toolDocumentation, value: toolDocumentation };
      const availableServices = this.availableServices();
      if (name === 'read_code' && Object.hasOwn(availableServices, requested)) {
        const declaration = `Host service object declaration (type only, not implementation or field values):\n` +
          `declare const ${requested}: ${undeclaredServiceType(availableServices[requested], 0, requested)};\n\n` +
          `These members are available to this call under its current service scope. Inspect the value in eval when needed.`;
        return { kind: 'ok', text: declaration, value: declaration };
      }
      // Say what the name is when it is not the program's: a model asks for the source of its tools and built-ins.
      const own = listCodebase(this.lam.codebase);
      const readable = own.length ? `a function of this program: ${own.join(', ')}` : 'a function of this program, and this program has none of its own';
      const what = NATIVE_TOOLS.includes(requested) ? this.offeredToolDocs.has(requested) ?
        `${requested} is one of your currently offered tools, not a function of this program; call it directly` :
        `${requested} is a native tool name but is not offered in this call` : undefined;
      throw new Reject([{ path: requested, code: 'no-such-function', expected: what ? `${readable}. ${what}` : readable }]);
    }
    const view = this.runtime.frame?.task.programView;
    const record = view ? view.record(found.record) : found.record;
    const expectedRevision = view?.revisionId ?? 0;
    if (record.kind === 'namespace')
      throw new Reject([{ path: requested, code: 'no-such-function', expected: `an item inside ${requested}: ${Object.keys(record.codebase).join(', ')}` }]);
    if (name === 'read_code') {
      let explanation = '';
      if (record.kind === 'natlang' && !this.explainedNaturalFunctions.has(record.id)) {
        this.explainedNaturalFunctions.add(record.id);
        explanation = 'Natural-language function source: its frontmatter declares parameter and return types; the body contains instructions executed line by line.\n\n';
      }
      return { kind: 'ok', text: explanation + record.text, value: record.text };
    }
    if (this.runtime.frame?.task.runtime.options.codeEdits === 'deny' || view?.binding?.artifact.policy.codeEdits === 'deny')
      throw new Reject([{ path: requested, code: 'bad-action', expected: 'code edits are denied by this evaluation policy' }]);
    if (args.expected_revision !== undefined && args.expected_revision !== expectedRevision)
      throw new Reject([{ path: requested, code: 'bad-action', expected: 'current source revision; reread after a concurrent edit' }]);
    const text = editTextContent(record.text, String(args.find ?? ''), String(args.replace_with ?? ''), args.fuzzy === true);
    const updated = record.kind === 'natlang' ? parseNatlang(record.source, text, record.types, PATH_ONLY) :
      parseModule(record.source, text, record.types, PATH_ONLY);
    updated.programId = record.programId;
    updated.codebase = record.codebase;
    let inlineRevision: { original: InlineLambdaPlan[]; previous: InlineLambdaPlan[]; current: InlineLambdaPlan[] } | undefined;
    if (updated.kind === 'module' && record.kind === 'module') {
      // Validate and compile before committing: existing closures retain their live cells and slots,
      // but subsequent invocations resolve the new static text or reject a changed contract.
      let siblings = this.lam.codebase as Record<string, ItemRecord>;
      for (const part of found.path.slice(0, -1)) siblings = siblings[part]?.codebase ?? {};
      const level = { ...siblings };
      level[record.name] = record;
      inlineRevision = { original: [], previous: [], current: [] };
      compileModule({ ...record, text: view?.program?.sources[record.source] ?? view?.original(record).text ?? record.text }, level,
        plans => { inlineRevision!.original = plans; });
      compileModule(record, level, plans => { inlineRevision!.previous = plans; });
      compileModule(updated, { ...level, [updated.name]: updated }, plans => { inlineRevision!.current = plans; });
    }
    view?.commit(updated, expectedRevision, inlineRevision);
    if (!this.originalSources.has(record.source)) this.originalSources.set(record.source, record.text);
    this.lam.codebase = replaceCodebaseItem(this.lam.codebase, found.path, updated);
    const revision = { function: found.path.join('.'), source: record.source, changed: true, revision: view?.revisionId ?? 0,
      originalHash: hexDigest(view?.program?.sources[record.source] ?? record.text), effectiveHash: hexDigest(record.text), patchedHash: hexDigest(updated.text) };
    return { kind: 'ok', text: JSON.stringify(revision), value: revision };
  }

  /** File tools for directory reducers, over the private folder copy. */
  private async fileTool(name: string, args: Record<string, unknown>): Promise<NativeResult> {
    const path = String(args.path ?? '');
    if (path.startsWith('/') || path === '..' || path.startsWith('../') || path.includes('/../'))
      throw new Reject([{ path, code: 'no-such-path', expected: 'a relative path in the current folder' }]);
    const folder = this.lam.projectTransaction?.folder;
    if (!folder) throw new Reject([{ path, code: 'bad-action', expected: 'a directory reducer folder' }]);
    if (['read_file', 'write_file', 'edit_file'].includes(name) && !path)
      throw new Reject([{ path, code: 'no-such-path', expected: 'a file in the current folder' }]);
    let value: unknown;
    if (name === 'list_files') value = folder.listFiles(path, args.pattern === undefined ? undefined : String(args.pattern));
    else if (name === 'search_files') value = await folder.search(String(args.query ?? ''), path,
      args.pattern === undefined ? undefined : String(args.pattern), args.regex === true);
    else if (name === 'read_file') value = await folder.readText(path,
      args.start_line === undefined ? undefined : Number(args.start_line), args.end_line === undefined ? undefined : Number(args.end_line));
    else if (name === 'write_file') { folder.writeText(path, String(args.content ?? '')); value = { path, changed: true }; }
    else if (name === 'edit_file') value = await folder.editText(path, String(args.find ?? ''), String(args.replace_with ?? ''), args.fuzzy === true);
    else value = folder.diffSync(path);
    const text = name === 'diff_files' ? fileDiffPreview(value as ReturnType<Folder['diffSync']>) :
      name === 'list_files' ? fileListingText(value as EntryStat[]) : typeof value === 'string' ? value : JSON.stringify(value, null, 1);
    return { kind: 'ok', text: this.show(text), value: value as Value };
  }

  private inferScopeType(value: unknown): string { return inferValueType(value); }

  /** Preserve a source-backed callable signature when it uses types visible in this call's scope. */
  private async sourceCallableType(value: unknown, availableTypes: Record<string, Type>): Promise<Type | undefined> {
    if (typeof value !== 'function') return;
    const { callableMeta } = await import('../runtime/callable.js');
    const meta = callableMeta(value);
    if (!meta || !['inline', 'named'].includes(meta.kind)) return;
    try {
      const returns = parseType(meta.definition.returns);
      const fields = meta.definition.params.map(parameter => {
        return { name: parameter.name, type: parseType(parameter.type), optional: parameter.optional === true };
      });
      const signature: Type = { kind: 'lambda', params: { kind: 'record', fields }, returns };
      this.env.child(availableTypes).checkNames(signature);
      return signature;
    } catch { return; }
  }


  private scopePath(expression: string): string {
    const match = /^([A-Za-z_$][\w$]*)(.*)$/.exec(expression.trim());
    if (!match) throw new Reject([{ path: 'expression', code: 'bad-action', expected: 'a variable and field/index selections' }]);
    const root = match[1]!;
    const base = Object.hasOwn(this.lam.let, root) ? `let/${root}` : Object.hasOwn(this.lam.args, root) ? `args/${root}` :
      root === 'result' && this.lam.return !== MISSING ? 'return' : '';
    if (!base) throw new Reject([{ path: root, code: 'no-such-path', expected: 'a scope variable' }]);
    let path = base, tail = match[2]!;
    while (tail) {
      const member = /^\.([A-Za-z_$][\w$]*)(.*)$/s.exec(tail);
      const index = /^\[(?:"([^"\\]+)"|'([^'\\]+)'|(\d+))\](.*)$/s.exec(tail);
      if (member) { path += `/${member[1]}`; tail = member[2]!; }
      else if (index) { path += `/${index[1] ?? index[2] ?? index[3]}`; tail = index[4]!; }
      else throw new Reject([{ path: 'expression', code: 'bad-action', expected: 'field and index selection without computation' }]);
    }
    return path;
  }

  /** Static type of an initializer expression, for locals declared without an annotation. */
  private scopeInitializerType(expression: string, locals: Record<string, Type>): Type | undefined {
    const source = expression.trim();
    // A reduce result has the accumulator's type, which need not match its list.
    if (/\.(?:reduce|reduceRight)\s*\(/.test(source)) return;
    const directCall = /^(?:await\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(/.exec(source);
    const directReturns = directCall ? returnTypeOf(this.lam.codebase, directCall[1]!) : undefined;
    if (directReturns) return parseType(directReturns);
    const mappedCall = /^await\s+Promise\.all\([\s\S]*\.map\([\s\S]*?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(/.exec(source);
    const mappedReturns = mappedCall ? returnTypeOf(this.lam.codebase, mappedCall[1]!) : undefined;
    if (mappedReturns) return parseType(`(${mappedReturns})[]`);
    const lastCollectionMethod = [...source.matchAll(/\.(find|filter|slice|map|flatMap)\s*\(/g)].at(-1)?.[1];
    const collection = ['find', 'filter', 'slice'].includes(lastCollectionMethod ?? '') ?
      /^(.*)\.(find|filter|slice)\s*\(.*\)$/s.exec(source) : null;
    const projected = /^(.*)\.map\s*\(\s*(?:\(\s*)?([A-Za-z_$][\w$]*)(?:\s*\))?\s*=>\s*\2((?:\.[A-Za-z_$][\w$]*|\[(?:\d+|"[^"]+"|'[^']+')\])+)\s*\)$/s.exec(source);
    const receiver = (collection?.[1] ?? projected?.[1] ?? source).trim();
    const selected = (base: Type, tail: string): Type | undefined => {
      let type = base;
      while (tail) {
        const member = /^\.([A-Za-z_$][\w$]*)(.*)$/s.exec(tail);
        const index = /^\[(?:"([^"\\]+)"|'([^'\\]+)'|(\d+))\](.*)$/s.exec(tail);
        const resolved = this.env.resolve(type);
        if (member && resolved.kind === 'record') {
          const field = resolved.fields.find(item => item.name === member[1]); if (!field) return;
          type = field.type; tail = member[2]!;
        } else if (index && (resolved.kind === 'list' || resolved.kind === 'dict')) { type = resolved.element; tail = index[4]!; }
        else if (index && resolved.kind === 'record') {
          const field = resolved.fields.find(item => item.name === (index[1] ?? index[2] ?? index[3])); if (!field) return;
          type = field.type; tail = index[4]!;
        } else return;
      }
      return type;
    };
    try {
      const root = /^[A-Za-z_$][\w$]*/.exec(receiver)?.[0]; if (!root) return;
      const type = root in locals ? selected(locals[root]!, receiver.slice(root.length)) : this.resolve(this.scopePath(receiver)).type;
      if (!type) return;
      const resolved = this.env.resolve(type);
      if (collection) return resolved.kind === 'list' ? (collection[2] === 'find' ? resolved.element : type) : undefined;
      if (projected) return resolved.kind === 'list' ?
        (() => { const item = selected(resolved.element, projected[3]!); return item ? { kind: 'list', element: item } as Type : undefined; })() : undefined;
      return type;
    } catch { return; }
  }

  /** Portable snapshot of the scope for failure debugging. */
  private scopeSnapshot(): Record<string, unknown> {
    const locals: Record<string, Value> = { ...this.lam.let };
    if (!Object.hasOwn(locals, 'result') && this.lam.return !== MISSING) locals.result = this.lam.return;
    const view = (values: Record<string, Value>) => Object.fromEntries(Object.entries(values).slice(0, 128).map(([name, value]) =>
      [name, diagnosticValue(value, name, this.runtime.displayLiveId)]));
    return { inputs: view(this.lam.args), locals: view(locals) };
  }

  /** Whether this call's scope can hold soft values, so eval code is always checked for their opacity. */
  private holdsNeuralese(): boolean {
    const mentions = (text: string) => /\bNeuralese</.test(text);
    return mentions(formatType(this.lam.type)) || Object.values(this.lam.typesSrc).some(mentions) ||
      Object.values(this.lam.letTypes).some(type => mentions(formatType(type))) ||
      Object.values(this.lam.captures ?? {}).some(cell => mentions(cell.type));
  }

  /** Execute one eval action as an atomic scope transaction. */
  private async evaluate(written: string, timeoutMs?: number, complete = false, toolCallId?: string): Promise<NativeResult> {
    if (!written.trim()) throw new Reject([{ path: 'code', code: 'bad-action', expected: 'a TypeScript statement or expression' }]);
    // Literals the model wrote are block markers in its text; compiled, each is a call the checker types from context.
    const code = desugarNlCalls(sourceWithLiteralCalls(written));
    const scopeBefore = this.scopeSnapshot(), traceMark = this.runtime.trace.events.length;
    const inputNames = this.lam.type.kind === 'lambda' ? this.lam.type.params.fields.map(field => field.name) : [];
    const localNames = Object.keys(this.lam.let).filter(name => !isPending(this.lam.let[name]!));
    const localBindings = localNames.map(name => ({ name, mutable: this.scopeLocalMutability.get(name) ?? true,
      annotation: this.lam.letTypes[name] && this.lam.letTypes[name]!.kind !== 'host' ? formatType(this.lam.letTypes[name]!) : undefined }));
    const callableNames = Object.keys(this.lam.codebase);
    const opaqueNames: string[] = [];
    if (this.lam.projectTransaction) opaqueNames.push('folder');
    // read_inputs() returns the call's arguments (a frozen copy), which the opening eval also holds as `inputs`;
    // a parameter or local of either name takes precedence.
    const taken = (name: string) => inputNames.includes(name) || Object.hasOwn(this.lam.let, name);
    const inputsBinding = !taken('read_inputs'), inputsObject = !taken('inputs');
    if (inputsBinding) opaqueNames.push('read_inputs');
    if (inputsObject) opaqueNames.push('inputs');
    // transcript: this call's earlier tool calls and their outputs, read-only, unless the name is taken.
    const transcriptBinding = !taken('transcript');
    if (transcriptBinding) opaqueNames.push('transcript');
    // return_result also works as a function in eval: return_result(value) stages a result, and
    // return_result(value, status, reason) carries out the tool's request once the eval has succeeded.
    // A finisher name the snippet declares itself stays the snippet's own variable.
    const declaredHere = (name: string) => new RegExp(`\\b(?:const|let|var|function|class)\\s+${name}\\b|[{,]\\s*${name}\\s*[,}=]`).test(code);
    const finishers = ['return_result'].filter(name => !taken(name) && !declaredHere(name));
    opaqueNames.push(...finishers);
    let requested: { tool: string; args: Record<string, unknown> } | undefined;
    const captureCells = this.lam.captures ?? {};
    const captureRead = Object.fromEntries(Object.entries(captureCells).map(([name, cell]) => [name, cell.get()]));
    const serviceNames = Object.keys(this.availableServices()).filter(name => !inputNames.includes(name) &&
      !callableNames.includes(name) && !Object.hasOwn(captureCells, name));
    // `read_code` is normally a native tool. Give eval the same pure lookup for code/docs so code can
    // inspect a declaration without leaving TypeScript. Existing scope names and real program callables win.
    const readCodeBinding = !taken('read_code') && !declaredHere('read_code') &&
      !callableNames.includes('read_code') && !serviceNames.includes('read_code') && !Object.hasOwn(captureCells, 'read_code');
    if (readCodeBinding) opaqueNames.push('read_code');
    // decide(fn, ...args) calls a function with a finite result type and returns its answer with the probability of
    // every allowed value (runtime.decide in eval), unless the name is taken.
    const decideBinding = !taken('decide') && !callableNames.includes('decide') && !serviceNames.includes('decide') &&
      !Object.hasOwn(captureCells, 'decide');
    if (decideBinding) opaqueNames.push('decide');
    const savedHelpers = [...this.persistentScopeHelpers.values()].filter(helper =>
      hexDigest(helper.source) === helper.sourceHash && Number.isSafeInteger(helper.declarationSpan.start) &&
      helper.declarationSpan.start >= 0 && helper.declarationSpan.end - helper.declarationSpan.start === helper.source.length &&
      !declaredHere(helper.name) && !inputNames.includes(helper.name) && !localNames.includes(helper.name) &&
      !callableNames.includes(helper.name) && !serviceNames.includes(helper.name) && !opaqueNames.includes(helper.name) &&
      !Object.hasOwn(captureCells, helper.name));
    const savedHelperNames = savedHelpers.map(helper => helper.name);
    const visibleTypes = { ...this.lam.typesSrc, ...Object.fromEntries(Object.keys(this.localTypes).map(name => [name, ''])) };
    const currentTypes = evalTypeDeclarations(code);
    const persistentTypeNames = [...new Set([...Object.keys(visibleTypes), ...Object.keys(currentTypes)])];
    const hooks = this.runtime.hooks;
    const compiled = compileScopeSnippet(code, { inputBindings: inputNames, localBindings, helperBindings: callableNames,
      persistentHelpers: savedHelpers, persistentHelperNames: savedHelperNames, persistentTypeNames,
      opaqueBindings: opaqueNames,
      captureBindings: Object.values(captureCells).map(cell => ({ name: cell.name, mutable: cell.mutable })),
      serviceBindings: serviceNames, analyze: source => hooks.analyze(this, source), neuralese: this.holdsNeuralese(),
      guardPrefix: `eval:${this.runtime.options.runId}`, ...this.runtime.environment.scopeCapabilities });
    // Model-written literals in this eval are graph nodes: the block, its contextual type, the reference written.
    if (compiled.ok) {
      const turn = [...this.runtime.trace.events].reverse().find(event => event.kind === 'model_turn')?.node as string | undefined;
      for (const literal of compiled.literals ?? []) graphNode(this.runtime.trace, 'literal', { call_id: this.runtime.options.runId,
        block: literal.id, type: literal.type, expression: `__neuralese(${JSON.stringify(literal.id)})` },
        [blockInput(literal.id, 'block'), ...(turn ? [{ node: turn, port: 'turn' }] : [])]);
      for (const plan of compiled.plans ?? []) if (plan.softBody) graphNode(this.runtime.trace, 'literal', { call_id: this.runtime.options.runId,
        block: plan.softBody, type: `Neuralese<(${plan.parameters.map(item => `${item.name}: ${item.type.natlang ?? item.type.text}`).join(', ')}) => ${plan.returns.natlang ?? plan.returns.text}>`,
        expression: `__neuralese.body(${JSON.stringify(plan.softBody)})`,
        captures: plan.captures.filter(capture => !capture.shadowedByParameter).map(capture => capture.name),
        shadowed_capture_initializers: plan.captures.filter(capture => capture.shadowedByParameter)
          .map(({ name, mode }) => ({ name, mode: mode ?? 'snapshot' })) },
        [blockInput(plan.softBody, 'body'), ...(turn ? [{ node: turn, port: 'turn' }] : [])]);
    }
    if (!compiled.ok || !compiled.program) {
      const text = compiled.diagnostics.map(item => `${item.line}:${item.column} ${item.code}: ${item.message}`).join('\n');
      return { kind: 'rejected', text: text + this.captureScopeFailure('compile', code, scopeBefore, text, compiled.diagnostics),
        codes: [...new Set(compiled.diagnostics.map(item => item.code))] };
    }
    const inputs = splitScope(this.lam.args);
    const locals = splitScope(Object.fromEntries(localNames.map(name => [name, this.lam.let[name]!])));
    let finished: unknown;
    let evalStop: AbortController | undefined;
    const plans = compiled.plans ?? [];
    const origin: InlineInstructionOrigin | undefined = toolCallId && this.runtime.currentCallId ? {
      parentInvocationId: this.runtime.currentCallId, toolCallId, actionOrdinal: this.actions,
      writtenCodeSha256: hexDigest(written), checkedCodeSha256: hexDigest(code),
      ...(() => {
        const bodyPlan = plans.find(plan => plan.softBody);
        if (!bodyPlan?.softBody) return {};
        const literal = `\`\uE000${bodyPlan.softBody}\uE001\``;
        const start = written.indexOf(literal);
        return start >= 0 && written.indexOf(literal, start + 1) < 0 ?
          { sourceTemplateSpan: { start, end: start + literal.length } } : {};
      })() } : undefined;
    this.activeScopeLocals = new Map();
    const live = { inputs: inputs.live, locals: locals.live, captures: captureRead, callables: this.callables(),
      services: this.availableServices(), folder: this.lam.projectTransaction?.folder.root(),
      readNeuralese: readNeuraleseForCurrentTask,
      readNeuraleseIfReference,
      readCode: (input: unknown) => {
        let name: string | undefined;
        if (typeof input === 'string') name = input;
        else if (isPlainRecord(input) && Object.keys(input).length === 1 && Object.hasOwn(input, 'name')) {
          const descriptor = Object.getOwnPropertyDescriptor(input, 'name');
          if (descriptor && 'value' in descriptor && typeof descriptor.value === 'string') name = descriptor.value;
        }
        if (name === undefined) throw new Reject([{ path: 'read_code', code: 'type-mismatch',
          expected: 'a name string or exactly { name: string }',
          got: input === null ? 'null' : Array.isArray(input) ? 'array' : typeof input }]);
        try {
          const result = this.functionTool('read_code', { name });
          this.runtime.trace.emit('eval_code_read', { call_id: this.runtime.currentCallId ?? this.runtime.options.runId,
            action: toolCallId ?? null, name, outcome: result.kind, text_sha256: hexDigest(result.text) });
          return result.text;
        } catch (error) {
          this.runtime.trace.emit('eval_code_read', { call_id: this.runtime.currentCallId ?? this.runtime.options.runId,
            action: toolCallId ?? null, name, outcome: error instanceof Reject ? 'rejected' : 'error',
            error_sha256: hexDigest(error instanceof Error ? error.message : String(error)) });
          throw error;
        }
      },
      concatNeuralese,
      joinNeuralese,
      arrayToStringNeuralese,
      mapNeuraleseReadout,
      invokeWithReceiver,
      rebindInline: (value: unknown, captures: Record<string, unknown>, sources: Record<string, 'input' | 'local' | 'block' | 'handle'>) =>
        rebindInlineCallable(value, captures, origin, sources),
      callInputs: inputsBinding || inputsObject ? frozenCopy(this.lam.args) : undefined,
      transcript: transcriptBinding ? new TranscriptView(this.transcript.slice()) : undefined,
      decide: (fn: (...args: unknown[]) => Promise<unknown>, args: unknown[]) => {
        const frame = currentFrame() ?? this.runtime.frame!;
        return import('../runtime/runtime.js').then(module => module.decideInFrame(frame, fn, args));
      },
      request: (tool: string, args: Record<string, unknown>) => { requested ??= { tool, args }; },
      bindLocal: (name: string, get: () => unknown, set?: (value: unknown) => void) => {
        this.activeScopeLocals?.set(name, [get, set]);
      },
      inline: (index: number, values: unknown[], accessors: Record<string, unknown>) => {
        const plan = plans[index];
        if (!plan) throw new Error('internal error: unknown inline plan');
        const bound = { ...accessors };
        for (const capture of plan.captures) {
          if (capture.shadowedByParameter) continue;
          if (!capture.mutable || capture.source === 'block' ||
              !(localNames.includes(capture.name) || compiled.bindings.some(binding => binding.name === capture.name))) continue;
          const accessor = accessors[capture.name] as [() => unknown, ((value: unknown) => void)?];
          if (!accessor) continue;
          this.activeScopeLocals?.set(capture.name, accessor);
          bound[capture.name] = [
            () => { const active = this.activeScopeLocals?.get(capture.name); return active ? active[0]() : this.lam.let[capture.name]; },
            (value: unknown) => {
              const active = this.activeScopeLocals?.get(capture.name);
              if (active?.[1]) active[1](value);
              else this.lam.let[capture.name] = coerce(value, this.lam.letTypes[capture.name]!, this.env, `let/${capture.name}`);
            },
          ];
        }
        return hooks.inline(this, plan, values, bound, origin);
      },
      finite: hooks.finite, finiteArrayIterator: hooks.finiteArrayIterator, finiteAsync: hooks.finiteAsync, guard: hooks.guard,
      iterateOn: (step: unknown, initial: unknown, ...args: unknown[]) => hooks.iterateOn(this, step, initial, ...args),
      finish: (value: unknown) => { finished = value; } };
    const prologue = [
      ...(this.lam.projectTransaction ? ['const folder = __live.folder;'] : []),
      ...(inputsBinding ? ['const read_inputs = () => __live.callInputs;'] : []),
      ...(inputsObject ? ['const inputs = __live.callInputs;'] : []),
      ...(transcriptBinding ? ['const transcript = __live.transcript;'] : []),
      ...(readCodeBinding ? ['const read_code = (input: string | { name: string }) => __live.readCode(input);'] : []),
      ...(decideBinding ? ['const decide = (fn: any, ...args: any[]) => __live.decide(fn, args);'] : []),
      ...finishers.map(name => `const ${name} = (value?: unknown, status: string = 'success', reason?: string) => ` +
        `{ __live.request(${JSON.stringify(name)}, { value, status, reason }); };`),
    ].join('\n');
    const source = `${SCOPE_RUNTIME_PRELUDE}${prologue}\n${compiled.program}\n` +
      `return await ${compiled.entrypoint}(Object.assign({}, self.inputs, __live.inputs), ` +
      `Object.assign({}, self.locals, __live.locals), __live.captures);`;
    try {
      const callFrame = this.runtime.frame;
      evalStop = callFrame ? new AbortController() : undefined;
      const evaluated = await this.runtime.evaluate(this.lam, source,
        { inputs: inputs.portable, locals: locals.portable }, live, timeoutMs,
        evalStop ? AbortSignal.any([callFrame!.signal ?? callFrame!.task.signal, evalStop.signal]) : undefined);
      const parentCallId = callFrame?.parentCallId;
      if (parentCallId && callFrame!.task.hasPendingChildren(parentCallId)) {
        // Calls that lost a Promise.race or Promise.any are stopped quietly; any other call still running was dropped.
        const dropped = callFrame!.task.pendingChildCalls(parentCallId).filter(call => !racedCalls.has(call)).length;
        evalStop!.abort(new Error('the eval that started this call finished without awaiting it'));
        await callFrame!.task.drainChildren(parentCallId);
        if (dropped) throw new Error(`eval finished while ${dropped} natural-language call${dropped === 1 ? '' : 's'} it started ` +
          `${dropped === 1 ? 'was' : 'were'} still running; ${dropped === 1 ? 'it was' : 'they were'} stopped. Await every call ` +
          '(for example with Promise.all(...)) before the eval ends.');
      }
      const raw = finished as { result?: unknown; returned?: boolean;
        bindings?: Record<string, unknown>; captures?: Record<string, unknown> } | undefined;
      if (!raw || typeof raw !== 'object' || !raw.bindings || typeof raw.bindings !== 'object')
        throw new Reject([{ path: 'code', code: 'bad-action', expected: 'an atomic scope transaction result' }]);
      // A malformed compiled result must not poison every later eval with numeric locals.
      // This can happen when a comma expression is passed as two function arguments.
      if (Object.keys(raw.bindings).some(name => !/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name)))
        throw new Reject([{ path: 'code', code: 'bad-action',
          expected: 'named eval bindings; wrap a final comma expression in parentheses' }]);
      // Plain data is rebuilt in this realm; captured values keep their identity for write-back.
      const output = { ...hostCopy({ result: raw.result, bindings: raw.bindings }) as
        { result?: unknown; bindings: Record<string, unknown> }, captures: raw.captures };
      const thenable = (value: unknown) => !!value && (typeof value === 'object' || typeof value === 'function') &&
        typeof (value as PromiseLike<unknown>).then === 'function';
      const unsettled = [['the result', output.result], ...Object.entries(output.bindings),
        ...Object.entries(output.captures ?? {})].find(([, value]) => thenable(value));
      if (unsettled) throw new Error(`${unsettled[0]} holds a promise that could not be kept: await the call that made it`);
      const annotations = new Map(compiled.bindings.map(binding => [binding.name, binding.annotation]));
      // A local annotated with a type an eval declared stores that type's definition, which later evals can read.
      const typesHere = evalTypeDeclarations(code), localTypes = { ...evalTypeDeclarations(Object.values(this.runtime.declarations).join("\n")), ...this.localTypes, ...typesHere };
      const initializers = new Map(compiled.bindings.map(binding => [binding.name, binding.initializer]));
      const mutability = new Map(compiled.bindings.map(binding => [binding.name, binding.mutable]));
      const staged: [string, Type, Value][] = [];
      const inferred: Record<string, Type> = { ...this.lam.letTypes };
      for (const [name, value] of Object.entries(output.bindings)) {
        if (Object.hasOwn(this.lam.args, name) || Object.hasOwn(this.lam.codebase, name))
          throw new Reject([{ path: name, code: 'not-writable', expected: 'a local variable' }]);
        // A name this eval declares is a new binding, as in a REPL: it takes its type afresh. A local that so far only
        // held null (let x = null) takes the type of what is assigned to it, as TypeScript's evolving let does.
        const stored = this.lam.letTypes[name];
        let type = annotations.has(name) || (stored?.kind === 'prim' && stored.name === 'null' && value !== null) ? undefined : stored;
        const annotation = annotations.get(name);
        if (annotation) type = inlineDeclaredTypes(parseType(annotation), localTypes);
        if (!type && initializers.get(name)) type = this.scopeInitializerType(initializers.get(name)!, inferred);
        if (!type) type = await this.sourceCallableType(value, localTypes);
        if (!type) {
          try { type = parseType(this.inferScopeType(value)); }
          catch (error) {
            if (this.lam.type.kind !== 'lambda' || !compiled.resultBindings?.includes(name)) throw error;
            type = this.lam.type.returns;
          }
        }
        inferred[name] = type;
        // A local holds what eval computed (a service's or a callee's value): as in TypeScript, fields its type does
        // not list are kept, so storing a provider value never asks for a retyped copy.
        staged.push([name, type, coerce(value, type, this.env, `let/${name}`, { preserveRecordExtras: true })]);
      }
      // A top-level return proposes the call's result; it is taken only if it has the declared type.
      let functionResult: Value | undefined, notResult = '';
      if ((raw.returned || complete) && this.lam.type.kind === 'lambda') try {
        functionResult = await this.coerceReturn(output.result, complete ? 'eval-finish' : 'eval-return', true);
      } catch (error) {
        if (!(error instanceof Reject)) throw error;
        notResult = `\nThis is not a valid ${formatType(this.lam.type.returns)}, so it is not the result: ` +
          error.message;
      }
      const captureWrites: [string, unknown][] = [];
      for (const [name, value] of Object.entries(output.captures ?? {})) {
        const cell = captureCells[name];
        if (!cell?.mutable || Object.is(value, captureRead[name])) continue;
        if (!Object.is(cell.get(), captureRead[name]))
          throw new Reject([{ path: name, code: 'capture-conflict',
            expected: `${name} was changed by another caller while this eval ran; run the eval again with its current value` }]);
        captureWrites.push([name, value]);
      }
      for (const [name, value] of captureWrites) captureCells[name]!.set!(value);
      // Report only locals this eval declared or changed.
      // Compare staged portable values structurally. JSON.stringify(dump(value)) allocated a
      // second copy of every prior binding on each eval and overflowed on large model-built data.
      const before = new Map(Object.entries(this.lam.let));
      const changed = staged.filter(([name, , value]) => !before.has(name) ||
        (containsLive(value) ? before.get(name) !== value : sameValueWithin(before.get(name), value) !== true));
      Object.assign(this.localTypes, typesHere);
      // A successful transaction replaces helpers by their source. An ordinary or transient declaration
      // with the same name shadows the old source-backed definition for later evals.
      for (const binding of compiled.bindings) this.persistentScopeHelpers.delete(binding.name);
      for (const helper of compiled.persistentHelpers ?? []) {
        const exactSource = helper.source;
        if (code.slice(helper.declarationSpan.start, helper.declarationSpan.end) !== exactSource)
          throw new Error(`Saved helper ${helper.name} source span did not match the successfully compiled eval source`);
        this.persistentScopeHelpers.set(helper.name, { ...helper, sourceHash: hexDigest(exactSource),
          declaredAction: { ...(toolCallId ? { toolCallId } : {}), actionOrdinal: this.actions,
            writtenCodeSha256: hexDigest(written) } });
      }
      for (const [name, type, value] of staged) {
        this.lam.letTypes[name] = type;
        this.lam.let[name] = value;
        if (mutability.has(name)) this.scopeLocalMutability.set(name, mutability.get(name)!);
      }
      this.activeScopeLocals = undefined;
      if (functionResult !== undefined) {
        this.lam.return = functionResult;
        this.failureDebug = undefined;
      }
      this.evalDetail = { value: output.result ?? null, console: (evaluated.logs ?? []).join('\n') };
      const rendered = isLive(output.result) || isHandle(output.result) ? livePreview(output.result as object, this.runtime.displayLiveId) :
        this.showValue(output.result ?? null);
      const status = functionResult !== undefined ?
        stagedMessage(functionResult, this.runtime.displayLiveId) : notResult;
      const stored = changed.map(([name, , value]) => `local ${name} = ${oneLine(value, name, this.runtime.displayLiveId)}`);
      const storedStatus = (stored.length ? `\nStored ${stored.join('; ')}.` : '') +
        ((compiled.transientHelpers?.length ?? 0) ? `\n${compiled.transientHelpers!.map(helper =>
          `${helper.name} is available only in this eval because ${helper.reason}; use it here or move its dependencies into persistent scope.`).join('\n')}` : '');
      // A local without a value cannot be kept, so say so here rather than let a later eval fail on its name.
      const unset = compiled.bindings.filter(binding => binding.mutable && !binding.initializer).map(binding => binding.name)
        .filter(name => output.bindings[name] === undefined && !Object.hasOwn(this.lam.let, name));
      const unsetStatus = unset.length ? `\nNot kept: ${unset.join(', ')} ${unset.length > 1 ? 'have' : 'has'} no value, and a ` +
        `local without a value is not kept; declare it with one (let ${unset[0]} = …).` : '';
      const logStatus = evaluated.logs?.length ? `console:\n${this.show(evaluated.logs.join('\n'))}\n` : '';
      // return_result in eval stages its value like a top-level return: the value was computed, so the model
      // sees it before the call finishes. The blocker and error reports carry the model's own text and end the call.
      // return_result({ status, value | reason }) in eval is the tool's request written as code; read it as that when
      // the object is not itself a value of the return type.
      const shaped = requested?.args.value as Record<string, unknown> | undefined;
      if (requested?.tool === 'return_result' && requested.args.status === 'success' && requested.args.reason === undefined &&
          shaped && typeof shaped === 'object' && !Array.isArray(shaped) && ['success', 'blocked', 'failed'].includes(String(shaped.status)) &&
          Object.keys(shaped).every(key => ['status', 'value', 'reason'].includes(key)) && this.lam.type.kind === 'lambda') {
        let fits = true;
        try { coerce(shaped as Value, this.lam.type.returns, this.env, 'return'); } catch { fits = false; }
        if (!fits) requested = { tool: 'return_result', args: { status: shaped.status, value: shaped.value, reason: shaped.reason } };
      }
      if (requested?.tool === 'return_result' && requested.args.status === 'success' && this.lam.type.kind === 'lambda') {
        let staged: Value | undefined, refusal = '';
        try { staged = await this.coerceReturn(requested.args.value, 'return_result', true); }
        catch (error) {
          if (!(error instanceof Reject)) throw error;
          refusal = error.message;
        }
        if (staged === undefined) return { kind: 'rejected', text: `${logStatus}${rendered}${storedStatus}\nreturn_result: this is not a valid ` +
          `${formatType(this.lam.type.returns)}, so it is not the result: ${refusal}`, codes: ['type-mismatch'] };
        this.lam.return = staged;
        this.failureDebug = undefined;
        if (complete) {
          const done=this.scopeTool('return_result',{status:'success',value:staged});
          return {...done,text:`${logStatus}${rendered}${storedStatus}${unsetStatus}\n${done.text}`};
        }
        return { kind: 'ok', text: `${logStatus}${rendered}${storedStatus}${unsetStatus}${stagedMessage(staged, this.runtime.displayLiveId)}`, value: staged };
      }
      if (requested) {
        const shown = logStatus + rendered + storedStatus + unsetStatus;
        try {
          const done = this.scopeTool(requested.tool, requested.args);
          return { ...done, text: `${shown}\n${done.text}` };
        } catch (error) {
          if (!(error instanceof Reject)) throw error;
          const refused = rejected(error);
          return { ...refused, text: `${shown}\n${requested.tool}: ${refused.text}` };
        }
      }
      if (complete) {
        if (functionResult === undefined) {
          const finishHint = notResult
            ? "\nCorrect this eval's value to match the declared result type; a previously staged value cannot finish this action."
            : '\nfinish:true requires a fresh value of the declared result type from this eval. Use a final expression or explicit return; an older staged result cannot finish this action.';
          return {kind:'rejected',text:logStatus+rendered+storedStatus+unsetStatus+notResult+finishHint,codes:['missing-fresh-result']};
        }
        // Finish with the staged value as checked above (computed values keep their extra fields), not a re-check.
        const done=this.scopeTool('return_result',{status:'success'});
        return {...done,text:logStatus+rendered+storedStatus+unsetStatus+'\n'+done.text};
      }
      return { kind: 'ok', text: logStatus + rendered + storedStatus + unsetStatus + status, value: (output.result ?? null) as Value,
        ...(compiled.repairs.length ? { codes: ['coerced-redundant-self-alias'] } : {}) };
    } catch (error) {
      // A failed eval (an error, a rejected value, its timeout) stops the natural-language calls it started.
      evalStop?.abort(new Error('the eval that started this call failed'));
      const message = error instanceof Error ? error.message : String(error);
      const note = this.captureScopeFailure(error instanceof EvalFailure ? 'runtime' : 'boundary',
        code, scopeBefore, message, error instanceof Reject ? error.diagnostics : [], error, traceMark);
      if (error instanceof Reject) { const result = rejected(error); return { ...result, text: result.text + note }; }
      return { kind: 'error', text: message + note };
    } finally {
      const frame = this.runtime.frame;
      const parentCallId = frame?.parentCallId;
      if (parentCallId) await frame!.task.drainChildren(parentCallId);
      this.activeScopeLocals = undefined;
    }
  }

  private resolve(path: string): Ref {
    const parts = path.split('/').filter(Boolean);
    const first = parts.shift();
    if (first === 'return') {
      const ref = itemRef(this.lam as unknown as Record<string, Value>, 'return',
        this.lam.type.kind === 'lambda' ? this.lam.type.returns : parseType('null'), this.env, 'return');
      return this.descend(ref, parts);
    }
    const name = parts.shift();
    if (!name) throw new Reject([{ path, code: 'no-such-path' }]);
    if (first === 'let') {
      const type = this.lam.letTypes[name];
      if (!type) throw new Reject([{ path, code: 'no-such-path' }]);
      return this.descend(itemRef(this.lam.let, name, type, this.env, `let/${name}`), parts);
    }
    if (first === 'args' && this.lam.type.kind === 'lambda') {
      const field = this.lam.type.params.fields.find(f => f.name === name);
      if (!field) throw new Reject([{ path, code: 'unknown-field' }]);
      return this.descend(itemRef(this.lam.args, name, field.type, this.env, `args/${name}`, 'not-writable'), parts);
    }
    throw new Reject([{ path, code: 'no-such-path' }]);
  }

  private descend(ref: Ref, parts: string[]): Ref {
    for (const part of parts) {
      const current = ref.get();
      let type = ref.env.resolve(ref.type!);
      if (type.kind === 'union') {
        const selected = type.members.map(member => ref.env.resolve(member)).find(member =>
          member.kind === 'record' ? member.fields.some(field => field.name === part) :
            member.kind === 'dict' ? current !== null && typeof current === 'object' && !Array.isArray(current) :
              member.kind === 'list' ? Array.isArray(current) : false);
        if (selected) type = selected;
      }
      let childType: Type;
      if (type.kind === 'record') {
        const field = type.fields.find(f => f.name === part);
        if (!field) throw new Reject([{ path: `${ref.path}/${part}`, code: 'unknown-field' }]);
        childType = field.type;
      } else if (type.kind === 'dict') childType = type.element;
      else if (type.kind === 'list') {
        if (!/^\d+$/.test(part)) throw new Reject([{ path: `${ref.path}/${part}`, code: 'no-such-path' }]);
        childType = type.element;
      } else throw new Reject([{ path: `${ref.path}/${part}`, code: 'no-such-path' }]);
      const parent = ref, key = part;
      ref = { path: `${parent.path}/${part}`, type: childType, env: parent.env, deny: parent.deny,
        get: () => {
          const value = parent.get();
          return value !== MISSING && value !== null && typeof value === 'object' && key in value ? (value as Record<string, Value>)[key]! : MISSING;
        },
        set: () => { throw new Reject([{ path: `${parent.path}/${key}`, code: 'not-writable' }]); }, del: () => {} };
    }
    return ref;
  }
}

/**
 * The type aliases of the callable items a call can use: the types its function listing shows. A name
 * that two items define differently is left out, since neither definition is the one in scope.
 */
export function callableTypes(codebase: Record<string, unknown>): Record<string, Type> {
  const texts = new Map<string, string | null>();
  const visit = (items: Record<string, unknown>) => {
    for (const raw of Object.values(items)) {
      const item = raw as { types?: Record<string, string>; codebase?: Record<string, unknown> };
      for (const [name, text] of Object.entries(item.types ?? {}))
        texts.set(name, !texts.has(name) || texts.get(name) === text ? text : null);
      if (item.codebase) visit(item.codebase);
    }
  };
  visit(codebase ?? {});
  const types: Record<string, Type> = {};
  for (const [name, text] of texts) if (text !== null) {
    try { types[name] = parseType(text); } catch { /* outside the portable grammar: checked by the compiler, not here */ }
  }
  return types;
}
import { hexDigest } from './hash.js';
