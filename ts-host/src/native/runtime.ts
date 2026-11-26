/**
 * The natural-language function interpreter: runs one invocation of a `.nl` or inline `nl`
 * function as a model-driven session over a persistent TypeScript scope.
 *
 * Portable data reaches eval as a frozen snapshot; live values (host objects, functions, folder
 * handles, captured bindings, callables, services) arrive by reference through `__live`.
 */
import {executionOccurrence,occurrenceArgumentHash} from './execution-occurrence.js';
import { aliasDeclaration } from './type-aliases.js';
import { writesFields, writesProblem, writesRefinementChecks, type WritesField } from '../runtime/writes.js';
import { COMPACTED_RESULT, PROMPT_EXAMPLE_SOURCE, PROMPT_EXAMPLE_TEXTS } from './prompt.js';
import { arrayToStringNeuralese, concatNeuralese, instantiate, invokeWithReceiver, joinNeuralese, mapNeuraleseReadout, readNeuraleseIfReference, rebindInlineCallable, type InlineInstructionOrigin } from '../runtime/lowered.js';
import { EvalFailure, type EvalEnvironment, type HostEvent } from './evaluator.js';
import { PageStore } from './pages.js';
import { NativeEffectOwner, isRecording, recordingServices, type EffectEvent } from './effects.js';
import { DEFAULT_DIALECT, TypeEnv, TypeSyntaxError, containsRefinement, formatType, parseType, parseTypeCached, typeAliasNames, type Type } from './types.js';
import { evalTypeDeclarations, inlineDeclaredTypes } from './eval-types.js';
import { MISSING, Reject, coerce, hostCopy, mapCopiedParts, withSourceIdentity, dump, isLive, isPending, liveLabel, problems, unboundParts, createLiveIdentity, scopedLiveIdentity,
  isPlainRecord, type LambdaNode, type Value } from './values.js';
import { changes, NativeTraceRecorder } from './trace.js';
import { FileHandle, Folder, FolderHandle, editTextContent, fileListingText, type EntryStat } from './scoped-fs.js';
import { fileDiffPreview } from './file-diff-preview.js';
import type { PythonHost } from './folder-python.js';
import { compileScopeSnippet, evalDeclaredNames, SCOPE_RUNTIME_PRELUDE, type PersistentScopeHelper } from '../scope-compiler.js';
import { livePreview, renderValue } from './agent.js';
import { UntrustedRegistry, containsUntrusted, hostUntrusted, scopedUntrusted } from './untrusted.js';
import type { InlineLambdaPlan, NatlangDiagnostic } from '../compiler/inline.js';
import { NEURALESE_TYPE_DOCUMENTATION } from '../compiler/intrinsics.js';
import { desugarNlCalls } from '../compiler/nl-call.js';
import { dialectBinding, isNeuraleseRef, isNoNeuraleseWriterFault, isTeacherTextNeuraleseId, neuraleseRef, neuraleseSentinel, nativeLiteralArguments, NeuraleseNoWriterError, NeuraleseUnsupportedError, sourceWithLiteralCalls,
  type NeuraleseRuntimeOptions } from './neuralese.js';
import { bindTraceScope, blockInput, FILE_CONTEXT, graphNode, invocationNodeId, producerScopeForTrace, valueInputs } from './graph.js';
import { producingCall } from './operator-frames.js';
import { canGenerateNl, currentFrame, racedCalls, startedCalls, runInFrame, type Frame } from '../runtime/context.js';
import { SAME_QUESTION_REFUSAL } from '../runtime/kernel.js';
import { isNatlangCallable } from '../runtime/callable.js';
import { withSelectedCaptureAccessors } from '../runtime/selected-captures.js';
import { declareSelectedSurface } from './selected-surface.js';
import { PATH_ONLY, isProtectedSource, parseModule, parseNatlang, treeProtectedSources, type ItemRecord } from '../runtime/loader.js';
import { compileModule, stdModule } from '../runtime/modules.js';
import { readNeuraleseForCurrentTask } from '../neuralese/combinators.js';
import { undeclaredServiceType } from './introspection.js';
import { NATIVE_TOOL_SCHEMA } from './tool-surface.js';
import { returnErrorText, returnValueRejection, type ReturnErrorText } from './return-errors.js';
import { literalText } from './pseudo-call.js';
import { STAGED_RESULT_NOTICE } from './staged-result.js';
import { modelFacing } from '../runtime/model-facing.js';
import type { Platform } from '../runtime/platform.js';
import ts from 'typescript';

function typeAliasesFromEnv(env: TypeEnv): Record<string, string> {
  const chain: TypeEnv[] = [];
  for (let current: TypeEnv | undefined = env; current; current = current.parent) chain.unshift(current);
  const aliases: Record<string, string> = {};
  for (const current of chain) for (const [name, type] of Object.entries(current.names)) aliases[name] = formatType(type);
  return aliases;
}

/** What the model is told when it reaches for the call it is executing (calling it, or reading its code): a judgment is
 * made here and returned, and eval serves the exact work the instructions need, not as a stand-in for the judgment. */
/** A value built only of strings, finite numbers, booleans, null, arrays and plain objects (no runtime values). */
function isPlainJson(value: unknown, depth = 0): boolean {
  if (depth > 32) return false;
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(item => isPlainJson(item, depth + 1));
  if (typeof value !== 'object') return false;
  return isPlainRecord(value) && Object.values(value).every(item => isPlainJson(item, depth + 1));
}

const SELF_CALL_ADVICE = 'Do what its instructions ask here. For a judgment, decide it yourself from the inputs and return the answer with ' +
  'return_result; use eval only for exact work the instructions need.';

/** Detect the unsupported pattern of wrapping a computed value in marker strings. Static marker literals remain valid. */
function computedNeuraleseMarkerConcatenation(code: string, codebase: Record<string, ItemRecord>, isNeuraleseType: (source: string) => boolean,
  neuraleseResult: boolean): ts.Node | undefined {
  const source = ts.createSourceFile('natlang-eval.ts', code, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const partsOf = (expression: ts.Expression): ts.Expression[] => ts.isBinaryExpression(expression) &&
    expression.operatorToken.kind === ts.SyntaxKind.PlusToken ? [...partsOf(expression.left), ...partsOf(expression.right)] : [expression];
  const hasComputedMarker = (expression: ts.Expression): boolean => {
    if (ts.isTemplateExpression(expression)) {
      const templateText = expression.head.text + expression.templateSpans.map(span => span.literal.text).join('');
      return expression.templateSpans.length > 0 && templateText.includes('<|neuralese|>') && templateText.includes('<|/neuralese|>');
    }
    const parts = partsOf(expression);
    const literalText = parts.filter(ts.isStringLiteralLike).map(part => part.text);
    const hasOpen = literalText.some(text => text.includes('<|neuralese|>'));
    const hasClose = literalText.some(text => text.includes('<|/neuralese|>'));
    const hasComputedPart = parts.some(part => !ts.isStringLiteralLike(part) &&
      !(ts.isNoSubstitutionTemplateLiteral(part)));
    return hasOpen && hasClose && hasComputedPart;
  };
  const expressionPath = (expression: ts.Expression): string[] | undefined => {
    if (ts.isIdentifier(expression)) return [expression.text];
    if (ts.isPropertyAccessExpression(expression)) {
      const base = expressionPath(expression.expression);
      return base ? [...base, expression.name.text] : undefined;
    }
    return undefined;
  };
  const callableArgumentTypes = (expression: ts.Expression): string[] | undefined => {
    const path = expressionPath(expression);
    if (!path) return undefined;
    let items: Record<string, ItemRecord> = codebase;
    let item: ItemRecord | undefined;
    for (let index = 0; index < path.length; index++) {
      item = items[path[index]!];
      if (!item) return undefined;
      const next = path[index + 1];
      if (item.kind === 'module' && next && item.exports[next]?.kind === 'function')
        return Object.values(item.exports[next]!.args);
      if (index + 1 === path.length) return item.kind === 'natlang' ? Object.values(item.args) : undefined;
      items = item.codebase;
    }
    return undefined;
  };
  let found: ts.Node | undefined;
  const computedVariables = new Set<string>();
  const collectComputedVariables = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.initializer && hasComputedMarker(node.initializer))
      computedVariables.add(node.name.text);
    ts.forEachChild(node, collectComputedVariables);
  };
  collectComputedVariables(source);
  const visit = (node: ts.Node) => {
    if (found) return;
    if (ts.isReturnStatement(node) && node.expression && neuraleseResult &&
        (hasComputedMarker(node.expression) || ts.isIdentifier(node.expression) && computedVariables.has(node.expression.text))) {
      found = node.expression;
      return;
    }
    if (ts.isVariableDeclaration(node) && node.type && isNeuraleseType(node.type.getText(source)) &&
        node.initializer && hasComputedMarker(node.initializer)) {
      found = node.initializer;
      return;
    }
    if (ts.isCallExpression(node)) {
      const parameters = callableArgumentTypes(node.expression);
      for (const [index, argument] of node.arguments.entries()) if (parameters?.[index] && isNeuraleseType(parameters[index]!) &&
          (hasComputedMarker(argument) || ts.isIdentifier(argument) && computedVariables.has(argument.text))) {
        found = argument;
        return;
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return found;
}

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
  analyze(session: NativeSession, source: string, bindings?: readonly string[]): { plans: InlineLambdaPlan[]; diagnostics: NatlangDiagnostic[];
    neuralese?: import('../compiler/neuralese.js').NeuraleseLiteral[]; readouts?: import('../compiler/neuralese.js').NeuraleseReadout[];
    instantiations?: import('../compiler/neuralese.js').GenericInstantiation[] };
  /** Reads of fields a value's declared type does not have, checked in every eval. */
  checkFields?(session: NativeSession, source: string, bindings?: readonly string[]): NatlangDiagnostic[];
  /** The snippet with typed `refine<Is<T, "p">>(value)` / `assume<R>(value)` calls in their predicate form. */
  lowerRefinements?(session: NativeSession, source: string): string;
};
export type NativeOutcome = { kind: 'done' | 'quiesced'; detail: string; value?: Value };
/** A tool call's result. `entry` is its index in the session's transcript. */
export type NativeResult = { kind: string; text: string; value?: Value; codes?: string[]; entry?: number;
  /** The full texts of outputs this result shows cut off (the learned compaction operator reads them; agent.ts). */
  cutOff?: string[] };
export type NativeAgent = (session: NativeSession) => Promise<string | void> | string | void;
export type NativeRuntimeOptions = { environment: EvalEnvironment; hooks: NativeRuntimeHooks; agent?: NativeAgent;
  /** Sees each service call this runtime wraps, with its exact arguments and result (a standalone run's call record). */
  observeEffect?: (event: EffectEvent) => void;
  maxActions?: number; maxToolCalls?: number;
  checkEvalBindings?: boolean;
  /** Check unresolved asynchronous values before reading their result fields (default true). */
  checkAwaitedFields?: boolean;
  /** Join accepted Native operations at the eval boundary (default true). */
  settleNativeEffects?: boolean;
  /**
   * The root call's ID: unique per run (it keys the call record, its children's parent link and the trace). Without one a
   * fresh `native-run:<uuid>` is used. `runLabel` is the logical name shared by runs of one kind (`curriculum-reference`),
   * kept on the call record; derived seeds use `seedId`, else the label, else the run ID, so a unique ID leaves them as
   * they were under the shared name.
   */
  runId?: string; runLabel?: string; seedId?: string; signal?: AbortSignal; timeoutMs?: number;
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
  /** Producer custody shared with the actual parent trace, independently of task budgets. */
  sharedProducerScope?: object;
  /** Wording of return_result's value rejections (native/return-errors.ts); default the platform's (Node: NATLANG_RETURN_ERROR_TEXT), then the adopted one. */
  returnErrorText?: ReturnErrorText;
  seedPolicy?: { mode: 'derived' | 'backend'; root?: number } };

type Ref = { path: string; type?: Type; env: TypeEnv; deny?: string;
  get(): Value; set(value: Value): void; del(): void };

/** A one-line summary of a value for status lines; `holder` names where all of it is (see renderValue). */
const oneLine = (value: unknown, holder?: string, liveIdentity?: (value: object) => number, untrusted?: UntrustedRegistry) =>
  renderValue(value, { holder, budget: 80, liveIdentity, untrusted });
function diagnosticValue(value: Value, holder?: string, liveIdentity?: (value: object) => number): unknown {
  if (portableSizeAtMost(value, TRANSCRIPT_VALUE_CHARS) && !containsLive(value)) return dump(value);
  return { $diagnostic_preview: oneLine(value, holder, liveIdentity), complete: false, holder };
}
function diagnosticArgument(value: unknown, holder: string, liveIdentity?: (value: object) => number): unknown {
  if (portableSizeAtMost(value, TRANSCRIPT_VALUE_CHARS) && !containsLive(value)) return value;
  return { $diagnostic_preview: oneLine(value, holder, liveIdentity), complete: false, holder };
}

const NL_WITH_HINT = 'With `nl.with<T>`, T is the child result type; use `nl.with<CaptureRecord, Result>` only when you want to type both the captures and result. The child input is passed separately.';
const DIAGNOSTIC_HINTS: Record<string, string> = {
  'type-mismatch': 'Pass the value itself with the type shown as expected, not wrapped in another object: for boolean use `true`, for number use `42.5`, for string use text, and for a record use an object with exactly its fields.',
  'nl-type-arguments': NL_WITH_HINT,
  'nl-explicit-captures': NL_WITH_HINT,
  'callable-not-invoked': 'This is a natlang child function, not its result. Call it with its declared input, then return or store the value it produces.',
  'unknown-field': 'Use one of the fields listed as expected.',
  'no-such-path': 'Use a variable or field that exists in the scope.',
  'capture-conflict': 'Another caller changed that captured variable; run the eval again with its current value.',
};
/** Every tool the runtime can apply; a call's offer is a subset. */
/** A return_result value that is a single `${...}` placeholder: return_result sends its value as written. */
export const PLACEHOLDER_VALUE = /^\$\{[^{}]*\}$/;
export const PLACEHOLDER_ADVICE = 'the value itself: return_result fills in no placeholder, so write out the text or ' +
  'value you mean';
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

/**
 * The eval binding `NatlangContext`: the runtime's Context class (runtime/contexts.ts installs it under a global symbol,
 * as for callable-folder modules) through `empty()` and `ofCallable(fn)`. Undefined parts fail when used, with one sentence.
 */
function evalContextValue(): { empty(): unknown; ofCallable(fn: unknown): unknown } {
  const contextClass = () => {
    const context = (globalThis as Record<symbol, unknown>)[Symbol.for('natlang.context-class')] as
      { empty(): unknown; ofCallable(fn: unknown): unknown } | undefined;
    if (!context) throw new Error('NatlangContext is unavailable in this runtime: it has no context support loaded.');
    return context;
  };
  return Object.freeze({
    empty: () => contextClass().empty(),
    ofCallable: (fn: unknown) => contextClass().ofCallable(fn),
  });
}

/**
 * The standard library's combinators as eval built-ins (2026-11-25): a call composes its judgments with them by name,
 * and they run in the call's frame, so their child calls nest under it in `std:<name>` spans. The two workspace
 * combinators take a folder and are offered where the call has one. NATLANG_EVAL_STD=0 turns them off (eval then has
 * only the built-ins it had before).
 */
export const EVAL_STD_COMBINATORS = ['semanticMap', 'semanticFilter', 'semanticPartition', 'semanticReduce', 'semanticMatch',
  'semanticRank', 'semanticRepair', 'fixedPoint', 'semanticMerge', 'escalate', 'withContext', 'compileRule'] as const;
export const EVAL_STD_FOLDER_COMBINATORS = ['proposeAndVerify', 'branchAndSelect'] as const;
/** Whether eval offers the std combinators (default on; NATLANG_EVAL_STD=0, off or false turns them off). */
export function evalStdEnabled(): boolean {
  const setting = typeof process === 'undefined' ? undefined : process.env?.NATLANG_EVAL_STD;
  return !/^(?:0|off|false|no)$/i.test((setting ?? '').trim());
}

/**
 * Eval locals keep JavaScript object identity across evals (one copy of the whole scope, input parts mapped back to the
 * host's values, coerced locals keeping their objects); NATLANG_EVAL_IDENTITY=0 copies each value on its own as before.
 */
export function evalIdentityEnabled(): boolean {
  const setting = typeof process === 'undefined' ? undefined : process.env?.NATLANG_EVAL_IDENTITY;
  return !/^(?:0|off|false|no)$/i.test((setting ?? '').trim());
}

/** read_code documentation of the std built-ins (the declarations are those of `@natlang/node/std`). */
export const STD_BUILT_IN_DOCS: Record<string, string> = {
  semanticMap: `semanticMap(items, fn, { concurrency? }): call fn(item, index) on every item, concurrently, and return the
results in the items' order. fn is a function of this program, an nl function or an arrow function.
  const summaries = await semanticMap(tickets, summarize);
  const labels = await semanticMap(rows, row => nl<'bug' | 'question'>\`Label the row's text.\`(row.text), { concurrency: 4 });
Each call runs as a child of this call, under a std:semanticMap span.`,
  semanticFilter: `semanticFilter(items, criterionOrPredicate, { floor? }): the items that pass, in order. A string criterion is
judged per item by the shipped meets judgment; a function returns a boolean. With floor, each judgment is scored and an
item passes when its probability of true reaches floor.
  const urgent = await semanticFilter(tickets, 'needs a reply today');
  const valid = await semanticFilter(rows, isValidRow);`,
  semanticPartition: `semanticPartition(items, classifier, { floor? }): group items by the label classifier(item) returns.
  const { groups, uncertain } = await semanticPartition(tickets, routeTicket);
groups maps each label to its items; uncertain lists items whose label's probability fell below floor.`,
  semanticReduce: `semanticReduce(items, initial, step, { mode?: 'linear' | 'tree', combine? }): fold the items into one state.
step(state, item, index) returns the next state; with mode 'tree', step makes one state per item from initial and
combine(left, right) joins them pairwise.
  const digest = await semanticReduce(notes, '', (summary, note) => nl<string>\`Fold note into summary.\`(summary, note));
Each step runs as a child of this call, under a std:semanticReduce span.`,
  semanticMatch: `semanticMatch(left, right, { relation?, match?, block?, floor? }): pair items of left with items of right.
relation is a sentence judged per pair (shipped relates judgment), or match(l, r) decides; block(l, r) skips pairs cheaply.
  const { pairs, unmatchedLeft, unmatchedRight } = await semanticMatch(invoices, payments, { relation: 'pays this invoice' });`,
  semanticRank: `semanticRank(items, goal, { debiased? }): order items by goal from pairwise judgments (shipped beats judgment).
  const { ranked, scores } = await semanticRank(drafts, 'clearest answer to the question');`,
  semanticRepair: `semanticRepair(value, { check, fix, attempts }): check(value) returns the problems it finds (strings, none when
it passes); while there are problems, fix(value, problems) returns a new value, at most attempts times.
  const repaired = await semanticRepair(draft, { check: d => problemsOf(d), fix: (d, problems) => reviseDraft(d, problems), attempts: 3 });
  // { ok, value, problems, attempts, history }`,
  fixedPoint: `fixedPoint(step, initial, { settled?, maxSteps? }): apply step to its own output until the value stops changing
in meaning (the shipped settled judgment, or settled(previous, current)); returns { value, steps, settled }. With an exact
settled test give maxSteps.
  const { value } = await fixedPoint(tighten, draft);`,
  semanticMerge: `semanticMerge(base, ours, theirs, { check? }): merge two edits of base key by key; conflicting edits are judged
by intent and combined when compatible. Returns { value, merged, unresolved }.`,
  escalate: `escalate(fn, { to, floor?, risk?, policy? }): fn's decision, sent to the stronger model to when its confidence is below
floor (or a risky label's probability passes risk[label]). Returns a function; its result is a decision record.
  const verdict = await escalate(classify, { to: 'teacher', floor: 0.8 })(item);
  // verdict.value, verdict.confidence, verdict.escalated`,
  withContext: `withContext(fn, entries): fn bound to its own context with these data entries added (fn.in(...)); not awaited.
  const strict = withContext(review, { policy: strictPolicy });
  const verdict = await strict(item);`,
  compileRule: `compileRule(rule, { examples?, check?, attempts? }): a function (item) => Promise<result> written from the rule,
checked against the examples ({ input, expected }) and revised until they pass.
  const isSpam = await compileRule('the message asks for money from a stranger', { examples });`,
  proposeAndVerify: `proposeAndVerify(folder, reducer, { verify, attempts, args? }): run reducer on a private copy of folder,
verify(proposal) returns the problems it finds; a failing proposal is proposed again with its problems as reducer's last
argument. Only a passing proposal is accepted into folder. Returns { accepted, proposal, value, problems, attempts, history }.`,
  branchAndSelect: `branchAndSelect(folder, branches, { verify?, goal?, select? }): run each branch (a reducer, or { reducer, args })
on its own private copy of folder, keep the ones verify passes, choose one by goal (pairwise judgments) or select, and
accept it. Returns { chosen, proposal, value, candidates }.`,
};

export const BUILT_IN_DOCS: Record<string, string> = {
  Neuralese: NEURALESE_TYPE_DOCUMENTATION,
  read: 'Read an opaque Neuralese<T> value using the configured typed reader. In eval, call `await read(value)`; it returns Promise<T> and records the ordinary typed readout in the execution graph. Already-crisp strings pass through unchanged; other non-reference values are rejected. Use FileHandle.readText() or FileHandle.readJson() for files.',
  'neuralese.read': 'Compatibility access through the configured neuralese service. Prefer the shared eval builtin `await read(value)` for Neuralese<T>; it returns Promise<T> and records the ordinary typed readout in the execution graph. Already-crisp strings pass through unchanged; other non-reference values are rejected. `neuralese.bodies.read` is only the configured reader body ID.',
  'neuralese.textReadSource': `neuralese.textReadSource is read-only standard-library metadata, not a callable API or a value to invoke.
  read_code("neuralese") shows the available host service declaration; read_code("Neuralese") shows the opaque compile-time type.
The descriptor identifies the configured read instruction body (export: "read") and its provenance. Do not call
neuralese.textReadSource.read or inspect/copy its fields to read a value. Use the supported typed operation
read(value), or a supported JavaScript text-conversion context such as String(value) for
Neuralese<string>. The runtime performs the configured typed readout; the descriptor itself does not perform it.`,
  NatlangContext: `NatlangContext: build a context to run a function under. fn.in(context) is the same function bound to it,
a new function; the function's instructions read the context's data entries by their names.
  const strict = verdict.in(NatlangContext.empty().with({ policy: strictPolicy }));
  const scoped = verdict.in(NatlangContext.ofCallable(verdict).with({ policy: strictPolicy }));
NatlangContext.empty() is the context with nothing in it; NatlangContext.ofCallable(fn) starts from fn's own context
(keeping its callable helpers); .with({ name: value, ... }) adds or replaces data entries (a new context each time),
.without(...names) removes them, .pick(...names) keeps only those, and .data reads them. A function keeps every other
part of its signature under .in; it must still find each item its instructions name.`,
  refine: `refine(value, predicate[, { context }]) or refine<Is<T, "predicate">>(value[, { context }]):
check that value satisfies a natural-language predicate and return it typed as Is<T, predicate>.
  const summary = await refine(draft, 'one concise sentence');
  const grounded = await refine(answer, 'supported by source', { context: { source } });
Throws refinement-unsatisfied (or refinement-undecided inside an uncertainty band) when the check fails.`,
  assume: `assume(value, predicate) or assume<Is<T, "predicate">>(value):
declare that value already satisfies predicate without calling the judge, recording the assumption in the trace.`,
  decide: `decide(fn, ...args): call a function in this eval with those arguments and inspect its decision result.
  const verdict: Neuralese<(statement: string) => Promise<"allow" | "deny">> = nl\`Choose allow or deny for the statement.\`;
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
JavaScript template interpolation runs when you create the callable. Mention a future argument by its parameter
name in the instructions; interpolate it only inside a callback where that variable already exists.
When a child needs fields from a record, pass the full record or capture it by name with nl.with, rather than only
passing an ID or prior decision. Reading a file in the parent does not by itself pass its contents to the child.
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
  search_code: `search_code(query[, limit]): search this call's codebase, types.ts aliases, and service declarations in eval.
  const matches = search_code("refund");
  // returns { name, kind, signature?, description?, path? }[] (up to 20 matches by default)
Matches case-insensitively across callable names, parameter and return types, types.ts aliases, descriptions/doc comments,
and service declarations, ranked by relevance. Call read_code(match.name) to inspect any result in full.
Available in eval whenever the call's scope has a codebase.`,
};

function rejected(error: Reject): NativeResult {
  const softMismatch = error.diagnostics.some(diagnostic => diagnostic.code === 'type-mismatch' &&
    /\bNeuralese\s*</.test(diagnostic.expected ?? ''));
  const nlWithIssue = error.diagnostics.some(diagnostic =>
    diagnostic.code === 'nl-type-arguments' || diagnostic.code === 'nl-explicit-captures' ||
    /\bnl\.with\b/.test(`${diagnostic.path} ${diagnostic.expected ?? ''} ${diagnostic.text ?? ''}`));
  const hint = softMismatch ? 'At a declared Neuralese<string> argument or final-result slot, an ordinary string is written as that typed value through the configured port; argument conversion applies only to the direct parameter, not nested fields. For a final Neuralese<T> result with a concrete JSON-serializable T, return a value of T or its JSON text; the runtime validates it and writes its canonical JSON through the configured port. These conversions do not apply to locals or other payload types: preserve typed values or use one exact block marker in an explicitly typed position. A displayed [[Neuralese text block ...]] label is a preview, not a JavaScript string value or a reference to copy; do not turn it into a string or object.' :
    nlWithIssue ? NL_WITH_HINT :
    error.diagnostics.map(diagnostic => DIAGNOSTIC_HINTS[diagnostic.code]).find(Boolean);
  return { kind: 'rejected', text: `rejected\n${error.message}${hint ? `\nhint: ${hint}` : ''}`,
    codes: error.diagnostics.map(diagnostic => diagnostic.code) };
}

/** How a value of a structured type is written ('an object', 'an array'), or undefined when the type can be text. */
function structuredShape(type: Type, env: TypeEnv): string | undefined {
  const resolved = env.resolve(type);
  if (resolved.kind === 'refined' || resolved.kind === 'untrusted') return structuredShape(resolved.base, env);
  if (resolved.kind === 'record' || resolved.kind === 'dict') return 'an object';
  if (resolved.kind === 'list' || resolved.kind === 'tuple') return 'an array';
  if (resolved.kind !== 'union' || !resolved.members.length) return undefined;
  const shapes = new Set(resolved.members.map(member => structuredShape(member, env)));
  if (shapes.has(undefined)) return undefined;
  return shapes.size === 1 ? [...shapes][0] : 'an object or an array';
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
const stagedMessage = (value: Value, liveIdentity?: (value: object) => number, untrusted?: UntrustedRegistry) =>
  `\nStaged ${stagedText(value, liveIdentity, untrusted)}${STAGED_RESULT_NOTICE}`;

/** A staged value, in full when it is small, so it can be checked (and never needs retyping); long ones are cut by structure. */
function stagedText(value: Value, liveIdentity?: (value: object) => number, untrusted?: UntrustedRegistry): string {
  return renderValue(value, { budget: 1500, liveIdentity, untrusted });
}
/** A deep-frozen copy of portable data; live values and handles are kept by reference. */
function frozenCopy(value: Record<string, Value>): Record<string, unknown> {
  const copy = (item: unknown): unknown => Array.isArray(item) ? Object.freeze(item.map(copy)) :
    item && typeof item === 'object' && !isLive(item) && !isHandle(item) && isPlainRecord(item) ?
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
function splitScope(values: Record<string, Value>, shared?: Map<object, unknown>): { portable: Record<string, unknown>; live: Record<string, unknown> } {
  const portable: Record<string, unknown> = {}, live: Record<string, unknown> = {};
  for (const [name, value] of Object.entries(values)) {
    if (value === MISSING || isPending(value)) continue;
    if (containsLive(value)) live[name] = value;
    else portable[name] = dump(value, false, shared);
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
/**
 * The scope type of a portable value, as a local holding it is declared. A list or record that holds a live value is one
 * live value, unless `liveFields`: then it is typed field by field and only the live value itself is a live type (for a
 * host that reads the fields around a handle, such as an eval cell's `scope`).
 */
export function inferValueType(value: unknown, options: { liveFields?: boolean } = {}): string {
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
    if (!options.liveFields && containsLive(value)) return 'Live<"array", "tag", "Array">';
    const types = [...new Set(value.map(item => inferValueType(item, options)))];
    return types.length === 1 ? `(${types[0]})[]` : `(${types.join(' | ')})[]`;
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value);
    if (!entries.length) return 'Record<string, unknown>';
    if (!options.liveFields && containsLive(value)) return 'Live<"object", "any", "">';
    // An object keyed by data (ids, emails, names) is a dictionary: Record<string, T>, so later evals can add keys.
    // Only a few identifier keys read as a record with fixed fields.
    if (entries.length > 12 || entries.some(([key]) => !/^[A-Za-z_$][\w$]*$/.test(key))) {
      const types = [...new Set(entries.map(([, item]) => inferValueType(item, options)))];
      return `Record<string, ${types.length === 1 ? types[0] : types.join(' | ')}>`;
    }
    return `{ ${entries.map(([key, item]) => item === undefined ? `${key}?: unknown` : `${key}: ${inferValueType(item, options)}`).join(', ')} }`;
  }
  if (isNatlangCallable(value)) {
    throw new Reject([{ path: 'value', code: 'callable-not-invoked',
      expected: 'the portable result of the natlang callable', got: 'an uninvoked natlang callable' }]);
  }
  throw new Reject([{ path: 'value', code: 'type-mismatch', expected: 'a portable value' }]);
}

const itemRef = (container: Record<string, Value>, key: string, type: Type, env: TypeEnv, path: string, deny = ''): Ref =>
  ({ path, type, env, deny,
    get: () => Object.hasOwn(container, key) ? container[key]! : MISSING,
    set: value => { container[key] = value; },
    del: () => { delete container[key]; } });

/** The source of type alias `name` as the program's items declare it (their folders' types.ts), nearest item first. */
export function codebaseTypeAlias(codebase: Record<string, unknown>, name: string): string | undefined {
  if (!/^[A-Za-z_$][\w$]*$/.test(name)) return undefined;
  let level = Object.values(codebase as Record<string, ItemRecord>);
  for (let depth = 0; level.length && depth < 64; depth++) {
    for (const item of level) if (item.kind !== 'namespace' && Object.hasOwn(item.types ?? {}, name)) return item.types[name];
    level = level.flatMap(item => Object.values(item.codebase ?? {}));
  }
  return undefined;
}

/** Find a codebase item by dotted path; module exports resolve to their module's source. */
/** The edit_code refusal for an item that judges the call (`protect:`; WS4 §8 F7): one sentence that names the fix. */
export const protectedEditError = (source: string): string =>
  `${source} decides whether this call succeeded, so it cannot be edited here; fix the code that produces the value it rejects.`;
export const PROTECTED_TYPES_ERROR = 'types.ts declares the types this call\'s result is checked against, so it cannot be edited here; ' +
  'fix the code that produces the value they reject.';
const CONTRACT_EDIT_ERROR = (source: string): string =>
  `The ensures:, writes:, measure: and recursive: lines of ${source} judge its calls, so an edit cannot change them; fix the code that produces the value they reject.`;
const PROTECT_LIST_ERROR = (source: string): string =>
  `The protect: list of ${source} names what judges its calls, so an edit cannot change it; fix the code that produces the value they reject.`;

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

/** One structured match returned by `search_code(query)`. */
export type CodeSearchMatch = {
  name: string;
  kind: 'natlang' | 'function' | 'value' | 'namespace' | 'type' | 'service';
  signature?: string;
  description?: string;
  path?: string;
};

type CodeSearchEntry = CodeSearchMatch & { raw?: string };

const formatArgList = (args: Record<string, string>, prefixFolder = false): string => {
  const parts = Object.entries(args).map(([raw, type]) => `${raw}: ${type}`);
  if (prefixFolder) parts.unshift('folder: Folder');
  return parts.join(', ');
};

const firstDocLine = (text: string | undefined): string | undefined => {
  if (!text) return undefined;
  const cleaned = text.replace(/\s+/g, ' ').trim();
  if (!cleaned) return undefined;
  return cleaned.length > 200 ? `${cleaned.slice(0, 197)}...` : cleaned;
};

/** Extract declared member entries (functions, properties, types) and doc comments from a service declaration string. */
function indexServiceDeclaration(serviceName: string, declaration: string): CodeSearchEntry[] {
  const out: CodeSearchEntry[] = [];
  const docMatch = /\/\*\*([\s\S]*?)\*\//.exec(declaration);
  const topDoc = docMatch ? firstDocLine(docMatch[1]!.replace(/^\s*\*\s?/gm, '')) : undefined;
  out.push({ name: serviceName, kind: 'service', ...(topDoc ? { description: topDoc } : {}), raw: declaration });
  // Match individual declared functions, methods, consts, types and interfaces inside the declaration block.
  const memberPattern = /(?:\/\*\*([\s\S]*?)\*\/\s*)?(?:export\s+|declare\s+)?(?:(function)\s+([A-Za-z_$][\w$]*)\s*(<[^>{]*>)?\s*\(([^)]*)\)\s*:\s*([^\n;{]+)|(const|let|var)\s+([A-Za-z_$][\w$]*)\s*:\s*([^\n;]+)|(type)\s+([A-Za-z_$][\w$]*)\s*=\s*([^\n;]+)|(interface)\s+([A-Za-z_$][\w$]*)\s*\{([^}]*)\}|([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*:\s*([^\n;{]+))/g;
  const seen = new Set<string>();
  for (const match of declaration.matchAll(memberPattern)) {
    const doc = match[1] ? firstDocLine(match[1].replace(/^\s*\*\s?/gm, '')) : undefined;
    if (match[2] === 'function' && match[3]) {
      const name = `${serviceName}.${match[3]}`;
      if (seen.has(name)) continue; seen.add(name);
      const signature = `${match[4] ?? ''}(${(match[5] ?? '').replace(/\s+/g, ' ').trim()}) => ${(match[6] ?? '').replace(/\s+/g, ' ').trim()}`;
      out.push({ name, kind: 'service', signature, ...(doc ? { description: doc } : {}) });
    } else if (match[7] && match[8]) {
      if (match[8] === serviceName) continue;
      const name = `${serviceName}.${match[8]}`;
      if (seen.has(name)) continue; seen.add(name);
      out.push({ name, kind: 'service', signature: (match[9] ?? '').replace(/\s+/g, ' ').trim(), ...(doc ? { description: doc } : {}) });
    } else if (match[10] === 'type' && match[11]) {
      const name = `${serviceName}.${match[11]}`;
      if (seen.has(name)) continue; seen.add(name);
      out.push({ name, kind: 'type', signature: `type ${match[11]} = ${(match[12] ?? '').replace(/\s+/g, ' ').trim()}`, ...(doc ? { description: doc } : {}) });
    } else if (match[13] === 'interface' && match[14]) {
      const name = `${serviceName}.${match[14]}`;
      if (seen.has(name)) continue; seen.add(name);
      out.push({ name, kind: 'type', signature: `interface ${match[14]} { ${(match[15] ?? '').replace(/\s+/g, ' ').trim()} }`, ...(doc ? { description: doc } : {}) });
    } else if (match[16] && !['if', 'for', 'while', 'switch', 'catch', 'return'].includes(match[16])) {
      const name = `${serviceName}.${match[16]}`;
      if (seen.has(name)) continue; seen.add(name);
      const signature = `(${(match[17] ?? '').replace(/\s+/g, ' ').trim()}) => ${(match[18] ?? '').replace(/\s+/g, ' ').trim()}`;
      out.push({ name, kind: 'service', signature, ...(doc ? { description: doc } : {}) });
    }
  }
  return out;
}

/**
 * Lexical + signature search across a codebase tree, its `types.ts` aliases, and service declarations.
 * Case-insensitive tokenized substring matching ranked by relevance, capped at `limit` (default 20).
 */
export function searchCodebase(
  query: string,
  codebase: Record<string, unknown>,
  options: {
    types?: Record<string, string>;
    declarations?: Record<string, string>;
    services?: Record<string, object>;
    resolveItem?: (item: ItemRecord) => ItemRecord;
    limit?: number;
  } = {},
): CodeSearchMatch[] {
  const limit = Math.max(1, Math.min(Number.isFinite(options.limit) ? Math.floor(options.limit!) : 20, 100));
  const entries: CodeSearchEntry[] = [];
  const seenTypes = new Map<string, string>(); // alias -> definition text (deduplicates identical shared aliases)
  const addTypeAlias = (alias: string, text: string, ownerName?: string, path?: string) => {
    const existing = seenTypes.get(alias);
    if (existing === text) return;
    if (existing === undefined) {
      seenTypes.set(alias, text);
      entries.push({ name: alias, kind: 'type', signature: `type ${alias} = ${text}`, ...(path ? { path } : {}) });
    } else if (ownerName) {
      entries.push({ name: `${ownerName}.${alias}`, kind: 'type', signature: `type ${alias} = ${text}`, ...(path ? { path } : {}) });
    }
  };
  const visit = (level: Record<string, ItemRecord>, prefix: string) => {
    for (const [key, rawItem] of Object.entries(level)) {
      if (!rawItem || typeof rawItem !== 'object' || !('kind' in rawItem)) continue;
      const item = options.resolveItem ? options.resolveItem(rawItem) : rawItem;
      const fullName = prefix ? `${prefix}.${key}` : key;
      if (item.kind === 'namespace') {
        entries.push({ name: fullName, kind: 'namespace', ...(item.source ? { path: item.source } : {}) });
        visit(item.codebase ?? {}, fullName);
        continue;
      }
      for (const [alias, text] of Object.entries(item.types ?? {})) addTypeAlias(alias, text, fullName, item.source);
      if (item.kind === 'natlang') {
        const signature = `(${formatArgList(item.args, item.subtype === 'directory-reducer')}) => ${item.returns}`;
        const description = firstDocLine(item.description || item.instructions.split(/\r?\n/).find(line => line.trim()));
        entries.push({ name: fullName, kind: 'natlang', signature, ...(description ? { description } : {}), path: item.source, raw: item.instructions });
      } else if (item.kind === 'module') {
        const def = item.exports.default;
        if (def?.kind === 'function') {
          const signature = `(${formatArgList(def.args)}) => ${def.async ? `Promise<${def.returns}>` : def.returns}`;
          const description = firstDocLine(def.doc);
          entries.push({ name: fullName, kind: 'function', signature, ...(description ? { description } : {}), path: item.source, raw: item.text });
        }
        for (const [exportName, spec] of Object.entries(item.exports)) {
          if (exportName === 'default') continue;
          const memberName = `${fullName}.${exportName}`;
          if (spec.kind === 'function') {
            const signature = `(${formatArgList(spec.args)}) => ${spec.async ? `Promise<${spec.returns}>` : spec.returns}`;
            const description = firstDocLine(spec.doc);
            entries.push({ name: memberName, kind: 'function', signature, ...(description ? { description } : {}), path: item.source });
          } else {
            entries.push({ name: memberName, kind: 'value', ...(spec.type ? { signature: spec.type } : {}), path: item.source });
          }
        }
        for (const [declName, declText] of Object.entries(item.declarations ?? {})) {
          if (!seenTypes.has(declName)) {
            seenTypes.set(declName, declText);
            entries.push({ name: declName, kind: 'type', signature: firstDocLine(declText), path: item.source, raw: declText });
          }
        }
      }
      visit(item.codebase ?? {}, fullName);
    }
  };
  visit(codebase as Record<string, ItemRecord>, '');
  for (const [alias, text] of Object.entries(options.types ?? {})) addTypeAlias(alias, text);
  const declaredServices = new Set<string>();
  for (const [serviceName, declaration] of Object.entries(options.declarations ?? {})) {
    declaredServices.add(serviceName);
    entries.push(...indexServiceDeclaration(serviceName, declaration));
  }
  for (const [serviceName, value] of Object.entries(options.services ?? {})) {
    if (declaredServices.has(serviceName)) continue;
    entries.push({ name: serviceName, kind: 'service', signature: undeclaredServiceType(value, 0, serviceName) });
  }

  const trimmed = query.trim().toLowerCase();
  const toMatch = ({ raw: _raw, ...match }: CodeSearchEntry): CodeSearchMatch => match;
  if (!trimmed) {
    const kindOrder: Record<CodeSearchMatch['kind'], number> = { natlang: 0, function: 1, service: 2, type: 3, value: 4, namespace: 5 };
    return entries
      .slice()
      .sort((a, b) => kindOrder[a.kind] - kindOrder[b.kind] || a.name.localeCompare(b.name))
      .slice(0, limit)
      .map(toMatch);
  }
  const tokens = [...new Set(trimmed.split(/[\s/.,;:()<>[\]{}|&=?!]+/).filter(Boolean))];
  const scored: { entry: CodeSearchEntry; score: number }[] = [];
  for (const entry of entries) {
    const nameLower = entry.name.toLowerCase();
    const baseNameLower = nameLower.split('.').at(-1) ?? nameLower;
    const segments = nameLower.split(/[._-]+/).filter(Boolean);
    const sigLower = (entry.signature ?? '').toLowerCase();
    const descLower = (entry.description ?? '').toLowerCase();
    const pathLower = (entry.path ?? '').toLowerCase();
    const rawLower = (entry.raw ?? '').toLowerCase();
    let score = 0;
    let matchedTokens = 0;
    if (baseNameLower === trimmed || nameLower === trimmed) score += 1000;
    else if (baseNameLower.startsWith(trimmed) || nameLower.startsWith(trimmed)) score += 400;
    else if (nameLower.includes(trimmed)) score += 250;
    if (sigLower.includes(trimmed)) score += 120;
    if (descLower.includes(trimmed)) score += 80;
    for (const token of tokens) {
      let tokenScore = 0;
      if (segments.includes(token)) tokenScore += 120;
      else if (baseNameLower.includes(token)) tokenScore += 90;
      else if (nameLower.includes(token)) tokenScore += 60;
      if (sigLower.includes(token)) tokenScore += 45;
      if (descLower.includes(token)) tokenScore += 30;
      if (pathLower.includes(token)) tokenScore += 20;
      if (tokenScore === 0 && rawLower.includes(token)) tokenScore += 10;
      if (tokenScore > 0) { matchedTokens++; score += tokenScore; }
    }
    if (matchedTokens === 0) continue;
    if (matchedTokens === tokens.length) score += 300;
    else score += Math.round((matchedTokens / tokens.length) * 100);
    scored.push({ entry, score });
  }
  scored.sort((a, b) => b.score - a.score || a.entry.name.length - b.entry.name.length || a.entry.name.localeCompare(b.entry.name));
  return scored.slice(0, limit).map(item => toMatch(item.entry));
}
function replaceCodebaseItem(codebase: Record<string, unknown>, path: string[], record: ItemRecord): Record<string, unknown> {
  const [head, ...rest] = path;
  const current = (codebase as Record<string, ItemRecord>)[head!]!;
  return { ...codebase, [head!]: rest.length ? { ...current, codebase: replaceCodebaseItem(current.codebase, rest, record) } : record };
}
/** Declared return type of the selected callable or module export. */
function returnTypeOf(codebase: Record<string, unknown>, name: string, caller: TypeEnv): Type | undefined {
  const found = findCodebaseItem(codebase, name);
  if (!found || found.path.join('.') !== name.replace(/\//g, '.') && found.record.kind !== 'module') return;
  const record = found.record;
  if (record.kind === 'namespace') return;
  if (record.kind === 'natlang') return transferCallableType(record, record.returns, caller);
  const exportName = name.split('.').at(-1)!;
  const spec = record.exports[exportName === record.name ? 'default' : exportName] ?? record.exports.default;
  return spec?.kind === 'function' ? transferCallableType(record, spec.returns, caller) : undefined;
}

const callableAliases = new WeakMap<ItemRecord, Record<string, Type>>();

/** Transfer a callable's signature using its own types, rather than looking up private names in the caller. */
function transferCallableType(record: Exclude<ItemRecord, { kind: 'namespace' }>, source: string, caller: TypeEnv): Type | undefined {
  let type: Type;
  try { type = parseTypeCached(source); }
  catch (error) {
    if (!(error instanceof TypeSyntaxError)) throw error;
    return;
  }
  if (!typeAliasNames(type).length) return type;
  let aliases = callableAliases.get(record);
  if (!aliases) {
    aliases = callableTypes(record.codebase);
    for (const [alias, text] of Object.entries(record.types)) {
      try { aliases[alias] = parseTypeCached(text); }
      catch { /* TypeScript-only types stay outside portable checking. */ }
    }
    callableAliases.set(record, aliases);
  }
  const sourceAliases = aliases;
  // A name may be kept only if its entire dependency chain resolves identically. A shared
  // outer spelling can otherwise hide a conflicting, private, or caller-shadowed dependency.
  const sharedAlias = (alias: string, seen: ReadonlySet<string>): boolean => {
    if (seen.has(alias)) return true;
    const from = Object.hasOwn(sourceAliases, alias) ? sourceAliases[alias] : undefined;
    const to = caller.lookup(alias);
    if (!from || !to) return from === to;
    if (formatType(from) !== formatType(to)) return false;
    const next = new Set([...seen, alias]);
    return typeAliasNames(from).every(dependency => sharedAlias(dependency, next));
  };
  const privateAliases: Record<string, Type> = Object.create(null);
  const seen = new Set<string>();
  const collect = (signature: Type) => {
    for (const alias of typeAliasNames(signature)) {
      if (seen.has(alias)) continue;
      seen.add(alias);
      if (!Object.hasOwn(sourceAliases, alias) || sharedAlias(alias, new Set())) continue;
      privateAliases[alias] = sourceAliases[alias]!;
      collect(privateAliases[alias]);
    }
  };
  collect(type);
  const expanded = inlineDeclaredTypes(type, privateAliases);
  // An incompatible recursive alias cannot be transferred as a finite structural type.
  // Infer the observed local without borrowing its name.
  if (typeAliasNames(expanded).some(alias => !sharedAlias(alias, new Set()))) return;
  try { caller.checkNames(expanded); }
  catch (error) {
    if (!(error instanceof TypeSyntaxError)) throw error;
    return;
  }
  return expanded;
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

/** The logical name of a run started without a run ID. */
export const DEFAULT_RUN_LABEL = 'native-run';

/**
 * A run's identity: its unique root call ID and its logical name. A run without an ID gets a fresh one under the label
 * `native-run`.
 */
export function runIdentity(options: { runId?: string; runLabel?: string }): { runId: string; runLabel?: string } {
  const runLabel = options.runLabel ?? (options.runId === undefined ? DEFAULT_RUN_LABEL : undefined);
  const runId = options.runId ?? `${runLabel}:${globalThis.crypto.randomUUID()}`;
  return { runId, ...(runLabel !== undefined ? { runLabel } : {}) };
}

/** What a run's derived seeds are keyed by: its seed ID, else its logical name, else its run ID (see `runIdentity`). */
export const runSeedKey = (options: { runId: string; runLabel?: string; seedId?: string }): string =>
  options.seedId ?? options.runLabel ?? options.runId;

export class NativeRuntime {
  readonly events: HostEvent[] = [];
  readonly nativeEffects: NativeEffectOwner;
  readonly trace: NativeTraceRecorder;
  readonly options: { checkEvalBindings?: boolean; checkAwaitedFields?: boolean; maxActions?: number; maxToolCalls?: number; runId: string; runLabel?: string; seedId?: string; parentCallId?: string; returnErrorText?: ReturnErrorText };
  readonly seedPolicy: { mode: 'derived' | 'backend'; root?: number };
  readonly environment: EvalEnvironment;
  readonly agent?: NativeAgent;
  readonly episodeBudget: { limit?: number; used: number };
  frame?: Frame;
  readonly displayLiveId: (value: object) => number;
  /** Where untrusted text came from; shared by the calls of one task (native/untrusted.ts). */
  readonly untrusted: UntrustedRegistry;
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
    this.nativeEffects = options.frame?.nativeEffects ?? new NativeEffectOwner(options.settleNativeEffects !== false);
    this.displayLiveId = options.frame ? scopedLiveIdentity(options.frame.task) : createLiveIdentity();
    this.untrusted = options.frame ? scopedUntrusted(options.frame.task) : new UntrustedRegistry(hostUntrusted);
    const run = runIdentity(options);
    this.options = { checkEvalBindings: options.checkEvalBindings, checkAwaitedFields: options.checkAwaitedFields, maxActions: options.maxActions, maxToolCalls: options.maxToolCalls, runId: run.runId,
      ...(run.runLabel !== undefined ? { runLabel: run.runLabel } : {}), seedId: options.seedId,
      parentCallId: options.parentCallId, returnErrorText: options.returnErrorText };
    for (const [name, value] of Object.entries({ maxActions: this.options.maxActions, maxToolCalls: this.options.maxToolCalls }))
      if (value !== undefined && (!Number.isInteger(value) || value < 1)) throw new RangeError(`${name} must be a positive integer`);
    this.episodeBudget = options.sharedEpisodeBudget ?? { used: 0 };
    this.seedPolicy = options.seedPolicy??{mode:'derived',root:0};
    if(this.seedPolicy.mode==='derived'){this.seedPolicy={...this.seedPolicy,root:this.seedPolicy.root??0};if(!Number.isSafeInteger(this.seedPolicy.root))throw new RangeError('seed root must be a safe integer');}
    if (this.seedPolicy.mode === 'derived' && !Number.isInteger(this.seedPolicy.root))
      throw new TypeError('derived seed policy requires an integer root');
    this.trace = new NativeTraceRecorder({ run_id: this.options.runId, tool_schema: NATIVE_TOOL_SCHEMA,
      ...(options.sourceRevision ? { source_revision: options.sourceRevision } : {}),
      ...(options.parentCallId ? { parent_call_id: options.parentCallId } : {}),
      environment: { mode: options.environment.mode, authority: options.environment.authority, native_state_replayable: false },
      seed_policy: this.seedPolicy, coverage: 'natlang-state-and-observed-host-effects', ...(options.manifest ?? {}) });
    bindTraceScope(this.trace, options.sharedProducerScope ?? this.episodeBudget);
    this.exactHostTraceCapture = options.exactHostTraceCapture;
    const capture = this.exactHostTraceCapture;
    if (capture && (!Number.isSafeInteger(capture.maxBytes) || capture.maxBytes < 1 || capture.maxBytes > 8_000_000 ||
        !Array.isArray(capture.definitionSources) || (!capture.definitionSources.length && capture.captureAllOutputs !== true) ||
        (capture.captureAllOutputs !== undefined && typeof capture.captureAllOutputs !== 'boolean') ||
        capture.definitionSources.some(source => typeof source !== 'string' || !source.endsWith('.nl')) ||
        !Array.isArray(capture.inputArguments) || capture.inputArguments.some(name => typeof name !== 'string' || !name)))
      throw new RangeError('exact host trace capture requires source allowlist, argument names, and a bounded positive byte budget');
    // `offered` is emitted by NativeToolAgent only when it actually builds the opening shown to the model.
    this.hooks = options.hooks;
    // Every service call is recorded as an effect, so a failed eval can say what already happened. Services a caller
    // has already wrapped (the kernel records with its own call IDs) are used as given.
    const services = options.services ?? {};
    const observe = options.observeEffect;
    this.services = isRecording(services) ? services : recordingServices(services, ({ exact, ...event }) => {
      const persisted = observe?.({ ...event, exact });
      if(event.phase === 'requested') this.trace.emit('effect', {call_id:this.currentCallId??null,capability:`${event.service}.${event.method}`,...event});
      else graphNode(this.trace,'effect',{call_id:this.currentCallId??null,capability:`${event.service}.${event.method}`,...event},[{node:invocationNodeId(this.options.runId),port:'caller'}]);
      return persisted;
    }, undefined, options.frame ? {task:options.frame.task,callId:options.frame.parentCallId} : undefined, this.nativeEffects);
    // TS native imports and eval share this invocation's checked, journaled ports.
    // Copy the frame: callers and concurrent siblings may share the supplied one.
    this.frame = options.frame ? { ...options.frame, nativeEffects:this.nativeEffects, services: this.services } : undefined;
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
      try { aliases[name] = parseTypeCached(source); } catch { /* Keep TypeScript-only aliases outside portable checks. */ }
    }
    const env = new TypeEnv(aliases);
    env.dialects = dialectBinding(this.neuralese);
    const normalized = [...inputs];
    for (let index = 0; index < Math.min(definition.params.length, normalized.length); index++) {
      const input = normalized[index];
      if (typeof input !== 'string') continue;
      const parameter = definition.params[index]!;
      let wanted: Type;
      try { wanted = env.resolve(parseTypeCached(parameter.type)); }
      catch { continue; }
      const element = wanted.kind === 'neuralese' ? env.resolve(wanted.element) : undefined;
      if (wanted.kind !== 'neuralese' || element?.kind !== 'prim' || element.name !== 'string') continue;

      const callerInput = { node: parentCallId ? invocationNodeId(parentCallId) : 'external', port: `argument:${parameter.name}` };
      normalized[index] = await this.writeNeuraleseText(input, wanted, {
        sourceKind: 'typed-text-argument', source: 'call-argument', markerContext: 'typed-argument',
        argumentName: parameter.name, inputs: [callerInput], producerCallId: parentCallId ?? null,
        typeAliases: Object.fromEntries(Object.entries(aliases).map(([name, alias]) => [name, formatType(alias)])),
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
    typeAliases?: Record<string, string>;
    resultPath?: (string | number)[];
    argumentName?: string;
    /** The call whose latest turn produced the text (default: the current call; null: none, an outside input). */
    producerCallId?: string | null;
  }): Promise<Value> {
    const port = this.neuralese?.port;
    const type = formatType(wanted), bodySha = hexDigest(text);
    if (!port) throw new NeuraleseNoWriterError(`${metadata.sourceKind === 'typed-json-result' ? 'plain JSON' : 'plain-text'} ` +
      `${type} cannot be written`);
    // The port writes only its own dialect; a slot of another dialect cannot hold what it writes.
    const wantedDialect = wanted.kind === 'neuralese' && wanted.dialect !== DEFAULT_DIALECT ? wanted.dialect : port.dialect;
    if (wantedDialect !== port.dialect) throw new Reject([{ path: metadata.argumentName ? `argument/${metadata.argumentName}` : 'return',
      code: 'neuralese-dialect-mismatch', expected: `${type} in dialect ${JSON.stringify(wantedDialect)}`,
      got: `text, which this runtime's port writes in dialect ${JSON.stringify(port.dialect)}; pass a value already in ` +
        `${JSON.stringify(wantedDialect)}, or declare the slot as Neuralese<T> (the runtime's dialect)` }]);
    const callId = this.currentCallId ?? this.options.runId;
    const producer = { source_kind: metadata.sourceKind, source: metadata.source, marker_context: metadata.markerContext,
      call_id: callId, ...(metadata.argumentName ? { argument_type: type } : { result_type: type }), text_body_sha256: bodySha,
      ...(this.neuralese?.teacherText && metadata.typeAliases ? { type_aliases: metadata.typeAliases } : {}),
      ...(metadata.resultPath ? { result_path: metadata.resultPath } : {}),
      ...(metadata.argumentName ? { argument_name: metadata.argumentName } : {}) };
    // Written at the end of its producing call (the latest turn of the call that returned or passed the text); a value
    // with no producing call gets the writer's own producer call (DECISIONS 2026-11-25).
    const context = producingCall(metadata.producerCallId === undefined ? callId : metadata.producerCallId);
    const block = await port.write(text, { type, producer, ...(context ? { context } : {}) });
    const teacherText = isTeacherTextNeuraleseId(block.id) && block.producer?.kind === 'teacher-text-emulation';
    if (isTeacherTextNeuraleseId(block.id) && (!teacherText || !this.neuralese?.teacherText?.get(block.id)))
      throw new NeuraleseUnsupportedError('teacher-text artifact has no exact case-local body store');
    const sourceTurn = metadata.inputs.find(input => /#turn\d+$/.test(input.node))?.node ?? '';
    const writerNode = graphNode(this.trace, 'block_write', { call_id: callId, block: block.id, length: block.length, turn: sourceTurn, stops: [],
      truncated: !!block.truncated, producer: block.producer?.kind ?? null, source_kind: producer.source_kind,
      source: metadata.source, marker_context: metadata.markerContext,
      ...(metadata.argumentName ? { argument_type: type } : { result_type: type }), text_body_sha256: bodySha,
      ...(producer.result_path ? { result_path: producer.result_path } : {}),
      ...(producer.argument_name ? { argument_name: producer.argument_name } : {}),
      ...(teacherText ? { format: 'teacher-text', producer: 'teacher-text-emulation', learned_vectors: false, tokenizer_ids: false } : {}),
      ...(block.native_state ? { format: 'native-state', native_identity: {
        architecture: block.native_state.architecture, checkpoint_sha256: block.native_state.checkpoint_sha256,
        tokenizer_fingerprint: block.native_state.tokenizer_fingerprint,
        embedding_weight_sha256: block.native_state.embedding_weight_sha256 },
        source_capture: block.producer?.native_source_capture } : {}) }, metadata.inputs);
    if (teacherText && writerNode) this.neuralese?.teacherText?.recordWriter(block.id, writerNode);
    return neuraleseRef(type, block.id, false, writerNode, teacherText);
  }

  async evaluate(node: LambdaNode, code: string, scope: Record<string, unknown>, live: Record<string, unknown>, timeoutMs?: number,
    signal?: AbortSignal) {
    this.checkInterruption();
    try {
      const request = { code, body: true, path: 'eval', scope, live,
        ...(signal ? { signal } : {}), ...(timeoutMs === undefined ? {} : { timeoutMs }) };
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
      ...Object.entries(node.args).flatMap(([name, value]) => valueInputs(value, `arg:${name}`, producerScopeForTrace(this.trace))),
      ...Object.values(node.captures ?? {}).flatMap(cell => { try { return valueInputs(cell.get(), `capture:${cell.name}`, producerScopeForTrace(this.trace)); } catch { return []; } })];
    const definitionId=String(manifest.definition_id ?? node.functionName), revision=String(manifest.source_revision ?? '');
    const context=String(manifest.context_id ?? FILE_CONTEXT), signature=formatType(node.type);
    const argumentBindingSha=occurrenceArgumentHash(dump(node.args,true));
    const parentOccurrence=manifest.parent_occurrence&&typeof manifest.parent_occurrence==='object'?
      manifest.parent_occurrence as Record<string,unknown>:undefined;
    const siblingOrdinal=Number.isSafeInteger(manifest.sibling_ordinal)?Number(manifest.sibling_ordinal):0;
    const occurrence=executionOccurrence({definitionId,revision,context,signature,parent:parentOccurrence,ordinal:siblingOrdinal,argumentHash:argumentBindingSha});
    return { node: invocationNodeId(callId), inputs,
      definition: { id: definitionId, revision }, native_occurrence:occurrence,
      context, signature,
      captures: Object.fromEntries(Object.values(node.captures ?? {}).map(cell => [cell.name,
        { mode: cell.mutable ? 'live' : 'snapshot', type: cell.type, ...(cell.skill ? { skill: cell.skill } : {}) }])) };
  }

  /** Run one invocation to completion or quiescence. */
  async run(node: LambdaNode): Promise<{ outcome: NativeOutcome; value: Value }> {
    this.checkInterruption();
    this.root = node;
    // The prepared root node owns the skills selected by its program/file context. Inline `nl` functions
    // are authored inside this call, so make only those bound skill documents available on its frame.
    // Named file-backed calls still load their own companion-folder skills in invokeDefinition.
    if (this.frame && !this.frame.skillFiles && node.skills?.files) {
      const skillFiles = node.skills.files;
      if (Object.keys(skillFiles).length) this.frame = { ...this.frame, skillFiles };
    }
    const before = this.stateSummary(node);
    this.trace.emit('state', { phase: 'initial', value: before });
    this.captureExactInputs(node);
    this.lastObserved = before;
    const env = new TypeEnv(callableTypes(node.codebase)).child(node.types);
    env.classes = node.hostClasses;
    env.dialects = dialectBinding(this.neuralese);
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
  constructor(entries: readonly TranscriptEntry[]) {
    this.#entries = entries;
    declareSelectedSurface(this, 'TranscriptView', { search: 'synchronous', entry: 'synchronous',
      toString: 'synchronous', toJSON: 'synchronous' } as const, ['length']);
  }
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
  /** Program guidance written from its text, to be primed in this call once its opening exists (agent.ts). */
  guidancePrime?: { placeholder: string; text: string; settle: (id: string) => void };
  completed = false;
  /**
   * Judges the refined slots (`Is<T, P>`) of a value before the session accepts it as the result or as a local. Set by
   * the agent loop (native/agent.ts); a failure is `{ code, message }` and is reported to the model like a type error.
   */
  refinementCheck?: (value: Value, type: Type, path: string) => Promise<{ code: string; message: string } | undefined>;
  actions = 0;
  toolCalls = 0;
  readonly surfaceName = NATIVE_TOOL_SCHEMA;
  failureSerial = 0;
  failureDebug?: ScopeFailureDebug;
  private readonly scopeLocalMutability = new Map<string, boolean>();
  private readonly explainedNaturalFunctions = new Set<string>();
  private readonly originalSources = new Map<string, string>();
  /** Sources this call cannot edit (protect: lists), fixed at its first edit (functionTool). */
  private protectedList?: string[];
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
  /** The std built-ins the eval being compiled has (hooks.ts evalDeclarations types them from `@natlang/node/std`). */
  evalStdBindings: readonly string[] = [];
  /** The call's type environment; a recursive type an eval declares joins it (bindLocalTypeNames). */
  env: TypeEnv;
  constructor(readonly runtime: NativeRuntime, readonly lam: LambdaNode, env: TypeEnv) { this.env = env; this.markUntrusted(); }

  /**
   * An eval's types are inlined into a local's type where it is used; a recursive one (`type Node = { kids: Node[] }`)
   * cannot be, so the names a type still refers to join this call's type environment, with the local types they reach.
   */
  private bindLocalTypeNames(type: Type, localTypes: Record<string, Type>): void {
    const pending = typeAliasNames(type), add: Record<string, Type> = {};
    while (pending.length) {
      const name = pending.pop()!;
      if (Object.hasOwn(add, name) || !Object.hasOwn(localTypes, name) || this.env.lookup(name)) continue;
      add[name] = localTypes[name]!;
      pending.push(...typeAliasNames(localTypes[name]!));
    }
    if (Object.keys(add).length) this.env = this.env.child(add);
  }

  /** The standard library eval's std built-ins come from: the platform's instance, unless NATLANG_EVAL_STD turns them off. */
  private stdLibrary(): Record<string, unknown> | undefined {
    return evalStdEnabled() ? stdModule(this.runtime.frame?.task.runtime.platform) : undefined;
  }
  /** The std combinators this call offers in eval: all of them, the workspace ones where the call has a folder. */
  private stdCombinators(): string[] {
    const library = this.stdLibrary();
    if (!library) return [];
    return [...EVAL_STD_COMBINATORS, ...(this.lam.projectTransaction ? EVAL_STD_FOLDER_COMBINATORS : [])]
      .filter(name => typeof library[name] === 'function');
  }
  /** The std combinators an eval of `source` has: those no input, local, function, service, capture or declaration of it takes. */
  private evalStdNames(source: string): string[] {
    const inputs = this.lam.type.kind === 'lambda' ? this.lam.type.params.fields.map(field => field.name) : [];
    const services = this.availableServices(), captures = this.lam.captures ?? {};
    const declared = evalDeclaredNames(source);
    return this.stdCombinators().filter(name => !inputs.includes(name) && !Object.hasOwn(this.lam.let, name) &&
      !Object.hasOwn(this.lam.codebase, name) && !Object.hasOwn(services, name) && !Object.hasOwn(captures, name) &&
      !this.persistentScopeHelpers.has(name) && !declared.has(name));
  }

  executionOccurrence(): import('../contracts.js').NativeExecutionOccurrence | undefined {
    const event=this.runtime.trace.events.find(item=>item.kind==='invocation'&&item.phase==='start'&&
      item.call_id===this.runtime.options.runId);
    const occurrence=event?.native_occurrence;
    return occurrence&&typeof occurrence==='object'?structuredClone(occurrence) as import('../contracts.js').NativeExecutionOccurrence:undefined;
  }

  /**
   * Records where the strings of this call's `Untrusted<...>` arguments and captured variables came from, so the renderer
   * shows them as data (native/untrusted.ts). A text already known keeps the source that first named it (a service).
   */
  markUntrusted(): void {
    const lam = this.lam, registry = this.runtime.untrusted;
    if (lam.type.kind === 'lambda') for (const field of lam.type.params.fields) {
      if (Object.hasOwn(lam.args, field.name)) registry.markTyped(lam.args[field.name], field.type, this.env,
        `argument ${field.name}${lam.functionName ? ` of ${lam.functionName}` : ''}`);
    }
    for (const cell of Object.values(lam.captures ?? {})) {
      try {
        const type = parseTypeCached(cell.type);
        if (containsUntrusted(type, this.env)) registry.markTyped(cell.get(), type, this.env, `variable ${cell.name}`);
      } catch { /* an unreadable capture has nothing to mark */ }
    }
  }

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
  private readonly analysisCallableTypes = new WeakMap<ItemRecord, Map<string, string>>();

  /** Use the same lexical signature as runtime local inference in the static eval checker. */
  analysisCallableType(record: Exclude<ItemRecord, { kind: 'namespace' }>, source: string): string {
    let types = this.analysisCallableTypes.get(record);
    if (!types) { types = new Map(); this.analysisCallableTypes.set(record, types); }
    const known = types.get(source);
    if (known !== undefined) return known;
    const transferred = transferCallableType(record, source, this.env);
    const text = transferred ? formatType(transferred) : source;
    types.set(source, text);
    return text;
  }

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
    // The same code failing the same way again: the model is retrying without a change (it would retry forever).
    const repeated = this.failureDebug?.code === code.slice(0, 16000) && this.failureDebug?.message === message;
    this.failureDebug = { version: 'natlang.scope_failure/1', serial: ++this.failureSerial,
      kind, message, code: code.slice(0, 16000), scope, trace, diagnostics,
      logs: logs.slice(0, 32).map(line => line.slice(0, 2000)),
      ...(sourceStack ? { stack: sourceStack.split('\n').slice(0, 12).join('\n').slice(0, 4000) } : {}) };
    this.runtime.trace.emit('scope_failure', { failure_kind: kind, serial: this.failureSerial,
      message: message.slice(0, 1200), diagnostic_count: diagnostics.length });
    // Report what the model needs to repair the eval: its console output and any effect that actually completed.
    const effectEvents = traceMark === undefined ? [] : this.runtime.trace.events.slice(traceMark)
      .filter(event => event.kind === 'effect' || event.kind === 'host');
    const completedEffectEvents = effectEvents.filter(event =>
      event.kind === 'effect' ? event.phase === 'completed' :
        event.phase === undefined || event.phase === 'completed');
    const effects = completedEffectEvents
      .map(event => String(event.capability ?? event.operation ?? event.kind));
    const receipts = effectEvents.filter(event => event.kind === 'effect' && event.phase === 'completed').slice(-8)
      .map(event => `${String(event.capability ?? event.operation ?? 'service call')} completed: ${this.show(typeof event.result === 'string' ? event.result : JSON.stringify(event.result) ?? 'null')}`);
    this.evalDetail = { console: logs.join('\n') };
    // A model reaching for something the scope does not have (a name, a Node module, eval) is shown what it does have,
    // or it goes on probing the environment.
    // The checker's unbound-name rejection (before anything runs) reaches as a runtime ReferenceError does.
    const allText = message + diagnostics.map(item => String(item.message ?? '')).join('\n');
    const reaching = /is not defined|unavailable in eval|is not a function|Cannot find name|is not bound in this call's eval scope/.test(allText);
    const unbound = /\b(\w+) is not bound in this call's eval scope/.exec(allText)?.[1];
    // A tool named in eval code: the tools are called as tools, next to eval, not from inside it.
    const tool = /\b(\w+) is not defined/.exec(message)?.[1] ??
      (unbound && (NATIVE_TOOLS.includes(unbound) || unbound === this.lam.functionName) ? unbound : undefined);
    const toolNote = tool && NATIVE_TOOLS.includes(tool) && tool !== 'return_result' ?
      `\n${tool} is one of your tools: call it as a tool, not from eval code.` : '';
    // The call's own name is not a function in its scope: the model is executing it (it tries to call itself).
    const bindingNote = tool && tool === this.lam.functionName ?
      `\n${tool} is the call you are executing, not a function in its scope: ${SELF_CALL_ADVICE}` :
      tool && !NATIVE_TOOLS.includes(tool) ?
      `\n${tool} must be bound in this eval's scope before you use it. Mentioning it in instructions does not declare it.` +
      // Only an interpolated template can be the cause of the template advice.
      (code.includes('${') ? ' JavaScript template interpolation runs when the template is created: create an item-specific template inside the callback or loop that binds the item, ' +
        'or mention a future child parameter by name in literal instructions and pass its value when calling the child.' : '') : '';
    // An nl call asking this call's own question again (kernel SAME_QUESTION_REFUSAL): LFM3 read the refusal as a
    // failed child and retried it in new spellings 6-15 times (triage suite, 2026-11-13), so it says what this call is.
    const sameQuestionNote = message.includes(SAME_QUESTION_REFUSAL) ?
      `\nThat question is this call's own: you are the call that answers it. Any nl call with these instructions and inputs is refused the same way. ${SELF_CALL_ADVICE}` : '';
    const repeatNote = repeated ? '\nThis is the code of your previous eval, and it failed the same way: change it before running it again.' : '';
    return toolNote + bindingNote + sameQuestionNote + repeatNote + (reaching ? `\n${this.scopeGuide()}` : '') + (logs.length ? `\nconsole:\n${this.show(logs.join('\n'))}` : '') +
      (effects.length ? `\nAlready performed before the failure (not undone): ${[...new Set(effects)].join(', ')}.` : '') +
      (receipts.length ? `\nCompleted service calls succeeded:\n${receipts.join('\n')}\nRepair subsequent work without repeating an already completed write.` : '') +
      // Emitted 1,179 times in 512 held-v2 episodes (2026-11-13): one sentence, the prefix kept for the collectors.
      '\nNothing else from this eval was kept; writes to files, services or live values stay done, so check them before writing again.';
  }

  /** The runtime or the bound evaluation policy denies code edits. */
  codeEditsDenied(): boolean {
    return this.runtime.frame?.task.runtime.options.codeEdits === 'deny' ||
      this.runtime.frame?.task.programView?.binding?.artifact.policy.codeEdits === 'deny';
  }

  /** The call can change code: its program has functions of its own and code edits are not denied (edit_code, diff_code). */
  codeEditable(): boolean {
    return Object.keys(this.lam.codebase).length > 0 && !this.codeEditsDenied();
  }

  /** Whether `name` is a parameter of this call declared Neuralese<T>: its state is in the call's opening message. */
  neuraleseParameter(name: string): boolean {
    if (this.lam.type.kind !== 'lambda') return false;
    const field = this.lam.type.params.fields.find(item => item.name === name);
    return !!field && this.env.resolve(field.type).kind === 'neuralese';
  }

  /** Whether this call's result is text (a string or Neuralese<string>, with nothing staged). */
  expectsTextResult(): boolean {
    if (this.lam.type.kind !== 'lambda' || this.lam.return !== MISSING) return false;
    const resolved = this.env.resolve(this.lam.type.returns);
    const element = resolved.kind === 'neuralese' ? this.env.resolve(resolved.element) : resolved;
    return element.kind === 'prim' && element.name === 'string';
  }

  /**
   * What is wrong with the files this call wrote under its definition's `writes:` type (runtime/writes.ts), in one
   * sentence; undefined when they fit or nothing is declared.
   */
  writesProblem(): string | undefined {
    const tx = this.lam.projectTransaction;
    if (!this.lam.writes || !tx?.open) return undefined;
    this.writesChecked ??= writesFields(this.lam.writes, this.env);
    return writesProblem(this.writesChecked, this.env, this.lam.functionName || 'this call', tx.folder.diffSync().changes);
  }
  /** Asynchronous `writes:` check: enforces both structural file contracts and `Is<string, P>` file refinements. */
  async writesProblemAsync(): Promise<string | undefined> {
    const structural = this.writesProblem();
    if (structural) return structural;
    const tx = this.lam.projectTransaction;
    if (!this.lam.writes || !tx?.open || !this.refinementCheck || !this.writesChecked) return undefined;
    const checks = writesRefinementChecks(this.writesChecked, this.env, tx.folder.diffSync().changes);
    for (const check of checks) {
      const failure = await this.refinementCheck(check.text, check.type, `files/${check.path}`);
      if (failure) return `${check.path} failed ${check.pattern} (${failure.code}): ${failure.message}`;
    }
    return undefined;
  }
  private writesChecked?: WritesField[];

  /** Complete the invocation if an eval has returned a value of the declared type. */
  finish(): boolean {
    if (this.lam.return === MISSING || this.lam.type.kind !== 'lambda') return false;
    if (problems(this.lam.return, this.lam.type.returns, this.env, 'return').holes.length) return false;
    if (this.writesProblem()) return false;
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
    const shown = renderValue(value, { root, holder, budget: this.pages.budget, liveIdentity: this.runtime.displayLiveId, untrusted: this.runtime.untrusted });
    // Preserve the existing page route for modest portable values, proven small by a bounded
    // estimator. Large/cyclic values remain intact in eval state but are not fully serialized.
    if (!portableSizeAtMost(value, TRANSCRIPT_VALUE_CHARS) || containsLive(value)) return shown;
    const full = renderValue(value, { root, budget: Infinity, liveIdentity: this.runtime.displayLiveId, untrusted: this.runtime.untrusted });
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
    const cutOff = this.cuts.filter(cut => result.text.includes(cut.shown)).map(cut => cut.full);
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
    return { ...result, entry, ...(cutOff.length ? { cutOff } : {}) };
  }
  private actionLimitReached(): boolean {
    return (this.runtime.options.maxActions !== undefined && this.actions >= this.runtime.options.maxActions) ||
      (this.runtime.options.maxToolCalls !== undefined && this.toolCalls >= this.runtime.options.maxToolCalls);
  }

  /** Record an agent-owned control action under the same action and transcript custody as other tools. */
  hostAction(name: string, args: Record<string, unknown>, execute: () => NativeResult, toolCallId?: string): NativeResult {
    this.runtime.checkInterruption();
    if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' }, toolCallId);
    if (this.actionLimitReached()) return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' }, toolCallId);
    this.toolCalls++; this.actions++; this.lam.steps++;
    return this.record(name, args, execute(), toolCallId);
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
    const reencoded = sourceEvent ? this.runtime.trace.events.filter(event => event.kind === 'literal' &&
      event.source_kind === 'cross-model-output-source-text' && event.turn === sourceEvent.node &&
      event.text_sha256 === hexDigest(text) && typeof event.node === 'string') : [];
    const ref = await this.runtime.writeNeuraleseText(text, wanted, { sourceKind, source, markerContext: 'return-result',
      inputs: [input, ...reencoded.map(event => ({ node: event.node as string, port: 'reencoded-source' }))], typeAliases: typeAliasesFromEnv(this.env),
      ...(resultPath ? { resultPath: ['return', ...resultPath] } : {}) });
    return coerce(ref, wanted, this.env, 'return');
  }

  /** Materialize plain text or validated JSON only when the declared Neuralese element type makes that representation exact. */
  private async coerceReturn(value: unknown, source: 'return_result' | 'eval-return' | 'eval-finish',
    parseJsonFallback = false): Promise<Value> {
    const wanted = this.env.resolve(this.lam.type.kind === 'lambda' ? this.lam.type.returns : parseTypeCached('null'));
    const element = wanted.kind === 'neuralese' ? this.env.resolve(wanted.element) : undefined;
    if (typeof value === 'string' && wanted.kind === 'neuralese' && element?.kind === 'prim' && element.name === 'string') {
      try { return coerce(value, wanted, this.env, 'return'); }
      catch (error) { if (!(error instanceof Reject)) throw error; }
      if (parseJsonFallback) try { return coerce(JSON.parse(value), wanted, this.env, 'return'); }
      catch (error) { if (!(error instanceof Reject) && !(error instanceof SyntaxError)) throw error; }
      return this.writeNeuraleseResult(value, wanted, source, 'typed-text-result');
    }
    const resultType = this.lam.type.kind === 'lambda' ? this.lam.type.returns : parseTypeCached('null');
    // Final results (`return_result` and `eval` with `finish: true`) must match the declared record fields strictly;
    // intermediate locals keep extra fields so provider/callee values can be inspected and projected cleanly.
    const preserveRecordExtras = source === 'eval-return';
    try { return coerce(value, resultType, this.env, 'return', preserveRecordExtras ? { preserveRecordExtras: true } : {}); }
    catch (first) {
      if (!(first instanceof Reject)) throw first;
      if (wanted.kind === 'neuralese' && element && !(element.kind === 'prim' && element.name === 'string') &&
          jsonSerializableType(element, this.env)) {
        let typedValue: Value;
        try { typedValue = coerceCompleteJson(value, element, this.env, 'return'); }
        catch (directError) {
          if (!(directError instanceof Reject)) throw directError;
          if (!parseJsonFallback || typeof value !== 'string') throw directError;
          const parsed = literalText(value);
          if (!parsed) throw directError;
          typedValue = coerceCompleteJson(parsed.value, element, this.env, 'return');
        }
        const text = typedJsonSource(typedValue);
        if (text === undefined) throw first;
        return this.writeNeuraleseResult(text, wanted, source, 'typed-json-result');
      }
      // A structured result sent as its JSON or JS-literal text is read as the data it spells out (never run).
      const spelled = parseJsonFallback && typeof value === 'string' ? literalText(value) : undefined;
      const candidate = spelled ? spelled.value : value;
      let second: unknown;
      try { return coerce(candidate, resultType, this.env, 'return'); }
      catch (error) {
        second = error;
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
      const shape = parseJsonFallback && typeof value === 'string' ? structuredShape(resultType, this.env) : undefined;
      if (shape) {
        // Text that spells out an object or array: what is wrong inside it is the fix (the text itself is accepted).
        if (spelled && second instanceof Reject && spelled.value !== null && typeof spelled.value === 'object') throw second;
        throw new Reject([{ path: 'value', code: 'type-mismatch', text:
          `return_result needs the ${formatType(resultType)} itself as its value argument, written as ${shape} in the arguments rather than as text.` }]);
      }
      throw first;
    }
  }

  /** Apply one tool call. */
  async applyAsync(name: string, args: Record<string, unknown>, toolCallId?: string): Promise<NativeResult> {
    this.runtime.checkInterruption();
    const record = (result: NativeResult) => this.record(name, args, result, toolCallId);
    if (this.completed) return record({ kind: 'error', text: 'the task has already finished' });
    if (!NATIVE_TOOLS.includes(name))
      return record(rejected(new Reject([this.notATool(name, args)])));
    if (this.lam.withheldTools?.includes(name))
      return record(rejected(new Reject([{ path: name, code: 'withheld-tool', text:
        `${name} is withheld from calls of ${this.lam.functionName || 'this function'} by its definition; do this step with the tools this call offers` }])));
    if (this.actionLimitReached()) return record({ kind: 'budget', text: 'action or tool-call budget exhausted' });
    this.toolCalls++;
    this.actions++; this.lam.steps++;
    try {
      if (name === 'eval') {
        const timeout = args.timeout_ms;
        if (timeout !== undefined && (!Number.isInteger(timeout) || (timeout as number) < 1))
          throw new Reject([{ path: 'timeout_ms', code: 'bad-action', expected: 'a positive whole number of milliseconds' }]);
        if (args.finish !== undefined && typeof args.finish !== 'boolean')
          throw new Reject([{ path: 'finish', code: 'bad-action', expected: "a boolean: true completes this eval's fresh typed result" }]);
        return record(await this.evaluate(String(args.code ?? ''), timeout as number | undefined, args.finish === true, toolCallId));
      }
      if (name === 'return_result' && (args.status ?? 'success') === 'success' && this.lam.type.kind === 'lambda') {
        if (Object.hasOwn(args, 'value')) {
          // A value that is one `${...}` placeholder would be kept as those characters: return_result fills nothing in
          // (held-v3 after sft-v5: natlang.read calls returned the text `${value}` 23 of 68 times).
          if (typeof args.value === 'string' && PLACEHOLDER_VALUE.test(args.value.trim()))
            throw new Reject([{ path: 'value', code: 'bad-action', expected: PLACEHOLDER_ADVICE }]);
          const value = await this.coerceReturn(args.value, 'return_result', true);
          const unwritten = await this.writesProblemAsync();
          if (unwritten) throw new Reject([{ path: 'files', code: 'writes-unsatisfied', text: `writes-unsatisfied: ${unwritten}` }]);
          return record(this.scopeTool(name, { ...args, value }));
        }
        const unwritten = await this.writesProblemAsync();
        if (unwritten) throw new Reject([{ path: 'files', code: 'writes-unsatisfied', text: `writes-unsatisfied: ${unwritten}` }]);
        return record(this.scopeTool(name, args));
      }
      if (CODE_TOOLS.includes(name)) return record(this.functionTool(name, args));
      if (name === 'bash') {
        const folder = this.lam.projectTransaction?.folder;
        if (!folder) throw new Reject([{ path: name, code: 'bad-action', expected: 'a folder call' }]);
        const runFolderBash = this.folderTool('runFolderBash', name);
        const { invokeDefinition } = await import('../runtime/kernel.js');
        const value = await runFolderBash(folder, String(args.command ?? ''),
          { network: this.runtime.environment.scopeCapabilities?.allowNetwork !== false,
            python: await this.pythonHost(),
            // model-call-kind: runtime-authored-call (bash ask) -- training/model-call-kinds.json
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
        return record({ kind: value.exitCode === 0 ? 'ok' : 'error',
          text: this.show(JSON.stringify(value)), value: value as unknown as Value });
      }
      if (name === 'python') {
        const folder = this.lam.projectTransaction?.folder;
        if (!folder) throw new Reject([{ path: name, code: 'bad-action', expected: 'a folder call' }]);
        const timeout = args.timeout_ms === undefined ? undefined : Number(args.timeout_ms);
        if (timeout !== undefined && (!Number.isSafeInteger(timeout) || timeout < 1))
          throw new Reject([{ path: 'timeout_ms', code: 'bad-action', expected: 'a positive timeout in milliseconds' }]);
        const runFolderPython = this.folderTool('runFolderPython', name);
        const value = await runFolderPython(folder, String(args.code ?? ''), await this.pythonHost(), timeout);
        return record({ kind: 'ok', text: this.show(JSON.stringify(value)), value: value as unknown as Value });
      }
      if (name === 'delegate') {
        const folder = this.lam.projectTransaction?.folder;
        if (!folder) throw new Reject([{ path: name, code: 'bad-action', expected: 'a folder call' }]);
        const path = String(args.path ?? ''), instructions = String(args.instructions ?? '');
        if (!instructions.trim()) throw new Reject([{ path: 'instructions', code: 'bad-action', expected: 'nonempty child instructions' }]);
        const handle = folder.dir(path), transaction = await handle.beginTransaction(true);
        const { invokeDefinition } = await import('../runtime/kernel.js');
        // model-call-kind: runtime-authored-call (delegate tool) -- training/model-call-kinds.json
        const value = await invokeDefinition(this.runtime.frame!, {
          id: `delegate:${this.runtime.options.runId}:${this.lam.steps}`, name: 'delegate',
          body: instructions, params: [], returns: String(args.returns ?? 'unknown'),
          types: this.lam.typesSrc, codebase: this.lam.codebase, subtype: 'directory-reducer',
        }, [handle], { folder: { transaction, mode: 'apply' }, manifest: { delegate: true, path } });
        return record({ kind: 'ok', text: this.show(JSON.stringify(value) ?? 'null'), value: value as Value });
      }
      if (name === 'editor') return record(await this.editorTool(args));
      if (['list_files', 'search_files', 'read_file', 'write_file', 'edit_file', 'diff_files'].includes(name))
        return record(await this.fileTool(name, args));
      return record(this.scopeTool(name, args));
    } catch (error) {
      // No action can repair a runtime without a Neuralese writer: the run ends rather than the model retrying.
      if (isNoNeuraleseWriterFault(error)) throw error;
      if (error instanceof Reject) return record(rejected(error));
      return record({ kind: 'error', text: error instanceof Error ? error.message : String(error) });
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
  /** A folder tool the platform provides (Node and browsers do; the workerd core does not), or why the call cannot have it. */
  private folderTool<K extends 'runFolderBash' | 'runFolderPython'>(tool: K, name: string): NonNullable<Platform[K]> {
    const run = this.runtime.frame?.task.runtime.platform[tool];
    if (!run) throw new Reject([{ path: name, code: 'bad-action', expected: `a tool this platform provides (${name} is not available on it)` }]);
    return run as NonNullable<Platform[K]>;
  }
  private async pythonHost(): Promise<PythonHost> {
    const { invokeDefinition } = await import('../runtime/kernel.js');
    const { iterateOn } = await import('../runtime/iterate.js');
    return {
      // model-call-kind: runtime-authored-call (nl@python) -- training/model-call-kinds.json
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
      const wording = returnErrorText(this.runtime.options.returnErrorText);
      if (!Object.hasOwn(args, 'value')) {
        // A result sent under another argument name (e.g. { status, verdict: {...} }) or none at all, nothing staged.
        const misplaced = Object.keys(args).filter(key => !['status', 'value', 'reason'].includes(key));
        if (this.lam.return === MISSING) throw new Reject([returnValueRejection(wording,
          misplaced.length ? { kind: 'misplaced', fields: misplaced } : { kind: 'missing' }, formatType(this.lam.type.returns))]);
        // Reuse the staged object as-is. finish() below still checks completeness, live holes,
        // pending transactions, and the normal terminal validation path.
        value = this.lam.return;
      } else {
        try { value = coerce(args.value, this.lam.type.returns, this.env, 'return'); }
        catch (first) {
          // Some models send structured values as JSON or JS-literal text.
          const spelled = typeof args.value === 'string' ? literalText(args.value) : undefined;
          if (!spelled) throw first;
          try { value = coerce(spelled.value, this.lam.type.returns, this.env, 'return'); } catch { throw first; }
        }
      }
      // The files a directory reducer declares it writes (`writes:`) hold it to them before its result is taken.
      const unwritten = this.writesProblem();
      if (unwritten) throw new Reject([{ path: 'files', code: 'writes-unsatisfied', text: `writes-unsatisfied: ${unwritten}` }]);
      this.lam.return = value;
      if (!this.finish()) throw new Reject([returnValueRejection(wording, { kind: 'incomplete' }, formatType(this.lam.type.returns))]);
      return { kind: 'completed', text: `Returned ${oneLine(value, undefined, this.runtime.displayLiveId, this.runtime.untrusted)}.`, value };
    }
    throw new Reject([{ path: name, code: 'bad-action', expected: 'a scope-eval tool' }]);
  }

  /**
   * A tool call whose name is something in the eval scope (transcript.search, an input, a function of this call, a
   * service): the base LFM3 sent transcript.search/transcript.entry/read_function as tool calls in 46 of 512 held-v2
   * episodes and got only "expected a scope-eval tool". Name the eval form instead.
   */
  private notATool(name: string, args: Record<string, unknown>): { path: string; code: string; text?: string; expected?: string } {
    const root = name.split('.')[0]!;
    const hasCodebase = Object.keys(this.lam.codebase).length > 0;
    const builtins = ['transcript', 'nl', 'iterateOn', 'decide', 'read', 'refine', 'assume', 'NatlangContext', ...(hasCodebase ? ['search_code'] : []),
      ...this.stdCombinators()];
    const inputs = this.lam.type.kind === 'lambda' ? this.lam.type.params.fields.map(field => field.name) : [];
    const scope = new Set([...builtins, ...inputs, ...Object.keys(this.lam.captures ?? {}), ...Object.keys(this.lam.let),
      ...this.persistentScopeHelpers.keys(), ...Object.keys(this.lam.codebase), ...Object.keys(this.availableServices())]);
    if (!/^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*$/.test(name) || !scope.has(root))
      return { path: name, code: 'bad-action', expected: 'a scope-eval tool' };
    const first = Object.values(args).find(value => typeof value === 'string' || typeof value === 'number');
    const shown = first === undefined ? '' : JSON.stringify(first);
    // Built-ins that return no promise (escalate and withContext return a function).
    const unawaited = ['transcript', 'NatlangContext', 'search_code', 'assume', 'escalate', 'withContext'];
    const awaited = builtins.includes(root) && !unawaited.includes(root) || Object.hasOwn(this.lam.codebase, root) ||
      this.persistentScopeHelpers.has(root);
    const value = !awaited && !unawaited.includes(root) && !Object.hasOwn(this.availableServices(), root);
    const example = name === 'transcript' ? 'transcript.search("…")' : value && name === root ? name :
      `${awaited ? 'await ' : ''}${name}(${shown})`;
    return { path: name, code: 'bad-action', text: `${name}: bad-action, ${name} is not a tool: ${root} is in this call's eval scope. ` +
      `Use it in code, for example eval with code: ${example}${builtins.includes(root) ? `; read_code("${root}") shows how to use it` : ''}.` };
  }

  /** Lexical + signature search across this call's codebase, type aliases, and service declarations. */
  searchCode(query: string, limit?: number): CodeSearchMatch[] {
    const view = this.runtime.frame?.task.programView;
    const types: Record<string, string> = {
      ...(this.lam.typesSrc ?? {}),
      ...Object.fromEntries(Object.entries(this.localTypes).map(([name, type]) => [name, formatType(type)])),
    };
    return searchCodebase(query, this.lam.codebase, {
      types,
      declarations: this.availableServiceDeclarations(),
      services: this.availableServices(),
      resolveItem: view ? item => view.record(item) : undefined,
      limit,
    });
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
    const builtIns = [
      ...(canGenerateNl(this.runtime.frame) ? ['nl'] : []),
      'read', 'iterateOn', 'decide', 'refine', 'assume', 'transcript', 'NatlangContext',
      ...(Object.keys(this.lam.codebase).length > 0 ? ['search_code'] : []),
    ];
    const builtInList = `${builtIns.slice(0, -1).join(', ')} and ${builtIns.at(-1)}`;
    const std = this.stdCombinators().filter(name => !names.includes(name));
    const stdList = std.length ? `, the semantic combinators ${std.slice(0, -1).join(', ')} and ${std.at(-1)}` : '';
    return `This call's eval scope has ${names.length ? names.join(', ') : 'no names of its own'}, the built-ins ${builtInList}${stdList} ` +
      '(read_code shows how to use them), and standard JavaScript; nothing else (no Node modules, no require).';
  }

  availableServices(): Record<string, object> {
    // This session owns its invocation's authority even when a host callback has
    // no ambient frame, or happens to run inside an unrelated invocation.
    const frame = this.runtime.frame;
    const chain = frame?.serviceChain ?? frame?.chain ?? [];
    return Object.fromEntries(Object.entries(this.runtime.services).filter(([name]) => {
      const scope = this.runtime.serviceScopes[name];
      return !scope || scope.some(path => path.endsWith('/**')
        ? chain.includes(`nl:${path.slice(0,-3)}`) || chain.includes(`ts:${path.slice(0,-3)}`)
        : chain.at(-1)===`nl:${path}` || chain.at(-1)===`ts:${path}`);
    }).map(([name, service]) => [name, name === 'neuralese' && service && typeof service === 'object' ?
      Object.create(Object.getPrototypeOf(service), {
        // Descriptors, not a spread: a getter stays lazy (it is never read for a declaration) and prototype methods stay.
        ...Object.getOwnPropertyDescriptors(service),
        read: { value: (value: unknown) => readNeuraleseForCurrentTask(value), enumerable: true, configurable: true, writable: false } }) : service]));
  }

  /** Types and symbol lookup expose precisely the ports admitted to this invocation. */
  availableServiceDeclarations(): Record<string, string> {
    const services = this.availableServices();
    return Object.fromEntries(Object.entries(this.runtime.declarations).filter(([name]) => Object.hasOwn(services, name)));
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
    const stdDoc = this.stdCombinators().includes(requested) ? STD_BUILT_IN_DOCS[requested] : undefined;
    if (Object.hasOwn(this.availableServiceDeclarations(), service) && !findCodebaseItem(this.lam.codebase, requested) &&
        !Object.hasOwn(BUILT_IN_DOCS, requested) && stdDoc === undefined) {
      if (name === 'edit_code') throw new Reject([{ path: requested, code: 'external', expected:
        `a function of this program; ${service} is an external service: its declaration can be read, but it runs outside this program and cannot be changed` }]);
      const declaration = this.runtime.declarations[service]!;
      return { kind: 'ok', text: `${declaration}\n// ${service} is an external service: this declaration is all of it there is to read.`, value: declaration };
    }
    const found = findCodebaseItem(this.lam.codebase, requested);
    // A built-in of eval is documented, not defined, here: read shows its documentation; it cannot be edited.
    if (!found && stdDoc !== undefined) {
      if (name === 'edit_code') throw new Reject([{ path: requested, code: 'built-in', expected:
        `a function of this program's codebase; ${requested} is built into eval and cannot be changed` }]);
      return { kind: 'ok', text: stdDoc, value: stdDoc };
    }
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
    // The program's types.ts is read (its aliases above) but never a helper to change: the result is checked against it.
    if (!found && name === 'edit_code' && /^(?:\.\/)?(?:[\w$-]+\/)*types(?:\.ts)?$/.test(requested))
      throw new Reject([{ path: requested, code: 'protected', text: PROTECTED_TYPES_ERROR }]);
    // edit_code is offered only where codeEditable() holds (agent.ts toolsScope); a call that reaches this tool without
    // code of its own (a scripted or replayed action) has nothing to edit.
    if (!found && name === 'edit_code' && !Object.keys(this.lam.codebase).length) throw new Reject([{ path: requested, code: 'no-own-code',
      expected: 'a function in this call\'s callable folder; this function has none, so there is no code of its own to edit' }]);
    if (!found) {
      // The call's own name: LFM3 read it first in 278 of 1,190 held-v2 invocations and got a no-such-function error
      // (2026-11-13). Its source is the instructions it is executing, so that is what reading it shows.
      if (name === 'read_code' && this.lam.functionName && requested === this.lam.functionName) {
        const body = (this.lam.originalBody ?? this.lam.body).replace(/^\n+|\n+$/g, '');
        const text = `${requested} is the call you are executing. Its instructions, as in the opening message:\n${body}\n\n${SELF_CALL_ADVICE}`;
        return { kind: 'ok', text, value: body };
      }
      const toolDocumentation = name === 'read_code' ? this.toolDocumentation(requested) : undefined;
      if (toolDocumentation !== undefined) return { kind: 'ok', text: toolDocumentation, value: toolDocumentation };
      // A large scope shows entry signatures first. Their type names can collide across
      // branches, so a qualified lookup uses the exact declaring callable's aliases.
      const separator = requested.lastIndexOf('.');
      if (name === 'read_code' && separator > 0) {
        const parent = findCodebaseItem(this.lam.codebase, requested.slice(0, separator))?.record;
        const alias = requested.slice(separator + 1);
        const record = parent && (this.runtime.frame?.task.programView?.record(parent) ?? parent);
        if (record && record.kind !== 'namespace' && Object.hasOwn(record.types, alias)) {
          const declaration = (record.kind === 'module' ? record.declarations?.[alias] : undefined) ??
            aliasDeclaration(alias, record.types[alias]!);
          return { kind: 'ok', text: declaration, value: declaration };
        }
      }
      const availableServices = this.availableServices();
      if (name === 'read_code' && Object.hasOwn(availableServices, requested)) {
        const declaration = `Host service object declaration (type only, not implementation or field values):\n` +
          `declare const ${requested}: ${undeclaredServiceType(availableServices[requested], 0, requested)};\n\n` +
          `These members are available to this call under its current service scope. Inspect the value in eval when needed.`;
        return { kind: 'ok', text: declaration, value: declaration };
      }
      // A type alias of this call (`Schedule`) is shown as its definition.
      if (name === 'read_code' && Object.hasOwn(this.lam.typesSrc ?? {}, requested)) {
        const alias = aliasDeclaration(requested, this.lam.typesSrc[requested]!);
        return { kind: 'ok', text: alias, value: alias };
      }
      // A type alias of the program in the callable folder (its types.ts), which this call's own scope need not have:
      // read_code("Money") answered no-such-function and the declaration was nowhere to read (teacher-gen gap-v1
      // field-type tasks ended blocked or guessed).
      const programAlias = name === 'read_code' ? codebaseTypeAlias(this.lam.codebase, requested) : undefined;
      if (programAlias !== undefined) {
        const alias = aliasDeclaration(requested, programAlias);
        return { kind: 'ok', text: alias, value: alias };
      }
      // A bare method name of a scope service (`certificate` of greenhouse) is read as that service's declaration.
      const owner = name === 'read_code' && /^[A-Za-z_$][\w$]*$/.test(requested) ? Object.entries(availableServices)
        .find(([service, value]) => Object.hasOwn(this.runtime.declarations, service) && value !== null && typeof value === 'object' &&
          requested in (value as object))?.[0] : undefined;
      if (owner) {
        const declaration = this.runtime.declarations[owner]!;
        // A class or other exported type of the service is not called through it: name it as part of the declaration.
        const isFunction = new RegExp(`\\bfunction\\s+${requested}\\s*[<(]`).test(declaration);
        const text = isFunction ? `${requested} is a function of the ${owner} service; call it in eval as ${owner}.${requested}(...).\n${declaration}` :
          `${requested} is declared by the ${owner} service:\n${declaration}`;
        return { kind: 'ok', text, value: declaration };
      }
      // A source file name (`resolve.js`, `ticket_kind.nl`) names the function it defines: show that function.
      const stem = /^(?:\.\/)?(.+?)(?:\.(?:nl|ts|js|mjs|mts))?$/.exec(requested)?.[1];
      if (stem && stem !== requested && findCodebaseItem(this.lam.codebase, stem)) return this.functionTool(name, { ...args, name: stem });
      // Say what the name is when it is not the program's: a model asks for the source of its tools and built-ins,
      // of the very call it is executing, and of the values in its scope.
      const own = listCodebase(this.lam.codebase);
      const readable = own.length ? `a function of this program: ${own.join(', ')}` : 'a function of this program, and this program has none of its own';
      const inScope = (this.lam.type.kind === 'lambda' ? this.lam.type.params.fields.map(field => field.name) : [])
        .concat(Object.keys(this.lam.captures ?? {}), Object.keys(this.lam.let));
      const what = NATIVE_TOOLS.includes(requested) ? this.offeredToolDocs.has(requested) ?
        `${requested} is one of your currently offered tools, not a function of this program; call it directly` :
        `${requested} is a native tool name but is not offered in this call` :
        requested === this.lam.functionName ? `${requested} is the call you are executing: the opening message has all of its ` +
          `signature and instructions. ${SELF_CALL_ADVICE.replace(/\.$/, '')}` :
        this.neuraleseParameter(requested) ? `${requested} is this call's Neuralese input: its content is in your opening ` +
          `message, so use it from there, or write await read(${requested}) in eval for an ordinary copy` :
        inScope.includes(requested) ? `${requested} is a value in this call's eval scope: inspect it with eval, for example console.log(${requested})` :
        // A text-typed call with no functions of its own (a writer) has nothing to read: it answers with its text.
        !own.length && this.expectsTextResult() ? 'This call returns text: send the text itself as the value of return_result' :
        undefined;
      throw new Reject([{ path: requested, code: 'no-such-function', expected: what ? `${readable}. ${what}` : readable }]);
    }
    const view = this.runtime.frame?.task.programView;
    const record = view ? view.record(found.record) : found.record;
    const expectedRevision = view?.revisionId ?? 0;
    if (record.kind === 'namespace' && name === 'read_code') {
      const paths = Object.keys(record.codebase).map(child => `${requested}.${child}`);
      const text = `Callable namespace ${requested}:\n${paths.join('\n')}`;
      return { kind: 'ok', text: this.show(text), value: paths };
    }
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
    if (this.codeEditsDenied())
      throw new Reject([{ path: requested, code: 'bad-action', expected: 'code edits are denied by this evaluation policy' }]);
    // What judges the call is not the call's to change (protect:, WS1 L3): a stuck run edited its verifier to pass.
    this.protectedList ??= [...(this.lam.protectedSources ?? []), ...treeProtectedSources(this.lam.codebase as Record<string, ItemRecord>)];
    if (isProtectedSource(record.source, this.protectedList))
      throw new Reject([{ path: requested, code: 'protected', text: protectedEditError(record.source) }]);
    if (args.expected_revision !== undefined && args.expected_revision !== expectedRevision)
      throw new Reject([{ path: requested, code: 'bad-action', expected: 'current source revision; reread after a concurrent edit' }]);
    const text = editTextContent(record.text, String(args.find ?? ''), String(args.replace_with ?? ''), args.fuzzy === true);
    const updated = record.kind === 'natlang' ? parseNatlang(record.source, text, record.types, PATH_ONLY) :
      parseModule(record.source, text, record.types, PATH_ONLY);
    updated.programId = record.programId;
    updated.codebase = record.codebase;
    if (record.kind === 'natlang' && updated.kind === 'natlang' &&
        JSON.stringify(record.protect ?? []) !== JSON.stringify(updated.protect ?? []))
      throw new Reject([{ path: requested, code: 'protected', text: PROTECT_LIST_ERROR(record.source) }]);
    // Postconditions and termination measures judge the calls too: an edit cannot weaken them.
    if (record.kind === 'natlang' && updated.kind === 'natlang' &&
        JSON.stringify([record.ensures ?? [], record.measure ?? null, record.recursive ?? null, record.writes ?? null]) !==
        JSON.stringify([updated.ensures ?? [], updated.measure ?? null, updated.recursive ?? null, updated.writes ?? null]))
      throw new Reject([{ path: requested, code: 'protected', text: CONTRACT_EDIT_ERROR(record.source) }]);
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
    // The call record keeps each committed revision with its text (calls/recorder.ts code_revisions): a good fix is a
    // proposal that `natlang improve adopt-call` can install, not something that dies with the run.
    this.runtime.trace.emit('code_revision', { ...revision, patch: { find: String(args.find ?? ''), replace_with: String(args.replace_with ?? ''),
      ...(args.fuzzy === true ? { fuzzy: true } : {}) }, text: updated.text });
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
    // A read_file result is the file's text and nothing else: an appended hint read as file content (and was written into files).
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
      const returns = parseTypeCached(meta.definition.returns);
      const fields = meta.definition.params.map(parameter => {
        return { name: parameter.name, type: parseTypeCached(parameter.type), optional: parameter.optional === true };
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
    // A call's type belongs to the entire initializer only when that expression
    // is the call itself. A prefix call can instead be a predicate, logical
    // operand or receiver, whose result is not the value assigned to the local.
    const parsed = ts.createSourceFile('initializer.ts', `(${source}\n)`, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
    const statement = parsed.statements[0];
    const unwrap = (node: ts.Expression): ts.Expression =>
      ts.isParenthesizedExpression(node) || ts.isAwaitExpression(node) ? unwrap(node.expression) : node;
    const expressionNode = statement && ts.isExpressionStatement(statement) ? unwrap(statement.expression) : undefined;
    if (expressionNode?.kind === ts.SyntaxKind.NullKeyword) return { kind: 'prim', name: 'null' };
    if (expressionNode && (ts.isStringLiteral(expressionNode) || ts.isNoSubstitutionTemplateLiteral(expressionNode)))
      return { kind: 'prim', name: 'string' };
    if (expressionNode && ts.isNumericLiteral(expressionNode)) return { kind: 'prim', name: 'number' };
    if (expressionNode?.kind === ts.SyntaxKind.TrueKeyword || expressionNode?.kind === ts.SyntaxKind.FalseKeyword)
      return { kind: 'prim', name: 'boolean' };
    if (expressionNode && ts.isConditionalExpression(expressionNode)) {
      const branches = [expressionNode.whenTrue, expressionNode.whenFalse].map(branch =>
        this.scopeInitializerType(branch.getText(parsed), locals));
      if (branches.every((branch): branch is Type => branch !== undefined)) {
        const members = branches.flatMap(branch => branch.kind === 'union' ? branch.members : [branch]);
        const unique = [...new Map(members.map(member => [formatType(member), member])).values()];
        return unique.length === 1 ? unique[0] : { kind: 'union', members: unique };
      }
      return;
    }
    if (expressionNode && ts.isCallExpression(expressionNode)) {
      const callee = expressionNode.expression.getText(parsed);
      if (/^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*$/.test(callee)) {
        const directReturns = returnTypeOf(this.lam.codebase, callee, this.env);
        if (directReturns) return directReturns;
      }
    }
    // Only a map callback that is the call itself (`x => f(x)`, optionally async/await) gives the list the callee's
    // return type; a callback that wraps the call in an object or expression has a type this cannot see.
    const mappedCall = /^await\s+Promise\.all\(\s*[\s\S]*?\.map\(\s*(?:async\s+)?(?:\([\w$,\s]*\)|[A-Za-z_$][\w$]*)\s*=>\s*(?:await\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(/.exec(source);
    let mappedReturns: Type | undefined;
    if (mappedCall) {
      let depth = 1, at = mappedCall[0].length;
      for (; at < source.length && depth > 0; at++) depth += source[at] === '(' ? 1 : source[at] === ')' ? -1 : 0;
      if (depth === 0 && /^\s*\)\s*\)\s*$/.test(source.slice(at))) mappedReturns = returnTypeOf(this.lam.codebase, mappedCall[1]!, this.env);
    }
    if (mappedReturns) return { kind: 'list', element: mappedReturns };
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
    // A callable with a representation-generic result: each of its call sites is analyzed for the instance it runs.
    const generic = (level: Record<string, unknown>): boolean => Object.values(level as Record<string, ItemRecord>).some(item =>
      !!item && typeof item === 'object' && ((item.kind === 'natlang' && !!item.generic) || generic(item.codebase ?? {})));
    return mentions(formatType(this.lam.type)) || Object.values(this.lam.typesSrc).some(mentions) || generic(this.lam.codebase) ||
      Object.values(this.lam.letTypes).some(type => mentions(formatType(type))) ||
      Object.values(this.lam.captures ?? {}).some(cell => mentions(cell.type));
  }

  /** Execute one eval action as an atomic scope transaction. */
  private async evaluate(written: string, timeoutMs?: number, complete = false, toolCallId?: string): Promise<NativeResult> {
    if (!written.trim()) throw new Reject([{ path: 'code', code: 'bad-action', expected: 'a TypeScript statement or expression' }]);
    const declaredResult = this.lam.type.kind === 'lambda' ? this.env.resolve(this.lam.type.returns) : undefined;
    const neuraleseElement = declaredResult?.kind === 'neuralese' ? this.env.resolve(declaredResult.element) : undefined;
    const isNeuraleseType = (source: string): boolean => {
      try { return this.env.resolve(parseTypeCached(source)).kind === 'neuralese'; } catch { return false; }
    };
    const hasUnsupportedMarkerExpression = computedNeuraleseMarkerConcatenation(written, this.lam.codebase as Record<string, ItemRecord>, isNeuraleseType,
      declaredResult?.kind === 'neuralese');
    if (hasUnsupportedMarkerExpression && declaredResult?.kind === 'neuralese')
      throw new Reject([{ path: 'code', code: 'computed-neuralese-marker-concatenation',
        expected: `This function returns ${formatType(declaredResult)}. Marker bodies are literal and do not evaluate JavaScript concatenation. Return the computed value directly according to its declared type, without marker strings.` }]);
    if (hasUnsupportedMarkerExpression)
      throw new Reject([{ path: 'code', code: 'computed-neuralese-marker-concatenation',
        expected: 'This argument is declared Neuralese<T>. Marker bodies are literal and do not evaluate JavaScript concatenation. Return the computed typed value directly, without marker strings.' }]);
    // Literals the model wrote are block markers in its text; compiled, each is a call the checker types from context.
    const desugared = desugarNlCalls(sourceWithLiteralCalls(written));
    // The std combinators by name (typed for the checker through evalStdBindings), unless a binding of the call takes it.
    const stdLibrary = this.stdLibrary();
    const stdNames = stdLibrary ? this.evalStdNames(desugared) : [];
    this.evalStdBindings = stdNames;
    const code = this.runtime.hooks.lowerRefinements ? this.runtime.hooks.lowerRefinements(this, desugared) : desugared;
    const scopeBefore = this.scopeSnapshot(), traceMark = this.runtime.trace.events.length;
    const inputNames = this.lam.type.kind === 'lambda' ? this.lam.type.params.fields.map(field => field.name) : [];
    const localNames = Object.keys(this.lam.let).filter(name => !isPending(this.lam.let[name]!));
    const localBindings = localNames.map(name => ({ name, mutable: this.scopeLocalMutability.get(name) ?? true,
      annotation: this.lam.letTypes[name] && this.lam.letTypes[name]!.kind !== 'host' ? formatType(this.lam.letTypes[name]!) : undefined }));
    const callableNames = Object.keys(this.lam.codebase);
    const opaqueNames: string[] = [];
    if (this.lam.projectTransaction) opaqueNames.push('folder');
    // Calls with formal parameters expose their frozen arguments through `read_inputs()` and `inputs`.
    // With no formal parameters there is no caller input object to inspect, so keep both names available for locals.
    const taken = (name: string) => inputNames.includes(name) || Object.hasOwn(this.lam.let, name);
    const hasCallInputs = inputNames.length > 0;
    const inputsBinding = hasCallInputs && !taken('read_inputs'), inputsObject = hasCallInputs && !taken('inputs');
    if (inputsBinding) opaqueNames.push('read_inputs');
    if (inputsObject) opaqueNames.push('inputs');
    // return_result also works as a function in eval: return_result(value) stages a result, and
    // return_result(value, status, reason) carries out the tool's request once the eval has succeeded.
    // A finisher name the snippet declares itself stays the snippet's own variable.
    // Names this eval's own top-level declarations take; a name it only uses (an argument, a shorthand property) keeps the
    // saved helper or built-in of that name. A regex here once read `f(a, grade, b)` as declaring grade, and dropped the
    // saved helper grade (authored wave 1, 2026-11-25).
    const declaredNames = evalDeclaredNames(code);
    const declaredHere = (name: string) => declaredNames.has(name);
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
    // `search_code(query)` is offered in eval whenever this call's scope has a codebase, unless the name is taken.
    const searchCodeBinding = callableNames.length > 0 && !taken('search_code') && !declaredHere('search_code') &&
      !callableNames.includes('search_code') && !serviceNames.includes('search_code') && !Object.hasOwn(captureCells, 'search_code');
    if (searchCodeBinding) opaqueNames.push('search_code');
    // decide(fn, ...args) calls a function with a finite result type and returns its answer with the probability of
    // every allowed value (runtime.decide in eval), unless the name is taken.
    const decideBinding = !taken('decide') && !callableNames.includes('decide') && !serviceNames.includes('decide') &&
      !Object.hasOwn(captureCells, 'decide');
    if (decideBinding) opaqueNames.push('decide');
    // refine(value, predicate) / assume(value, predicate): the refinement helpers (Is<T, P>), unless the name is taken.
    const refinementBinding = (name: string) => !taken(name) && !declaredHere(name) && !callableNames.includes(name) &&
      !serviceNames.includes(name) && !Object.hasOwn(captureCells, name);
    const refineBinding = refinementBinding('refine'), assumeBinding = refinementBinding('assume');
    // NatlangContext: the runtime's context value, so eval can bind a function to another context
    // (`fn.in(NatlangContext.empty().with({ policy }))`), unless the name is taken. Packages export it as `Context`;
    // eval uses this name so that a program type named Context does not collide with it.
    const contextBinding = refinementBinding('NatlangContext');
    if (contextBinding) opaqueNames.push('NatlangContext');
    // transcript: this call's earlier tool calls and their outputs, read-only, unless the name is taken (an input, a
    // local, a function of the call's folder, a service or a capture: theirs wins, as for read_code and decide).
    const transcriptBinding = !taken('transcript') && !callableNames.includes('transcript') && !serviceNames.includes('transcript') &&
      !Object.hasOwn(captureCells, 'transcript');
    if (transcriptBinding) opaqueNames.push('transcript');
    const savedHelpers = [...this.persistentScopeHelpers.values()].filter(helper =>
      hexDigest(helper.source) === helper.sourceHash && Number.isSafeInteger(helper.declarationSpan.start) &&
      helper.declarationSpan.start >= 0 && helper.declarationSpan.end - helper.declarationSpan.start === helper.source.length &&
      !declaredHere(helper.name) && !inputNames.includes(helper.name) && !localNames.includes(helper.name) &&
      !callableNames.includes(helper.name) && !serviceNames.includes(helper.name) && !opaqueNames.includes(helper.name) &&
      !stdNames.includes(helper.name) &&
      !Object.hasOwn(captureCells, helper.name));
    const savedHelperNames = savedHelpers.map(helper => helper.name);
    const visibleTypes = { ...this.lam.typesSrc, ...Object.fromEntries(Object.keys(this.localTypes).map(name => [name, ''])) };
    const currentTypes = evalTypeDeclarations(code);
    const persistentTypeNames = [...new Set([...Object.keys(visibleTypes), ...Object.keys(currentTypes)])];
    const hooks = this.runtime.hooks;
    const boundValues = Object.fromEntries([...Object.entries(captureRead), ...inputNames.map(name => [name, this.lam.args[name]])]
      .filter(entry => isPlainJson(entry[1])));
    // An example's inline nl copied into a call that has nothing to do with it (read_code's nl documentation; the system
    // prompt's before 2026-11-23): refused before running, or the copy runs as a child that answers an unrelated question.
    const ownText = (this.lam.body) + '\n' + JSON.stringify(this.lam.args ?? {}) + '\n' + JSON.stringify(this.lam.let ?? {});
    const copied = PROMPT_EXAMPLE_TEXTS.find(text => code.includes(text) && !ownText.includes(text));
    if (copied) {
      const own = inputNames.length ? `this call's own instructions and inputs (${inputNames.join(', ')})` : "this call's own instructions";
      const text = `prompt-example: ${JSON.stringify(copied)} comes from ${PROMPT_EXAMPLE_SOURCE}, not from this call; ` +
        `work from ${own} instead.`;
      return { kind: 'rejected', text: text + this.captureScopeFailure('compile', code, scopeBefore, text), codes: ['prompt-example'] };
    }
    const compiled = compileScopeSnippet(code, { inputBindings: inputNames, boundValues, localBindings, helperBindings: callableNames,
      persistentHelpers: savedHelpers, persistentHelperNames: savedHelperNames, persistentTypeNames,
      opaqueBindings: [...opaqueNames, ...stdNames], ...(stdLibrary ? { stdBindings: stdNames } : {}),
      captureBindings: Object.values(captureCells).map(cell => ({ name: cell.name, mutable: cell.mutable })),
      serviceBindings: serviceNames, analyze: source => hooks.analyze(this, source, [...opaqueNames, ...savedHelperNames]), neuralese: this.holdsNeuralese(),
      ...(hooks.checkFields ? { checkFields: (source: string) => hooks.checkFields!(this, source, [...opaqueNames, ...savedHelperNames]) } : {}),
      guardPrefix: `eval:${this.runtime.options.runId}`, ...this.runtime.environment.scopeCapabilities });
    // Model-written literals in this eval are graph nodes: the block, its contextual type, the reference written.
    if (compiled.ok) {
      const turn = [...this.runtime.trace.events].reverse().find(event => event.kind === 'model_turn')?.node as string | undefined;
      for (const literal of compiled.literals ?? []) graphNode(this.runtime.trace, 'literal', { call_id: this.runtime.options.runId,
        block: literal.id, type: literal.type, expression: `__neuralese(${nativeLiteralArguments(literal.id,literal.stopGradient,literal.sourceOccurrence)})` },
        [blockInput(literal.id, 'block', literal.sourceOccurrence, producerScopeForTrace(this.runtime.trace)), ...(turn ? [{ node: turn, port: 'turn' }] : [])]);
      for (const plan of compiled.plans ?? []) if (plan.softBody) graphNode(this.runtime.trace, 'literal', { call_id: this.runtime.options.runId,
        block: plan.softBody, type: `Neuralese<(${plan.parameters.map(item => `${item.name}: ${item.type.natlang ?? item.type.text}`).join(', ')}) => ${plan.returns.natlang ?? plan.returns.text}>`,
        expression: `__neuralese.body(${nativeLiteralArguments(plan.softBody,plan.softBodyStopGradient,plan.softBodySourceOccurrence)})`,
        captures: plan.captures.filter(capture => !capture.shadowedByParameter).map(capture => capture.name),
        shadowed_capture_initializers: plan.captures.filter(capture => capture.shadowedByParameter)
          .map(({ name, mode }) => ({ name, mode: mode ?? 'snapshot' })) },
        [blockInput(plan.softBody, 'body', plan.softBodySourceOccurrence, producerScopeForTrace(this.runtime.trace)), ...(turn ? [{ node: turn, port: 'turn' }] : [])]);
    }
    if (!compiled.ok || !compiled.program) {
      const text = compiled.diagnostics.map(item => `${item.line}:${item.column} ${item.code}: ${item.message}`).join('\n');
      return { kind: 'rejected', text: text + this.captureScopeFailure('compile', code, scopeBefore, text, compiled.diagnostics),
        codes: [...new Set(compiled.diagnostics.map(item => item.code))] };
    }
    // One copy of inputs and locals together: an object two of them hold stays one object in eval (JavaScript identity).
    const identity = evalIdentityEnabled(), scopeCopies = identity ? new Map<object, unknown>() : undefined;
    const inputs = splitScope(this.lam.args, scopeCopies);
    const locals = splitScope(Object.fromEntries(localNames.map(name => [name, this.lam.let[name]!])), scopeCopies);
    let finished: unknown, scopeInputs: unknown;
    let evalStop: AbortController | undefined;
    const plans = compiled.plans ?? [];
    const origin: InlineInstructionOrigin | undefined = toolCallId && this.runtime.currentCallId ? {
      parentInvocationId: this.runtime.currentCallId, toolCallId, actionOrdinal: this.actions,
      writtenCodeSha256: hexDigest(written), checkedCodeSha256: hexDigest(code),
      ...(() => {
        const bodyPlan = plans.find(plan => plan.softBody);
        if (!bodyPlan?.softBody) return {};
        const literal = `\`${neuraleseSentinel(bodyPlan.softBody, bodyPlan.softBodyStopGradient,bodyPlan.softBodySourceOccurrence)}\``;
        const start = written.indexOf(literal);
        return start >= 0 && written.indexOf(literal, start + 1) < 0 ?
          { sourceTemplateSpan: { start, end: start + literal.length } } : {};
      })() } : undefined;
    this.activeScopeLocals = new Map();
    const live = { own: (value: Promise<unknown>) => {
      // NL calls already have their own cancellation/drain protocol. Keep a TS
      // helper's entire execution, including gaps between successive RPCs.
      if (!startedCalls.has(value) && this.runtime.nativeEffects.enabled) {
        this.runtime.nativeEffects.track(Promise.resolve(value));
        this.runtime.frame?.task.track(Promise.resolve(value));
      }
    }, inputs: inputs.live, locals: locals.live, captures: captureRead, callables: this.callables(),
      services: this.availableServices(), folder: this.lam.projectTransaction?.folder.root(),
      readNeuralese: readNeuraleseForCurrentTask,
      readNeuraleseIfReference,
      instantiate,
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
      searchCode: (input: unknown, limit?: unknown) => {
        let query: string | undefined;
        let cap: number | undefined;
        if (typeof input === 'string') {
          query = input;
          cap = typeof limit === 'number' ? limit : undefined;
        } else if (isPlainRecord(input) && typeof input.query === 'string') {
          query = input.query;
          cap = typeof input.limit === 'number' ? input.limit : undefined;
        }
        if (query === undefined) throw new Reject([{ path: 'search_code', code: 'type-mismatch',
          expected: 'a query string or { query: string, limit?: number }',
          got: input === null ? 'null' : Array.isArray(input) ? 'array' : typeof input }]);
        const matches = this.searchCode(query, cap);
        this.runtime.trace.emit('eval_code_search', { call_id: this.runtime.currentCallId ?? this.runtime.options.runId,
          action: toolCallId ?? null, query, matches: matches.length });
        return matches;
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
      refine: (value: unknown, predicate: string | readonly string[], options?: { context?: Record<string, unknown> }) => {
        const frame = currentFrame() ?? this.runtime.frame!;
        return import('../runtime/surface.js').then(module => runInFrame(frame, () => module.refine(value, predicate as string, options)));
      },
      NatlangContext: evalContextValue(),
      assume: (value: unknown, predicate: string | readonly string[]) => {
        const raw = Array.isArray(predicate) ? predicate : [predicate];
        const texts = raw.map(item => typeof item === 'string' ? item.replace(/\s+/g, ' ').trim() : '');
        if (!texts.length || texts.some(text => !text))
          throw new Error('refinement-predicate-invalid: assume(value, predicate) takes a nonempty string predicate, for example "one line of at most 60 characters".');
        const shown = oneLine(value as Value, undefined, this.runtime.displayLiveId, this.runtime.untrusted).slice(0, 400);
        for (const text of new Set(texts))
          this.runtime.trace.emit('refinement_assumed', { call_id: this.runtime.currentCallId ?? null, predicate: text, value: shown });
        return value;
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
              if (active?.[1]) return active[1](value);
              else this.lam.let[capture.name] = coerce(value, this.lam.letTypes[capture.name]!, this.env, `let/${capture.name}`);
            },
          ];
        }
        return this.runtime.environment.authority.startsWith('selected-semantic/') ?
          withSelectedCaptureAccessors(plan, bound, observed => hooks.inline(this, plan, values, observed, origin)) :
          hooks.inline(this, plan, values, bound, origin);
      },
      finite: hooks.finite, finiteArrayIterator: hooks.finiteArrayIterator, finiteAsync: hooks.finiteAsync, guard: hooks.guard,
      iterateOn: (step: unknown, initial: unknown, ...args: unknown[]) => hooks.iterateOn(this, step, initial, ...args),
      // Each std combinator runs in the frame of the code calling it (this call, or a combinator's span when nested),
      // so its std:<name> span and child calls nest under this call.
      std: Object.fromEntries(stdNames.map(name => [name, (...args: unknown[]) =>
        runInFrame(currentFrame() ?? this.runtime.frame, () => (stdLibrary![name] as (...items: unknown[]) => unknown)(...args))])),
      finish: (value: unknown) => { finished = value; },
      scopeInputs: (value: unknown) => { scopeInputs = value; } };
    const prologue = [
      ...(this.lam.projectTransaction ? ['const folder = __live.folder;'] : []),
      ...(inputsBinding ? ['const read_inputs = () => __live.callInputs;'] : []),
      ...(inputsObject ? ['const inputs = __live.callInputs;'] : []),
      ...(transcriptBinding ? ['const transcript = __live.transcript;'] : []),
      ...(readCodeBinding ? ['const read_code = (input: string | { name: string }) => __live.readCode(input);'] : []),
      ...(searchCodeBinding ? ['const search_code = (query: string | { query: string; limit?: number }, limit?: number) => __live.searchCode(query, limit);'] : []),
      ...(refineBinding ? ['const refine = (value: any, predicate: string | string[], options?: any) => __live.refine(value, predicate, options);'] : []),
      ...(contextBinding ? ['const NatlangContext = __live.NatlangContext;'] : []),
      ...(assumeBinding ? ['const assume = (value: any, predicate: string | string[]) => __live.assume(value, predicate);'] : []),
      ...(decideBinding ? ['const decide = (fn: any, ...args: any[]) => __live.decide(fn, args);'] : []),
      ...stdNames.map(name => `const ${name} = __live.std.${name};`),
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
      await this.runtime.nativeEffects.drain();
      if(this.runtime.nativeEffects.enabled) this.runtime.checkInterruption();
      const raw = finished as { result?: unknown; returned?: boolean; valueless?: boolean;
        bindings?: Record<string, unknown>; captures?: Record<string, unknown> } | undefined;
      if (!raw || typeof raw !== 'object' || !raw.bindings || typeof raw.bindings !== 'object')
        throw new Reject([{ path: 'code', code: 'bad-action', expected: 'an atomic scope transaction result' }]);
      // A malformed compiled result must not poison every later eval with numeric locals.
      // This can happen when a comma expression is passed as two function arguments.
      if (Object.keys(raw.bindings).some(name => !/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name)))
        throw new Reject([{ path: 'code', code: 'bad-action',
          expected: 'named eval bindings; wrap a final comma expression in parentheses' }]);
      // Plain data is rebuilt in this realm; captured values keep their identity for write-back.
      // A part of an input that a local or the result holds is the host's own input value, not a copy of it.
      const inputParts = new Map<object, unknown>();
      if (identity) mapCopiedParts(inputParts, scopeInputs, this.lam.args);
      const output = { ...hostCopy({ result: raw.result, bindings: raw.bindings }, inputParts) as
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
        if (annotation) type = inlineDeclaredTypes(parseTypeCached(annotation), localTypes);
        if (!type && initializers.get(name)) type = this.scopeInitializerType(initializers.get(name)!, inferred);
        // An unannotated let initialized to null can evolve before this same eval finishes, just as it can in a
        // later eval. Its initializer describes the opening value, not a constraint on the computed final value.
        if (!annotation && mutability.get(name) && type?.kind === 'prim' && type.name === 'null' && value !== null)
          type = undefined;
        if (!type) type = await this.sourceCallableType(value, localTypes);
        if (!type) {
          try { type = parseType(this.inferScopeType(value)); }
          catch (error) {
            if (this.lam.type.kind !== 'lambda' || !compiled.resultBindings?.includes(name)) throw error;
            type = this.lam.type.returns;
          }
        }
        inferred[name] = type;
        this.bindLocalTypeNames(type, localTypes);
        // A local holds what eval computed (a service's or a callee's value): as in TypeScript, fields its type does
        // not list are kept, so storing a provider value never asks for a retyped copy.
        // The coerced copy gives way to the value itself wherever they agree, so the objects a local holds stay the same objects.
        const coerced = coerce(value, type, this.env, `let/${name}`, { preserveRecordExtras: true, functions: 'keep' });
        staged.push([name, type, identity ? withSourceIdentity(value, coerced) : coerced]);
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
      // A proposed result (or a staged local) with a refined type must satisfy its predicate before it is kept.
      let refinementFailure: { code: string; message: string } | undefined;
      if (this.refinementCheck) {
        if (functionResult !== undefined && this.lam.type.kind === 'lambda') {
          refinementFailure = await this.refinementCheck(functionResult, this.lam.type.returns, 'return');
          if (refinementFailure) { functionResult = undefined; notResult = `\nThis is not the result: ${refinementFailure.code}: ${refinementFailure.message}`; }
        }
        for (const [name, type, value] of staged) {
          const failure = containsRefinement(type, this.env) ? await this.refinementCheck(value, type, name) : undefined;
          if (failure) throw new Reject([{ path: `let/${name}`, code: failure.code, expected: failure.message }]);
        }
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
      // An eval that completes without a value shows `undefined`, as a JavaScript REPL does; `null` is shown only for a
      // value that is null. A trailing `null` after console output read as part of that output (an edit-stream run appended it).
      const rendered = isLive(output.result) || isHandle(output.result) ? livePreview(output.result as object, this.runtime.displayLiveId) :
        raw.valueless === true && !isHandle(output.result) ? 'undefined' : this.showValue(output.result ?? null);
      const status = functionResult !== undefined ?
        stagedMessage(functionResult, this.runtime.displayLiveId, this.runtime.untrusted) : notResult;
      const shownLogs = evaluated.logs?.length ? this.show(evaluated.logs.join('\n')) : '';
      const logStatus = shownLogs ? `console:\n${shownLogs}\n` : '';
      const stored = changed.map(([name, , value]) => {
        const preview = oneLine(value, name, this.runtime.displayLiveId, this.runtime.untrusted);
        const shown = typeof value === 'string' && preview.includes('<<cut off') && shownLogs && shownLogs.includes(value)
          ? `(full string printed above, ${value.length} chars)`
          : preview;
        return `local ${name} = ${shown}` +
          // A Neuralese value is shown as its state; the words beside it say what it is and how to use it.
          (isNeuraleseRef(value) ? ` (${value.$neuralese.type}: pass it on as is, or write await read(${name}) for an ordinary copy)` : '');
      });
      // Variant effect-receipt: a successful eval lists the service calls it completed, as a failed one does, so a model
      // does not rerun an effectful eval that already succeeded.
      const performed = new Map<string, number>();
      if (modelFacing('effect-receipt'))
        for (const event of this.runtime.trace.events.slice(traceMark))
          if (event.kind === 'effect' && event.phase === 'completed') {
            const name = String(event.capability ?? 'service call');
            performed.set(name, (performed.get(name) ?? 0) + 1);
          }
      const effectStatus = performed.size ?
        `\nService calls completed by this eval (not undone; do not repeat a write): ${[...performed].map(([name, count]) => count > 1 ? `${name} x${count}` : name).join(', ')}.` : '';
      const storedStatus = effectStatus + (stored.length ? `\nStored ${stored.join('; ')}.` : '') +
        ((compiled.transientHelpers?.length ?? 0) ? `\n${compiled.transientHelpers!.map(helper =>
          `${helper.name} is available only in this eval because ${helper.reason}; use it here or move its dependencies into persistent scope.`).join('\n')}` : '');
      // A local without a value cannot be kept, so say so here rather than let a later eval fail on its name.
      const unset = compiled.bindings.filter(binding => binding.mutable && !binding.initializer).map(binding => binding.name)
        .filter(name => output.bindings[name] === undefined && !Object.hasOwn(this.lam.let, name));
      const unsetStatus = unset.length ? `\nNot kept: ${unset.join(', ')} ${unset.length > 1 ? 'have' : 'has'} no value, and a ` +
        `local without a value is not kept; declare it with one (let ${unset[0]} = …).` : '';
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
        const unrefined = this.refinementCheck ? await this.refinementCheck(staged, this.lam.type.returns, 'return') : undefined;
        if (unrefined) return { kind: 'rejected', text: `${logStatus}${rendered}${storedStatus}\nreturn_result: this is not the result: ` +
          `${unrefined.code}: ${unrefined.message}`, codes: [unrefined.code] };
        this.lam.return = staged;
        this.failureDebug = undefined;
        if (complete) {
          const unwritten = await this.writesProblemAsync();
          if (unwritten) throw new Reject([{ path: 'files', code: 'writes-unsatisfied', text: `writes-unsatisfied: ${unwritten}` }]);
          const done=this.scopeTool('return_result',{status:'success',value:staged});
          return {...done,text:`${logStatus}${rendered}${storedStatus}${unsetStatus}\n${done.text}`};
        }
        return { kind: 'ok', text: `${logStatus}${rendered}${storedStatus}${unsetStatus}${stagedMessage(staged, this.runtime.displayLiveId, this.runtime.untrusted)}`, value: staged };
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
            : '\nFor inspection or staging, omit finish or set false; finish:true needs a fresh value of the declared result type from this eval as its final expression or explicit return, and cannot reuse an older staged value.';
          return {kind:'rejected',text:logStatus+rendered+storedStatus+unsetStatus+notResult+finishHint,codes:[refinementFailure?.code ?? 'missing-fresh-result']};
        }
        const unwritten = await this.writesProblemAsync();
        if (unwritten) throw new Reject([{ path: 'files', code: 'writes-unsatisfied', text: `writes-unsatisfied: ${unwritten}` }]);
        // Finish with the staged value as checked strictly above.
        const done=this.scopeTool('return_result',{status:'success'});
        return {...done,text:logStatus+rendered+storedStatus+unsetStatus+'\n'+done.text};
      }
      return { kind: 'ok', text: logStatus + rendered + storedStatus + unsetStatus + status, value: (output.result ?? null) as Value,
        ...(refinementFailure ? { codes: [refinementFailure.code] } : compiled.repairs.length ? { codes: ['coerced-redundant-self-alias'] } : {}) };
    } catch (error) {
      // A failed eval (an error, a rejected value, its timeout) stops the natural-language calls it started.
      evalStop?.abort(new Error('the eval that started this call failed'));
      // A runtime without a Neuralese writer (here or in a call the eval started) is not the model's to repair.
      if (isNoNeuraleseWriterFault(error)) throw error;
      const message = error instanceof Error ? error.message : String(error);
      const note = this.captureScopeFailure(error instanceof EvalFailure ? 'runtime' : 'boundary',
        code, scopeBefore, message, error instanceof Reject ? error.diagnostics : [], error, traceMark);
      if (error instanceof Reject) { const result = rejected(error); return { ...result, text: result.text + note }; }
      return { kind: 'error', text: message + note };
    } finally {
      const frame = this.runtime.frame;
      const parentCallId = frame?.parentCallId;
      if (parentCallId) await frame!.task.drainChildren(parentCallId);
      await this.runtime.nativeEffects.drain();
      this.activeScopeLocals = undefined;
    }
  }

  private resolve(path: string): Ref {
    const parts = path.split('/').filter(Boolean);
    const first = parts.shift();
    if (first === 'return') {
      const ref = itemRef(this.lam as unknown as Record<string, Value>, 'return',
        this.lam.type.kind === 'lambda' ? this.lam.type.returns : parseTypeCached('null'), this.env, 'return');
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
    try { types[name] = parseTypeCached(text); } catch { /* outside the portable grammar: checked by the compiler, not here */ }
  }
  return types;
}
import { hexDigest } from './hash.js';
