import { handoffTurns, type Turn } from './replay.js';
import { hexDigest } from '../native/hash.js';
import { nativeDecisionTargetDigest, validNativeDecisionApproval, type NativeDecisionApproval } from '../native/decision-review.js';
import { sourceConversionProblems, retiredWorkflowEvaluationReleased } from './source-conversion.js';
import { trainingQualityReason, runtimeFailureReason, quarantineReason, retiredFamily } from './curriculum-policy.js';
import type { ProgramRecord } from './program.js';
import { sourceWithLiteralCalls } from '../native/neuralese.js';
import { desugarNlCalls } from '../compiler/nl-call.js';

export const NATIVE_TEACHER_TRAJECTORY_VERSION = 'natlang.teacher_trajectory.native/1';
export const NATIVE_TEACHER_TURN_VERSION = 'natlang.teacher_training_turn.native/1';

type Dict = Record<string, unknown>;
type NativeRow = Dict & {
  version: string;
  id: string;
  task: Dict;
  provenance: Dict;
  outcome: Dict & { accepted?: boolean };
  trajectory: Dict[];
};

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as Dict).sort(([a], [b]) =>
    a.localeCompare(b)).map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}`;
  return JSON.stringify(value);
}

export function nativeRowDigest(value: unknown): string {
  // Lineage refers to the saved JSON artifact, including omitted undefined fields.
  return hexDigest(canonical(JSON.parse(JSON.stringify(value))));
}

function record(value: unknown, label: string): Dict {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`${label} must be an object`);
  return value as Dict;
}

function oracleLevel(value: unknown): 'exact' | 'normalized' | 'span' | 'agreement' | 'judged' | 'constraints' | undefined {
  const level = typeof value === 'string' ? value : value && typeof value === 'object' && !Array.isArray(value) ?
    (value as Dict).level : undefined;
  return ['exact', 'normalized', 'span', 'agreement', 'judged', 'constraints'].includes(String(level)) ?
    level as 'exact' | 'normalized' | 'span' | 'agreement' | 'judged' | 'constraints' : undefined;
}

function messages(value: unknown, label: string): Dict[] {
  if (!Array.isArray(value) || value.some(item => !item || typeof item !== 'object' || Array.isArray(item)))
    throw new TypeError(`${label} must be an array of messages`);
  return value as Dict[];
}

function parseArguments(value: unknown): unknown {
  if (typeof value !== 'string') return structuredClone(value ?? {});
  try { return JSON.parse(value); }
  catch { return { __unparsed__: value }; }
}

function normalizeContextMessage(raw: Dict): Dict {
  const role = raw.role;
  if (role === 'assistant') {
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls.map((item, index) => {
      const call = record(item, `context tool call ${index}`), fn = record(call.function ?? {}, 'context function');
      const name = String(fn.name ?? '');
      return { call_id: call.id ?? null, tool: name, source_tool: name,
        arguments: parseArguments(fn.arguments) };
    }) : [];
    return { role, content: raw.content ?? '', reasoning: raw.reasoning_content ?? raw.reasoning ?? raw.thinking ?? null,
      calls };
  }
  if (role === 'tool') return { role, call_id: raw.tool_call_id ?? null,
    name: raw.name ?? null, content: raw.content ?? '' };
  return { role: role ?? 'unknown', content: raw.content ?? '' };
}

function toolSchemas(value: unknown, label: string): Dict[] {
  if (!Array.isArray(value)) throw new TypeError(`${label} must be an array`);
  return value.map((raw, index) => {
    const tool = record(raw, `${label}[${index}]`);
    if (tool.type === 'function') {
      const fn = record(tool.function, `${label}[${index}].function`);
      return { name: fn.name, description: fn.description ?? '', parameters: structuredClone(fn.parameters ?? {}) };
    }
    return structuredClone(tool);
  });
}

function publicValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(publicValue);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value as Dict)
    .filter(([key]) => !key.startsWith('x-'))
    .map(([key, child]) => [key, publicValue(child)]));
}

function trainingTarget(assistant: Dict, calls: Dict[], decisionIndex: number): Dict {
  const target: Dict = { role: 'assistant', content: assistant.content ?? '' };
  if (calls.length) target.tool_calls = calls.map((call, index) => ({
    id: `teacher_${decisionIndex}_${index}`, type: 'function',
    function: { name: call.source_tool, arguments: JSON.stringify(call.arguments ?? {}) },
  }));
  return target;
}

function callMatches(call: Dict, event: Dict): boolean {
  return call.source_tool === event.name && canonical(call.arguments) === canonical(event.arguments);
}

/** Match the one exact marker-to-sentinel rewrite performed when a raw model soft-body call enters the runtime. */
function softBodyActionMatch(modelArguments: unknown, actionArguments: unknown): 'complete' | 'truncated' | undefined {
  const model = modelArguments && typeof modelArguments === 'object' && !Array.isArray(modelArguments) ? modelArguments as Dict : {};
  const action = actionArguments && typeof actionArguments === 'object' && !Array.isArray(actionArguments) ? actionArguments as Dict : {};
  const rawCode = model.code, actionCode = action.code;
  if (typeof rawCode !== 'string' || typeof actionCode !== 'string') return undefined;
  const open = '<|neuralese|>', close = '<|/neuralese|>';
  const starts = [...rawCode.matchAll(/<\|neuralese\|>/g)];
  const sentinels = [...actionCode.matchAll(/(nz1_[a-z2-7]{20,})/g)];
  if (starts.length !== 1 || sentinels.length !== 1) return undefined;
  const start = starts[0]!.index!, bodyStart = start + open.length;
  const closeAt = rawCode.indexOf(close, bodyStart);
  // writeLiterals has an explicit, observable EOF rule: without a close marker it writes the whole remaining
  // source suffix as a truncated body, replaces that suffix with one sentinel, and stops. Match that exact rewrite;
  // never infer the intended body from punctuation or a likely variable name.
  const rewrittenCode = closeAt < 0 ? rawCode.slice(0, start) + sentinels[0]![0] :
    rawCode.slice(0, start) + sentinels[0]![0] + rawCode.slice(closeAt + close.length);
  if (rewrittenCode !== actionCode || canonical({ ...model, code: rewrittenCode }) !== canonical(action)) return undefined;
  return closeAt < 0 ? 'truncated' : 'complete';
}

function softBodyActionMatches(modelArguments: unknown, actionArguments: unknown, row: NativeRow,
  invocationId: string | undefined): boolean {
  const mode = softBodyActionMatch(modelArguments, actionArguments);
  if (!mode) return false;
  if (mode === 'complete') return true;
  if (!invocationId) return false;
  const model = modelArguments as Dict, action = actionArguments as Dict;
  const rawCode = model.code as string, actionCode = action.code as string;
  const openAt = rawCode.indexOf('<|neuralese|>');
  const body = rawCode.slice(openAt + '<|neuralese|>'.length);
  const block = /(nz1_[a-z2-7]{20,})/.exec(actionCode)?.[1];
  const graph = Array.isArray(row.outcome.execution_graph) ? row.outcome.execution_graph as Dict[] : [];
  // A truncated rewrite is linkable only when this invocation's runtime trace certifies the exact body digest
  // and block ID. Preserve the malformed proposal without accepting an arbitrary replacement sentinel.
  return !!block && graph.some(event => event.kind === 'block_write' && event.call_id === invocationId &&
    event.block === block && event.truncated === true && event.text_body_sha256 === hexDigest(body));
}

/** Match an authored Neuralese result marker only when runtime trace proves that exact child wrote its block. */
function softReturnResultActionMatches(modelArguments: unknown, actionArguments: unknown,
  row: NativeRow, invocationId: string | undefined): boolean {
  if (!invocationId) return false;
  const model = modelArguments && typeof modelArguments === 'object' && !Array.isArray(modelArguments) ? modelArguments as Dict : {};
  const action = actionArguments && typeof actionArguments === 'object' && !Array.isArray(actionArguments) ? actionArguments as Dict : {};
  if (model.status !== 'success' || typeof model.value !== 'string' || action.status !== 'success' || typeof action.value !== 'string')
    return false;
  const markers = [...model.value.matchAll(/^<\|neuralese\|>([\s\S]*)<\|\/neuralese\|>$/g)];
  const sentinel = /^(nz1_[a-z2-7]{20,})$/.exec(action.value);
  if (markers.length !== 1 || !sentinel || /<\|neuralese\|>|<\|\/neuralese\|>/.test(markers[0]![1]!)) return false;
  const expectedAction: Dict = { ...model, value: action.value };
  if (canonical(expectedAction) !== canonical(action)) return false;

  const invocationLedger = Array.isArray(row.outcome.invocation_ledger) ? row.outcome.invocation_ledger as Dict[] : [];
  const invocation = invocationLedger.find((entry: Dict) => entry.invocation_id === invocationId);
  const hostResult = invocation?.host_result && typeof invocation.host_result === 'object' ? invocation.host_result as Dict : undefined;
  const resultValue = hostResult?.value && typeof hostResult.value === 'object' ? hostResult.value as Dict : undefined;
  const resultRef = resultValue?.$neuralese && typeof resultValue.$neuralese === 'object' ? resultValue.$neuralese as Dict : undefined;
  const declaredType = (invocation?.inline_instruction_site as Dict | undefined)?.returns;
  const runtimeType = declaredType && typeof declaredType === 'object' ? (declaredType as Dict).natlang : undefined;
  if (hostResult?.kind !== 'host_capture' || hostResult.capture_kind !== 'invocation_output' ||
      hostResult.call_id !== invocationId || hostResult.complete !== true ||
      hostResult.result_type !== 'Neuralese<string>' || runtimeType !== 'Neuralese<string>' ||
      resultRef?.type !== 'Neuralese<string>' || resultRef.id !== sentinel[1]) return false;
  const actionLedger = Array.isArray(row.outcome.action_ledger) ? row.outcome.action_ledger as Dict[] : [];
  const returnAction = actionLedger.find((event: Dict) =>
    (event as Dict).call_id === invocationId && (event as Dict).name === 'return_result' &&
    canonical((event as Dict).arguments) === canonical(action) && ['ok', 'completed'].includes(String((event as Dict).outcome)));
  const terminalSeq = returnAction?.seq;
  if (!returnAction || !Number.isSafeInteger(terminalSeq) || hostResult.terminal_action_seq !== terminalSeq) return false;
  const executionGraph = Array.isArray(row.outcome.execution_graph) ? row.outcome.execution_graph as Dict[] : [];
  const currentTurn = executionGraph.filter((event: Dict) => event.kind === 'model_turn' && event.call_id === invocationId &&
    typeof event.node === 'string' && Number.isSafeInteger(event.seq) && (event.seq as number) < (terminalSeq as number))
    .sort((a: Dict, b: Dict) => Number(a.seq) - Number(b.seq)).at(-1);
  if (!currentTurn || typeof currentTurn.node !== 'string') return false;
  return executionGraph.some((writer: Dict) => {
    return writer.kind === 'block_write' && writer.call_id === invocationId && writer.block === sentinel[1] &&
      typeof writer.node === 'string' && writer.turn === currentTurn.node && Number.isSafeInteger(writer.seq) &&
      (writer.seq as number) > (currentTurn.seq as number) && (writer.seq as number) < (terminalSeq as number) &&
      Array.isArray(writer.inputs) && writer.inputs.some(input => input && typeof input === 'object' &&
        (input as Dict).node === currentTurn.node && (input as Dict).port === 'turn');
  });
}

/** A bounded diagnostic preview is never executable data unless the exact value came from a raw model call. */
function containsIncompleteDiagnostic(value: unknown): boolean {
  const pending = [value], seen = new Set<object>(); let visited = 0;
  while (pending.length) {
    if (++visited > 4096) return true;
    const item = pending.pop();
    if (!item || typeof item !== 'object' || seen.has(item)) continue;
    seen.add(item);
    if (Object.hasOwn(item, '$diagnostic_preview') && (item as Dict).complete === false) return true;
    if (Array.isArray(item)) for (const child of item) pending.push(child);
    else for (const key in item) if (Object.hasOwn(item, key)) pending.push((item as Dict)[key]);
  }
  return false;
}

function hasExactRawModelCall(source: Dict, call: Dict): boolean {
  const response = source.model_response && typeof source.model_response === 'object' ? source.model_response as Dict : {};
  const rawCalls = Array.isArray(response.raw_calls) ? response.raw_calls : [];
  return rawCalls.some(raw => {
    if (!raw || typeof raw !== 'object') return false;
    const fn = (raw as Dict).function;
    if (!fn || typeof fn !== 'object') return false;
    const name = (fn as Dict).name, args = (fn as Dict).arguments;
    if (name !== call.source_tool || typeof args !== 'string') return false;
    try { return canonical(JSON.parse(args)) === canonical(call.arguments); }
    catch { return false; }
  });
}

function validateRow(raw: unknown): NativeRow {
  const row = record(raw, 'native teacher row') as NativeRow;
  if (row.version !== NATIVE_TEACHER_TRAJECTORY_VERSION)
    throw new Error(`unsupported native teacher row version: ${String(row.version)}`);
  if (typeof row.id !== 'string' || !row.id) throw new Error('native teacher row is missing id');
  record(row.task, 'task'); record(row.provenance, 'provenance'); record(row.outcome, 'outcome');
  if (!Array.isArray(row.trajectory)) throw new TypeError('trajectory must be an array');
  if (typeof row.outcome.accepted !== 'boolean') throw new Error(`${row.id}: outcome.accepted must be boolean`);
  return row;
}

/** The system and user messages plus the runtime's pre-filled scope exchanges (tool calls with `scope_` ids). */
function openingLength(context: Dict[]): number {
  const calls = (message: Dict | undefined) => ((message?.tool_calls ?? []) as Dict[]);
  let length = 2;
  while (context[length]?.role === 'assistant' && calls(context[length]).length &&
      calls(context[length]).every(call => String(call.id).startsWith('scope_')))
    length += 1 + calls(context[length]).length;
  return length;
}

/** A checker's verdict on an attempt ({ ok, certificate, problem }), refusing it. */
const CHECKER_REFUSAL = /"?ok"?\s*:\s*false\s*,\s*"?certificate"?\s*:/;

/**
 * Convert accepted native teacher runs to one self-contained model decision per row.
 * Context is copied from that exact native request, never assembled by appending one decision onto another.
 * Rows with checkpoint turns (conversation rollover, since retired) are rejected. With `failedRuns`, runs that were not
 * accepted are materialized too. Their decisions require an explicit exact-target quality review to be approved.
 */
export type NativeDecisionHold = { trajectory_id: string; source_row_sha256: string;
  decision_index: number; reason: string; evidence: string[] };

/**
 * Scripted authored-source references can prove that a declared source world
 * runs, but they are not teacher observations or training approvals. Keep the
 * materializer's execution evidence and targets intact while giving these
 * newly generated rows an explicit pending/no-admission disposition.
 */
export function markAuthoredStaticReferencePending(turns: Dict[]): Dict[] {
  return turns.map(turn => {
    const training = record(turn.training_admission, 'training_admission');
    const trace = record(turn.trace_admission, 'trace_admission');
    const decision = record(turn.decision, 'decision');
    const provenance = record(turn.provenance, 'provenance');
    return { ...turn,
      provenance: { ...provenance, training_provenance: 'authored-source-static-reference;zero-provider-calls' },
      training_admission: { ...training, kind: 'authored-static-reference-pending-review', approved: false,
        reason: 'scripted source reference requires independent semantic and training review' },
      trace_admission: { ...trace, admitted: false, kind: 'authored-static-reference-not-teacher-trace' },
      decision: { ...decision, training_approved: false },
    };
  });
}

export function materializeNativeRows(input: unknown[], options: { directAnswers?: boolean; failedRuns?: boolean;
  decisionHolds?: readonly NativeDecisionHold[]; decisionApprovals?: readonly NativeDecisionApproval[] } = {}): {
  turns: Dict[]; acceptedRows: number; rejectedRows: number;
  unlinked: { id: string; outcomes: number; reason?: string }[];
} {
  const turns: Dict[] = [];
  const unlinked: { id: string; outcomes: number }[] = [];
  let acceptedRows = 0, rejectedRows = 0;
  for (const candidate of input) {
    const row = validateRow(candidate);
    const rowDigest = nativeRowDigest(row);
    const reviewedHolds = new Map<number, NativeDecisionHold>();
    for (const hold of options.decisionHolds ?? []) if (hold.trajectory_id === row.id) {
      if (hold.source_row_sha256 !== rowDigest) throw new Error(`semantic review source hash mismatch: ${row.id}`);
      if (!Number.isSafeInteger(hold.decision_index) || hold.decision_index < 0 ||
          hold.decision_index >= row.trajectory.length || !row.trajectory[hold.decision_index]?.assistant ||
          typeof hold.reason !== 'string' || !hold.reason.trim() || !Array.isArray(hold.evidence) ||
          !hold.evidence.length || hold.evidence.some(value => typeof value !== 'string' || !value.trim()))
        throw new Error(`invalid semantic decision review: ${row.id}`);
      if (reviewedHolds.has(hold.decision_index)) throw new Error(`duplicate semantic decision review: ${row.id}`);
      reviewedHolds.set(hold.decision_index, structuredClone(hold));
    }

    const reviewedApprovals = new Map<number, NativeDecisionApproval>();
    for (const approval of options.decisionApprovals ?? []) if (approval.trajectory_id === row.id) {
      if (approval.source_row_sha256 !== rowDigest) throw new Error(`semantic approval source hash mismatch: ${row.id}`);
      if (!Number.isSafeInteger(approval.decision_index) || approval.decision_index < 0 ||
          approval.decision_index >= row.trajectory.length || !row.trajectory[approval.decision_index]?.assistant)
        throw new Error(`invalid semantic decision approval: ${row.id}`);
      if (reviewedApprovals.has(approval.decision_index) || reviewedHolds.has(approval.decision_index))
        throw new Error(`duplicate or conflicting semantic decision approval: ${row.id}`);
      reviewedApprovals.set(approval.decision_index, structuredClone(approval));
    }

    if (sourceConversionProblems(row).length) { rejectedRows++; continue; }
    const taskIr = record(row.task.program_ir, `${row.id}.task.program_ir`);
    const program = taskIr as unknown as ProgramRecord;
    // Direct exports and failed-run pair discovery must honor the same source holds as collection/admission.
    if (trainingQualityReason(row) || runtimeFailureReason(row) || quarantineReason(program) || retiredFamily(program)) { rejectedRows++; continue; }
    // Rows from before conversation rollover was retired contain checkpoint notes and cut contexts.
    const rolledOver = row.trajectory.some(turn => (turn as Dict | undefined)?.phase === 'checkpoint');
    if ((!row.outcome.accepted && !options.failedRuns) || rolledOver) { rejectedRows++; continue; }
    acceptedRows++;
    const role = String(row.provenance.collection_role ?? 'teacher');
    const semantics = taskIr.semantics && typeof taskIr.semantics === 'object' && !Array.isArray(taskIr.semantics) ?
      taskIr.semantics as Dict : {};
    const evidenceOracle = oracleLevel(semantics.oracle);
    // A handoff row (teacher/handoff.ts) replays another model's turns up to where a teacher took over.
    const handoff = taskIr.handoff as Parameters<typeof handoffTurns>[1] | undefined;
    const replayedTurns = new Set(handoff ? handoffTurns(row.trajectory as unknown as Turn[], handoff).prefix : []);
    const ledger = Array.isArray(row.outcome.action_ledger) ? row.outcome.action_ledger.map((event, index) =>
      record(event, `${row.id}.outcome.action_ledger[${index}]`)) : [];
    // Keep the original model text for an already-written soft body. The decoded action contains only the
    // content-addressed sentinel; the paired raw call is the only exact source text available to the converter.
    const rawEvalArguments = new Map<string, Dict>();
    for (const turn of row.trajectory) {
      const calls = Array.isArray((turn.assistant as Dict | undefined)?.calls) ?
        (turn.assistant as Dict).calls as Dict[] : [];
      const rawCalls = Array.isArray((turn.model_response as Dict | undefined)?.raw_calls) ?
        (turn.model_response as Dict).raw_calls as Dict[] : [];
      calls.forEach((call, callIndex) => {
        if (call.source_tool !== 'eval' || !call.arguments || typeof call.arguments !== 'object') return;
        const actions = ledger.filter(event => event.call_id === turn.invocation_id && event.name === 'eval' &&
          (canonical(event.arguments) === canonical(call.arguments) || softBodyActionMatches(call.arguments, event.arguments,
            row, typeof turn.invocation_id === 'string' ? turn.invocation_id : undefined)));
        if (actions.length !== 1 || typeof actions[0]!.tool_call_id !== 'string') return;
        const raw = rawCalls[callIndex], fn = raw && typeof raw.function === 'object' ? raw.function as Dict : undefined;
        if (fn?.name !== 'eval' || typeof fn.arguments !== 'string') return;
        const args = parseArguments(fn.arguments);
        if (args && typeof args === 'object' && !Array.isArray(args))
          rawEvalArguments.set(`${String(actions[0]!.call_id)}\u0000${String(actions[0]!.tool_call_id)}`, args as Dict);
      });
    }
    // Each call (the root and every nl child) has its own actions, in order; the trajectory interleaves the calls'
    // decisions. A call's decisions, recognised by their opening (with the inputs it lists), claim the one action log whose next action their
    // first call matches, and are linked to it in order from then on.
    const parents = new Map<string, string>();
    const hostOutputs = new Map<string, Dict>();
    const instructionSites = new Map<string, Dict>();
    const lastInvocationDecision = new Map<string, number>();
    for (let i = 0; i < row.trajectory.length; i++) {
      const turn = row.trajectory[i]!;
      if (typeof turn.invocation_id === 'string') lastInvocationDecision.set(turn.invocation_id, i);
    }
    for (const entry of Array.isArray(row.outcome.invocation_ledger) ? row.outcome.invocation_ledger : []) {
      const item = record(entry, 'invocation ledger entry');
      if (typeof item.invocation_id === 'string' && typeof item.parent_invocation_id === 'string')
        parents.set(item.invocation_id, item.parent_invocation_id);
      if (typeof item.invocation_id === 'string' && item.inline_instruction_site && typeof item.inline_instruction_site === 'object') {
        const site = structuredClone(record(item.inline_instruction_site, 'inline instruction site'));
        const origin = site.origin && typeof site.origin === 'object' ? record(site.origin, 'inline instruction origin') : {};
        const actions = ledger.filter(event => event.call_id === origin.parentInvocationId &&
          event.tool_call_id === origin.toolCallId && event.name === 'eval');
        const code = actions.length === 1 && typeof (actions[0]!.arguments as Dict | undefined)?.code === 'string' ?
          (actions[0]!.arguments as Dict).code as string : undefined;
        const span = site.template_span && typeof site.template_span === 'object' ? site.template_span as Dict : {};
        const segments = Array.isArray(site.template_segments) ? site.template_segments : [];
        const holes = Array.isArray(site.interpolations) ? site.interpolations : [];
        const softBodySite = typeof site.soft_body_id === 'string';
        const bindingsValid = segments.length === holes.length + 1 && segments.every(value => typeof value === 'string') &&
          holes.every(value => value && typeof value === 'object' && typeof (value as Dict).rendered === 'string');
        let realized = bindingsValid ? String(segments[0]) + holes.map((value, index) =>
          String((value as Dict).rendered) + String(segments[index+1])).join('') : undefined;
        if (realized !== undefined && !realized.endsWith('\n')) realized += '\n';
        const reasons = [
          ...(site.schema !== 'natlang.inline_instruction_site/1' ? ['site-schema-mismatch'] : []),
          ...(!softBodySite && !bindingsValid ? ['interpolation-bindings-missing'] : []),
          ...(site.soft_body_id === undefined && (realized === undefined || realized !== site.realized_instruction) ? ['realized-instruction-mismatch'] : []),
          ...(!origin.toolCallId || origin.parentInvocationId !== item.parent_invocation_id ? ['parent-action-lineage-missing'] : []),
          ...(actions.length !== 1 ? ['parent-action-ambiguous-or-missing'] : []),
          ...(actions.length === 1 && !['ok','completed'].includes(String(actions[0]!.outcome)) ? ['parent-action-failed'] : []),
          ...(code === undefined || hexDigest(code) !== origin.writtenCodeSha256 ? ['written-code-hash-mismatch'] : []),
          ...(origin.checkedCodeSha256 !== origin.writtenCodeSha256 &&
            !(typeof site.soft_body_id === 'string' && site.checked_template_span && origin.sourceTemplateSpan) ?
            ['checked-source-requires-span-mapping'] : []),
          ...(code === undefined || !Number.isInteger(span.start) || !Number.isInteger(span.end) ||
            Number(span.start) < 0 || Number(span.end) > code.length || Number(span.end) <= Number(span.start) ||
            code[Number(span.start)] !== '`' || code[Number(span.end)-1] !== '`' ? ['template-span-mismatch'] : []),
        ];
        if (typeof site.soft_body_id === 'string') {
          const rawArgs = typeof origin.toolCallId === 'string' && typeof origin.parentInvocationId === 'string' ?
            rawEvalArguments.get(`${origin.parentInvocationId}\u0000${origin.toolCallId}`) : undefined;
          const rawCode = typeof rawArgs?.code === 'string' ? rawArgs.code : undefined;
          const markers = rawCode ? [...rawCode.matchAll(/<\|neuralese\|>([\s\S]*?)<\|\/neuralese\|>/g)] : [];
          const sentinel = `\uE000${site.soft_body_id}\uE001`;
          const paired = /^nz1_[a-z2-7]{20,}$/.test(site.soft_body_id) && rawCode !== undefined && code !== undefined && markers.length === 1 &&
            rawCode.replace(markers[0]![0], sentinel) === code;
          if (!paired) reasons.push('soft-body-source-unavailable-or-unpaired');
          else {
            site.raw_body_source = markers[0]![1];
            site.raw_body_source_sha256 = hexDigest(markers[0]![1]!);
            const checked = desugarNlCalls(sourceWithLiteralCalls(code));
            if (hexDigest(checked) !== origin.checkedCodeSha256)
              reasons.push('checked-source-transform-mismatch');
            const spanStart = Number(span.start), spanEnd = Number(span.end);
            if (code.slice(spanStart, spanEnd) !== `\`${sentinel}\`` ||
                canonical(origin.sourceTemplateSpan) !== canonical({ start: spanStart, end: spanEnd }))
              reasons.push('soft-body-source-span-mismatch');
            const checkedSpan = site.checked_template_span as Dict | undefined;
            const checkedStart = Number(checkedSpan?.start), checkedEnd = Number(checkedSpan?.end);
            if (!Number.isSafeInteger(checkedStart) || !Number.isSafeInteger(checkedEnd) ||
                checked.slice(checkedStart, checkedEnd) !== `\`\${__neuralese.body(${JSON.stringify(site.soft_body_id)})}\``)
              reasons.push('checked-soft-body-span-mismatch');
          }
        }
        if (item.captures && typeof item.captures === 'object')
          site.runtime_captures = structuredClone(item.captures);
        instructionSites.set(item.invocation_id, { site: structuredClone(site),
          validation: { valid: !reasons.length, reasons },
          scope: 'compiler/runtime provenance only; not yet a converted instruction writer/read or training admission' });
      }
      if (typeof item.invocation_id === 'string' && item.host_result && typeof item.host_result === 'object') {
        const capture = record(item.host_result, 'observed host result');
        const lineage = item.completion_status === 'done' && capture.capture_kind === 'invocation_output' &&
          capture.call_id === item.invocation_id && capture.parent_call_id === (item.parent_invocation_id ?? null);
        const exact = capture.complete !== true || (Object.hasOwn(capture, 'value') &&
          capture.value_sha256 === hexDigest(JSON.stringify(capture.value)) &&
          capture.bytes === new TextEncoder().encode(JSON.stringify(capture.value)).byteLength);
        hostOutputs.set(item.invocation_id, { capture: structuredClone(capture),
          validation: { valid: lineage && exact, ...(!lineage ? { reason: 'invocation lineage mismatch' } :
            !exact ? { reason: 'value hash or byte count mismatch' } : {}) },
          scope: 'observed host result metadata; not a model target or training admission' });
      }
    }
    const logs = new Map<string, Dict[]>();
    for (const event of ledger) { const key = String(event.call_id ?? ''); logs.set(key, [...logs.get(key) ?? [], event]); }
    // An unterminated text marker is faithfully materialized by the runtime as a truncated body. Preserve its
    // action for diagnostics, but hold every later invocation whose observed block_read consumed that block.
    const truncatedSoftBlocks = new Set<string>();
    for (const sourceValue of row.trajectory) {
      const source = record(sourceValue, `${row.id}.trajectory`);
      if (typeof source.invocation_id !== 'string') continue;
      for (const callValue of Array.isArray((source.assistant as Dict | undefined)?.calls) ?
        (source.assistant as Dict).calls as Dict[] : []) {
        const call = record(callValue, 'assistant call');
        if (call.source_tool !== 'eval' || !hasExactRawModelCall(source, call)) continue;
        const events = logs.get(source.invocation_id) ?? [];
        for (const event of events) {
          if (event.name !== 'eval' || softBodyActionMatch(call.arguments, event.arguments) !== 'truncated' ||
              !softBodyActionMatches(call.arguments, event.arguments, row, source.invocation_id)) continue;
          const code = (event.arguments as Dict | undefined)?.code;
          if (typeof code !== 'string') continue;
          const sentinel = /(nz1_[a-z2-7]{20,})/.exec(code);
          if (sentinel) truncatedSoftBlocks.add(sentinel[1]!);
        }
      }
    }
    const truncatedSoftReaderInvocations = new Set<string>();
    if (truncatedSoftBlocks.size) for (const event of Array.isArray(row.outcome.execution_graph) ? row.outcome.execution_graph : []) {
      if (!event || typeof event !== 'object' || (event as Dict).kind !== 'block_read' ||
          typeof (event as Dict).call_id !== 'string' || typeof (event as Dict).block !== 'string' ||
          !truncatedSoftBlocks.has((event as Dict).block as string)) continue;
      truncatedSoftReaderInvocations.add((event as Dict).call_id as string);
    }
    const next = new Map<string, number>(), claimed = new Map<string, string>(), owners = new Set<string>();
    // Per call (by its opening): each call signature already made, with the result it got.
    const sentBefore = new Map<string, Map<string, string>>();
    // Once per row: every decision names the row it came from, and hashing a long row per decision is quadratic.
    const chunkRewrite = row.provenance.student_chunk_rewrite as Dict | undefined;
    const cutoff = chunkRewrite?.supervision_cutoff_decision;
    if (cutoff !== undefined && cutoff !== null &&
        (!Number.isSafeInteger(cutoff) || Number(cutoff) < 0 || Number(cutoff) >= row.trajectory.length))
      throw new Error(`${row.id}: invalid chunk-rewrite supervision cutoff`);
    const outcomeDigest = nativeRowDigest(row.outcome);
    let linked = 0, diagnosticArgsUnlinked = false;
    const rowTurns: Dict[] = [];
    for (let index = 0; index < row.trajectory.length; index++) {
      const source = record(row.trajectory[index], `${row.id}.trajectory[${index}]`);
      const contextSource = messages(source.context, `${row.id}.trajectory[${index}].context`);
      if (!contextSource.length || contextSource[0]?.role !== 'system' || contextSource[1]?.role !== 'user')
        throw new Error(`${row.id}: decision ${index} lacks a fresh system/user opening context`);

      const offered = toolSchemas(source.tools_offered ?? [], `${row.id}.trajectory[${index}].tools_offered`);
      const assistant = record(source.assistant, `${row.id}.trajectory[${index}].assistant`);
      const planningAttempted = Object.hasOwn(assistant, 'execution_plan');
      const executionPlan = typeof assistant.execution_plan === 'string' && assistant.execution_plan.trim() ?
        assistant.execution_plan : null;
      const retainedReasoning = planningAttempted ? executionPlan : assistant.reasoning ?? null;
      // A call's whole opening, its inputs included: parallel nl calls can share their instructions word for word.
      const invocation = typeof source.invocation_id === 'string' ? source.invocation_id : undefined;
      const caller = invocation ?? JSON.stringify(contextSource.slice(0, openingLength(contextSource)).map(normalizeContextMessage));
      // A decision that sees nothing but its opening starts a call: even a call identical to an earlier one (the
      // same instructions on the same input) has its own actions.
      if (!invocation && contextSource.length === openingLength(contextSource)) { claimed.delete(caller); sentBefore.delete(caller); }
      const calls = Array.isArray(assistant.calls) ? assistant.calls.map((value, callIndex) => {
        const call = record(value, `${row.id}.trajectory[${index}].assistant.calls[${callIndex}]`);
        const normalized: Dict = { tool: String(call.tool ?? ''), source_tool: String(call.source_tool ?? call.tool ?? ''),
          arguments: structuredClone(call.arguments ?? {}), call_id: call.call_id ?? null };
        const exactRaw = hasExactRawModelCall(source, normalized);
        const eventMatches = (candidate: Dict | undefined) => !!candidate &&
          (callMatches(normalized, candidate) || (exactRaw && candidate.name === normalized.source_tool &&
            softBodyActionMatches(normalized.arguments, candidate.arguments, row, invocation)) || (exactRaw && candidate.name === normalized.source_tool &&
            normalized.source_tool === 'return_result' && softReturnResultActionMatches(normalized.arguments,
              candidate.arguments, row, invocation)) || (exactRaw && candidate.name === normalized.source_tool &&
            containsIncompleteDiagnostic(candidate.arguments)));
        const projectedOutcome = [...logs.keys()].some(key => !owners.has(key) &&
          logs.get(key)![next.get(key) ?? 0]?.name === normalized.source_tool &&
          containsIncompleteDiagnostic(logs.get(key)![next.get(key) ?? 0]?.arguments));
        let log = claimed.get(caller);
        if (log === undefined) {
          log = invocation ? (logs.has(invocation) ? invocation : undefined) :
            [...logs.keys()].find(key => !owners.has(key) && eventMatches(logs.get(key)![next.get(key) ?? 0]));
          if (log !== undefined) { claimed.set(caller, log); owners.add(log); }
        }
        const at = log === undefined ? 0 : next.get(log) ?? 0, event = log === undefined ? undefined : logs.get(log)![at];
        const projected = projectedOutcome || containsIncompleteDiagnostic(normalized.arguments) ||
          (event !== undefined && containsIncompleteDiagnostic(event.arguments));
        if (projected && !exactRaw) {
          diagnosticArgsUnlinked = true;
          normalized.outcome = { event_index: null, trace_seq: null, name: normalized.source_tool,
            arguments: structuredClone(normalized.arguments), status: 'unlinked_incomplete_diagnostic_arguments',
            result: null, diagnostics: [] };
        } else if (event && eventMatches(event)) {
          const eventArguments = containsIncompleteDiagnostic(event.arguments) ? normalized.arguments : event.arguments;
          const softBodyMode = exactRaw && event.name === 'eval' ?
            softBodyActionMatch(normalized.arguments, event.arguments) : undefined;
          const eventDiagnostics = Array.isArray(event.diagnostics) ? structuredClone(event.diagnostics) : [];
          normalized.outcome = { event_index: ledger.indexOf(event), trace_seq: event.seq ?? null,
            ...(typeof event.tool_call_id === 'string' ? { tool_call_id: event.tool_call_id } : {}),
            name: event.name, arguments: structuredClone(eventArguments ?? {}),
            ...(containsIncompleteDiagnostic(event.arguments) ? { arguments_source: 'exact_raw_model_call' } : {}),
            status: event.outcome ?? null, result: event.result_text ?? null,
            diagnostics: [...eventDiagnostics,
              ...(softBodyMode === 'truncated' ? ['coerced-truncated-neuralese-marker'] : []),
              ...(invocation && truncatedSoftReaderInvocations.has(invocation) ?
                ['unreviewed-truncated-neuralese-dependency'] : [])] };
          next.set(log!, at + 1); linked++;
        } else normalized.outcome = { event_index: null, trace_seq: null,
          name: normalized.source_tool, arguments: structuredClone(normalized.arguments),
          // No action log to link to at all: rows collected before child calls' actions were recorded. The call may
          // well have run; its outcome is unknown, so it is not a training target, and says why.
          status: log === undefined && [...logs.keys()].every(key => owners.has(key)) ? 'not_recorded' : 'not_executed',
          result: null, diagnostics: [] };
        return normalized;
      }) : [];

      const programId = taskIr.id ?? null;
      const target = trainingTarget(assistant, calls, index);
      const skill = calls.length ? calls.map(call => String(call.source_tool)).join('+') : 'reply';
      const badStatuses = new Set(['rejected', 'refused', 'error', 'not_executed', 'not_recorded']);
      const fromStudentPrefix = replayedTurns.has(index);
      const ranCleanly = calls.every(call =>
        !badStatuses.has(String(record(call.outcome, 'call outcome').status)) &&
        !(record(call.outcome, 'call outcome').diagnostics as unknown[] ?? [])
          .some(code => String(code).startsWith('coerced-') ||
            code === 'unreviewed-truncated-neuralese-dependency'));
      // Two kinds of step run cleanly and still teach nothing to repeat: a call this call already made with the same
      // result (a detour), and an attempt the task's own checker rejected (the code worked; the attempt did not).
      // Both stay in the context of later steps, where they are what the model recovers from.
      const earlier = sentBefore.get(caller) ?? sentBefore.set(caller, new Map()).get(caller)!;
      const signature = (call: Dict) => JSON.stringify([call.tool, call.arguments]);
      const resultOf = (call: Dict) => String(record(call.outcome, 'call outcome').result ?? '');
      const detour = calls.length > 0 && calls.every(call => earlier.get(signature(call)) === resultOf(call));
      const redundantSkillRead = calls.some(call => call.tool === 'read_code' &&
        (record(call.outcome, 'call outcome').diagnostics as unknown[] ?? []).includes('skill-unchanged-read'));
      const refusedAttempt = calls.some(call => call.tool === 'eval' && CHECKER_REFUSAL.test(resultOf(call)));
      // A decision that failed: an action the runtime rejected, refused or that raised, or an attempt the checker refused.
      const failedAction = refusedAttempt || calls.some(call =>
        ['rejected', 'refused', 'error'].includes(String(record(call.outcome, 'call outcome').status)));
      for (const call of calls) earlier.set(signature(call), resultOf(call));
      // An answer given without reasoning towards it (a scripted conclusion behind a one-line note) teaches a reasoning
      // student to answer without reasoning; a student that answers directly is trained on it (options.directAnswers).
      // Released typed labels have no authored chain of thought: explicitly train their answer,
      // masking action-note reasoning. This exception requires a verified pinned source replay.
      const typedSourceAnswer = retiredWorkflowEvaluationReleased(taskIr) &&
        row.provenance.synthetic_reasoning === 'action-notes/1' &&
        (row.provenance.source_conversion as Dict | undefined)?.adapter === 'natlang.workflowevals_adapter/1';
      const heldDirect = assistant.direct_answer === true && !options.directAnswers && !typedSourceAnswer;
      // A corrected variant (teacher/corrections.ts) trains its fix only; the rest repeats its parent's turns.
      const variant = row.provenance.variant as { decision?: number } | undefined;
      const variantContext = variant !== undefined && index !== variant.decision;
      const afterChunkCutoff = cutoff !== undefined && cutoff !== null && index > Number(cutoff);
      const semanticHold = reviewedHolds.get(index);
      const semanticApproval = reviewedApprovals.get(index);
      if (semanticApproval && !validNativeDecisionApproval(semanticApproval, { trajectory_id: row.id,
          source_row_sha256: rowDigest, decision_index: index, target_sha256: nativeDecisionTargetDigest(target) }))
        throw new Error(`semantic decision approval target or evidence mismatch: ${row.id}:${index}`);
      const decisionApproved = !semanticHold && !afterChunkCutoff && (row.outcome.accepted || !!semanticApproval) && !fromStudentPrefix && ranCleanly && !detour && !redundantSkillRead && !refusedAttempt &&
        !heldDirect && !variantContext;
      rowTurns.push({ version: NATIVE_TEACHER_TURN_VERSION,
        id: `${row.id}:decision:${String(index).padStart(4, '0')}`,
        source_ref: { trajectory_id: row.id, source_row_sha256: rowDigest,
          ...(semanticApproval ? { native_target_sha256: semanticApproval.target_sha256 } : {}),
          ...(invocation ? { invocation_id: invocation, ...(instructionSites.has(invocation) ? { inline_instruction_site: instructionSites.get(invocation) } : {}), ...(parents.has(invocation) ? { parent_invocation_id: parents.get(invocation) } : {}),
            ...(lastInvocationDecision.get(invocation) === index && hostOutputs.has(invocation) ?
              { host_result_capture: hostOutputs.get(invocation) } : {}) } : {}),
          program_ir_id: programId },
        provenance: row.provenance,
        task: row.task,
        program_id: programId,
        family: `${role}_program`,
        task_family: taskIr.family ?? null,
        task_kind: taskIr.kind ?? null,
        task_modality: taskIr.task_modality ?? (typeof semantics.root === 'string' && semantics.files &&
          typeof semantics.files === 'object' && /(?:^|\n)kind: directory-reducer(?:\n|$)/.test(
            String((semantics.files as Dict)[semantics.root] ?? '')) ? 'directory-reducer' : 'function'),
        ...(taskIr.split ? { split: taskIr.split } : {}),
        ...(taskIr.source_ids ? { source_ids: taskIr.source_ids } : {}),
        ...(taskIr.source_revisions ? { source_revisions: taskIr.source_revisions } : {}),
        ...(evidenceOracle ? { oracle_level: evidenceOracle } : {}),
        skill,
        provisional_gold: false,
        source_program_ids: programId === null ? [] : [programId],
        source_groups: [...new Set([...(programId === null ? [] : [programId]),
          ...(Array.isArray(taskIr.source_groups) ? taskIr.source_groups.filter((group): group is string => typeof group === 'string') : [])])],
        source: `${role}-native`,
        gold_sources: [...(Array.isArray(taskIr.gold_sources) ? taskIr.gold_sources : []), `checked-${role}-trajectory`, 'exact-runtime-oracle'],
        license: taskIr.license ?? 'project-generated',
        messages: publicValue(contextSource),
        tools: publicValue(source.tools_offered ?? []),
        target,
        teacher_reasoning: retainedReasoning,
        // Only reasoning a model wrote is a target: rationales a model wrote for scripted actions (static
        // demonstrations) train like a teacher's reasoning. Authored action plans and stock action notes are
        // template text: kept in context, never trained (owner 2026-10-06).
        ...(row.provenance.synthetic_reasoning ? { teacher_reasoning_trained:
          row.provenance.synthetic_reasoning === 'rationalized-actions/1' } : {}),
        teacher_execution_plan: executionPlan,
        teacher_trajectory_id: row.id,
        teacher_trajectory_digest: rowDigest,
        // Keep the final verdict with each decision for downstream continuation validation, without copying files.
        outcome: { accepted: row.outcome.accepted, status: row.outcome.status,
          ...(row.outcome.oracle ? { oracle: row.outcome.oracle } : {}) },
        training_admission: { kind: semanticApproval ? 'reviewed-native-decision' : 'exact-native-runtime-oracle', approved: decisionApproved,
          ...(semanticApproval ? { semantic_review: semanticApproval } : {}),
          ...(semanticHold ? { semantic_review: semanticHold } : {}),
          ...(evidenceOracle ? { oracle_level: evidenceOracle } : {}),
          ...(decisionApproved ? {} : { reason: semanticHold ? semanticHold.reason : afterChunkCutoff ? 'beyond verified chunk-rewrite supervision cutoff' : variantContext ? 'context of a corrected variant' :
            heldDirect ? 'an answer given without reasoning towards it' :
            (fromStudentPrefix ? 'student replay prefix is not a teacher correction' :
            calls.some(call => record(call.outcome, 'call outcome').status === 'not_recorded') ?
              'the outcome of a call in this decision was not recorded' :
            !ranCleanly ? 'decision contains a failed or unexecuted proposal' :
            redundantSkillRead ? 'retrieves unchanged skill instructions again' :
            detour ? 'repeats an earlier call of this call with the same result' :
            refusedAttempt ? "the task's checker rejected this attempt" : 'the run was not accepted') }) },
        trace_admission: { admitted: true, kind: 'exact-native-runtime-oracle',
          final_outcome_sha256: outcomeDigest },
        decision: { index,
          durable_opening: contextSource.slice(0, openingLength(contextSource)).map(normalizeContextMessage),
          tool_schemas: offered,
          assistant: { content: assistant.content ?? '', reasoning: retainedReasoning,
            execution_plan: executionPlan,
            calls },
          failed_action: failedAction,
          training_approved: decisionApproved,
          source_raw_response_sha256: source.raw_response_sha256 ?? null,
          source_tools_offered: structuredClone(source.tools_offered ?? []) },
      });
    }
    // A decision linked to another call's outcome would train the wrong target, so a row whose outcomes cannot all
    // be linked is not used (concurrent calls with the very same opening cannot be told apart), and is reported.
    if (linked !== ledger.length) {
      acceptedRows--; rejectedRows++;
      unlinked.push({ id: row.id, outcomes: ledger.length - linked,
        ...(diagnosticArgsUnlinked ? { reason: 'incomplete_diagnostic_arguments_without_exact_raw_model_call' } : {}) });
      continue;
    }
    turns.push(...rowTurns);
  }
  return { turns, acceptedRows, rejectedRows, unlinked };
}
