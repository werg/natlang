import { handoffTurns, type Turn } from './replay.js';
import { hexDigest } from '../native/hash.js';
import { nativeDecisionTargetDigest, validNativeDecisionApproval, type NativeDecisionApproval } from '../native/decision-review.js';
import { sourceConversionProblems, retiredWorkflowEvaluationReleased } from './source-conversion.js';
import { trainingQualityReason, runtimeFailureReason, quarantineReason, retiredFamily } from './curriculum-policy.js';
import type { ProgramRecord } from './program.js';
import { sourceWithLiteralCalls } from '../native/neuralese.js';
import { desugarNlCalls } from '../compiler/nl-call.js';
import { pureLiteralEvalReturn } from '../compiler/neuralese-conversion.js';
import { formatType, parseType, type Type } from '../native/types.js';

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

export type PureLiteralTypedTextWitness = {
  trajectory_id: string;
  source_row_sha256: string;
  source_result_row_sha256: string;
  generation_turn: { trajectory_index: number; invocation_id: string; turn: number;
    request_sha256: string; raw_response_sha256: string };
  target_code_sha256: string;
  writer_call_id: string;
  writer_node: string;
  block_id: string;
  body_source: string;
  body_sha256: string;
  result_type: string;
  source: string;
  marker_context: string;
  source_kind: string;
};

/**
 * Make a held derived target only for an authenticated pure-literal eval return.
 * The normal native materializer still emits the original eval action unchanged;
 * this view is explicitly transformed to the equivalent typed return tool call.
 */
export function derivePureLiteralTypedTextTarget(rowValue: unknown,
  witnessValue: PureLiteralTypedTextWitness): (NativeRow & Dict) | undefined {
  const row = record(rowValue, 'derived typed-text source row');
  const target = record(row.target, 'derived typed-text source target');
  const sourceRef = record(row.source_ref, 'derived typed-text source reference');
  const preview = record(row.preview_source_selection, 'derived typed-text preview selection');
  const generationTurn = record(preview.target_generation_turn, 'derived typed-text generation turn');
  const decision = record(row.decision, 'derived typed-text source decision');
  const assistantDecision = decision.assistant && typeof decision.assistant === 'object' ?
    decision.assistant as Dict : undefined;
  const decisionCalls = Array.isArray(assistantDecision?.calls) ? assistantDecision.calls as Dict[] : [];
  const writeReceipts = decisionCalls.flatMap(call => {
    const outcome = call.outcome && typeof call.outcome === 'object' ? call.outcome as Dict : undefined;
    return Array.isArray(outcome?.typed_result_writes) ? outcome.typed_result_writes as Dict[] : [];
  });
  const targetCalls = Array.isArray(target.tool_calls) ? target.tool_calls : [];
  if (target.role !== 'assistant' || targetCalls.length !== 1 ||
      !targetCalls[0] || typeof targetCalls[0] !== 'object') return undefined;
  const originalCall = record(targetCalls[0], 'derived typed-text target call');
  const fn = record(originalCall.function, 'derived typed-text target function');
  if (fn.name !== 'eval' || typeof fn.arguments !== 'string') return undefined;
  let originalArgs: Dict;
  try { originalArgs = record(JSON.parse(fn.arguments), 'derived typed-text eval arguments'); }
  catch { return undefined; }
  if (originalArgs.finish !== true) return undefined;
  const parsed = pureLiteralEvalReturn(originalArgs.code);
  if (!parsed) return undefined;
  const matchingWrites = writeReceipts.filter(write => write.schema === 'natlang.typed-result-write/1' &&
    write.trajectory_id === witnessValue.trajectory_id && write.source_row_sha256 === witnessValue.source_row_sha256 &&
    write.invocation_id === witnessValue.generation_turn.invocation_id &&
    write.writer_call_id === witnessValue.writer_call_id && write.writer_node === witnessValue.writer_node &&
    write.block_id === witnessValue.block_id && write.source_kind === witnessValue.source_kind &&
    write.source === witnessValue.source && write.result_type === witnessValue.result_type &&
    write.body_sha256 === witnessValue.body_sha256 &&
    write.request_sha256 === witnessValue.generation_turn.request_sha256 &&
    write.raw_response_sha256 === witnessValue.generation_turn.raw_response_sha256);
  const sha256 = (value: unknown): value is string => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
  const authenticatedWriter = witnessValue.source === 'eval-finish' && matchingWrites.length === 1 &&
    witnessValue.marker_context === 'return-result';
  if (!parsed || parsed.value !== witnessValue.body_source ||
      !sha256(witnessValue.body_sha256) || !sha256(witnessValue.source_row_sha256) ||
      !sha256(witnessValue.source_result_row_sha256) || !sha256(witnessValue.target_code_sha256) ||
      !sha256(witnessValue.generation_turn.request_sha256) || !sha256(witnessValue.generation_turn.raw_response_sha256) ||
      !Number.isInteger(witnessValue.generation_turn.trajectory_index) ||
      !Number.isInteger(witnessValue.generation_turn.turn) || witnessValue.generation_turn.turn < 0 ||
      !witnessValue.trajectory_id || !witnessValue.generation_turn.invocation_id ||
      !witnessValue.writer_call_id || !witnessValue.writer_node || !/^nz1_[a-z0-9]{20,}$/.test(witnessValue.block_id) ||
      !authenticatedWriter || witnessValue.trajectory_id !== sourceRef.trajectory_id ||
      witnessValue.source_row_sha256 !== sourceRef.source_row_sha256 ||
      hexDigest(witnessValue.body_source) !== witnessValue.body_sha256 ||
      witnessValue.result_type !== 'Neuralese<string>' || witnessValue.source_kind !== 'typed-text-result' ||
      witnessValue.generation_turn.invocation_id !== sourceRef.invocation_id ||
      witnessValue.writer_call_id !== sourceRef.invocation_id ||
      !witnessValue.writer_node.startsWith(`${witnessValue.writer_call_id}#`) ||
      witnessValue.generation_turn.raw_response_sha256 !== decision.source_raw_response_sha256 ||
      witnessValue.generation_turn.invocation_id !== generationTurn.invocation_id ||
      witnessValue.generation_turn.trajectory_index !== generationTurn.trajectory_index ||
      witnessValue.generation_turn.request_sha256 !== generationTurn.request_sha256 ||
      witnessValue.generation_turn.raw_response_sha256 !== generationTurn.raw_response_sha256 ||
      hexDigest(originalArgs.code as string) !== witnessValue.target_code_sha256) return undefined;
  const site = sourceRef.inline_instruction_site && typeof sourceRef.inline_instruction_site === 'object' ?
    (sourceRef.inline_instruction_site as Dict).site as Dict | undefined : undefined;
  const returns = site && typeof site === 'object' ? site.returns as Dict | undefined : undefined;
  if (returns?.natlang !== witnessValue.result_type) return undefined;
  const tools = Array.isArray(row.tools) ? row.tools as Dict[] : [];
  if (!tools.some(tool => (tool.function as Dict | undefined)?.name === 'return_result')) return undefined;
  if (row.training_admission && typeof row.training_admission === 'object' &&
      (row.training_admission as Dict).approved !== false) return undefined;
  const derivedCallId = `derived_${hexDigest(`${row.id}\u0000${witnessValue.writer_node}\u0000${witnessValue.body_sha256}`).slice(0, 24)}`;
  const derivedTarget = { role: 'assistant', content: '', tool_calls: [{ id: derivedCallId, type: 'function',
    function: { name: 'return_result', arguments: JSON.stringify({ status: 'success', value: witnessValue.body_source }) } }] };
  const derived = structuredClone(row) as NativeRow & Dict;
  derived.id = `held-neuralese-derived-typed-text:${String(row.id)}:${witnessValue.body_sha256.slice(0, 12)}`;
  derived.target = derivedTarget;
  derived.review_disposition = 'held_derived_equivalent_typed_text_target';
  derived.training_admission = { approved: false, status: 'held-review-only',
    reason: 'derived equivalent target; root review pending' };
  derived.decision = { ...decision, training_approved: false, target_representation: 'derived-equivalent-typed-text-return' };
  derived.derived_target = {
    schema: 'natlang.derived-equivalent-typed-text-target/1',
    transform_revision: 'pure-terminal-eval-finish-to-typed-return/4',
    derivation_role: 'derived_target_not_original_assistant_action',
    original_row_id: row.id,
    original_target_sha256: nativeRowDigest(target),
    original_target: structuredClone(target),
    original_messages_sha256: nativeRowDigest(row.messages),
    original_messages: structuredClone(row.messages),
    original_code_sha256: hexDigest(originalArgs.code as string),
    parsed_binding_name: parsed.binding_name,
    parsed_literal_kind: parsed.literal_kind,
    source_result_type: witnessValue.result_type,
    source_result: { trajectory_id: witnessValue.trajectory_id, source_row_sha256: witnessValue.source_row_sha256,
      source_result_row_sha256: witnessValue.source_result_row_sha256, generation_turn: witnessValue.generation_turn,
      writer_call_id: witnessValue.writer_call_id, writer_node: witnessValue.writer_node,
      block_id: witnessValue.block_id, body_sha256: witnessValue.body_sha256, source: witnessValue.source,
      marker_context: witnessValue.marker_context, source_kind: witnessValue.source_kind },
    derived_target_sha256: nativeRowDigest(derivedTarget),
    target_equivalence: 'The original pure-literal eval returns this exact constant string as its declared Neuralese<string> result; the derived target calls return_result with that same value.',
    original_runtime_target_preserved_in_source_row: true,
    runtime_gradient_qualification: false,
    training_approved: false,
  };
  derived.preview_source_selection = { ...preview, derived_target: derived.derived_target };
  return derived;
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

type StatusOnlySuccessProof = { schema: 'natlang.status-only-success-proof/1';
  basis: 'same-invocation-staged-result' | 'declared-void-return';
  invocation_id?: string; staged_call_id?: string; staged_action_linkage?: 'tool-call-id' | 'unique-invocation-arguments-output';
  staged_action_seq?: number; terminal_action_seq?: number; staged_output_sha256?: string; declared_return_type?: string };

function exactTranscriptText(value: unknown): string | undefined {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return;
  const parts: string[] = [];
  for (const part of value) {
    if (!part || typeof part !== 'object' || Array.isArray(part)) return;
    const item = part as Dict;
    if (item.type === 'text' && typeof item.text === 'string') parts.push(item.text);
    else if (item.type === 'neuralese' && typeof item.id === 'string' && /^nz1_[a-z2-7]{20,}$/.test(item.id))
      parts.push(`${item.id}`);
    else return;
  }
  return parts.join('');
}

function exactSampledEvalStageCall(row: NativeRow, invocationId: string, toolCallId: string,
  modelArguments: unknown): boolean {
  const matches: Dict[] = [];
  for (const stepValue of row.trajectory) {
    const step = stepValue as Dict;
    if (step.phase !== 'action' || step.invocation_id !== invocationId) continue;
    const calls = Array.isArray((step.assistant as Dict | undefined)?.calls) ?
      (step.assistant as Dict).calls as Dict[] : [];
    const rawCalls = Array.isArray((step.model_response as Dict | undefined)?.raw_calls) ?
      (step.model_response as Dict).raw_calls as Dict[] : [];
    for (const call of calls) {
      if (call.source_tool !== 'eval' || canonical(call.arguments) !== canonical(modelArguments) ||
          !hasExactRawModelCall(step, call)) continue;
      const idMatches = rawCalls.filter(raw => {
        const fn = raw && typeof raw.function === 'object' ? raw.function as Dict : undefined;
        if (raw?.id !== toolCallId || fn?.name !== 'eval' || typeof fn.arguments !== 'string') return false;
        try { return canonical(JSON.parse(fn.arguments)) === canonical(modelArguments); }
        catch { return false; }
      });
      if (idMatches.length === 1) matches.push(step);
    }
  }
  return matches.length === 1;
}

/** A status-only success can finish a non-void call only when the same call has just staged a
 * runtime-accepted typed result. The runtime's own linked eval output is the receipt: arbitrary
 * host captures, parent outcomes, prior tool outputs, and unlinked prose cannot stand in for it. */
function transformedSoftEvalStageProof(modelArguments: unknown, actionArguments: unknown, row: NativeRow,
  invocationId: string, toolCallId: string, stagedValue: string, actionEvent: Dict): boolean {
  if (softBodyActionMatch(modelArguments, actionArguments) !== 'complete' ||
      !softBodyActionMatches(modelArguments, actionArguments, row, invocationId) ||
      !exactSampledEvalStageCall(row, invocationId, toolCallId, modelArguments)) return false;
  const model = modelArguments as Dict;
  const rawCode = model.code as string, runtimeCode = (actionArguments as Dict).code as string;
  const open = '<|neuralese|>', close = '<|/neuralese|>';
  const start = rawCode.indexOf(open), bodyStart = start + open.length;
  const closeAt = rawCode.indexOf(close, bodyStart);
  if (start < 0 || closeAt < bodyStart || rawCode.indexOf(open, start + open.length) >= 0 ||
      rawCode.indexOf(close, closeAt + close.length) >= 0) return false;
  const body = rawCode.slice(bodyStart, closeAt);
  const sentinel = /(nz1_[a-z2-7]{20,})/.exec(runtimeCode);
  if (!sentinel || stagedValue !== sentinel[0] ||
      [...runtimeCode.matchAll(/(nz1_[a-z2-7]{20,})/g)].length !== 1) return false;
  const graph = Array.isArray(row.outcome.execution_graph) ? row.outcome.execution_graph as Dict[] : [];
  const writes = graph.filter(event => event.kind === 'block_write' && event.call_id === invocationId &&
    event.block === sentinel[1] && event.result_type === 'Neuralese<string>' &&
    event.marker_context === 'eval-code' && event.truncated === false &&
    event.text_body_sha256 === hexDigest(body) && Number.isSafeInteger(event.seq) &&
    Number(event.seq) < Number(actionEvent.seq));
  if (writes.length !== 1) return false;
  const write = writes[0]!;
  const inputs = Array.isArray(write.inputs) ? write.inputs as Dict[] : [];
  const turnNodes = inputs.filter(input => input.port === 'turn' && typeof input.node === 'string')
    .map(input => input.node as string);
  if (turnNodes.length !== 1) return false;
  const turns = graph.filter(event => event.kind === 'model_turn' && event.call_id === invocationId &&
    event.node === turnNodes[0] && Array.isArray(event.calls) && (event.calls as unknown[]).includes('eval') &&
    Number.isSafeInteger(event.seq) && Number(event.seq) < Number(write.seq));
  if (turns.length !== 1 || write.turn !== turns[0]!.node) return false;
  const turnInputs = Array.isArray(turns[0]!.inputs) ? turns[0]!.inputs as Dict[] : [];
  return turnInputs.some(input => input.node === `call:${invocationId}` && input.port === 'invocation');
}

function statusOnlySuccessProof(args: unknown, context: readonly Dict[], invocationId: string | undefined,
  declaredReturnType: unknown, actionLedger: readonly Dict[], terminalSeq: unknown,
  row: NativeRow): StatusOnlySuccessProof | undefined {
  if (!args || typeof args !== 'object' || Array.isArray(args)) return;
  const result = args as Dict;
  if ((result.status ?? 'success') !== 'success' || Object.hasOwn(result, 'value')) return;
  const returnType = typeof declaredReturnType === 'string' ? declaredReturnType :
    declaredReturnType && typeof declaredReturnType === 'object' && !Array.isArray(declaredReturnType) ?
      String((declaredReturnType as Dict).natlang ?? '') : '';
  if (returnType.trim() === 'void') return { schema: 'natlang.status-only-success-proof/1',
    basis: 'declared-void-return', ...(invocationId ? { invocation_id: invocationId } : {}), declared_return_type: returnType };
  if (!returnType.trim() || !invocationId || context.length < 2) return;
  const exactInvocationId = invocationId;
  // The complete runtime notice is the final line of a staged result. Requiring that suffix rejects
  // eval console output that merely imitates the notice. Earlier staged values remain in the
  // invocation across ordinary reads and unsuccessful evals, so inspect the whole current context
  // and bind its latest authentic staged eval to the successful action ledger.
  const stagePattern = /(?:^|\n)Staged ([\s\S]+) as the result\. If this is the result of the task you were given and you are satisfied with it, reply done to return exactly this value without a tool call, or call return_result with status "success" and omit value to finish using this exact stored result\. You can keep working and return a different value later\.$/;
  const stagedCandidates: { callId: string; output: string; event: Dict; linkage: 'tool-call-id' | 'unique-invocation-arguments-output' }[] = [];
  for (let outputIndex = 0; outputIndex < context.length; outputIndex++) {
    const message = context[outputIndex]!;
    if (message.role !== 'tool' || typeof message.tool_call_id !== 'string') continue;
    const stagedToolCallId = message.tool_call_id as string;
    const output = exactTranscriptText(message.content);
    if (output === undefined) continue;
    const stage = stagePattern.exec(output);
    if (!stage || stage[1]!.includes('<<cut off')) continue;
    const producer = context.slice(0, outputIndex).reverse().find(candidate => candidate.role === 'assistant' &&
      Array.isArray(candidate.tool_calls) && (candidate.tool_calls as Dict[]).some(call => call.id === stagedToolCallId));
    const call = (producer?.tool_calls as Dict[] | undefined)?.find(item => item.id === stagedToolCallId);
    const fn = call?.function && typeof call.function === 'object' ? call.function as Dict : undefined;
    if (fn?.name !== 'eval') continue;
    const stagedArgs = parseArguments(fn.arguments);
    const stagedEvents = actionLedger.filter(event => {
      if (event.call_id !== exactInvocationId || event.name !== 'eval' ||
          !(event.tool_call_id === stagedToolCallId || event.tool_call_id === undefined) ||
          !['ok', 'completed'].includes(String(event.outcome)) || event.result_text !== output ||
          (event.arguments && typeof event.arguments === 'object' && (event.arguments as Dict).finish === true) ||
          !Number.isSafeInteger(event.seq) || !Number.isSafeInteger(terminalSeq) || Number(event.seq) >= Number(terminalSeq))
        return false;
      if (canonical(event.arguments) === canonical(stagedArgs)) return true;
      // The runtime rewrites a single complete inline Neuralese literal to a block sentinel before eval.
      // Accept that one transformation only when the action call ID, staged notice, and execution graph
      // all identify the same exact typed block/body and producing eval turn.
      return event.tool_call_id === stagedToolCallId && stage[1] !== undefined &&
        transformedSoftEvalStageProof(stagedArgs, event.arguments, row, exactInvocationId, stagedToolCallId,
          stage[1], event);
    });
    if (stagedEvents.length === 1) stagedCandidates.push({ callId: stagedToolCallId, output,
      event: stagedEvents[0]!, linkage: stagedEvents[0]!.tool_call_id === stagedToolCallId ?
        'tool-call-id' : 'unique-invocation-arguments-output' });
  }
  const staged = stagedCandidates.at(-1);
  if (!staged) return;
  return { schema: 'natlang.status-only-success-proof/1', basis: 'same-invocation-staged-result',
    invocation_id: invocationId, staged_call_id: staged.callId, staged_action_linkage: staged.linkage,
    staged_action_seq: Number(staged.event.seq), terminal_action_seq: Number(terminalSeq),
    staged_output_sha256: hexDigest(staged.output),
    ...(returnType ? { declared_return_type: returnType } : {}) };
}

function contextDeclaredReturnType(context: readonly Dict[]): string | undefined {
  const user = context.find(message => message.role === 'user');
  const content = exactTranscriptText(user?.content) ?? '';
  const firstLine = content.split(/\r?\n/, 1)[0] ?? '';
  const signature = /^You are inside this call: .*$/.exec(firstLine)?.[0];
  if (!signature) return;
  const delimiter = signature.lastIndexOf('): ');
  return delimiter < 0 ? undefined : signature.slice(delimiter + 3).trim() || undefined;
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
  row: NativeRow, invocationId: string | undefined, actionEvent: Dict): boolean {
  if (!invocationId) return false;
  const model = modelArguments && typeof modelArguments === 'object' && !Array.isArray(modelArguments) ? modelArguments as Dict : {};
  const action = actionArguments && typeof actionArguments === 'object' && !Array.isArray(actionArguments) ? actionArguments as Dict : {};
  if (model.status !== 'success' || action.status !== 'success') return false;
  if (typeof model.value !== 'string' || typeof action.value !== 'string')
    return nestedNeuraleseReturnMatches(model, action, row, invocationId, actionEvent);
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

function neuraleseReference(value: unknown): Dict | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return;
  const ref = (value as Dict).$neuralese;
  if (!ref || typeof ref !== 'object' || Array.isArray(ref)) return;
  const record = ref as Dict;
  return typeof record.id === 'string' && typeof record.type === 'string' ? record : undefined;
}

function exactNeuraleseTextMarker(value: unknown): { id: string; type: string; body: string } | undefined {
  if (typeof value !== 'string') return;
  const match = /^\[\[Neuralese text block id=(nz1_[a-z2-7]{20,}) type=(.+); exact JSON string body=([\s\S]*)\]\]$/.exec(value);
  if (!match) return;
  try {
    const body = JSON.parse(match[3]!) as unknown;
    return typeof body === 'string' ? { id: match[1]!, type: match[2]!, body } : undefined;
  } catch { return; }
}

/**
 * Match an expanded text marker back to its typed value only at a declared Neuralese return slot.
 * The caller's raw tool call, host capture, current-turn read and graph writer must all name the
 * same content-addressed block. Ordinary string fields retain exact string comparison.
 */
function nestedNeuraleseReturnMatches(modelArguments: Dict, actionArguments: Dict, row: NativeRow,
  invocationId: string, actionEvent: Dict): boolean {
  if (typeof actionEvent.tool_call_id !== 'string' || !Number.isSafeInteger(actionEvent.seq)) return false;
  const invocationEntries = (Array.isArray(row.outcome.invocation_ledger) ? row.outcome.invocation_ledger : [])
    .filter((entry: Dict) => entry.invocation_id === invocationId);
  if (invocationEntries.length !== 1) return false;
  const invocation = invocationEntries[0] as Dict;
  const host = invocation.host_result && typeof invocation.host_result === 'object' ? invocation.host_result as Dict : undefined;
  const site = invocation.inline_instruction_site && typeof invocation.inline_instruction_site === 'object' ?
    invocation.inline_instruction_site as Dict : undefined;
  const returns = site?.returns && typeof site.returns === 'object' ? site.returns as Dict : undefined;
  if (host?.kind !== 'host_capture' || host.capture_kind !== 'invocation_output' || host.call_id !== invocationId ||
      host.complete !== true || host.terminal_action_seq !== actionEvent.seq || !Object.hasOwn(host, 'value') ||
      typeof returns?.natlang !== 'string' || typeof host.result_type !== 'string' ||
      typeof host.value_sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(host.value_sha256) ||
      host.value_sha256 !== hexDigest(JSON.stringify(host.value)) ||
      (typeof host.bytes === 'number' && host.bytes !== new TextEncoder().encode(JSON.stringify(host.value)).byteLength)) return false;
  let declared: Type, captured: Type;
  try { declared = parseType(returns.natlang); captured = parseType(host.result_type); }
  catch { return false; }
  if (canonical(declared) !== canonical(captured)) return false;

  const sourceSteps = (row.trajectory as Dict[]).filter(step => step.invocation_id === invocationId &&
    Array.isArray((step.model_response as Dict | undefined)?.raw_calls) &&
    ((step.model_response as Dict).raw_calls as Dict[]).some(raw => {
      if (raw?.id !== actionEvent.tool_call_id) return false;
      const fn = raw.function && typeof raw.function === 'object' ? raw.function as Dict : {};
      if (fn.name !== 'return_result' || typeof fn.arguments !== 'string') return false;
      try { return canonical(JSON.parse(fn.arguments)) === canonical(modelArguments); }
      catch { return false; }
    }));
  if (sourceSteps.length !== 1) return false;
  const sameInvocationSteps = (row.trajectory as Dict[]).filter(step => step.invocation_id === invocationId);
  const callTurn = sameInvocationSteps.indexOf(sourceSteps[0]!) + 1;
  if (callTurn <= 0) return false;
  const graph = Array.isArray(row.outcome.execution_graph) ? row.outcome.execution_graph as Dict[] : [];
  const modelTurns = graph.filter(event => event.kind === 'model_turn' && event.call_id === invocationId && event.turn === callTurn);
  if (modelTurns.length !== 1) return false;
  const modelTurn = modelTurns[0]!;

  const sameValue = (type: Type, raw: unknown, action: unknown, capturedValue: unknown): boolean => {
    if (type.kind === 'neuralese' && type.element.kind === 'prim' && type.element.name === 'string') {
      const marker = exactNeuraleseTextMarker(raw), actionRef = neuraleseReference(action), hostRef = neuraleseReference(capturedValue);
      const typeText = formatType(type);
      if (!marker || marker.type !== typeText || !actionRef || !hostRef ||
          actionRef.id !== marker.id || hostRef.id !== marker.id || actionRef.type !== typeText || hostRef.type !== typeText)
        return false;
      const bodySha = hexDigest(marker.body);
      const writers = graph.filter(event => event.kind === 'block_write' && typeof event.call_id === 'string' &&
        event.block === marker.id &&
        event.truncated === false && event.result_type === typeText && event.text_body_sha256 === bodySha &&
        ['typed-text-result', 'typed-text-result-field'].includes(String(event.source_kind)) &&
        ['return_result', 'eval-return', 'eval-finish'].includes(String(event.source)) &&
        event.marker_context === 'return-result' && typeof event.node === 'string');
      const reads = graph.filter(event => event.kind === 'block_read' && event.call_id === invocationId &&
        event.block === marker.id && event.turn === modelTurn.node && typeof event.node === 'string');
      const authenticatedRead = reads.some(read => Array.isArray(read.inputs) &&
        (read.inputs as Dict[]).some(input => writers.some(writer => input.node === writer.node &&
          input.block === marker.id && input.port === 'block')) &&
        Array.isArray(modelTurn.inputs) && (modelTurn.inputs as Dict[]).some(input =>
          input.node === read.node && input.block === marker.id && input.port === 'read'));
      return writers.length > 0 && authenticatedRead;
    }
    if (type.kind === 'record') {
      if (!raw || typeof raw !== 'object' || Array.isArray(raw) || !action || typeof action !== 'object' || Array.isArray(action) ||
          !capturedValue || typeof capturedValue !== 'object' || Array.isArray(capturedValue)) return false;
      const rawRecord = raw as Dict, actionRecord = action as Dict, hostRecord = capturedValue as Dict;
      const declaredNames = type.fields.map(field => field.name);
      if ([rawRecord, actionRecord, hostRecord].some(value => Object.keys(value).some(key => !declaredNames.includes(key)))) return false;
      for (const field of type.fields) {
        const hasRaw = Object.hasOwn(rawRecord, field.name), hasAction = Object.hasOwn(actionRecord, field.name), hasHost = Object.hasOwn(hostRecord, field.name);
        if (!(hasRaw || hasAction || hasHost) && field.optional) continue;
        if (hasRaw !== hasAction || hasRaw !== hasHost || !hasRaw ||
            !sameValue(field.type, rawRecord[field.name], actionRecord[field.name], hostRecord[field.name])) return false;
      }
      return true;
    }
    if (type.kind === 'list') {
      return Array.isArray(raw) && Array.isArray(action) && Array.isArray(capturedValue) &&
        raw.length === action.length && raw.length === capturedValue.length &&
        raw.every((value, index) => sameValue(type.element, value, action[index], capturedValue[index]));
    }
    if (type.kind === 'dict') {
      if (!raw || typeof raw !== 'object' || Array.isArray(raw) || !action || typeof action !== 'object' || Array.isArray(action) ||
          !capturedValue || typeof capturedValue !== 'object' || Array.isArray(capturedValue)) return false;
      const rawRecord = raw as Dict, actionRecord = action as Dict, hostRecord = capturedValue as Dict;
      const keys = Object.keys(rawRecord).sort();
      return canonical(keys) === canonical(Object.keys(actionRecord).sort()) && canonical(keys) === canonical(Object.keys(hostRecord).sort()) &&
        keys.every(key => sameValue(type.element, rawRecord[key], actionRecord[key], hostRecord[key]));
    }
    return canonical(raw) === canonical(action) && canonical(action) === canonical(capturedValue);
  };
  return modelArguments.status === 'success' && Object.hasOwn(modelArguments, 'value') &&
    sameValue(declared, modelArguments.value, actionArguments.value, host.value);
}

/** Count typed Neuralese references without treating quoted IDs as reads. */
function countTypedNeuraleseReferences(value: unknown, id: string): number {
  if (Array.isArray(value)) return value.reduce((count, item) => count + countTypedNeuraleseReferences(item, id), 0);
  if (!value || typeof value !== 'object') return 0;
  const item = value as Dict;
  let count = item.type === 'neuralese' && item.id === id ? 1 : 0;
  for (const [key, child] of Object.entries(item)) {
    if (key === 'arguments' && typeof child === 'string') {
      try { count += countTypedNeuraleseReferences(JSON.parse(child), id); } catch { /* Preserve opaque arguments. */ }
    } else count += countTypedNeuraleseReferences(child, id);
  }
  return count;
}

/** Track ID mentions in text leaves separately; they are diagnostic text, not typed reads. */
function countLiteralNeuraleseIdMentions(value: unknown, id: string): number {
  if (Array.isArray(value)) return value.reduce((count, item) => count + countLiteralNeuraleseIdMentions(item, id), 0);
  if (typeof value === 'string') return value.split(id).length - 1;
  if (!value || typeof value !== 'object') return 0;
  const item = value as Dict;
  let count = 0;
  for (const [key, child] of Object.entries(item)) {
    if (key === 'arguments' && typeof child === 'string') {
      try { count += countLiteralNeuraleseIdMentions(JSON.parse(child), id); }
      catch { count += child.split(id).length - 1; }
    } else if (key !== 'id' || item.type !== 'neuralese') count += countLiteralNeuraleseIdMentions(child, id);
  }
  return count;
}

function valueAtPath(value: unknown, path: readonly (string | number)[]): unknown {
  let current = value;
  for (const part of path) {
    if (typeof part === 'number') {
      if (!Array.isArray(current) || !Number.isSafeInteger(part) || part < 0 || part >= current.length) return undefined;
      current = current[part];
    } else {
      if (!current || typeof current !== 'object' || Array.isArray(current) || !Object.hasOwn(current, part)) return undefined;
      current = (current as Dict)[part];
    }
  }
  return current;
}

function typedJsonBody(value: unknown, active = new Set<object>()): string | undefined {
  if (value === null) return 'null';
  if (typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return Number.isFinite(value) ? Object.is(value, -0) ? '-0' : String(value) : undefined;
  if (Array.isArray(value)) {
    if (active.has(value)) return undefined;
    active.add(value);
    try {
      const values = value.map(item => typedJsonBody(item, active));
      return values.some(item => item === undefined) ? undefined : `[${values.join(',')}]`;
    } finally { active.delete(value); }
  }
  if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype) {
    if (active.has(value)) return undefined;
    active.add(value);
    try {
      const fields = Object.keys(value as Dict).sort().map(key => {
        const child = typedJsonBody((value as Dict)[key], active);
        return child === undefined ? undefined : `${JSON.stringify(key)}:${child}`;
      });
      return fields.some(item => item === undefined) ? undefined : `{${fields.join(',')}}`;
    } finally { active.delete(value); }
  }
  return undefined;
}

/** Prove direct typed-result rewrites from the exact model call, writer event, and captured invocation result. */
function typedResultActionProof(modelArguments: unknown, actionArguments: unknown, row: NativeRow,
  invocationId: string | undefined, actionEvent: Dict, requestSha256?: unknown,
  rawResponseSha256?: unknown): Dict[] | undefined {
  if (!invocationId || !['return_result', 'eval'].includes(String(actionEvent.name)) ||
      !['ok', 'completed'].includes(String(actionEvent.outcome))) return;
  const model = modelArguments && typeof modelArguments === 'object' && !Array.isArray(modelArguments) ? modelArguments as Dict : {};
  const action = actionArguments && typeof actionArguments === 'object' && !Array.isArray(actionArguments) ? actionArguments as Dict : {};
  if ((actionEvent.name === 'return_result' && !Object.hasOwn(model, 'value')) || canonical(model) !== canonical(action)) return;
  const ledger = Array.isArray(row.outcome.invocation_ledger) ? row.outcome.invocation_ledger as Dict[] : [];
  const outputs = ledger.filter(item => item.invocation_id === invocationId && item.host_result &&
    typeof item.host_result === 'object').map(item => item.host_result as Dict);
  if (outputs.length !== 1) return;
  const host = outputs[0]!;
  if (host.kind !== 'host_capture' || host.capture_kind !== 'invocation_output' || host.call_id !== invocationId ||
      host.complete !== true || !Object.hasOwn(host, 'value')) return;
  const terminalSeq = host.terminal_action_seq;
  const directCompletion = terminalSeq === actionEvent.seq;
  const stagedCompletion = directCompletion ? undefined :
    stagedEvalReturnCompletionProof(row, invocationId, actionEvent, terminalSeq, host.result_type);
  if (!directCompletion && (!stagedCompletion || actionEvent.name !== 'eval')) return;
  const graph = Array.isArray(row.outcome.execution_graph) ? row.outcome.execution_graph as Dict[] : [];
  const allowedSources = actionEvent.name === 'return_result' ? ['return_result'] : ['eval-return', 'eval-finish'];
  const writes = graph.filter(event => event.kind === 'block_write' && event.call_id === invocationId &&
    ['typed-text-result', 'typed-json-result', 'typed-text-result-field'].includes(String(event.source_kind)) &&
    allowedSources.includes(String(event.source)) && event.marker_context === 'return-result');
  if (!writes.length) return;
  const refs = new Map<string, { ref: Dict; receipt: Dict }>();
  const pathKeys = new Set<string>();
  const ambiguousPaths = new Set<string>();
  for (const write of writes) {
    const block = write.block, node = write.node, type = write.result_type, bodySha = write.text_body_sha256;
    if (typeof block !== 'string' || !/^nz1_[a-z2-7]{20,}$/.test(block) || typeof node !== 'string' ||
        typeof type !== 'string' || !type.startsWith('Neuralese<') || typeof bodySha !== 'string' ||
        !/^[0-9a-f]{64}$/.test(bodySha) || write.truncated !== false || !Number.isSafeInteger(write.seq) ||
        !Number.isSafeInteger(actionEvent.seq) || Number(write.seq) >= Number(actionEvent.seq)) continue;
    // A nonterminal eval-return can still be the actual typed writer when the same
    // invocation later commits that exact staged result with status-only success.
    // Keep this proof on the originating eval action; never retag the completion
    // helper as the writer or accept an overwritten earlier staged result.
    if (stagedCompletion && (write.source !== 'eval-return' || write.result_type !== host.result_type ||
        typeof actionEvent.result_text !== 'string' || !actionEvent.result_text.includes(block))) continue;
    const inputs = Array.isArray(write.inputs) ? write.inputs as Dict[] : [];
    const sourceNodes = inputs.filter(input => input.port === 'result-source' && typeof input.node === 'string')
      .map(input => input.node as string);
    if (sourceNodes.length !== 1) continue;
    const sourceTurns = graph.filter(event => event.kind === 'model_turn' && event.call_id === invocationId &&
      event.node === sourceNodes[0] && Number.isSafeInteger(event.seq) && Number(event.seq) < Number(write.seq));
    if (sourceTurns.length !== 1) continue;
    const precedingTurns = graph.filter(event => event.kind === 'model_turn' && event.call_id === invocationId &&
      Number.isSafeInteger(event.seq) && Number(event.seq) < Number(write.seq)).sort((a, b) => Number(a.seq) - Number(b.seq));
    if (precedingTurns.at(-1)?.node !== sourceNodes[0]) continue;
    let body: string | undefined;
    let bodyBasis = 'authenticated-final-host-output-reference';
    const exactRawValue = actionEvent.name === 'return_result' &&
      (model.status === undefined || model.status === 'success') && Object.hasOwn(model, 'value');
    if (exactRawValue && write.source_kind === 'typed-text-result' && type === 'Neuralese<string>' && write.result_path === undefined &&
        typeof model.value === 'string') {
      body = model.value; bodyBasis = 'exact-raw-model-result-string';
    } else if (exactRawValue && write.source_kind === 'typed-json-result' && write.result_path === undefined &&
        type !== 'Neuralese<string>') {
      const direct = typedJsonBody(model.value);
      if (direct !== undefined && hexDigest(direct) === bodySha) {
        body = direct; bodyBasis = 'canonical-json-of-exact-raw-model-result';
      } else if (typeof model.value === 'string') {
        try {
          const parsed = JSON.parse(model.value) as unknown;
          const normalized = typedJsonBody(parsed);
          if (normalized !== undefined && hexDigest(normalized) === bodySha) {
            body = normalized; bodyBasis = 'parsed-json-string-for-concrete-neuralese-result-type';
          }
        } catch { /* Invalid JSON text is not an alternate typed JSON source. */ }
      }
    } else if (exactRawValue && write.source_kind === 'typed-text-result-field' && type === 'Neuralese<string>' &&
        Array.isArray(write.result_path) && write.result_path[0] === 'return' && write.result_path.length > 1 &&
        write.result_path.slice(1).every(part => typeof part === 'string' || Number.isSafeInteger(part))) {
      const value = valueAtPath(model.value, write.result_path.slice(1) as (string | number)[]);
      if (typeof value !== 'string') continue;
      body = value; bodyBasis = 'exact-raw-model-result-field';
    }
    if (body !== undefined && hexDigest(body) !== bodySha) continue;
    if (body === undefined && exactRawValue) continue;
    const outputPath = Array.isArray(write.result_path) ? write.result_path[0] === 'return' ?
      write.result_path.slice(1) as (string | number)[] : undefined : [];
    if (!outputPath) continue;
    const capturedRef = neuraleseReference(valueAtPath(host.value, outputPath));
    if (!capturedRef || capturedRef.id !== block || capturedRef.type !== type) continue;
    const pathKey = JSON.stringify(outputPath);
    if (pathKeys.has(pathKey)) { refs.delete(pathKey); ambiguousPaths.add(pathKey); continue; }
    if (ambiguousPaths.has(pathKey)) continue;
    pathKeys.add(pathKey);
    const ref = { $neuralese: { id: block, type } };
    refs.set(pathKey, { ref, receipt: { schema: 'natlang.typed-result-write/1', trajectory_id: row.id,
      source_row_sha256: nativeRowDigest(row), invocation_id: invocationId, writer_call_id: invocationId,
      writer_node: node, block_id: block, source_kind: write.source_kind,
      source: write.source, result_type: type, result_path: write.result_path ?? ['return'],
      model_turn_node: sourceNodes[0], action_seq: actionEvent.seq, body_sha256: bodySha,
      request_sha256: typeof requestSha256 === 'string' ? requestSha256 : null,
      raw_response_sha256: typeof rawResponseSha256 === 'string' ? rawResponseSha256 : null,
      ...(write.source_kind === 'typed-json-result' && exactRawValue ?
        { raw_model_value_sha256: hexDigest(canonical(model.value)) } : {}),
      ...(body === undefined ? {} : { body_source: body }), body_source_basis: bodyBasis,
      ...(stagedCompletion ? { staged_success_completion: structuredClone(stagedCompletion) } : {}) } });
  }
  const receipts = [...refs.values()].map(entry => entry.receipt);
  return receipts.length ? receipts : undefined;
}

/** Bind a nonterminal eval-return writer to the later status-only completion of
 * that exact staged value in the same invocation. The terminal context must show
 * the staged result produced by this eval event as the latest staged candidate. */
function stagedEvalReturnCompletionProof(row: NativeRow, invocationId: string, actionEvent: Dict,
  terminalSeq: unknown, expectedResultType: unknown): Dict | undefined {
  if (actionEvent.name !== 'eval' || !Number.isSafeInteger(actionEvent.seq) ||
      !Number.isSafeInteger(terminalSeq) || Number(terminalSeq) <= Number(actionEvent.seq)) return;
  const ledger = Array.isArray(row.outcome.action_ledger) ? row.outcome.action_ledger as Dict[] : [];
  const terminalEvents = ledger.filter(event => event.call_id === invocationId && event.name === 'return_result' &&
    event.seq === terminalSeq && event.outcome === 'completed');
  if (terminalEvents.length !== 1) return;
  const terminalEvent = terminalEvents[0]!;
  const args = terminalEvent.arguments && typeof terminalEvent.arguments === 'object' &&
    !Array.isArray(terminalEvent.arguments) ? terminalEvent.arguments as Dict : {};
  if ((args.status ?? 'success') !== 'success' || Object.hasOwn(args, 'value')) return;
  const trajectory = Array.isArray(row.trajectory) ? row.trajectory as Dict[] : [];
  const terminalSteps = trajectory.filter(step => {
    const assistant = step.assistant && typeof step.assistant === 'object' && !Array.isArray(step.assistant) ?
      step.assistant as Dict : {};
    const calls = Array.isArray(assistant.calls) ? assistant.calls as Dict[] : [];
    return step.phase === 'action' && step.invocation_id === invocationId && calls.some(call =>
      call.source_tool === 'return_result' && canonical(call.arguments) === canonical(args) &&
      hasExactRawModelCall(step, call));
  });
  if (terminalSteps.length !== 1) return;
  const context = Array.isArray(terminalSteps[0]!.context) ? terminalSteps[0]!.context as Dict[] : [];
  const returnType = contextDeclaredReturnType(context);
  if (typeof expectedResultType !== 'string' || returnType !== expectedResultType) return;
  const proof = statusOnlySuccessProof(args, context, invocationId, returnType, ledger, terminalSeq, row);
  if (!proof || proof.basis !== 'same-invocation-staged-result' || proof.staged_action_seq !== actionEvent.seq ||
      (typeof actionEvent.tool_call_id === 'string' && proof.staged_call_id !== actionEvent.tool_call_id)) return;
  return proof;
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

/** Preserve provider-expanded Neuralese read bodies as context provenance.
 *
 * This is intentionally narrower than copying every provider expansion: only the exact body named by the
 * same response's typed readout is eligible. The execution graph must also show that this invocation read the
 * configured body block and that the read flowed into its model turn. The materialized row binds the copied
 * evidence to its source-row and trace digests below.
 */
function providerExpandedReadContexts(source: Dict, row: NativeRow, sourceRowSha256: string,
  invocationId: string | undefined, trajectoryIndex: number, target: Dict): Dict[] {
  if (!invocationId) return [];
  const response = source.model_response && typeof source.model_response === 'object' ? source.model_response as Dict : {};
  const transport = response.transport_provenance && typeof response.transport_provenance === 'object' ?
    response.transport_provenance as Dict : undefined;
  const readout = transport?.text_template_readout && typeof transport.text_template_readout === 'object' ?
    transport.text_template_readout as Dict : undefined;
  const expanded = transport?.expanded_input_blocks;
  const blocks = Array.isArray(expanded) ? expanded.filter(value => value && typeof value === 'object') as Dict[] : [];
  if (!transport || transport.learned_vectors !== false || transport.qualification_certificate !== false ||
      transport.training_admission !== false || typeof transport.raw_request_sha256 !== 'string' ||
      typeof transport.rendered_request_sha256 !== 'string' || typeof row.provenance.trace_sha256 !== 'string') return [];
  const context = Array.isArray(source.context) ? source.context : [];
  const graph = Array.isArray(row.outcome.execution_graph) ? row.outcome.execution_graph as Dict[] : [];
  const currentTurnNumber = row.trajectory.slice(0, trajectoryIndex + 1)
    .filter(step => step && typeof step === 'object' && (step as Dict).invocation_id === invocationId).length;
  const currentTurns = graph.filter(event => event.kind === 'model_turn' && event.call_id === invocationId &&
    event.turn === currentTurnNumber);
  if (currentTurns.length !== 1 || typeof currentTurns[0]!.node !== 'string') return [];
  const responseCalls = Array.isArray(response.calls) ? response.calls as unknown[] : [];
  const graphCalls = Array.isArray(currentTurns[0]!.calls) ? currentTurns[0]!.calls : [];
  const responseCallNames = responseCalls.map(value => Array.isArray(value) && typeof value[0] === 'string' ? value[0] : null);
  if (canonical(responseCallNames) !== canonical(graphCalls)) return [];
  const targetCalls = Array.isArray(target.tool_calls) ? target.tool_calls as Dict[] : [];
  const targetActionsAuthentic = targetCalls.every(item => {
    const fn = item.function && typeof item.function === 'object' ? item.function as Dict : {};
    if (typeof fn.name !== 'string' || typeof fn.arguments !== 'string') return false;
    try { return hasExactRawModelCall(source, { source_tool: fn.name, arguments: JSON.parse(fn.arguments) }); }
    catch { return false; }
  });
  if (!targetActionsAuthentic) return [];
  const sourceRequestSha256 = source.request_sha256;
  const sourceResponseSha256 = source.raw_response_sha256;
  if (typeof sourceRequestSha256 !== 'string' || !/^[0-9a-f]{64}$/.test(sourceRequestSha256) ||
      typeof sourceResponseSha256 !== 'string' || !/^[0-9a-f]{64}$/.test(sourceResponseSha256)) return [];
  const receipts: Dict[] = [];
  for (const block of blocks) {
    const blockId = block.id;
    if (typeof blockId !== 'string' || !/^nz1_[a-z2-7]{20,}$/.test(blockId) ||
        typeof block.type !== 'string' || !block.type.startsWith('Neuralese<') ||
        typeof block.body !== 'string' || block.body.includes('<|neuralese|>') || block.body.includes('<|/neuralese|>') ||
        hexDigest(block.body) !== block.body_sha256 || block.learned_vectors !== false) continue;
    const visibleCount = countTypedNeuraleseReferences(context, blockId);
    const literalIdMentions = countLiteralNeuraleseIdMentions(context, blockId);
    const reads = graph.filter(event => event.kind === 'block_read' && event.call_id === invocationId && event.block === blockId);
    if (!reads.length || reads.some(event => typeof event.node !== 'string')) continue;
    const readTurnPairs = reads.map(read => ({ read, turn: graph.find(event => event.kind === 'model_turn' &&
      event.call_id === invocationId && Array.isArray(event.inputs) && (event.inputs as Dict[]).some(input =>
        input.node === read.node && input.port === 'read' && input.block === blockId)) })).
      filter((pair): pair is { read: Dict; turn: Dict } => !!pair.turn);
    if (readTurnPairs.length !== reads.length) continue;
    const selectedTurnPairs = readTurnPairs.filter(pair => pair.turn.node === currentTurns[0]!.node);
    if (!selectedTurnPairs.length) continue;
    const primaryPair = selectedTurnPairs[0]!;
    const orderedReadTurnPairs = [...selectedTurnPairs,
      ...readTurnPairs.filter(pair => !selectedTurnPairs.includes(pair))];
    const definition = graph.filter(event => event.kind === 'invocation' && event.phase === 'start' &&
      event.call_id === invocationId && event.definition && typeof event.definition === 'object' &&
      (event.definition as Dict).id === `nz-fn:${blockId}`);
    const readoutMatch = !!readout && readout.read_body_id === blockId && readout.read_source_sha256 === block.body_sha256 &&
      readout.schema === 'natlang.text-template-readout/1' && readout.call === 'return_result' &&
      readout.value === 'decode' && readout.value_type === 'string';
    const readInputs = Array.isArray(primaryPair.read.inputs) ? primaryPair.read.inputs as Dict[] : [];
    // Sequence numbers are scoped to each invocation. The exact block input edge into this read is the causal
    // proof across invocations; comparing writer/read seq values from different calls would reject valid links.
    const legacyResultWitness = (event: Dict): { writerSourceClass: string; witness: Dict } | undefined => {
      const transportConfig = row.provenance.text_neuralese_transport;
      if (!transportConfig || typeof transportConfig !== 'object' ||
          (transportConfig as Dict).mode !== 'text-marker-standin/2' ||
          event.producer !== 'text-marker-emulation' || event.source_kind !== 'typed-text-result' ||
          event.marker_context !== 'return-result' || event.result_type !== block.type ||
          typeof event.call_id !== 'string') return;
      const ledger = Array.isArray(row.outcome.invocation_ledger) ? row.outcome.invocation_ledger as Dict[] : [];
      const entries = ledger.filter(item => item.invocation_id === event.call_id);
      if (entries.length !== 1) return;
      const entry = entries[0]!;
      const host = entry.host_result && typeof entry.host_result === 'object' ? entry.host_result as Dict : undefined;
      const ref = host?.value && typeof host.value === 'object' ? (host.value as Dict).$neuralese : undefined;
      if (entry.completion_status !== 'done' || entry.completion_source !== 'execution_graph' ||
          entry.completion_detail !== `\uE000${blockId}\uE001` || host?.kind !== 'host_capture' ||
          host.capture_kind !== 'invocation_output' || host.call_id !== event.call_id || host.complete !== true ||
          host.name !== 'return' || !ref || typeof ref !== 'object' || (ref as Dict).id !== blockId ||
          (ref as Dict).type !== block.type || host.result_type !== block.type ||
          typeof host.value_sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(host.value_sha256)) return;
      const steps = (row.trajectory as Dict[]).filter(step => step.invocation_id === event.call_id);
      const resultSourceInputs = Array.isArray(event.inputs) ? (event.inputs as Dict[]).filter(input =>
        input.port === 'result-source' && typeof input.node === 'string') : [];
      const sourceTurns = resultSourceInputs.length === 1 ? graph.filter(candidate =>
        candidate.kind === 'model_turn' && candidate.call_id === event.call_id &&
        candidate.node === resultSourceInputs[0]!.node && Number.isSafeInteger(candidate.turn)) : [];
      const sourceStep = sourceTurns.length === 1 ? steps[Number(sourceTurns[0]!.turn) - 1] :
        steps.length === 1 ? steps[0] : undefined;
      if (event.source === 'eval-finish') return { writerSourceClass: 'legacy-text-marker-standin-eval-finish',
        witness: { kind: 'completed-eval-finish-host-reference', source: 'eval-finish',
          host_result_call_id: event.call_id, host_result_type: block.type,
          host_result_value_sha256: host.value_sha256 } };
      if (event.source === 'eval-return') return { writerSourceClass: 'legacy-text-marker-standin-eval-return',
        witness: { kind: 'completed-eval-return-host-reference', source: 'eval-return',
          host_result_call_id: event.call_id, host_result_type: block.type,
          host_result_value_sha256: host.value_sha256,
          completion_status: entry.completion_status, completion_source: entry.completion_source,
          completion_detail: entry.completion_detail } };
      if (event.source !== 'return_result' || block.type !== 'Neuralese<string>') return;
      const response = sourceStep?.model_response && typeof sourceStep.model_response === 'object' ?
        sourceStep.model_response as Dict : {};
      const rawCalls = Array.isArray(response.raw_calls) ? response.raw_calls : [];
      const exact = rawCalls.filter(raw => {
        if (!raw || typeof raw !== 'object') return false;
        const fn = (raw as Dict).function;
        if (!fn || typeof fn !== 'object' || (fn as Dict).name !== 'return_result' ||
            typeof (fn as Dict).arguments !== 'string') return false;
        try {
          const args = JSON.parse((fn as Dict).arguments as string) as Dict;
          return (args.status === undefined || args.status === 'success') && args.value === block.body;
        } catch { return false; }
      });
      if (exact.length === 1) return { writerSourceClass: 'legacy-text-marker-standin-return-result',
        witness: { kind: 'raw-return-result-value-equals-expanded-body', source: 'return_result',
          host_result_call_id: event.call_id, host_result_type: block.type,
          host_result_value_sha256: host.value_sha256,
          source_model_turn: sourceTurns[0]?.turn ?? null,
          raw_response_sha256: sourceStep?.raw_response_sha256 ?? null } };
      // Some legacy runs returned an opaque typed handle from a completed eval.
      // The runtime then emitted a second return-result write for the same
      // successful host result. Authenticate that read context only when the
      // completed host capture names this exact typed block and an earlier
      // typed eval-return/eval-finish write in the same invocation carries the
      // exact expanded-body digest. This records the observed re-emission; it
      // does not create another semantic target or recurrence writer.
      const upstream = graph.filter(candidate => candidate.kind === 'block_write' &&
        candidate.call_id === event.call_id && candidate.block === blockId && candidate.node !== event.node &&
        candidate.truncated === false && candidate.producer === 'text-marker-emulation' &&
        candidate.source_kind === 'typed-text-result' && ['eval-return', 'eval-finish'].includes(String(candidate.source)) &&
        candidate.result_type === block.type && candidate.text_body_sha256 === block.body_sha256 &&
        Number.isSafeInteger(candidate.seq) && Number.isSafeInteger(event.seq) && Number(candidate.seq) < Number(event.seq));
      if (upstream.length !== 1) return;
      return { writerSourceClass: 'legacy-text-marker-standin-return-result-host-reference',
        witness: { kind: 'completed-return-result-host-reference-reemission', source: 'return_result',
          host_result_call_id: event.call_id, host_result_type: block.type,
          host_result_value_sha256: host.value_sha256,
          completion_status: entry.completion_status, completion_source: entry.completion_source,
          completion_detail: entry.completion_detail, upstream_typed_output_node: upstream[0]!.node,
          upstream_typed_output_source: upstream[0]!.source,
          upstream_typed_output_seq: upstream[0]!.seq,
          upstream_typed_output_body_sha256: upstream[0]!.text_body_sha256,
          source_model_turn: sourceTurns[0]?.turn ?? null,
          raw_response_sha256: sourceStep?.raw_response_sha256 ?? null } };
    };
    const writerSourceClass = (event: Dict): string | undefined => {
      const legacy = legacyResultWitness(event);
      if (legacy) return legacy.writerSourceClass;
      const transportConfig = row.provenance.text_neuralese_transport;
      if (transportConfig && typeof transportConfig === 'object' &&
          (transportConfig as Dict).mode === 'text-marker-standin/2' &&
          event.marker_context === 'return-result') return undefined;
      if (event.producer === 'text-marker-emulation' && event.source_kind === 'typed-text-result')
        return 'modern-typed-text-result';
      if (event.emulation_version === 'text-marker-standin/2' && event.marker_context === 'eval-code' &&
          event.learned_vectors === false)
        return 'legacy-text-marker-standin-eval-code';
      return undefined;
    };
    const writers = graph.filter(event => event.kind === 'block_write' && event.block === blockId &&
      typeof event.node === 'string' && readInputs.some(input => input.node === event.node && input.block === blockId) &&
      event.truncated === false && writerSourceClass(event) !== undefined && event.result_type === block.type &&
      event.text_body_sha256 === block.body_sha256);
    const origin = readoutMatch && definition.length === 1 && visibleCount <= 1 ? 'configured-function-definition' :
      writers.length === 1 && visibleCount >= 1 ? 'same-run-producer' : undefined;
    if (!origin || (origin === 'same-run-producer' && visibleCount < 1)) continue;
    const invocationLedger = Array.isArray(row.outcome.invocation_ledger) ? row.outcome.invocation_ledger as Dict[] : [];
    const invocationEntry = invocationLedger.find(item => item.invocation_id === invocationId);
    receipts.push({ schema: origin === 'same-run-producer' ? 'natlang.provider-expanded-read-context/2' :
        'natlang.provider-expanded-read-context/1', invocation_id: invocationId,
      parent_invocation_id: invocationEntry?.parent_invocation_id ?? null,
      source_row_sha256: sourceRowSha256, trace_sha256: row.provenance.trace_sha256,
      transport_provenance_sha256: hexDigest(canonical(transport)),
      raw_request_sha256: transport.raw_request_sha256, rendered_request_sha256: transport.rendered_request_sha256,
      source_request_sha256: sourceRequestSha256, source_response_sha256: sourceResponseSha256,
      source_trajectory_index: trajectoryIndex, source_action_target_sha256: nativeDecisionTargetDigest(target),
      prompt_revision: transport.prompt_revision ?? null, origin,
      readout: readoutMatch ? structuredClone(readout) : null, block: structuredClone(block),
      definition: definition.length === 1 ? structuredClone(definition[0]!.definition) : null,
      signature: definition.length === 1 ? definition[0]!.signature ?? null : null,
      block_read: structuredClone(primaryPair.read), model_turn: structuredClone(primaryPair.turn),
      ...(orderedReadTurnPairs.length > 1 ? { additional_read_turn_pairs: orderedReadTurnPairs.slice(1).map(pair => ({
        block_read: structuredClone(pair.read), model_turn: structuredClone(pair.turn) })) } : {}),
      context_occurrences: visibleCount,
      serialized_literal_id_mentions: literalIdMentions,
      producer_write: writers.length === 1 ? structuredClone(writers[0]) : null,
      ...(origin === 'same-run-producer' ? { writer_target_selected: false,
        writer_source_class: writers.length === 1 ? writerSourceClass(writers[0]!) : null,
        ...(writers.length === 1 ? { writer_witness: legacyResultWitness(writers[0]!)?.witness ?? null } : {}) } : {}),
      learned_vectors: false, qualification_certificate: false, training_admission: false });
  }
  return receipts;
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
  turns: Dict[]; authored_actions: Dict[]; acceptedRows: number; rejectedRows: number;
  unlinked: { id: string; outcomes: number; reason?: string }[];
} {
  const turns: Dict[] = [];
  const authoredActions: Dict[] = [];
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
              candidate.arguments, row, invocation, candidate)) || (exactRaw && candidate.name === normalized.source_tool &&
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
          const typedResultWrites = exactRaw ? typedResultActionProof(normalized.arguments, event.arguments, row,
            invocation, event, source.request_sha256, source.raw_response_sha256) : undefined;
          const eventDiagnostics = Array.isArray(event.diagnostics) ? structuredClone(event.diagnostics) : [];
          normalized.outcome = { event_index: ledger.indexOf(event), trace_seq: event.seq ?? null,
            ...(typeof event.tool_call_id === 'string' ? { tool_call_id: event.tool_call_id } : {}),
            name: event.name, arguments: structuredClone(eventArguments ?? {}),
            ...(containsIncompleteDiagnostic(event.arguments) ? { arguments_source: 'exact_raw_model_call' } : {}),
            ...(typedResultWrites?.length ? { typed_result_writes: structuredClone(typedResultWrites) } : {}),
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

      const invocationReturnType = invocation ?
        ((instructionSites.get(invocation) as Dict | undefined)?.returns ?? contextDeclaredReturnType(contextSource)) :
        contextDeclaredReturnType(contextSource);
      const statusOnlySuccessValidation = calls.flatMap(call => {
        if (call.source_tool !== 'return_result') return [];
        const args = call.arguments && typeof call.arguments === 'object' && !Array.isArray(call.arguments) ? call.arguments as Dict : {};
        if (args.status !== 'success' || Object.hasOwn(args, 'value')) return [];
        const proof = statusOnlySuccessProof(args, contextSource, invocation, invocationReturnType, ledger,
          (call.outcome as Dict | undefined)?.trace_seq, row);
        return [{ valid: proof !== undefined, ...(proof ? { proof } : {}),
          ...(proof ? {} : { reason: 'success without value has no same-invocation staged typed result or declared void return' }) }];
      });
      const invalidStatusOnlySuccess = statusOnlySuccessValidation.some(item => !item.valid);

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
      const authoredRootAction = record(source.action_provenance ?? {}, 'action provenance').kind === 'authored_reference_root_eval';
      const authoredGuidancePending = row.collection_guidance && typeof row.collection_guidance === 'object' &&
        (row.collection_guidance as Dict).training_admission === false;
      const authoredGuidanceBlocked = authoredGuidancePending && !semanticApproval;
      const decisionApproved = (!authoredRootAction || !!semanticApproval) && !authoredGuidanceBlocked && !semanticHold && !afterChunkCutoff &&
        (row.outcome.accepted || !!semanticApproval) && !fromStudentPrefix && ranCleanly && !detour && !redundantSkillRead && !refusedAttempt &&
        !heldDirect && !variantContext && !invalidStatusOnlySuccess;
      const expandedReadContexts = providerExpandedReadContexts(source, row, rowDigest, invocation, index, target);
      if (authoredRootAction && !decisionApproved) authoredActions.push({ version: 'natlang.authored_action_hold/1',
        id: `${row.id}:authored-action:${String(index).padStart(4, '0')}`,
        source_ref: { trajectory_id: row.id, source_row_sha256: rowDigest, invocation_id: invocation ?? null,
          program_ir_id: programId, action_provenance: structuredClone(record(source.action_provenance,
            `${row.id}.trajectory[${index}].action_provenance`)) }, provenance: row.provenance, task: row.task, collection_guidance: row.collection_guidance ?? null,
        context: publicValue(contextSource), tools: publicValue(source.tools_offered ?? []), target,
        actions: calls.map(call => ({ tool: call.tool, source_tool: call.source_tool, arguments: publicValue(call.arguments),
          call_id: call.call_id, outcome: publicValue(call.outcome) })),
        action_provenance: source.action_provenance, outcome: { accepted: row.outcome.accepted, status: row.outcome.status,
          ...(row.outcome.oracle ? { oracle: row.outcome.oracle } : {}) },
        training_admission: { approved: false, kind: 'authored-root-action-pending-review',
          reason: 'authored reference root action requires separate root training approval' },
        trace_admission: { admitted: false, kind: 'authored-root-action-pending-review' } });
      if (!authoredRootAction || decisionApproved) rowTurns.push({ version: NATIVE_TEACHER_TURN_VERSION,
        id: `${row.id}:decision:${String(index).padStart(4, '0')}`,
        source_ref: { trajectory_id: row.id, source_row_sha256: rowDigest,
          ...(source.action_provenance && typeof source.action_provenance === 'object' && !Array.isArray(source.action_provenance) ?
            { action_provenance: structuredClone(source.action_provenance) } : {}),
          ...(semanticApproval ? { native_target_sha256: semanticApproval.target_sha256 } : {}),
          ...(invocation ? { invocation_id: invocation, ...(instructionSites.has(invocation) ? { inline_instruction_site: instructionSites.get(invocation) } : {}), ...(parents.has(invocation) ? { parent_invocation_id: parents.get(invocation) } : {}),
            ...(lastInvocationDecision.get(invocation) === index && hostOutputs.has(invocation) ?
              { host_result_capture: hostOutputs.get(invocation) } : {}),
            ...(expandedReadContexts.length ? { provider_expanded_read_contexts: expandedReadContexts } : {}) } : {}),
          program_ir_id: programId },
        provenance: row.provenance,
        ...(row.collection_guidance ? { collection_guidance: row.collection_guidance } : {}),
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
        source: authoredRootAction ? 'authored-static-native' : `${role}-native`,
        gold_sources: [...(Array.isArray(taskIr.gold_sources) ? taskIr.gold_sources : []),
          authoredRootAction ? 'authored-reference-root-action' : `checked-${role}-trajectory`, 'exact-runtime-oracle'],
        license: taskIr.license ?? 'project-generated',
        messages: publicValue(contextSource),
        tools: publicValue(source.tools_offered ?? []),
        target,
        teacher_reasoning: authoredRootAction ? null : retainedReasoning,
        // Only reasoning a model wrote is a target: rationales a model wrote for scripted actions (static
        // demonstrations) train like a teacher's reasoning. Authored action plans and stock action notes are
        // template text: kept in context, never trained (owner 2026-10-06).
        ...(authoredRootAction ? { teacher_reasoning_trained: false } :
          row.provenance.synthetic_reasoning ? { teacher_reasoning_trained:
            row.provenance.synthetic_reasoning === 'rationalized-actions/1' } : {}),
        teacher_execution_plan: authoredRootAction ? null : executionPlan,
        teacher_trajectory_id: row.id,
        teacher_trajectory_digest: rowDigest,
        // Keep the final verdict with each decision for downstream continuation validation, without copying files.
        outcome: { accepted: row.outcome.accepted, status: row.outcome.status,
          ...(row.outcome.oracle ? { oracle: row.outcome.oracle } : {}) },
        training_admission: { kind: authoredRootAction && semanticApproval ? 'reviewed-authored-static-native-action' :
          authoredGuidanceBlocked ? 'authored-root-guided-pending-review' :
          semanticApproval ? 'reviewed-native-decision' : 'exact-native-runtime-oracle', approved: decisionApproved,
          ...(semanticApproval ? { semantic_review: semanticApproval } : {}),
          ...(semanticHold ? { semantic_review: semanticHold } : {}),
          ...(authoredGuidanceBlocked ? { reason: 'authored-root-guided sample awaits separate root training approval' } : {}),
          ...(evidenceOracle ? { oracle_level: evidenceOracle } : {}),
          ...(decisionApproved ? {} : { reason: authoredGuidanceBlocked ? 'authored-root-guided sample awaits separate root training approval' : semanticHold ? semanticHold.reason : invalidStatusOnlySuccess ? 'success return omitted value without an authenticated same-invocation staged typed result' : afterChunkCutoff ? 'beyond verified chunk-rewrite supervision cutoff' : variantContext ? 'context of a corrected variant' :
            heldDirect ? 'an answer given without reasoning towards it' :
            (fromStudentPrefix ? 'student replay prefix is not a teacher correction' :
            calls.some(call => record(call.outcome, 'call outcome').status === 'not_recorded') ?
              'the outcome of a call in this decision was not recorded' :
            !ranCleanly ? 'decision contains a failed or unexecuted proposal' :
            redundantSkillRead ? 'retrieves unchanged skill instructions again' :
            detour ? 'repeats an earlier call of this call with the same result' :
            refusedAttempt ? "the task's checker rejected this attempt" : 'the run was not accepted') }) },
        trace_admission: { admitted: !authoredGuidancePending && !authoredRootAction,
          kind: authoredRootAction ? 'authored-root-static-reference-not-provider-sampled' :
            authoredGuidancePending ? 'authored-root-guided-pending-review' : 'exact-native-runtime-oracle',
          final_outcome_sha256: outcomeDigest },
        decision: { index,
          ...(statusOnlySuccessValidation.length ? { status_only_success_validation: statusOnlySuccessValidation } : {}),
          durable_opening: contextSource.slice(0, openingLength(contextSource)).map(normalizeContextMessage),
          tool_schemas: offered,
          assistant: { content: assistant.content ?? '', reasoning: authoredRootAction ? null : retainedReasoning,
            execution_plan: authoredRootAction ? null : executionPlan,
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
  return { turns, authored_actions: authoredActions, acceptedRows, rejectedRows, unlinked };
}
