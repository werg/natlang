import { handoffTurns, type Turn } from './replay.js';
import { hexDigest } from '../native/hash.js';
import { sourceConversionProblems, retiredWorkflowEvaluationReleased } from './source-conversion.js';
import { trainingQualityReason, runtimeFailureReason, quarantineReason, retiredFamily } from './curriculum-policy.js';
import type { ProgramRecord } from './program.js';

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

export function nativeRowDigest(value: unknown): string { return hexDigest(canonical(value)); }

function record(value: unknown, label: string): Dict {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`${label} must be an object`);
  return value as Dict;
}

function oracleLevel(value: unknown): 'exact' | 'normalized' | 'span' | 'agreement' | 'judged' | undefined {
  const level = typeof value === 'string' ? value : value && typeof value === 'object' && !Array.isArray(value) ?
    (value as Dict).level : undefined;
  return ['exact', 'normalized', 'span', 'agreement', 'judged'].includes(String(level)) ?
    level as 'exact' | 'normalized' | 'span' | 'agreement' | 'judged' : undefined;
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
 * accepted are materialized too, none of their decisions approved, so their failed decisions can be found.
 */
export function materializeNativeRows(input: unknown[], options: { directAnswers?: boolean; failedRuns?: boolean } = {}): {
  turns: Dict[]; acceptedRows: number; rejectedRows: number;
  unlinked: { id: string; outcomes: number; reason?: string }[];
} {
  const turns: Dict[] = [];
  const unlinked: { id: string; outcomes: number }[] = [];
  let acceptedRows = 0, rejectedRows = 0;
  for (const candidate of input) {
    const row = validateRow(candidate);
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
    // Each call (the root and every nl child) has its own actions, in order; the trajectory interleaves the calls'
    // decisions. A call's decisions, recognised by their opening (with the inputs it lists), claim the one action log whose next action their
    // first call matches, and are linked to it in order from then on.
    const logs = new Map<string, Dict[]>();
    for (const event of ledger) { const key = String(event.call_id ?? ''); logs.set(key, [...logs.get(key) ?? [], event]); }
    const next = new Map<string, number>(), claimed = new Map<string, string>(), owners = new Set<string>();
    // Per call (by its opening): each call signature already made, with the result it got.
    const sentBefore = new Map<string, Map<string, string>>();
    // Once per row: every decision names the row it came from, and hashing a long row per decision is quadratic.
    const rowDigest = nativeRowDigest(row), outcomeDigest = nativeRowDigest(row.outcome);
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
            containsIncompleteDiagnostic(candidate.arguments)));
        let log = claimed.get(caller);
        if (log === undefined) {
          log = invocation ? (logs.has(invocation) ? invocation : undefined) :
            [...logs.keys()].find(key => !owners.has(key) && eventMatches(logs.get(key)![next.get(key) ?? 0]));
          if (log !== undefined) { claimed.set(caller, log); owners.add(log); }
        }
        const at = log === undefined ? 0 : next.get(log) ?? 0, event = log === undefined ? undefined : logs.get(log)![at];
        const projected = containsIncompleteDiagnostic(normalized.arguments) ||
          (event !== undefined && containsIncompleteDiagnostic(event.arguments));
        if (projected && !exactRaw) {
          diagnosticArgsUnlinked = true;
          normalized.outcome = { event_index: null, trace_seq: null, name: normalized.source_tool,
            arguments: structuredClone(normalized.arguments), status: 'unlinked_incomplete_diagnostic_arguments',
            result: null, diagnostics: [] };
        } else if (event && eventMatches(event)) {
          const eventArguments = containsIncompleteDiagnostic(event.arguments) ? normalized.arguments : event.arguments;
          normalized.outcome = { event_index: ledger.indexOf(event), trace_seq: event.seq ?? null,
            name: event.name, arguments: structuredClone(eventArguments ?? {}),
            ...(containsIncompleteDiagnostic(event.arguments) ? { arguments_source: 'exact_raw_model_call' } : {}),
            status: event.outcome ?? null, result: event.result_text ?? null,
            diagnostics: structuredClone(event.diagnostics ?? []) };
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
          .some(code => String(code).startsWith('coerced-')));
      // Two kinds of step run cleanly and still teach nothing to repeat: a call this call already made with the same
      // result (a detour), and an attempt the task's own checker rejected (the code worked; the attempt did not).
      // Both stay in the context of later steps, where they are what the model recovers from.
      const earlier = sentBefore.get(caller) ?? sentBefore.set(caller, new Map()).get(caller)!;
      const signature = (call: Dict) => JSON.stringify([call.tool, call.arguments]);
      const resultOf = (call: Dict) => String(record(call.outcome, 'call outcome').result ?? '');
      const detour = calls.length > 0 && calls.every(call => earlier.get(signature(call)) === resultOf(call));
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
      const decisionApproved = row.outcome.accepted && !fromStudentPrefix && ranCleanly && !detour && !refusedAttempt &&
        !heldDirect && !variantContext;
      rowTurns.push({ version: NATIVE_TEACHER_TURN_VERSION,
        id: `${row.id}:decision:${String(index).padStart(4, '0')}`,
        source_ref: { trajectory_id: row.id, source_row_sha256: rowDigest,
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
        // Reasoning written for the row rather than by a model (provenance.synthetic_reasoning) is context, not a target.
        ...(row.provenance.synthetic_reasoning ? { teacher_reasoning_trained: false } : {}),
        teacher_execution_plan: executionPlan,
        teacher_trajectory_id: row.id,
        teacher_trajectory_digest: rowDigest,
        // Keep the final verdict with each decision for downstream continuation validation, without copying files.
        outcome: { accepted: row.outcome.accepted, status: row.outcome.status,
          ...(row.outcome.oracle ? { oracle: row.outcome.oracle } : {}) },
        training_admission: { kind: 'exact-native-runtime-oracle', approved: decisionApproved,
          ...(evidenceOracle ? { oracle_level: evidenceOracle } : {}),
          ...(decisionApproved ? {} : { reason: variantContext ? 'context of a corrected variant' :
            heldDirect ? 'an answer given without reasoning towards it' :
            (fromStudentPrefix ? 'student replay prefix is not a teacher correction' :
            calls.some(call => record(call.outcome, 'call outcome').status === 'not_recorded') ?
              'the outcome of a call in this decision was not recorded' :
            !ranCleanly ? 'decision contains a failed or unexecuted proposal' :
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
