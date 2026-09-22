import { hexDigest } from '../native/hash.js';

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

function callMatches(call: Dict, event: Dict): boolean {
  return call.source_tool === event.name && canonical(call.arguments) === canonical(event.arguments);
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

/**
 * Convert accepted native teacher runs to one self-contained model decision per row.
 * Context is copied from that exact native request. It is never assembled by appending
 * one continuation segment to another; a checkpoint is followed by the fresh request
 * context captured by the collector.
 */
export function materializeNativeRows(input: unknown[]): {
  turns: Dict[]; acceptedRows: number; rejectedRows: number;
} {
  const turns: Dict[] = [];
  let acceptedRows = 0, rejectedRows = 0;
  for (const candidate of input) {
    const row = validateRow(candidate);
    if (!row.outcome.accepted) { rejectedRows++; continue; }
    acceptedRows++;
    const ledger = Array.isArray(row.outcome.action_ledger) ? row.outcome.action_ledger.map((event, index) =>
      record(event, `${row.id}.outcome.action_ledger[${index}]`)) : [];
    let actionIndex = 0, segment = 0;
    let segmentOpening: Dict[] | null = null;
    for (let index = 0; index < row.trajectory.length; index++) {
      const source = record(row.trajectory[index], `${row.id}.trajectory[${index}]`);
      const phase = source.phase === 'checkpoint' ? 'checkpoint' : 'action';
      const contextSource = messages(source.context, `${row.id}.trajectory[${index}].context`);
      if (!contextSource.length || contextSource[0]?.role !== 'system' || contextSource[1]?.role !== 'user')
        throw new Error(`${row.id}: decision ${index} lacks a fresh system/user opening context`);
      if (!segmentOpening) segmentOpening = structuredClone(contextSource.slice(0, 2));

      const offered = toolSchemas(source.tools_offered ?? [], `${row.id}.trajectory[${index}].tools_offered`);
      const assistant = record(source.assistant, `${row.id}.trajectory[${index}].assistant`);
      const calls = Array.isArray(assistant.calls) ? assistant.calls.map((value, callIndex) => {
        const call = record(value, `${row.id}.trajectory[${index}].assistant.calls[${callIndex}]`);
        const normalized: Dict = { tool: String(call.tool ?? ''), source_tool: String(call.source_tool ?? call.tool ?? ''),
          arguments: structuredClone(call.arguments ?? {}), call_id: call.call_id ?? null };
        if (phase !== 'action') throw new Error(`${row.id}: checkpoint decision unexpectedly contains a tool call`);
        const event = ledger[actionIndex];
        if (!event || !callMatches(normalized, event))
          throw new Error(`${row.id}: cannot link decision ${index} call ${callIndex} to the next action outcome`);
        normalized.outcome = { event_index: actionIndex, trace_seq: event.seq ?? null,
          name: event.name, arguments: structuredClone(event.arguments ?? {}),
          status: event.outcome ?? null, result: event.result_text ?? null,
          diagnostics: structuredClone(event.diagnostics ?? []) };
        actionIndex++;
        return normalized;
      }) : [];

      // Every decision owns the exact request snapshot, normalized to roles and semantic
      // calls. This preserves within-segment evidence while preventing cross-segment stitching.
      const context = contextSource.map(normalizeContextMessage);
      turns.push({ version: NATIVE_TEACHER_TURN_VERSION,
        id: `${row.id}:decision:${String(index).padStart(4, '0')}`,
        source: { trajectory_id: row.id, source_row_sha256: nativeRowDigest(row),
          program_ir_id: record(row.task.program_ir, `${row.id}.task.program_ir`).id ?? null },
        provenance: structuredClone(row.provenance),
        task: structuredClone(row.task),
        decision: { index, segment, phase,
          context, durable_opening: segmentOpening.map(normalizeContextMessage),
          tool_schemas: offered,
          assistant: { content: assistant.content ?? '', reasoning: assistant.reasoning ?? null,
            calls, checkpoint_note: phase === 'checkpoint' ? assistant.content ?? '' : null },
          source_raw_response_sha256: source.raw_response_sha256 ?? null,
          source_tools_offered: structuredClone(source.tools_offered ?? []) },
        outcome: structuredClone(row.outcome),
        capture_limits: structuredClone(row.capture_limits ?? []) });
      if (phase === 'checkpoint') { segment++; segmentOpening = null; }
    }
    if (actionIndex !== ledger.length)
      throw new Error(`${row.id}: ${ledger.length - actionIndex} action outcomes have no teacher decision link`);
  }
  return { turns, acceptedRows, rejectedRows };
}
