import { createHash } from 'node:crypto';

export const STUDIO_TEACHER_TRAJECTORY_VERSION = 'natlang.studio_teacher_trajectory/1';

type Dict = Record<string, unknown>;

function record(value: unknown, label: string): Dict {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new TypeError(`${label} must be an object`);
  return value as Dict;
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as Dict)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(',')}}`;
  return JSON.stringify(value);
}

export function studioDigest(value: unknown): string {
  return createHash('sha256').update(canonical(value)).digest('hex');
}

function publicValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(publicValue);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value as Dict)
    .filter(([key]) => !key.startsWith('x-'))
    .map(([key, child]) => [key, publicValue(child)]));
}

/** Project one exact-oracle Studio trajectory into exporter-ready decisions. */
export function materializeStudioRow(raw: unknown): Dict[] {
  const row = record(raw, 'Studio teacher trajectory');
  if (row.schema !== STUDIO_TEACHER_TRAJECTORY_VERSION)
    throw new Error(`unsupported Studio teacher trajectory: ${String(row.schema)}`);
  const outcome = record(row.outcome, 'outcome');
  if (outcome.accepted !== true) return [];
  const caseValue = record(row.case, 'case'), provenance = record(row.provenance, 'provenance');
  if (caseValue.source_revision !== provenance.source_revision)
    throw new Error('Studio source revision mismatch');
  const runs = record(row.runs, 'runs');
  const reducer = record(runs.reducer, 'runs.reducer'), view = record(runs.view, 'runs.view');
  const admission = { admitted: true, kind: 'exact-studio-state-oracle',
    source_revision: caseValue.source_revision,
    expected_sha256: studioDigest(caseValue.expected),
    reducer_trace_sha256: studioDigest(reducer.trace),
    view_trace_sha256: studioDigest(view.trace) };
  if (!Array.isArray(row.exchanges)) throw new TypeError('exchanges must be an array');
  return row.exchanges.map((rawExchange, index) => {
    const exchange = record(rawExchange, `exchanges[${index}]`);
    const assistant = record(exchange.assistant, `exchanges[${index}].assistant`);
    const request = record(exchange.request, `exchanges[${index}].request`);
    const calls = Array.isArray(assistant.calls) ? assistant.calls.map((value, callIndex) =>
      record(value, `exchanges[${index}].assistant.calls[${callIndex}]`)) : [];
    const target: Dict = { role: 'assistant', content: assistant.content ?? '' };
    if (calls.length) target.tool_calls = calls.map((call, callIndex) => ({
      id: `studio_${index}_${callIndex}`, type: 'function',
      function: { name: call.tool, arguments: JSON.stringify(call.arguments ?? {}) },
    }));
    return { id: `${String(row.id)}:${index}`, program_id: caseValue.id,
      family: 'teacher_studio', ir_version: row.schema, provisional_gold: false,
      trace_admission: admission, teacher_trajectory_id: row.id,
      teacher_trajectory_digest: studioDigest(row),
      training_admission: { kind: 'exact-studio-state-oracle', approved: true },
      source_program_ids: [caseValue.id], teacher_model: provenance.model,
      gold_sources: ['checked-teacher-trajectory', 'exact-host-oracle'],
      license: 'project-generated', split: caseValue.split, source: 'teacher-studio',
      source_groups: [caseValue.target], messages: publicValue(request.messages ?? []),
      tools: publicValue(request.tools ?? []), target,
      skill: calls.length ? calls.map(call => String(call.tool)).join('+') : 'reply',
      teacher_reasoning: assistant.reasoning ?? null };
  });
}
