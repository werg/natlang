/** Bind a self-hosted Qwen correction turn to an exact no-hint task replay. */
import { createHash } from 'node:crypto';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';

type TeacherTurn = { phase?: unknown; invocation_id?: unknown; context?: unknown; tools_offered?: unknown;
  assistant?: { content?: unknown; reasoning?: unknown; calls?: { tool: string; arguments: Record<string, unknown> }[] } };
type TeacherRow = { version?: unknown; id?: unknown; task?: unknown; trajectory?: unknown };
type SourceBinding = { sourcePath: string; sourceFileSha256: string; sourceView: 'unhinted' | 'privileged-hint';
  actorSourceSha256: string; executionIdentitySha256: string; model: 'nvidia/Qwen3.6-35B-A3B-NVFP4';
  revision: '1355db6a052410cfd62085d94b58866fd0f2c3c5' };

function canonical(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  return `{${Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}`;
}
function sha(value: string): string { return createHash('sha256').update(value).digest('hex'); }

function contentParts(value: unknown): unknown[] {
  if (typeof value !== 'string') return Array.isArray(value) ? value : [];
  const parts: unknown[] = []; const pattern = /<\|neuralese\|>([\s\S]*?)<\|\/neuralese\|>/g;
  let cursor = 0, match: RegExpExecArray | null;
  while ((match = pattern.exec(value))) {
    if (match.index > cursor) parts.push({ type: 'text', text: value.slice(cursor, match.index) });
    if (!match[1]) throw new Error('empty Qwen Neuralese text occurrence cannot be forced');
    parts.push({ type: 'teacher_native_text', text: match[1] }); cursor = pattern.lastIndex;
  }
  if (cursor < value.length) parts.push({ type: 'text', text: value.slice(cursor) });
  if (!parts.length && value) parts.push({ type: 'text', text: value });
  return parts;
}

/**
 * The returned driver requires the replay to reuse the teacher source run ID and exact invocation IDs.
 * It supplies targets as transport metadata; task messages are never copied or edited here.
 */
export function qwenTeacherForcedDriver(row: TeacherRow, binding: SourceBinding,
    send: (request: ModelTurnRequest) => Promise<ModelTurn>): { driver: (request: ModelTurnRequest) => Promise<ModelTurn>; runId: string } {
  if (row.version !== 'natlang.teacher_trajectory.native/1' || !Array.isArray(row.trajectory) || !row.task ||
      !row.id || typeof row.id !== 'string' || !/^[0-9a-f]{64}$/.test(binding.sourceFileSha256) ||
      !/^[0-9a-f]{64}$/.test(binding.actorSourceSha256) || !/^[0-9a-f]{64}$/.test(binding.executionIdentitySha256) ||
      binding.model !== 'nvidia/Qwen3.6-35B-A3B-NVFP4' || binding.revision !== '1355db6a052410cfd62085d94b58866fd0f2c3c5')
    throw new Error('Qwen source row or immutable actor/source binding is invalid');
  const turns = new Map<string, { turn: TeacherTurn; ordinal: number }[]>();
  for (const item of row.trajectory as TeacherTurn[]) {
    if (item.phase !== 'action' || typeof item.invocation_id !== 'string' || !item.assistant ||
        !Array.isArray(item.assistant.calls) || !Array.isArray(item.context))
      throw new Error('Qwen source trajectory contains an incomplete action turn');
    const entries = turns.get(item.invocation_id) ?? [];
    entries.push({ turn: item, ordinal: entries.length }); turns.set(item.invocation_id, entries);
  }
  const matched = new Map<string, number>();
  const runId = [...turns.keys()][0]?.split('/')[0];
  if (!runId || [...turns.keys()].some(id => !id.startsWith(`${runId}/`)))
    throw new Error('Qwen source trajectory spans multiple run identities');
  return { runId, driver: async request => {
    const id = request.invocation_id;
    if (typeof id !== 'string' || !id.startsWith(`${runId}/`))
      throw new Error('corrected replay invocation does not match exact Qwen task/run identity');
    const entries = turns.get(id); const ordinal = matched.get(id) ?? 0; const entry = entries?.[ordinal];
    if (!entry) throw new Error(`no exact Qwen source invocation/turn match for ${id}#${ordinal}`);
    matched.set(id, ordinal + 1);
    const turn = entry.turn, assistant = turn.assistant!;
    const sourcePrompt = { messages: turn.context, tools: turn.tools_offered ?? [] };
    const target = { role: 'assistant', content: contentParts(assistant.content),
      ...(assistant.reasoning ? { reasoning: assistant.reasoning } : {}),
      tool_calls: assistant.calls!.map((call, index) => ({ id: `teacher-${index}`, type: 'function',
        function: { name: call.tool, arguments: JSON.stringify(call.arguments) } })) };
    const receipt = { schema: 'natlang.teacher-turn-source/1', source_path: binding.sourcePath,
      source_file_sha256: binding.sourceFileSha256, source_view: binding.sourceView,
      row_id: row.id, task_sha256: sha(canonical(row.task)), invocation_id: id, ordinal: entry.ordinal,
      turn_sha256: sha(canonical(turn)), prompt_sha256: sha(canonical(sourcePrompt)),
      actor: { model: binding.model, revision: binding.revision, actor_source_sha256: binding.actorSourceSha256,
        execution_identity_sha256: binding.executionIdentitySha256 } };
    return send({ ...request, native_teacher_forced_turn: { schema: 'natlang.native-teacher-forced-turn/1',
      assistant: target, source_receipt: receipt } });
  } };
}
