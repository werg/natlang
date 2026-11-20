/** Canonical native decisions from an independently verified live workspace capture. */
type Fields = Record<string, unknown>;

function object(value: unknown): Fields {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('observed workspace evidence must be an object');
  return value as Fields;
}

type ObservedRow = {
  id: string;
  task: Fields;
  provenance: Fields;
  outcome: Fields;
  trajectory: Fields[];
};

export const OBSERVED_WORKSPACE_EXECUTION = 'observed-live-workspace';

export function materializeObservedWorkspace(row: ObservedRow, digest: string): Fields[] {
  const checks = Object.values(object(row.outcome.checks));
  const source = object(row.provenance.source_manifest);
  const verification = object(row.provenance.verification);
  if (row.outcome.accepted !== true || row.outcome.status !== 'passed' ||
      !checks.length || checks.some(value => value !== true) ||
      [source, verification].some(binding => typeof binding.path !== 'string' ||
        !/^[a-f\d]{64}$/u.test(String(binding.sha256 ?? ''))) ||
      object(row.task.interaction).kind !== 'open-user-goal' ||
      row.task.source !== OBSERVED_WORKSPACE_EXECUTION || !Array.isArray(row.outcome.invocation_ledger))
    throw new Error('observed workspace trajectory lacks verified open-goal evidence');

  const calls = new Map<string, Fields>();
  for (const invocation of row.outcome.invocation_ledger) {
    const call = object(invocation);
    if (typeof call.call_id !== 'string' || calls.has(call.call_id) || !Array.isArray(call.events))
      throw new Error('invalid observed invocation ledger');
    calls.set(call.call_id, call);
  }
  const ordinals = new Map<string, number>();
  const decisions: Fields[] = [];
  for (const [index, rawDecision] of row.trajectory.entries()) {
    const decision = object(rawDecision);
    const exchange = object(decision.live_exchange);
    const invocationId = decision.invocation_id;
    const call = typeof invocationId === 'string' ? calls.get(invocationId) : undefined;
    if (!call || typeof invocationId !== 'string' || call.parent_call_id !== decision.parent_invocation_id ||
        exchange.ordinal !== (ordinals.get(invocationId) ?? 0) ||
        object(call.artifact).sha256 !== object(exchange.call_artifact).sha256)
      throw new Error('observed decision differs from its capture ledger');
    ordinals.set(invocationId, Number(exchange.ordinal) + 1);
    if (object(decision.training_admission).approved !== true) continue;
    const assistant = object(decision.assistant);
    const gateway = object(exchange.gateway);
    const observed = new Map<string, Fields[]>();
    for (const value of call.events as unknown[]) {
      const event = object(value);
      if (event.kind !== 'action' || typeof event.tool_call_id !== 'string') continue;
      const events = observed.get(event.tool_call_id) ?? [];
      events.push(event);
      observed.set(event.tool_call_id, events);
    }
    const rawCalls = assistant.raw_calls;
    if (decision.phase !== 'action' || decision.error || call.outcome !== 'done' ||
        object(call.executor).kind === 'crisp' || !Array.isArray(decision.context) ||
        !Array.isArray(decision.tools_offered) || !Array.isArray(rawCalls) ||
        typeof assistant.content !== 'string' || typeof assistant.reasoning !== 'string' ||
        (!assistant.content && !rawCalls.length) || !gateway.model || exchange.student_sampling_claimed !== false) {
      throw new Error('observed workspace decision is ineligible for supervision');
    }
    const actionReceipts = rawCalls.map(value => {
      const id = object(value).id;
      const events = typeof id === 'string' ? observed.get(id) : undefined;
      if (events?.length !== 1 || !['ok', 'completed'].includes(String(events[0]?.outcome)))
        throw new Error('observed workspace decision lacks a successful tool action receipt');
      return events[0];
    });
    decisions.push({
      version: 'natlang.teacher_training_turn.native/1',
      id: `${row.id}:decision:${String(index).padStart(4, '0')}`,
      source_ref: { trajectory_id: row.id, source_row_sha256: digest, invocation_id: invocationId,
        parent_invocation_id: decision.parent_invocation_id, live_exchange: exchange },
      provenance: row.provenance,
      task: row.task,
      task_id: row.task.id,
      task_family: row.task.family,
      family: 'teacher_program',
      split: row.task.split,
      source: 'teacher-native',
      source_ids: row.task.source_ids,
      source_groups: row.task.source_groups,
      source_revisions: row.task.source_revisions,
      source_program_ids: [],
      messages: decision.context,
      tools: decision.tools_offered,
      teacher_reasoning: assistant.reasoning || null,
      target: { role: 'assistant', content: assistant.content,
        ...(assistant.reasoning ? { reasoning_content: assistant.reasoning } : {}),
        ...(rawCalls.length ? { tool_calls: rawCalls } : {}) },
      decision: { ...decision, action_receipts: actionReceipts },
      teacher_trajectory_id: row.id,
      teacher_trajectory_digest: digest,
      outcome: { accepted: true, status: row.outcome.status },
      training_admission: { kind: 'independent-observed-workspace-outcome', approved: true },
    });
  }
  if (!decisions.length) throw new Error('verified observed workspace has no supervised decisions');
  return decisions;
}
