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
  if (row.outcome.accepted !== true || !['passed', 'partial'].includes(String(row.outcome.status)) ||
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
  const selected = new Map<string, Set<number> | null>();
  if (row.outcome.status === 'partial') {
    const scope = object(row.outcome.training_scope);
    if (row.outcome.task_completed !== false || !['natlang.training-scope/1', 'natlang.training-scope/2'].includes(String(scope.schema)) ||
        !Array.isArray(scope.invocations) || !scope.invocations.length)
      throw new Error('partial observed decisions require independent training scope');
    const certified = scope.schema === 'natlang.training-scope/2';
    if (certified && !scope.assessor || scope.invocations.some(value => Object.hasOwn(object(value), 'certificate') !== certified))
      throw new Error('native certificates require training-scope/2 and its source-bound assessor');
    for (const value of scope.invocations) {
      const invocation = object(value);
      const checks = Object.values(object(invocation.checks));
      const indexes = invocation.ordinals;
      if (typeof invocation.call_id !== 'string' || !calls.has(invocation.call_id) || selected.has(invocation.call_id) ||
          !checks.length || checks.some(value => value !== true) || !Array.isArray(invocation.evidence) ||
          !invocation.evidence.length || indexes != null && (!Array.isArray(indexes) || !indexes.length ||
            new Set(indexes).size !== indexes.length || indexes.some(index => !Number.isSafeInteger(index) || index < 0)))
        throw new Error('invalid partial observed invocation scope');
      selected.set(invocation.call_id, indexes == null ? null : new Set(indexes as number[]));
    }
  } else if (typeof row.outcome.task_completed !== 'boolean' || row.outcome.training_scope != null) {
    throw new Error('completed observed outcome contradicts partial scope');
  }
  const excludedReturns = new Set<string>();
  for (const value of (row.outcome.excluded_return_decisions ?? []) as unknown[]) {
    const exclusion = object(value);
    const call = typeof exclusion.invocation_id === 'string' ? calls.get(exclusion.invocation_id) : undefined;
    if (!call || !Number.isSafeInteger(exclusion.ordinal) || Number(exclusion.ordinal) < 0)
      throw new Error('invalid independently excluded application return');
    excludedReturns.add(`${exclusion.invocation_id}:${exclusion.ordinal}`);
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
    // A reasoned readout's free generation supplies context, never an action target.
    if (decision.phase === 'decision_reasoning') continue;
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
    const selectedOrdinals = selected.get(invocationId);
    const scoped = selected.has(invocationId) && (selectedOrdinals === null || selectedOrdinals!.has(Number(exchange.ordinal)));
    if (row.outcome.status === 'partial' && !scoped || excludedReturns.has(`${invocationId}:${exchange.ordinal}`))
      throw new Error('observed decision is outside independent admission scope');
    if (decision.phase !== 'action' || decision.error ||
        call.outcome !== 'done' && row.outcome.task_completed !== true && !(scoped && selectedOrdinals !== null) ||
        object(call.executor).kind === 'crisp' || !Array.isArray(decision.context) ||
        !Array.isArray(decision.tools_offered) || !Array.isArray(rawCalls) ||
        typeof assistant.content !== 'string' || typeof assistant.reasoning !== 'string' ||
        (!assistant.content && !rawCalls.length) ||
        call.outcome !== 'done' && !scoped && !rawCalls.length || !gateway.model || exchange.student_sampling_claimed !== false) {
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
      outcome: { accepted: true, status: row.outcome.status,
        ...(row.outcome.task_completed !== undefined ? {task_completed: row.outcome.task_completed} : {}) },
      training_admission: { kind: 'independent-observed-workspace-outcome', approved: true },
    });
  }
  return decisions;
}
