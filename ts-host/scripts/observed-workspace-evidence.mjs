import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { isDeepStrictEqual } from 'node:util';

async function readBound(binding, root) {
  if (!binding?.path || !/^[a-f\d]{64}$/u.test(binding.sha256 ?? ''))
    throw new Error('invalid observed workspace artifact binding');
  const bytes = await readFile(resolve(root, binding.path));
  if (createHash('sha256').update(bytes).digest('hex') !== binding.sha256)
    throw new Error(`observed workspace artifact changed: ${binding.path}`);
  return bytes;
}

/** Verify the on-disk artifacts before allowing the generic native exporter to use an observed capture. */
export async function verifyObservedWorkspaceEvidence(row) {
  const binding = row.provenance.source_manifest;
  const root = dirname(binding.path);
  const manifest = JSON.parse(await readBound(binding, process.cwd()));
  const attestation = manifest.verification;
  const receipt = JSON.parse(await readBound(attestation, root));
  const equal = isDeepStrictEqual;
  if (manifest.schema !== 'natlang.live-teacher-task/1' || receipt.schema !== 'natlang.live-task-verification/1' ||
      !equal(attestation, row.provenance.verification) || receipt.task_id !== row.task.id ||
      (receipt.split ?? manifest.split) !== manifest.split ||
      receipt.run_id !== row.outcome.world.run_id || receipt.status !== 'passed' ||
      !['id', 'family', 'split', 'source_namespace', 'source_ids', 'source_groups', 'source_revisions', 'interaction'].every(
        key => equal(manifest[key], row.task[key])) ||
      !equal(receipt.interaction, row.task.interaction) || !equal(manifest.world, receipt.world) ||
      !equal(receipt.world, row.outcome.world) || !equal(receipt.checks, row.outcome.checks) ||
      !equal(receipt.root_call_ids, row.outcome.root_call_ids) ||
      !equal(manifest.code, row.provenance.code) || manifest.code?.dirty !== false)
    throw new Error('observed workspace trajectory differs from the independently verified manifest');
  const evidenceKinds = new Set();
  for (const evidence of receipt.evidence ?? []) {
    await readBound(evidence, root);
    evidenceKinds.add(evidence.kind);
  }
  if (!Array.isArray(manifest.required_effect_kinds) || !manifest.required_effect_kinds.length ||
      manifest.required_effect_kinds.some(kind => !evidenceKinds.has(kind)))
    throw new Error('observed workspace outcome evidence is incomplete');
  await readBound(receipt.validator, root);

  const decisions = Map.groupBy(row.trajectory, item => item.invocation_id);
  const captureScopes = new Map(receipt.calls.map(scope => [scope.call_id, scope]));
  const callFiles = new Map(manifest.calls.map(artifact => [artifact.path, artifact]));
  const roots = new Set(receipt.root_call_ids);
  const goal = row.task.interaction?.goal;
  if (roots.size !== 1 || typeof goal !== 'string' || !goal.trim())
    throw new Error('observed open goal requires one verified goal-only root');
  for (const call of row.outcome.invocation_ledger) {
    const original = JSON.parse(await readBound(call.artifact, root));
    const record = original.record;
    if (!equal(call.artifact, callFiles.get(call.artifact.path)) ||
        original.callId !== call.call_id || !equal(call.scope, captureScopes.get(call.call_id)) ||
        ['call_id', 'owner_kind', 'owner_key', 'run_id'].some(key => original.summary[key] !== call.scope[key]) ||
        original.summary.parent_call_id !== call.parent_call_id ||
        !equal(record.definition, call.definition) || !equal(record.executor, call.executor) ||
        !equal(record.parent_call_id, call.parent_call_id) || record.outcome !== call.outcome ||
        !equal(record.inputs ?? null, call.inputs) || !equal(record.output ?? null, call.output) ||
        !equal(record.graph_custody ?? null, call.graph_custody) ||
        !equal(original.events ?? [], call.events) || !equal(record.effects ?? [], call.effects))
      throw new Error('observed invocation differs from its captured call or verified scope');
    if (record.parent_call_id === null) {
      const input = record.inputs?.goal;
      const goalBytes = Buffer.from(JSON.stringify(goal));
      if (!roots.delete(call.call_id) || record.definition.name !== 'runUserGoal' ||
          record.outcome !== 'done' || Object.keys(record.inputs ?? {}).length !== 1 || input?.complete !== true ||
          input.bytes !== goalBytes.length || input.hash !== createHash('sha256').update(goalBytes).digest('hex'))
        throw new Error('observed root invocation differs from its attested user goal');
    }
    const actual = record.model_turns ?? [];
    const projected = decisions.get(call.call_id) ?? [];
    if (projected.length !== actual.length)
      throw new Error('observed trajectory omitted captured exchanges');
    for (const [index, decision] of projected.entries()) {
      const exchange = actual[index];
      if (index !== decision.live_exchange.ordinal ||
          decision.function !== record.definition.name || decision.definition_id !== record.definition.id ||
          !equal(decision.context, exchange.request?.messages ?? null) ||
          !equal(decision.tools_offered, exchange.request?.tools ?? []) ||
          !equal(decision.live_exchange.gateway, exchange.reply?.raw_response?.vibestudio ?? null) ||
          decision.phase !== (exchange.kind === 'generation' ? 'action' : exchange.kind) ||
          !equal(decision.error, exchange.error ?? null))
        throw new Error('observed decision differs from the recorded model exchange');
      if (decision.training_admission.approved &&
          (!Array.isArray(exchange.reply?.raw_calls) ||
           !equal(decision.assistant.raw_calls, exchange.reply.raw_calls) ||
           decision.assistant.content !== (exchange.reply.text ?? '') ||
           decision.assistant.reasoning !== (exchange.reply.reasoning ?? '')))
        throw new Error('observed decision target differs from the model reply');
    }
    captureScopes.delete(call.call_id);
    callFiles.delete(call.artifact.path);
    decisions.delete(call.call_id);
  }
  if (captureScopes.size || callFiles.size || decisions.size || roots.size)
    throw new Error('observed trajectory does not cover the full verified call tree');
}
