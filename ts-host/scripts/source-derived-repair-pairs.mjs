import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { responseTarget, PREFERENCE_VERSION } from '../dist/teacher/handoff.js';

const sha256 = value => createHash('sha256').update(value).digest('hex');
const canonical = value => JSON.stringify(value, (_key, nested) =>
  nested && typeof nested === 'object' && !Array.isArray(nested)
    ? Object.fromEntries(Object.entries(nested).sort(([a], [b]) => a.localeCompare(b))) : nested);

export function validateSourceDerivedRepairItem(item) {
  const fail = reason => ({ ok: false, reason });
  const source = item?.source, original = item?.original_provider_decision;
  const repair = item?.counterfactual_repair, contract = item?.pair_contract;
  if (!item?.id || !source || !original || !repair || !contract) return fail('missing_required_repair_fields');
  if (source.split !== 'train') return fail('source_not_train_split');
  if (!source.source_id || !source.source_group || !source.program_id || !source.source_row_sha256_including_lf)
    return fail('source_identity_or_row_pin_missing');
  if (!/^[a-f0-9]{64}$/.test(original.raw_result_sha256 ?? '') ||
      !/^[a-f0-9]{64}$/.test(original.request_sha256 ?? '') ||
      !/^[a-f0-9]{64}$/.test(original.raw_response_sha256 ?? '') ||
      !/^[a-f0-9]{64}$/.test(original.provider_context_canonical_sha256 ?? ''))
    return fail('provider_action_pins_missing');
  if (sha256(canonical(original.provider_context)) !== original.provider_context_canonical_sha256 ||
      contract.original_context_hash !== original.provider_context_canonical_sha256)
    return fail('provider_context_hash_mismatch');
  if (sha256(canonical(original.tools_offered)) !== original.tools_offered_canonical_sha256)
    return fail('offered_tool_schema_hash_mismatch');
  if (repair.target_is_counterfactual !== true || repair.target_is_model_observed !== false ||
      repair.original_model_hidden_states_equivalent !== false || repair.successful_task_completion_claimed !== false ||
      repair.training_admission !== false || repair.preference_pair_admission !== false ||
      contract.chosen_side_is_static_source_derived_counterfactual !== true ||
      contract.observed_rejected_side_is_original_provider_response !== true ||
      contract.no_grader_or_expected_output_in_provider_context !== true ||
      contract.counterfactual_changes_only_response_target_not_prompt_or_source !== true ||
      contract.same_exact_provider_context !== true || contract.same_offered_return_result_tool_schema !== true ||
      contract.preserves_source_group_and_split !== true)
    return fail('counterfactual_scope_flags_invalid');
  if (repair.target_arguments?.status !== 'success' || !Object.hasOwn(repair.target_arguments, 'value') ||
      canonical(repair.target_arguments) !== canonical(repair.target_tool_call?.function?.arguments
        ? JSON.parse(repair.target_tool_call.function.arguments) : null))
    return fail('typed_terminal_call_arguments_mismatch');
  if (repair.target_tool_call?.function?.name !== 'return_result' ||
      canonical(repair.target_value) !== canonical(repair.target_arguments.value))
    return fail('typed_terminal_call_invalid');
  const returnTool = original.tools_offered.find(tool => tool?.function?.name === 'return_result');
  const returnSchema = returnTool?.function?.parameters;
  const valueSchema = returnSchema?.properties?.value;
  if (!returnTool || returnSchema?.properties?.status?.enum?.includes('success') !== true ||
      !returnSchema.required?.includes('status')) return fail('return_result_schema_not_offered');
  if (typeof repair.target_value === 'boolean') {
    if (repair.target_arguments.value !== true || valueSchema?.type !== 'boolean')
      return fail('source_derived_boolean_target_not_true');
  } else {
    const value = repair.target_value;
    const valueProps = valueSchema?.properties ?? {};
    const sourceValue = source.eligibility_file_body;
    if (!value || typeof value !== 'object' || Array.isArray(value) ||
        Object.keys(value).sort().join(',') !== 'candidateId,eligible,evidence' || value.eligible !== true ||
        value.candidateId !== sourceValue?.candidateId || value.evidence !== sourceValue?.evidence ||
        valueSchema?.type !== 'object' || Object.keys(valueProps).sort().join(',') !== 'candidateId,eligible,evidence' ||
        valueProps.candidateId?.type !== 'string' || valueProps.eligible?.type !== 'boolean' || valueProps.evidence?.type !== 'string' ||
        valueSchema.required?.slice().sort().join(',') !== 'candidateId,eligible,evidence')
      return fail('source_derived_typed_fact_does_not_match_pinned_source');
  }
  if (sha256(canonical(source.eligibility_file_body)) !== source.eligibility_file_body_sha256 ||
      repair.source_oracle_lineage?.exact_eligibility_source_body_sha256 !== source.eligibility_file_body_sha256)
    return fail('source_evidence_hash_mismatch');
  const captured = original.typed_child_capture_record?.host_capture_value ??
    original.typed_child_capture_record?.returned_host_value;
  const observedFalse = captured === false || (captured && typeof captured === 'object' && captured.eligible === false);
  if (!observedFalse) return fail('observed_rejected_value_not_false');
  if (!Array.isArray(original.provider_context) || !Array.isArray(original.tools_offered) ||
      !Array.isArray(original.observed_model_response?.calls)) return fail('provider_context_or_observed_action_missing');
  const observedCalls = original.observed_model_response.calls;
  if (observedCalls.length !== 1 || typeof observedCalls[0]?.[0] !== 'string' || !observedCalls[0]?.[1])
    return fail('observed_terminal_action_missing');
  return { ok: true };
}

export async function loadSourceDerivedRepairCandidates(proposalPath) {
  const base = path.resolve(path.dirname(proposalPath), '../../../../');
  const proposalBytes = await readFile(proposalPath);
  const proposal = JSON.parse(proposalBytes.toString('utf8'));
  if (proposal.schema !== 'natlang.source-derived-counterfactual-action-repair-review/2' ||
      proposal.status !== 'held-review-packet-not-training-data' || !Array.isArray(proposal.items))
    throw new Error('unsupported source-derived repair proposal schema or disposition');
  const pinFile = async (pin, label) => {
    if (!pin?.path || !/^[a-f0-9]{64}$/.test(pin.sha256 ?? '')) throw new Error(`${label}: missing immutable path/hash pin`);
    const bytes = await readFile(path.resolve(base, pin.path));
    if (sha256(bytes) !== pin.sha256) throw new Error(`${label}: pinned file hash mismatch`);
    return bytes;
  };
  const audit = JSON.parse((await pinFile(proposal.audit_receipt, 'source audit')).toString('utf8'));
  const inventory = JSON.parse((await pinFile(proposal.source_inventory, 'source inventory')).toString('utf8'));
  if (inventory.source_sha256 !== proposal.source_inventory.source_sha256)
    throw new Error('source inventory does not bind the pinned source corpus');
  const sourceBytes = await readFile(path.resolve(base, inventory.source_path));
  if (sha256(sourceBytes) !== inventory.source_sha256) throw new Error('source corpus hash mismatch');
  const sourceLines = sourceBytes.toString('utf8').split(/(?<=\n)/);
  const resultCache = new Map();
  for (const item of proposal.items) {
    const checked = validateSourceDerivedRepairItem(item);
    if (!checked.ok) throw new Error(`${item?.id ?? 'unknown repair'}: ${checked.reason}`);
    if (item.source.source_file_sha256 !== inventory.source_sha256) throw new Error(`${item.id}: source corpus pin mismatch`);
    const sourceLine = sourceLines.find(line => line.endsWith('\n') &&
      (JSON.parse(line).source_ids ?? []).includes(item.source.source_id));
    if (!sourceLine || sha256(sourceLine) !== item.source.source_row_sha256_including_lf)
      throw new Error(`${item.id}: exact source row bytes do not match`);
    const sourceRow = JSON.parse(sourceLine);
    if (sourceRow.split !== item.source.split || !sourceRow.source_groups?.includes(item.source.source_group))
      throw new Error(`${item.id}: source split/group binding mismatch`);
    const folderFiles = sourceRow.semantics?.folder_files ?? {};
    const eligibilityText = folderFiles[item.source.eligibility_file];
    const noticeText = folderFiles[item.source.review_notice_file];
    const auditText = folderFiles[item.source.eligibility_audit_file];
    if (typeof eligibilityText !== 'string' || typeof noticeText !== 'string' || typeof auditText !== 'string' ||
        sha256(eligibilityText) !== item.source.eligibility_file_body_sha256 ||
        sha256(noticeText) !== item.source.review_notice_sha256 || sha256(auditText) !== item.source.eligibility_audit_sha256 ||
        canonical(JSON.parse(eligibilityText)) !== canonical(item.source.eligibility_file_body) ||
        noticeText !== item.source.review_notice_body || auditText !== item.source.eligibility_audit_body)
      throw new Error(`${item.id}: source evidence bytes/body do not match the pinned facts`);
    const providerContextText = canonical(item.original_provider_decision.provider_context);
    const expectedFiles = sourceRow.semantics?.expected_files ?? {};
    const expectedValues = [...Object.values(expectedFiles), sourceRow.semantics?.expected];
    if (expectedValues.some(expected => typeof expected === 'string' && expected && providerContextText.includes(expected)))
      throw new Error(`${item.id}: grader expected output appears in the provider prompt`);
    const resultPath = path.resolve(base, item.original_provider_decision.raw_result_path);
    let result = resultCache.get(resultPath);
    if (!result) {
      const resultBytes = await readFile(resultPath);
      result = { bytes: resultBytes, value: JSON.parse(resultBytes.toString('utf8')) };
      resultCache.set(resultPath, result);
    }
    if (sha256(result.bytes) !== item.original_provider_decision.raw_result_sha256)
      throw new Error(`${item.id}: raw provider result hash mismatch`);
    const events = result.value.trajectory.filter(event => event.invocation_id === item.original_provider_decision.invocation_id);
    if (events.length !== 1) throw new Error(`${item.id}: provider invocation is missing or ambiguous in pinned raw result`);
    const event = events[0], original = item.original_provider_decision;
    if (event.request_sha256 !== original.request_sha256 || event.raw_response_sha256 !== original.raw_response_sha256 ||
        canonical(event.context) !== canonical(original.provider_context) ||
        canonical(event.tools_offered) !== canonical(original.tools_offered) ||
        canonical(event.model_response) !== canonical(original.observed_model_response))
      throw new Error(`${item.id}: proposal request/action/context differs from raw provider invocation`);
    if (canonical(event.assistant) !== canonical(item.original_provider_decision.observed_assistant))
      throw new Error(`${item.id}: raw provider terminal action differs from pinned assistant action`);
  }
  if (proposal.items.length !== 3) throw new Error('expected exactly three source-audited repair candidates');
  return { proposal, proposal_sha256: sha256(proposalBytes), audit_sha256: proposal.audit_receipt.sha256,
    source_inventory_sha256: proposal.source_inventory.sha256, items: proposal.items, audit };
}

export function sourceDerivedRepairReviewPair(item) {
  const checked = validateSourceDerivedRepairItem(item);
  if (!checked.ok) return { status: 'held', reason: checked.reason, id: item?.id ?? null };
  const original = item.original_provider_decision;
  const index = Number(original.invocation_id.split('/').at(-1));
  const observed = original.observed_model_response;
  const rejectedTurn = { text: observed.text ?? '', calls: observed.calls, execution_plan: observed.execution_plan };
  const observedTarget = responseTarget(rejectedTurn, index);
  if (canonical(observedTarget.tool_calls?.[0]?.function) !== canonical(original.observed_model_response.raw_calls?.[0]?.function))
    return { status: 'held', reason: 'native_completion_target_serialization_mismatch', id: item.id };
  const chosenTarget = responseTarget({ text: '', calls: [['return_result', item.counterfactual_repair.target_arguments]] }, index);
  const proposalPin = item.id;
  const pair = { version: PREFERENCE_VERSION, id: `${item.id}:source-derived-repair`, kind: 'wrong_result',
    program_id: item.source.program_id, source_groups: [item.source.source_group],
    messages: original.provider_context, tools: original.tools_offered,
    chosen: { target: chosenTarget, reasoning: null, reasoning_trained: false },
    rejected: { target: observedTarget, reasoning: observed.execution_plan ?? null,
      ...(observed.execution_plan ? {} : { reasoning_trained: false }) },
    evidence: { kind: 'source-derived-action-repair', source_id: item.source.source_id,
      source_row_sha256: item.source.source_row_sha256_including_lf,
      source_group: item.source.source_group, split: item.source.split,
      request_sha256: original.request_sha256, raw_response_sha256: original.raw_response_sha256,
      observed_action_target_sha256: original.original_target_sha256,
      repair_target_sha256: item.counterfactual_repair.target_canonical_sha256,
      proposal_item_id: proposalPin } };
  return { status: 'held', training_admission: false, disposition: 'root_preference_pair_admission_pending',
    synthetic_target: true, observed_rejected_target: true,
    runtime_or_hidden_state_equivalence: false, successful_task_completion_claimed: false,
    pair };
}
