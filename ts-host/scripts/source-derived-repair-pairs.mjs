import { createHash } from 'node:crypto';
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import ts from 'typescript';
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
  if (!source.eligibility_file_body || typeof source.eligibility_file_body !== 'object' ||
      Array.isArray(source.eligibility_file_body) || !repair.target_tool_call?.function ||
      typeof repair.target_tool_call.function !== 'object') return fail('source_or_typed_target_shape_invalid');
  if (!Array.isArray(original.provider_context) || !Array.isArray(original.tools_offered) ||
      !Array.isArray(original.observed_model_response?.calls)) return fail('provider_context_or_observed_action_missing');
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
  if (!/^[a-f0-9]{64}$/.test(original.tools_offered_canonical_sha256 ?? '') ||
      sha256(canonical(original.tools_offered)) !== original.tools_offered_canonical_sha256)
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
  if (!repair.target_arguments || typeof repair.target_arguments !== 'object' || Array.isArray(repair.target_arguments))
    return fail('typed_terminal_call_arguments_invalid');
  let encodedArguments;
  try { encodedArguments = JSON.parse(repair.target_tool_call?.function?.arguments); }
  catch { return fail('typed_terminal_call_arguments_malformed_json'); }
  if (repair.target_arguments.status !== 'success' || !Object.hasOwn(repair.target_arguments, 'value') ||
      Object.keys(repair.target_arguments).sort().join(',') !== 'status,value' ||
      canonical(repair.target_arguments) !== canonical(encodedArguments))
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
  if (sha256(canonical(original.observed_model_response)) !== original.original_target_sha256)
    return fail('observed_target_hash_mismatch');
  if (sha256(canonical(repair.target_tool_call)) !== repair.target_canonical_sha256)
    return fail('counterfactual_target_hash_mismatch');
  const captured = original.typed_child_capture_record?.host_capture_value ??
    original.typed_child_capture_record?.returned_host_value;
  const observedFalse = captured === false || (captured && typeof captured === 'object' && captured.eligible === false);
  if (!observedFalse) return fail('observed_rejected_value_not_false');
  const observedCalls = original.observed_model_response.calls;
  if (observedCalls.length !== 1 || typeof observedCalls[0]?.[0] !== 'string' || !observedCalls[0]?.[1])
    return fail('observed_terminal_action_missing');
  return { ok: true };
}

export async function loadSourceDerivedRepairCandidates(proposalPath) {
  const base = await findRepositoryRoot(path.dirname(path.resolve(proposalPath)));
  const proposalBytes = await readFile(proposalPath);
  const proposal = JSON.parse(proposalBytes.toString('utf8'));
  if (proposal.schema === 'natlang.source-derived-causal-eligibility-repair-proposal/1')
    return loadCausalActionRepairCandidates({ base, proposal, proposalBytes });
  if (proposal.schema !== 'natlang.source-derived-counterfactual-action-repair-review/2' ||
      proposal.status !== 'held-review-packet-not-training-data' || !Array.isArray(proposal.items))
    throw new Error('unsupported source-derived repair proposal schema or disposition');
  const pinFile = async (pin, label) => {
    if (!pin?.path || !/^[a-f0-9]{64}$/.test(pin.sha256 ?? '')) throw new Error(`${label}: missing immutable path/hash pin`);
    const bytes = await readFile(resolveRepoArtifact(base, pin.path));
    if (sha256(bytes) !== pin.sha256) throw new Error(`${label}: pinned file hash mismatch`);
    return bytes;
  };
  const audit = JSON.parse((await pinFile(proposal.audit_receipt, 'source audit')).toString('utf8'));
  const inventory = JSON.parse((await pinFile(proposal.source_inventory, 'source inventory')).toString('utf8'));
  if (!audit || !Array.isArray(audit.cases) || !inventory ||
      !/^[a-f0-9]{64}$/.test(inventory.source_sha256 ?? '') ||
      inventory.source_sha256 !== proposal.source_inventory.source_sha256 ||
      audit.source?.sha256 !== inventory.source_sha256)
    throw new Error('source inventory does not bind the pinned source corpus');
  const sourceBytes = await readFile(resolveRepoArtifact(base, inventory.source_path));
  if (sha256(sourceBytes) !== inventory.source_sha256) throw new Error('source corpus hash mismatch');
  const sourceLines = sourceBytes.toString('utf8').split(/(?<=\n)/).filter(Boolean);
  const resultCache = new Map();
  const seen = new Set();
  for (const item of proposal.items) {
    const checked = validateSourceDerivedRepairItem(item);
    if (!checked.ok) throw new Error(`${item?.id ?? 'unknown repair'}: ${checked.reason}`);
    if (seen.has(item.id)) throw new Error(`${item.id}: duplicate proposal item ID`);
    seen.add(item.id);
    if (item.source.source_file_sha256 !== inventory.source_sha256) throw new Error(`${item.id}: source corpus pin mismatch`);
    const sourceLine = sourceLines.find(line => line.endsWith('\n') &&
      (JSON.parse(line).source_ids ?? []).includes(item.source.source_id));
    if (!sourceLine || sha256(sourceLine) !== item.source.source_row_sha256_including_lf)
      throw new Error(`${item.id}: exact source row bytes do not match`);
    const sourceRow = JSON.parse(sourceLine);
    if (sourceRow.id !== item.source.program_id || sourceRow.split !== item.source.split ||
        !Array.isArray(sourceRow.source_ids) || !sourceRow.source_ids.includes(item.source.source_id) ||
        !Array.isArray(sourceRow.source_groups) || !sourceRow.source_groups.includes(item.source.source_group))
      throw new Error(`${item.id}: source split/group binding mismatch`);
    const auditCases = audit.cases.filter(candidate => candidate?.source?.source_id === item.source.source_id);
    if (auditCases.length !== 1) throw new Error(`${item.id}: source audit case is missing or ambiguous`);
    const auditCase = auditCases[0];
    const auditSource = auditCase.source, auditRun = auditCase.sampled_execution, auditDecision = auditCase.first_wrong_decision;
    if (auditSource.program_id !== item.source.program_id || auditSource.split !== item.source.split ||
        auditSource.source_group !== item.source.source_group ||
        auditSource.source_row_sha256_including_lf !== item.source.source_row_sha256_including_lf ||
        auditSource.eligibility_file !== item.source.eligibility_file ||
        canonical(auditSource.eligibility_file_body) !== canonical(item.source.eligibility_file_body) ||
        auditSource.eligibility_file_body_sha256 !== item.source.eligibility_file_body_sha256 ||
        auditSource.review_notice_file !== item.source.review_notice_file ||
        auditSource.review_notice_body !== item.source.review_notice_body ||
        auditSource.review_notice_sha256 !== item.source.review_notice_sha256 ||
        auditSource.independent_eligibility_audit_file !== item.source.eligibility_audit_file ||
        auditSource.independent_eligibility_audit_body !== item.source.eligibility_audit_body ||
        auditSource.independent_eligibility_audit_sha256 !== item.source.eligibility_audit_sha256 ||
        auditSource.physical_jsonl_index !== item.source.physical_source_index ||
        auditRun.result_sha256 !== item.original_provider_decision.raw_result_sha256 ||
        auditDecision.call_id !== item.original_provider_decision.invocation_id ||
        canonical(auditDecision.returned_host_value) !== canonical(item.original_provider_decision.observed_host_value))
      throw new Error(`${item.id}: source audit receipt does not bind this source/action/result`);
    const proposalDecision = item.original_provider_decision.exact_callstore_first_wrong_decision;
    if (!proposalDecision || auditDecision.callstore_record_hash !== proposalDecision.callstore_record_hash ||
        auditDecision.callstore_events_hash !== proposalDecision.callstore_events_hash ||
        auditDecision.initial_state_sha256 !== proposalDecision.initial_state_sha256 ||
        canonical(auditDecision.terminal_eval_action) !== canonical(proposalDecision.terminal_eval_action) ||
        auditDecision.returned_value_sha256 !== proposalDecision.returned_value_sha256)
      throw new Error(`${item.id}: source audit does not bind the exact first wrong action/capture`);
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
    const resultPath = resolveRepoArtifact(base, item.original_provider_decision.raw_result_path);
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
  if (!Array.isArray(proposal.items)) throw new Error('source-derived repair items must be an array');
  return { proposal, proposal_sha256: sha256(proposalBytes), audit_sha256: proposal.audit_receipt.sha256,
    source_inventory_sha256: proposal.source_inventory.sha256, items: proposal.items, audit };
}

async function loadCausalActionRepairCandidates({ base, proposal, proposalBytes }) {
  const fail = message => { throw new Error(`causal action repair proposal: ${message}`); };
  if (proposal.status !== 'held review proposal with exact raw provider request context and explicit pair candidates; not a training preference dataset' ||
      !Array.isArray(proposal.items) || proposal.items.length === 0 ||
      proposal.limits?.training_admission !== false || proposal.limits?.dpo_admission !== false ||
      proposal.limits?.runtime_or_hidden_state_equivalence !== false)
    fail('unsupported schema or disposition');
  const pin = async (entry, label) => {
    if (!entry?.path || !/^[a-f0-9]{64}$/.test(entry.sha256 ?? '')) fail(`${label}: malformed path/hash pin`);
    const bytes = await readFile(resolveRepoArtifact(base, entry.path));
    if (sha256(bytes) !== entry.sha256) fail(`${label}: pinned bytes mismatch`);
    return bytes;
  };
  const sourceBytes = await pin(proposal.source_file, 'source corpus');
  const sourceRows = sourceBytes.toString('utf8').split(/(?<=\n)/).filter(Boolean);
  if (sourceRows.some(line => !line.endsWith('\n'))) fail('source corpus must be LF terminated');
  const seenIds = new Set();
  const loaded = [];
  for (const item of proposal.items) {
    const id = item?.candidate_id;
    if (typeof id !== 'string' || !id || seenIds.has(id)) fail('candidate IDs must be present and unique');
    seenIds.add(id);
    if (!item.source || !item.trace || !item.raw_result || !item.provider_request || !item.callstore ||
        !item.causal_repair || !item.preference_pair_candidate || !item.rejected_provider_response)
      fail(`${id}: required evidence is missing`);
    const { source, trace, raw_result: rawPin, provider_request: request, callstore, causal_repair: repair,
      preference_pair_candidate: pairCandidate, rejected_provider_response: rejected } = item;
    if (source.path !== proposal.source_file.path || source.sha256 !== proposal.source_file.sha256 ||
        !Number.isInteger(source.physical_jsonl_index) || source.physical_jsonl_index < 0 ||
        source.split !== 'train' || !source.source_groups?.length || !source.source_id ||
        source.program_id !== source.source_id || !/^[a-f0-9]{64}$/.test(source.row_sha256_including_lf ?? ''))
      fail(`${id}: source identity/split pin is malformed`);
    const sourceLine = sourceRows[source.physical_jsonl_index];
    if (!sourceLine || sha256(Buffer.from(sourceLine)) !== source.row_sha256_including_lf)
      fail(`${id}: exact source row bytes mismatch`);
    const sourceRow = JSON.parse(sourceLine);
    if (sourceRow.id !== source.program_id || sourceRow.id !== source.source_id || sourceRow.split !== 'train' ||
        !source.source_groups.every(group => sourceRow.source_groups?.includes(group) && sourceRow.source_ids?.includes(group)))
      fail(`${id}: source group/split binding mismatch`);
    const sourceBody = sourceRow.semantics?.folder_files?.[source.source_file];
    if (typeof sourceBody !== 'string' || sha256(Buffer.from(sourceBody)) !== source.source_file_body_sha256 ||
        canonical(JSON.parse(sourceBody)) !== canonical(source.source_file_body))
      fail(`${id}: exact source fact body mismatch`);
    if (repair.training_admission !== false || repair.runtime_or_hidden_state_equivalence !== false ||
        repair.same_exact_request_context?.verified !== true || repair.synthetic_chosen_tool?.is_observed !== false ||
        repair.synthetic_chosen_tool?.is_source_derived !== true ||
        pairCandidate.version !== PREFERENCE_VERSION || pairCandidate.kind !== 'wrong_result' ||
        canonical(pairCandidate.source_groups) !== canonical(source.source_groups))
      fail(`${id}: candidate scope/disposition is invalid`);

    const rawBytes = await pin(rawPin, `${id} raw result`);
    const raw = JSON.parse(rawBytes.toString('utf8'));
    if (rawPin.bytes !== rawBytes.length || !/^[a-f0-9]{64}$/.test(request.request_sha256 ?? '') ||
        !/^[a-f0-9]{64}$/.test(request.raw_response_sha256 ?? '') ||
        request.raw_response_sha256 !== rejected.raw_response_sha256 ||
        repair.same_exact_request_context.request_sha256 !== request.request_sha256)
      fail(`${id}: raw result/request/response pins mismatch`);
    const matchingEvents = (raw.trajectory ?? []).filter(event => event.invocation_id === request.invocation_id &&
      event.request_sha256 === request.request_sha256 && event.raw_response_sha256 === request.raw_response_sha256);
    if (matchingEvents.length !== 1) fail(`${id}: exact invocation/request/response join is missing or ambiguous`);
    const event = matchingEvents[0];
    if (event.phase !== request.phase || canonical(event.context) !== canonical(request.captured_context) ||
        canonical(event.tools_offered) !== canonical(request.tools_offered) ||
        canonical(event.assistant) !== canonical(request.assistant_turn) ||
        canonical(event.model_response) !== canonical(request.model_response))
      fail(`${id}: exact provider prompt, schema, or response does not match raw trajectory`);
    const terminalCalls = (event.model_response?.raw_calls ?? []).filter(call =>
      call.id === request.terminal_tool_call_id && call.id === rejected.provider_call_id);
    if (terminalCalls.length !== 1 || canonical(terminalCalls[0]) !== canonical(rejected.raw_call))
      fail(`${id}: terminal provider tool-call ID/response join is missing or ambiguous`);
    const terminalFunction = terminalCalls[0]?.function;
    if (terminalFunction?.name !== 'eval' || typeof terminalFunction.arguments !== 'string')
      fail(`${id}: rejected terminal action is not the captured eval call`);
    const traceBytes = await pin(trace, `${id} execution trace`);
    const traceEvents = traceBytes.toString('utf8').split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
    const traceStart = traceEvents.filter(row => row.kind === 'invocation' && row.phase === 'start' && row.call_id === trace.call_id);
    const traceRequest = traceEvents.filter(row => row.kind === 'model_request' && row.call_id === trace.call_id &&
      row.phase === 'start' && row.seq === trace.model_request_event?.seq);
    const traceTerminal = traceEvents.filter(row => row.kind === 'action' && row.call_id === trace.call_id &&
      row.tool_call_id === request.terminal_tool_call_id && row.outcome === 'completed');
    const traceCapture = traceEvents.filter(row => row.kind === 'host_capture' && row.call_id === trace.call_id &&
      row.seq === trace.host_capture?.seq);
    if (traceStart.length !== 1 || traceRequest.length !== 1 || traceTerminal.length !== 1 || traceCapture.length !== 1 ||
        canonical(traceStart[0]) !== canonical(trace.invocation_event) ||
        canonical(traceRequest[0]) !== canonical(trace.model_request_event) ||
        canonical(traceTerminal[0]) !== canonical(trace.terminal_action) ||
        canonical(traceCapture[0]) !== canonical(trace.host_capture) ||
        canonical(traceTerminal[0].arguments) !== canonical(JSON.parse(terminalFunction.arguments)) ||
        traceTerminal[0].name !== terminalFunction.name)
      fail(`${id}: request call is not joined to its exact completed trace action`);
    if (!/^[a-f0-9]{64}$/.test(callstore.record_hash ?? '') || !/^[a-f0-9]{64}$/.test(callstore.events_hash ?? '') ||
        trace.call_id !== request.invocation_id || trace.terminal_action?.tool_call_id !== request.terminal_tool_call_id)
      fail(`${id}: CallStore/trace identity pins are malformed`);

    const offeredEval = request.tools_offered.find(tool => tool?.function?.name === 'eval');
    const evalParameters = offeredEval?.function?.parameters;
    if (!offeredEval || evalParameters?.additionalProperties !== false ||
        canonical(Object.keys(evalParameters.properties ?? {}).sort()) !== canonical(['code', 'finish', 'timeout_ms']) ||
        canonical(evalParameters.required) !== canonical(['code']) || evalParameters.properties.code?.type !== 'string' ||
        evalParameters.properties.finish?.type !== 'boolean' || evalParameters.properties.timeout_ms?.type !== 'integer')
      fail(`${id}: exact offered eval schema is absent or unsupported`);
    const chosen = pairCandidate.chosen?.target?.tool_calls;
    const rejectedTarget = pairCandidate.rejected?.target?.tool_calls;
    if (pairCandidate.messages === undefined || canonical(pairCandidate.messages) !== canonical(request.captured_context) ||
        !Array.isArray(chosen) || chosen.length !== 1 || !Array.isArray(rejectedTarget) || rejectedTarget.length !== 1 ||
        rejectedTarget[0].function?.name !== terminalFunction.name)
      fail(`${id}: candidate pair target/context is not aligned to the captured provider action`);
    let observedArguments, rejectedPairArguments;
    try {
      observedArguments = JSON.parse(terminalFunction.arguments);
      rejectedPairArguments = JSON.parse(rejectedTarget[0].function.arguments);
    } catch { fail(`${id}: rejected eval arguments are malformed JSON`); }
    if (Object.keys(observedArguments).sort().join(',') !== 'code,finish' || observedArguments.finish !== true ||
        canonical(rejectedPairArguments) !== canonical(observedArguments))
      fail(`${id}: rejected target does not exactly preserve the terminal eval arguments`);
    let chosenArgs;
    try { chosenArgs = JSON.parse(chosen[0].function.arguments); }
    catch { fail(`${id}: chosen eval arguments are malformed JSON`); }
    const sourceDerived = repair.synthetic_chosen_tool;
    if (chosen[0].function.name !== 'eval' || chosenArgs.finish !== true ||
        Object.keys(chosenArgs).sort().join(',') !== 'code,finish' ||
        canonical(chosenArgs) !== canonical(sourceDerived.arguments) || typeof chosenArgs.code !== 'string')
      fail(`${id}: chosen target differs from the pinned source-derived eval action`);
    const modelResponseCalls = request.model_response?.calls ?? [];
    const normalizedTerminal = modelResponseCalls.filter(([name, args]) => name === 'eval' &&
      canonical(args) === canonical(observedArguments));
    if (normalizedTerminal.length !== 1) fail(`${id}: normalized response does not agree with the exact raw tool call`);
    const syntax = checkEvalCodeAgainstCapturedScope(chosenArgs.code, request.captured_context);
    if (!syntax.ok) fail(`${id}: chosen eval code ${syntax.reason}`);
    loaded.push({ ...item, _causal_action_v5_validated: true,
      _causal_action_v5_proposal_sha256: sha256(proposalBytes),
      _causal_action_v5_code_validation: 'typescript syntax and names bound in the captured opening scope only' });
  }
  return { proposal, proposal_sha256: sha256(proposalBytes), audit_sha256: null,
    source_inventory_sha256: proposal.source_file.sha256, items: loaded, audit: null };
}

function checkEvalCodeAgainstCapturedScope(code, messages) {
  const fail = reason => ({ ok: false, reason });
  const opening = messages.flatMap(message => message?.tool_calls ?? [])
    .find(call => call?.id === 'scope_0' && call?.function?.name === 'eval');
  if (!opening) return fail('captured_scope_opening_missing');
  let openingArgs;
  try { openingArgs = JSON.parse(opening.function.arguments); }
  catch { return fail('captured_scope_opening_malformed'); }
  if (typeof openingArgs.code !== 'string') return fail('captured_scope_source_missing');
  const fileName = '/__source_derived_repair_scope_check__.ts';
  const chosenMarker = '\nasync function __source_derived_candidate__(){\n';
  const prefix = `${openingArgs.code}\ndeclare function nl<T>(parts: TemplateStringsArray, ...values: unknown[]): (input?: unknown) => Promise<T>;\n`;
  const sourceText = `${prefix}${chosenMarker}${code}\n}\n`;
  const chosenStart = prefix.length + chosenMarker.length;
  const sourceFile = ts.createSourceFile(fileName, sourceText, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  if (sourceFile.parseDiagnostics.length) return fail('has_typescript_syntax_error');
  const options = { noEmit: true, target: ts.ScriptTarget.ES2022, skipLibCheck: true, types: [] };
  const host = ts.createCompilerHost(options);
  const baseGetSourceFile = host.getSourceFile.bind(host);
  host.getSourceFile = (file, languageVersion, ...rest) => file === fileName
    ? sourceFile : baseGetSourceFile(file, languageVersion, ...rest);
  host.fileExists = ((base => file => file === fileName || base(file))(host.fileExists.bind(host)));
  host.readFile = ((base => file => file === fileName ? sourceText : base(file))(host.readFile.bind(host)));
  const program = ts.createProgram([fileName], options, host);
  const scopedDiagnostics = ts.getPreEmitDiagnostics(program).filter(diagnostic =>
    diagnostic.file?.fileName === fileName && (diagnostic.start ?? 0) >= chosenStart &&
    [2304, 2552, 2307, 2451, 2454].includes(diagnostic.code));
  return scopedDiagnostics.length ? fail('references_unavailable_or_conflicting_scope') : { ok: true };
}

async function findRepositoryRoot(start) {
  let current = start;
  while (true) {
    try {
      await Promise.all([access(path.join(current, 'scripts/coordination_inbox.py')),
        access(path.join(current, 'training/neuralese_corpora.json'))]);
      return current;
    } catch { /* keep walking toward filesystem root */ }
    const parent = path.dirname(current);
    if (parent === current) throw new Error(`could not locate repository root from ${start}`);
    current = parent;
  }
}

function resolveRepoArtifact(root, relativePath) {
  if (typeof relativePath !== 'string' || !relativePath)
    throw new Error('artifact locator must be a non-empty path');
  const resolved = path.isAbsolute(relativePath) ? path.resolve(relativePath) : path.resolve(root, relativePath);
  const relative = path.relative(root, resolved);
  if (!relative || relative.startsWith(`..${path.sep}`) || relative === '..')
    throw new Error(`artifact locator escapes repository root: ${relativePath}`);
  return resolved;
}

export function sourceDerivedRepairReviewPair(item) {
  if (item?._causal_action_v5_validated === true) {
    const pair = structuredClone(item.preference_pair_candidate);
    pair.program_id = item.source.program_id;
    pair.split = item.source.split;
    pair.source_groups = [...item.source.source_groups];
    pair.tools = item.provider_request.tools_offered;
    pair.evidence = { kind: 'source-derived-causal-action-repair', proposal_item_id: item.candidate_id,
      source_id: item.source.source_id, source_row_sha256: item.source.row_sha256_including_lf,
      source_groups: item.source.source_groups, split: item.source.split,
      raw_result_sha256: item.raw_result.sha256, request_sha256: item.provider_request.request_sha256,
      raw_response_sha256: item.provider_request.raw_response_sha256,
      invocation_id: item.provider_request.invocation_id,
      terminal_tool_call_id: item.provider_request.terminal_tool_call_id,
      synthetic_target: true, observed_rejected_target: true };
    return { status: 'held', training_admission: false, disposition: 'root_per_item_preference_admission_pending',
      synthetic_target: true, observed_rejected_target: true, runtime_or_hidden_state_equivalence: false,
      successful_task_completion_claimed: false,
      chosen_code_validation: item._causal_action_v5_code_validation,
      provenance: { proposal_sha256: item._causal_action_v5_proposal_sha256,
        candidate_id: item.candidate_id, source_file: item.source, raw_result: item.raw_result,
        trace: item.trace, callstore: item.callstore, provider_request: {
          invocation_id: item.provider_request.invocation_id,
          request_sha256: item.provider_request.request_sha256,
          raw_response_sha256: item.provider_request.raw_response_sha256,
          terminal_tool_call_id: item.provider_request.terminal_tool_call_id } }, pair };
  }
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
