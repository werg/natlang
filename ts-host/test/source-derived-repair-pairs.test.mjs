import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { mkdir, mkdtemp, rm, writeFile, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { loadSourceDerivedRepairCandidates, sourceDerivedRepairReviewPair,
  validateSourceDerivedRepairItem } from '../scripts/source-derived-repair-pairs.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const proposalPath = path.join(root, 'runs/luna-authored-root-return-guidance-fivecase-20261009-v1/evidence/counterfactual-eligibility-repairs-v2/proposal.json');
const causalProposalPath = path.join(root, 'runs/luna-authored-root-return-guidance-fivecase-20261009-v1/evidence/leaf-eligibility-repair-audit-v1/causal-boundary-repairs-v5.json');

test('renders the three pinned source-derived repairs as held native preference reviews', async () => {
  const loaded = await loadSourceDerivedRepairCandidates(proposalPath);
  assert.equal(loaded.items.length, 3);
  const reviews = loaded.items.map(sourceDerivedRepairReviewPair);
  assert.deepEqual(reviews.map(row => row.status), ['held', 'held', 'held']);
  assert.ok(reviews.every(row => row.training_admission === false && row.disposition === 'root_preference_pair_admission_pending'));
  for (const review of reviews) {
    const item = loaded.items.find(candidate => candidate.id === review.pair.evidence.proposal_item_id);
    assert.equal(review.pair.version, 'natlang.preference_pair/2');
    assert.equal(review.pair.kind, 'wrong_result');
    assert.equal(review.pair.evidence.kind, 'source-derived-action-repair');
    assert.equal(review.pair.chosen.target.tool_calls[0].function.name, 'return_result');
    assert.equal(review.pair.chosen.target.tool_calls[0].function.arguments.includes('"status":"success"'), true);
    assert.equal(review.pair.rejected.target.role, 'assistant');
    assert.deepEqual(review.pair.messages, item.original_provider_decision.provider_context);
    assert.deepEqual(review.pair.tools, item.original_provider_decision.tools_offered);
    assert.equal(review.runtime_or_hidden_state_equivalence, false);
    assert.equal(review.successful_task_completion_claimed, false);
    assert.deepEqual(review.pair.source_groups, [review.pair.evidence.source_group]);
  }
  assert.deepEqual(reviews.map(row => row.pair.chosen.target.tool_calls[0].function.arguments), [
    '{"status":"success","value":true}',
    '{"status":"success","value":{"candidateId":"RWA-1C","eligible":true,"evidence":"is within the current drought cap and has a valid meter certificate"}}',
    '{"status":"success","value":true}',
  ]);
});

test('holds target, prompt, evidence, and source-group tampering', async () => {
  const { items } = await loadSourceDerivedRepairCandidates(proposalPath);
  const item = structuredClone(items[0]);
  item.counterfactual_repair.target_arguments.value = false;
  assert.equal(validateSourceDerivedRepairItem(item).reason, 'typed_terminal_call_arguments_mismatch');
  const changedContext = structuredClone(items[0]);
  changedContext.original_provider_decision.provider_context[0].content += ' gold';
  assert.equal(validateSourceDerivedRepairItem(changedContext).reason, 'provider_context_hash_mismatch');
  const invalidScope = structuredClone(items[0]);
  invalidScope.pair_contract.no_grader_or_expected_output_in_provider_context = false;
  assert.equal(validateSourceDerivedRepairItem(invalidScope).reason, 'counterfactual_scope_flags_invalid');
  const extraArgument = structuredClone(items[0]);
  extraArgument.counterfactual_repair.target_arguments.unexpected = true;
  extraArgument.counterfactual_repair.target_tool_call.function.arguments = JSON.stringify(extraArgument.counterfactual_repair.target_arguments);
  assert.equal(validateSourceDerivedRepairItem(extraArgument).reason, 'typed_terminal_call_arguments_mismatch');
  const badStatus = structuredClone(items[0]);
  badStatus.counterfactual_repair.target_arguments.status = 'blocked';
  badStatus.counterfactual_repair.target_tool_call.function.arguments = JSON.stringify(badStatus.counterfactual_repair.target_arguments);
  assert.equal(validateSourceDerivedRepairItem(badStatus).reason, 'typed_terminal_call_arguments_mismatch');
  const wrongGroupContract = structuredClone(items[0]);
  wrongGroupContract.pair_contract.preserves_source_group_and_split = false;
  assert.equal(validateSourceDerivedRepairItem(wrongGroupContract).reason, 'counterfactual_scope_flags_invalid');
});

test('rejects malformed source-derived typed fact shape', async () => {
  const { items } = await loadSourceDerivedRepairCandidates(proposalPath);
  const river = structuredClone(items[1]);
  river.counterfactual_repair.target_value.rationale = 'extra field';
  river.counterfactual_repair.target_arguments.value.rationale = 'extra field';
  river.counterfactual_repair.target_tool_call.function.arguments = JSON.stringify(river.counterfactual_repair.target_arguments);
  assert.equal(validateSourceDerivedRepairItem(river).reason, 'source_derived_typed_fact_does_not_match_pinned_source');
  const malformedJson = structuredClone(items[0]);
  malformedJson.counterfactual_repair.target_tool_call.function.arguments = '{bad';
  assert.equal(validateSourceDerivedRepairItem(malformedJson).reason, 'typed_terminal_call_arguments_malformed_json');
  const alteredTargetPin = structuredClone(items[0]);
  alteredTargetPin.counterfactual_repair.target_canonical_sha256 = '0'.repeat(64);
  assert.equal(validateSourceDerivedRepairItem(alteredTargetPin).reason, 'counterfactual_target_hash_mismatch');
});

test('resolves artifact paths from repo markers at varied depth and accepts a one-item review fixture', async () => {
  const loaded = await loadSourceDerivedRepairCandidates(proposalPath);
  const temporaryRoot = await mkdtemp(path.join(root, '.source-derived-repair-depth-'));
  const nested = path.join(temporaryRoot, 'a', 'b', 'c', 'proposal.json');
  try {
    await mkdir(path.dirname(nested), { recursive: true });
    await writeFile(nested, JSON.stringify({ ...loaded.proposal, items: [loaded.items[0]] }));
    const subset = await loadSourceDerivedRepairCandidates(nested);
    assert.equal(subset.items.length, 1);
    assert.equal(subset.items[0].id, loaded.items[0].id);

    const tampered = structuredClone(subset.proposal);
    tampered.items[0].original_provider_decision.request_sha256 = '0'.repeat(64);
    await writeFile(nested, JSON.stringify(tampered));
    await assert.rejects(loadSourceDerivedRepairCandidates(nested), /proposal request\/action\/context differs/);
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test('loads reviewed causal-action v5 joins and keeps every candidate held', async () => {
  const loaded = await loadSourceDerivedRepairCandidates(causalProposalPath);
  assert.equal(loaded.items.length, 3);
  const reviews = loaded.items.map(sourceDerivedRepairReviewPair);
  assert.deepEqual(reviews.map(row => row.status), ['held', 'held', 'held']);
  assert.ok(reviews.every(row => row.training_admission === false &&
    row.disposition === 'root_per_item_preference_admission_pending'));
  assert.deepEqual(reviews.map(row => row.pair.evidence.terminal_tool_call_id),
    loaded.items.map(row => row.provider_request.terminal_tool_call_id));
  assert.ok(reviews.every(row => row.pair.evidence.synthetic_target === true &&
    row.pair.evidence.observed_rejected_target === true && row.pair.messages.length > 0));
  const repeated = loaded.items[0];
  const raw = JSON.parse(await readFile(path.join(root, repeated.raw_result.path), 'utf8'));
  const sameInvocation = raw.trajectory.filter(event => event.invocation_id === repeated.provider_request.invocation_id);
  const exactRequest = sameInvocation.filter(event => event.request_sha256 === repeated.provider_request.request_sha256 &&
    event.raw_response_sha256 === repeated.provider_request.raw_response_sha256);
  assert.ok(sameInvocation.length > 1, 'fixture exercises repeated invocation IDs');
  assert.equal(exactRequest.length, 1, 'request and response hashes select one exact provider turn');
  assert.equal(exactRequest[0].model_response.raw_calls[0].id, repeated.provider_request.terminal_tool_call_id);
});

test('causal-action v5 rejects request/response and repeated-invocation tool-call mismatches', async () => {
  const proposal = JSON.parse(await readFile(causalProposalPath, 'utf8'));
  const temporaryRoot = await mkdtemp(path.join(root, '.causal-action-repair-'));
  const nested = path.join(temporaryRoot, 'deep', 'proposal.json');
  try {
    await mkdir(path.dirname(nested), { recursive: true });
    const write = async changed => writeFile(nested, JSON.stringify(changed));
    const requestMismatch = structuredClone(proposal);
    requestMismatch.items[0].provider_request.request_sha256 = '0'.repeat(64);
    await write(requestMismatch);
    await assert.rejects(loadSourceDerivedRepairCandidates(nested), /raw result\/request\/response pins mismatch/);

    const responseMismatch = structuredClone(proposal);
    responseMismatch.items[0].provider_request.raw_response_sha256 = '0'.repeat(64);
    await write(responseMismatch);
    await assert.rejects(loadSourceDerivedRepairCandidates(nested), /raw result\/request\/response pins mismatch/);

    // The same invocation can occur more than once. The join must still identify the terminal call by its ID.
    const toolCallMismatch = structuredClone(proposal);
    toolCallMismatch.items[0].provider_request.terminal_tool_call_id = 'different-terminal-call';
    await write(toolCallMismatch);
    await assert.rejects(loadSourceDerivedRepairCandidates(nested), /terminal provider tool-call ID/);

    const contextMismatch = structuredClone(proposal);
    contextMismatch.items[0].preference_pair_candidate.messages[0].content += ' altered';
    await write(contextMismatch);
    await assert.rejects(loadSourceDerivedRepairCandidates(nested), /candidate pair target\/context/);

    const terminalArgumentsMismatch = structuredClone(proposal);
    terminalArgumentsMismatch.items[0].preference_pair_candidate.rejected.target.tool_calls[0].function.arguments =
      JSON.stringify({ code: 'return false;', finish: true, extra: true });
    await write(terminalArgumentsMismatch);
    await assert.rejects(loadSourceDerivedRepairCandidates(nested), /rejected target does not exactly preserve/);
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test('causal-action v5 checks chosen eval syntax and names against the captured scope', async () => {
  const proposal = JSON.parse(await readFile(causalProposalPath, 'utf8'));
  const temporaryRoot = await mkdtemp(path.join(root, '.causal-action-scope-'));
  const nested = path.join(temporaryRoot, 'proposal.json');
  try {
    const write = async changed => writeFile(nested, JSON.stringify(changed));
    const syntaxMismatch = structuredClone(proposal);
    const syntaxItem = syntaxMismatch.items[0];
    syntaxItem.causal_repair.synthetic_chosen_tool.arguments.code = 'return (;';
    syntaxItem.preference_pair_candidate.chosen.target.tool_calls[0].function.arguments =
      JSON.stringify(syntaxItem.causal_repair.synthetic_chosen_tool.arguments);
    await write(syntaxMismatch);
    await assert.rejects(loadSourceDerivedRepairCandidates(nested), /typescript_syntax_error/);

    const scopeMismatch = structuredClone(proposal);
    const scopeItem = scopeMismatch.items[0];
    scopeItem.causal_repair.synthetic_chosen_tool.arguments.code = 'return missingEligibilityFact;';
    scopeItem.preference_pair_candidate.chosen.target.tool_calls[0].function.arguments =
      JSON.stringify(scopeItem.causal_repair.synthetic_chosen_tool.arguments);
    await write(scopeMismatch);
    await assert.rejects(loadSourceDerivedRepairCandidates(nested), /unavailable_or_conflicting_scope/);
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});
