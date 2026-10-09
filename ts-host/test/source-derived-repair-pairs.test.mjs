import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadSourceDerivedRepairCandidates, sourceDerivedRepairReviewPair,
  validateSourceDerivedRepairItem } from '../scripts/source-derived-repair-pairs.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const proposalPath = path.join(root, 'runs/luna-authored-root-return-guidance-fivecase-20261009-v1/evidence/counterfactual-eligibility-repairs-v2/proposal.json');

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
});
