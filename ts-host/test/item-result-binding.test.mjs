import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const { checkItemResultBinding } = await import(process.env.NATLANG_ITEM_BINDING_MODULE ??
  '../dist/teacher/item-result-binding.js');

const proposal = JSON.parse(readFileSync(new URL('../../plans/semantic_stack/ASTRA_ECOLOGY_ITEM_RESULT_V18_PROPOSAL.json', import.meta.url)));
const site = proposal.item_result_binding.sites[0];
const spec = { schema: 'semantic-actual-child-result-binding/1', sites: [site] };
const relation = { cause: 'hares', effect: 'grass', sourceIds: ['r03'], witnesses: [{ sourcePath: 'plot/r03.txt', quote: 'Hares eat grass.' }] };
const decision = { cause: 'hares', effect: 'grass', usable: true, sign: -1, reason: 'consumption', witnesses: relation.witnesses };
const prior = { pending: [{ relation }], completedInputs: [], decisions: [] };
const next = { pending: [], completedInputs: [{ relation }], decisions: [decision] };
const inputCapture = (name, value, definition_source = null) => ({ name, complete: true, value, definition_source });
const outputCapture = (value, definition_source, result_type) => ({ complete: true, value, definition_source, result_type });
const ledger = [
  { invocation_id: 'parent', completion_status: 'done', host_inputs: [inputCapture('state', prior, site.owner_source_path)],
    host_result: outputCapture(next, site.owner_source_path, 'ScopeFrontier') },
  { invocation_id: 'child', parent_invocation_id: 'parent', completion_status: 'done',
    inline_instruction_site: { returns: { natlang: 'ScopeDecision' } },
    host_inputs: [inputCapture('input', { relation })], host_result: outputCapture(decision, 'eval', 'ScopeDecision') },
];

test('actual scoped result is consumed once with its original item', () => {
  assert.equal(checkItemResultBinding(spec, ledger).status, 'pass');
});
test('missing child is a provenance failure, missing input is an unverified hold', () => {
  assert.equal(checkItemResultBinding(spec, ledger.slice(0, 1)).status, 'fail');
  const incomplete = structuredClone(ledger);
  incomplete[1].host_inputs[0].complete = false;
  assert.equal(checkItemResultBinding(spec, incomplete).status, 'unverified_hold');
});
test('altered semantic field or foreign witness cannot qualify', () => {
  const altered = structuredClone(ledger);
  altered[0].host_result.value.decisions[0] = { ...altered[0].host_result.value.decisions[0], sign: 1 };
  assert.equal(checkItemResultBinding(spec, altered).status, 'fail');
  const foreign = structuredClone(ledger);
  foreign[0].host_result.value.decisions[0] = { ...foreign[0].host_result.value.decisions[0],
    witnesses: [{ sourcePath: 'elsewhere', quote: 'different' }] };
  assert.equal(checkItemResultBinding(spec, foreign).status, 'fail');
});
test('scope permits only the declared exact owner-witness union', () => {
  const augmented = structuredClone(ledger);
  const owner = { sourcePath: 'plot/r04.txt', quote: 'Hares reduce grass.' };
  augmented[1].host_inputs[0].value.relation.witnesses.push(owner);
  augmented[1].host_result.value = { ...augmented[1].host_result.value,
    witnesses: [relation.witnesses[0]] };
  augmented[0].host_result.value.decisions[0] = { ...augmented[0].host_result.value.decisions[0],
    witnesses: [relation.witnesses[0], owner] };
  assert.equal(checkItemResultBinding(spec, augmented).status, 'pass');
});
test('effect finding must be the actual child value for the newly completed entity', () => {
  const effectSite = proposal.item_result_binding.sites[1];
  const effectSpec = { schema: spec.schema, sites: [effectSite] };
  const finding = { entity: 'grass', status: 'unresolved', reason: 'opposing signed paths',
    witnesses: [{ sourcePath: 'plot/r04.txt', quote: 'Hares reduce grass.' }] };
  const effectPrior = { remainingEntityKeys: ['grass'], completedEntityKeys: [], nodeFindings: [], edges: [], seeds: [] };
  const effectNext = { remainingEntityKeys: [], completedEntityKeys: ['grass'], nodeFindings: [finding], edges: [], seeds: [] };
  const rows = [
    { invocation_id: 'effect-parent', completion_status: 'done',
      host_inputs: [inputCapture('state', effectPrior, effectSite.owner_source_path)],
      host_result: outputCapture(effectNext, effectSite.owner_source_path, 'EffectFrontier') },
    { invocation_id: 'effect-child', parent_invocation_id: 'effect-parent', completion_status: 'done',
      inline_instruction_site: { returns: { natlang: 'EffectNodeFinding' } },
      host_inputs: [inputCapture('input', { entity: 'grass' })],
      host_result: outputCapture(finding, 'eval', 'EffectNodeFinding') },
  ];
  assert.equal(checkItemResultBinding(effectSpec, rows).status, 'pass');
  const bypass = structuredClone(rows);
  bypass[0].host_result.value.nodeFindings[0] = { ...bypass[0].host_result.value.nodeFindings[0],
    status: 'increase' };
  assert.equal(checkItemResultBinding(effectSpec, bypass).status, 'fail');
});

test('joined seed coverage requires actual judgments for prior and current candidates', () => {
  const seedSpec = { schema: spec.schema, sites: [], joined_seed_coverage: {
    schema: 'semantic-joined-seed-coverage/1', owner_source_path: 'solve/auditEffectGraph.nl',
    candidate_array_pointers: ['/context/priorCompletedState/interactionAssertions', '/prior/interactionAssertions'],
    candidate_roles: ['perturbation', 'unresolved'], child_return_type: 'SeedDecision',
    candidate_source_id_pointer: '/sourceId', candidate_role_pointer: '/sourceRole',
    child_input_source_id_pointer: '/assertion/sourceId', child_result_source_id_pointer: '/sourceId',
  } };
  const joinedInput = { context: { priorCompletedState: { interactionAssertions: [
    { sourceId: 'r01', sourceRole: 'perturbation' },
  ] } }, prior: { interactionAssertions: [
    { sourceId: 'r07', sourceRole: 'interaction' }, { sourceId: 'r08', sourceRole: 'unresolved' },
  ] } };
  const seedChild = (sourceId, callId) => ({ invocation_id: callId, parent_invocation_id: 'audit',
    completion_status: 'done', inline_instruction_site: { returns: { natlang: 'SeedDecision' } },
    host_inputs: [inputCapture('input', { assertion: { sourceId } })],
    host_result: outputCapture({ sourceId, kind: 'interaction' }, 'eval', 'SeedDecision') });
  const rows = [{ invocation_id: 'audit', completion_status: 'done',
    host_inputs: [inputCapture('input', joinedInput, seedSpec.joined_seed_coverage.owner_source_path)],
    host_result: outputCapture({}, seedSpec.joined_seed_coverage.owner_source_path, 'FinalSemanticState') },
  seedChild('r01', 'seed-prior'), seedChild('r08', 'seed-current')];
  assert.equal(checkItemResultBinding(seedSpec, rows).status, 'pass');
  assert.equal(checkItemResultBinding(seedSpec, [rows[0], rows[2]]).status, 'fail');
  assert.equal(checkItemResultBinding(seedSpec, [rows[0], rows[1], seedChild('r07', 'wrong-role')]).status, 'fail');
  const incomplete = structuredClone(rows);
  incomplete[1].host_inputs[0].complete = false;
  assert.equal(checkItemResultBinding(seedSpec, incomplete).status, 'unverified_hold');
});
