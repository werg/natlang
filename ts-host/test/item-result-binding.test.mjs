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
