import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { parentEvalChain, parentEvalCode, projectDeclaredChildResult } from './lib/decision-child-projection.mjs';

const site = code => {
  const start = code.indexOf('nl.with');
  return { source_span: { file: 'eval', start, end: start + code.slice(start).indexOf('`') + 1 } };
};
const objectCode = `const classify = nl.with<{decision:string}>({})\`choose\`; const result = await classify(item.state); return [item.id, result.decision] as const;`;
assert.deepEqual(projectDeclaredChildResult('{ decision: string; }', { decision: 'order_status' }, objectCode, site(objectCode)), {
  accepted: true, value: 'order_status', projection: 'declared-record-decision-field', result_binding: 'result',
  projection_code_index: 0,
});
const reassignedObjectCode = `const classify = nl.with<{decision:string}>({})\`choose\`; let result = await classify(item.state); result = {decision:'other'}; return result.decision;`;
assert.equal(projectDeclaredChildResult('{decision:string}', { decision: 'order_status' }, reassignedObjectCode, site(reassignedObjectCode)).accepted, false,
  'a reassigned result binding cannot certify the child output');
const mutatedObjectCode = `const classify = nl.with<{decision:string}>({})\`choose\`; const result = await classify(item.state); result.decision = 'other'; return result.decision;`;
assert.equal(projectDeclaredChildResult('{decision:string}', { decision: 'order_status' }, mutatedObjectCode, site(mutatedObjectCode)).accepted, false,
  'a mutated projected field cannot certify the child output');
const wrongObjectCode = `const classify = nl.with<{decision:string}>({})\`choose\`; const result = await classify(item.state); return [item.id, result] as const;`;
assert.equal(projectDeclaredChildResult('{ decision: string; }', { decision: 'order_status' }, wrongObjectCode, site(wrongObjectCode)).accepted, false,
  'record decisions require the exact caller to consume the declared decision field');

const arrayCode = `const classify = nl.with<string[]>({})\`choose\`; const answer = await classify(item.state); return [item.id, answer[0]] as const;`;
assert.deepEqual(projectDeclaredChildResult('string[]', ['order_status'], arrayCode, site(arrayCode)), {
  accepted: true, value: 'order_status', projection: 'declared-string-array-index-zero', result_binding: 'answer',
  projection_code_index: 0,
});
const wrongArrayCode = `const classify = nl.with<string[]>({})\`choose\`; const answer = await classify(item.state); return [item.id, answer];`;
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], wrongArrayCode, site(wrongArrayCode)).accepted, false,
  'array projection requires the caller to select its first value');
const commentOnlyProjection = `const classify = nl.with<string[]>({})\`choose\`; const answer = await classify(item.state); // return answer[0]\nreturn answer;`;
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], commentOnlyProjection, site(commentOnlyProjection)).accepted, false,
  'comments cannot certify a result projection');
const stringOnlyProjection = `const classify = nl.with<string[]>({})\`choose\`; const answer = await classify(item.state); const note = "return answer[0]"; return answer;`;
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], stringOnlyProjection, site(stringOnlyProjection)).accepted, false,
  'string literals cannot certify a result projection');
const shadowedProjection = `const classify = nl.with<string[]>({})\`choose\`; const answer = await classify(item.state); function unrelated() { const answer = ['wrong']; return answer[0]; } return answer;`;
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], shadowedProjection, site(shadowedProjection)).accepted, false,
  'a same-named variable in a nested function cannot certify this result projection');
const nestedConsumerOnly = `const classify = nl.with<string[]>({})\`choose\`; const answer = await classify(item.state); function later() { return answer[0]; } return answer;`;
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], nestedConsumerOnly, site(nestedConsumerOnly)).accepted, false,
  'an unused deferred closure cannot certify the result was consumed');
const transformedCall = `const classify = nl.with<{decision:string}>({})\`choose\`; const result = await classify(item.state); return options.includes(result.decision);`;
assert.equal(projectDeclaredChildResult('{decision:string}', { decision: 'order_status' }, transformedCall, site(transformedCall)).accepted, false,
  'predicate calls do not preserve a selected label as the task result');
const transformedOperator = `const classify = nl.with<string[]>({})\`choose\`; const answer = await classify(item.state); return Boolean(answer[0]);`;
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], transformedOperator, site(transformedOperator)).accepted, false,
  'operator/call transformations do not certify the projected label');
assert.equal(projectDeclaredChildResult('string[]', ['order_status', 'oos'], arrayCode, site(arrayCode)).accepted, false,
  'array output is not flattened when it contains more than one value');

const aggregateCall = `const classifyEach = await Promise.all(records.map(async ({data}) => { const caseId = data.id; const classify = nl.with<string[]>({ caseId })\`choose\`; return [caseId, await classify()]; })); console.log(classifyEach);`;
const aggregateSite = site(aggregateCall);
const aggregateProjection = `const decisions = Object.fromEntries(classifyEach.map(([id, answer]) => [id, answer[0]])); return decisions;`;
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], [aggregateCall, aggregateProjection], aggregateSite).accepted, true,
  'a later same-invocation map may project the exact aggregate created by this child call');
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], [aggregateCall,
  'const otherDecisions = Object.fromEntries(otherResults.map(([id, answer]) => [id, answer[0]])); return otherDecisions;'], aggregateSite).accepted, false,
  'an unrelated later projection cannot prove the child result was consumed');
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], [aggregateCall,
  'const alias = classifyEach; const decisions = Object.fromEntries(classifyEach.map(([id, answer]) => [id, answer[0]])); return decisions;'], aggregateSite).accepted, false,
  'a cross-action alias breaks the exact aggregate-to-projection chain');
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], [aggregateCall,
  'function unrelated() { const classifyEach = []; return Object.fromEntries(classifyEach.map(([id, answer]) => [id, answer[0]])); }'], aggregateSite).accepted, false,
  'a shadowed aggregate in a later function cannot certify trace continuity');
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], [aggregateCall,
  'Object.fromEntries(classifyEach.map(([id, answer]) => [id, answer[0]]));'], aggregateSite).accepted, false,
  'an unused aggregate projection does not establish a supervised result');
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], [aggregateCall,
  'function unused() { const decisions = Object.fromEntries(classifyEach.map(([id, answer]) => [id, answer[0]])); return decisions; }'], aggregateSite).accepted, false,
  'a return from an uncalled nested helper cannot certify the trace result');
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], [aggregateCall,
  'const decisions = Object.fromEntries(classifyEach.map(([id, answer]) => [id, answer[0]])); await folder.file(task.output_path).writeJson(decisions); const verified = await folder.file(task.output_path).readJson(); return verified;'], aggregateSite).accepted, true,
  'an exact map saved and read back from the same path preserves the projected values');
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], [aggregateCall,
  'const decisions = Object.fromEntries(classifyEach.map(([id, answer]) => [id, answer[0]])); await folder.file(task.output_path).writeJson(decisions); const verified = await folder.file(otherPath).readJson(); return verified;'], aggregateSite).accepted, false,
  'a different readback path does not preserve the projected aggregate');
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], [aggregateCall,
  'classifyEach[0] = ["wrong"]; const decisions = Object.fromEntries(classifyEach.map(([id, answer]) => [id, answer[0]])); return decisions;'], aggregateSite).accepted, false,
  'aggregate element writes between trace actions invalidate continuity');
const wrongFunctionAggregate = aggregateCall.replace('await classify()', 'await anotherClassifier()');
assert.equal(projectDeclaredChildResult('string[]', ['order_status'], [wrongFunctionAggregate, aggregateProjection], site(wrongFunctionAggregate)).accepted, false,
  'a different child function in the call action cannot bind this result');

const scalarCode = `const classify = nl.with<string>({})\`choose\`; const answer = await classify(item.state); return { decision: answer };`;
assert.deepEqual(projectDeclaredChildResult('string', 'order_status', scalarCode, site(scalarCode)), {
  accepted: true, value: 'order_status', projection: 'declared-string-return', result_binding: 'answer',
  projection_code_index: 0,
});
assert.equal(projectDeclaredChildResult('string', { decision: 'order_status' }, scalarCode, site(scalarCode)).accepted, false,
  'host value must match the declared scalar return type');
assert.equal(projectDeclaredChildResult('unknown', 'order_status', scalarCode, site(scalarCode)).accepted, false,
  'unsupported or inferred result types are not normalized');

const twoChildCode = `const first = nl.with<{decision:string}>({})\`first\`; const second = nl.with<{decision:string}>({})\`second\`; const answer = await first(item.state); const other = await second(item.state); return other.decision;`;
const firstSite = { source_span: { file: 'eval', start: twoChildCode.indexOf('nl.with'), end: twoChildCode.indexOf('`', twoChildCode.indexOf('nl.with')) + 1 } };
assert.equal(projectDeclaredChildResult('{decision:string}', { decision: 'order_status' }, twoChildCode, firstSite).accepted, false,
  'a different child result cannot prove this child was projected');

const codeA = `const classify = nl.with<string[]>({})\`choose\`; const answer = await classify(); return answer[0];`;
const codeB = `const other = nl.with<{decision:string}>({})\`other\`; const result = await other(); return result.decision;`;
const codeHash = code => createHash('sha256').update(code).digest('hex');
const row = { trajectory: [
  { invocation_id: 'parent', phase: 'action', assistant: { calls: [
    { tool: 'eval', arguments: { code: codeA } }, { tool: 'eval', arguments: { code: codeB } },
  ] } },
] };
const childEntry = { invocation_id: 'child', parent_invocation_id: 'parent', inline_instruction_site: {
  ...site(codeA), origin: { parentInvocationId: 'parent', writtenCodeSha256: codeHash(codeA),
    checkedCodeSha256: codeHash(codeA), actionOrdinal: 1 },
} };
const ledger = [childEntry, { invocation_id: 'parent', parent_invocation_id: null }];
assert.equal(parentEvalCode(row, 'parent'), null, 'ambiguous parent actions are not accepted without callsite evidence');
const chain = parentEvalChain(row, ledger, childEntry);
assert.equal(chain.length, 1);
assert.equal(chain[0].code, codeA, 'only the caller action whose written-code hash is pinned by the child site is included');
assert.equal(projectDeclaredChildResult('string[]', ['flip_coin'], chain.flatMap(item => [item.code, ...item.followup_codes]), childEntry.inline_instruction_site).accepted, true);
const noMatch = { ...childEntry, inline_instruction_site: { ...childEntry.inline_instruction_site,
  origin: { ...childEntry.inline_instruction_site.origin, writtenCodeSha256: '0'.repeat(64) } } };
assert.deepEqual(parentEvalChain(row, ledger, noMatch), [], 'a later unrelated consumer cannot prove an earlier child projection');

console.log('decision child projection checks passed');
