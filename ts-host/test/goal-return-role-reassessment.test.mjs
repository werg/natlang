import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {reassessGoalReturnRoles as assess} from '../../scripts/neuralese_data/reassess_goal_return_roles.mjs';
const binding=value=>{const bytes=Buffer.from(JSON.stringify(value));return {complete:true,bytes:bytes.length,hash:createHash('sha256').update(bytes).digest('hex')}};
function packet(kind){
 const observation={contextId:'owner',operationId:'op',host:{kind:'desktop'},published:{id:'browser',runtime:'web',abi:'web-host-1'}};
 const source={observation,disposition:'published-contract'},host={status:'supported'};
 const search={queries:['native methods'],hits:[[{id:'vcs.status'}]],described:[{id:'vcs.status',scope:'owner'}]};
 const report=kind==='desktop'?{evidence:{admission:{source,host,decision:'check-declaration'}},decision:{answer:'continue'}}:{search};
 const effects=kind==='desktop'?[{service:'clientAdmissionRead',method:'observe',args:binding(['op']),result:binding(observation)}]:[
 {service:'nativeIntentDiscovery',method:'search',args:binding(search.queries),result:binding(search.hits[0])},
 {service:'nativeIntentDiscovery',method:'describe',args:binding(['vcs.status']),result:binding(search.described[0])}];
 const prior={caseId:kind==='desktop'?'desktop-readiness-test':'intent-discovery-test',contextId:'owner',desk:{operationId:'op'},
  checks:{[kind==='desktop'?'exactDesktopClientSource':'installedIntentDiscovery']:false,requestedAnswer:true},actualResult:report,
  exactAdmission:{source,host,decision:'reviewed-ready'},search,journal:[{service:'clientAdmissionRead',method:'observe',args:['op'],result:observation}]};
 return {result:{passed:false},execution:{diagnostics:{openGoalVerification:prior,natlangTeacher:{report,rootCallIds:['root'],calls:[{callId:'root',record:{outcome:'done',definition:{name:'reviewDesktopReadiness',source:'reviewDesktopReadiness.nl'},output:binding(report),effects},events:kind==='intent'?[{kind:'effect',service:'nativeIntentDiscovery',method:'search',result:JSON.stringify(search.hits[0])}]:[]}]}}}};
}
test('desktop completion retains root-only tree and masks exact incorrect returns without requiring children',()=>{
 const result=assess(packet('desktop'));assert.ok(Object.values(result.checks).every(Boolean));
 const selectors=result.outcomeDimensions.incidentalValues.incorrectReturns;
 assert.deepEqual(selectors.map(item=>item.required),[false,false,true]);
 assert.equal(selectors.at(-1).definition.source,'reviewDesktopReadiness.nl');
});
for(const kind of ['desktop','intent'])test(`${kind} rejects forged witnesses, wrong application value, and unrelated goal failure`,()=>{
 const original=packet(kind);assert.ok(assess(original));
 const missing=structuredClone(original);missing.execution.diagnostics.natlangTeacher.calls[0].record.effects=[];
 assert.throws(()=>assess(missing),/witness/);
 const wrong=structuredClone(original);wrong.execution.diagnostics.natlangTeacher.calls[0].record.output.hash='0'.repeat(64);
 assert.throws(()=>assess(wrong),/output identity/);
 const failed=structuredClone(original);failed.execution.diagnostics.openGoalVerification.checks.requestedAnswer=false;
 assert.throws(()=>assess(failed),/remain unproved/);
});
test('historical noncanonical unused hit transcript retains goal completion with exact producer/root exclusions',()=>{
 const raw=packet('intent'),native=raw.execution.diagnostics.natlangTeacher;
 native.report.search.hits=[[]];native.calls[0].record.output=binding(native.report);
 const result=assess(raw);
 assert.ok(Object.values(result.checks).every(Boolean));
 assert.equal(result.outcomeDimensions.incidentalValues.passed,false);
 assert.deepEqual(result.outcomeDimensions.incidentalValues.incorrectReturns.map(item=>item.required),[false,true]);
});
