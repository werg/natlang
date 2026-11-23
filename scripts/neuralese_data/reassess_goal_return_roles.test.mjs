import {createHash} from 'node:crypto';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {reassessGoalReturnRoles} from './reassess_goal_return_roles.mjs';
const bind=value=>{const bytes=Buffer.from(JSON.stringify(value));return {complete:true,hash:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length};};
function draft(){
 const goal='Retain the private working draft for local review without committing.';
 const before={mainEventId:'main',committed:{kind:'event',eventId:'private-before'}};
 const workingHead={kind:'application',applicationId:'draft'};
 const snapshot={head:JSON.stringify(workingHead)};
 const report={decision:{action:'draft'},draft:{snapshot,review:{ready:false,unsupportedClaims:['A malformed citation needs local review.']}},
  final:{outcome:'review-needed',candidate:snapshot,event:null},explanation:'The private draft is retained; correct its citation before approval.'};
 const journal=['contentProjectSources.observe','contentProjectScope.confirm','contentProjectDraft.prepare','contentProjectDraft.write'].map((key,index)=>{
  const [service,method]=key.split('.');return {service,method,args:[index],result:{receipt:index}};});
 journal.at(-1).result=snapshot;
 const checks=Object.fromEntries(['exactOwnedSourceDiscovery','genuineEvidenceDisposition','exactScopedAuthorization','exactArtifactOrReadOnlyHold','protectedPrivateBoundary'].map(name=>[name,true]));
 checks.correctOutcomeReport=false;
 const entry={callable:'prepareContentProject',arguments:{goal}};
 const proof={actualResult:report,caseId:'different-draft-domain',contextId:'private',goal,checks,preparedContentProject:{
  contextId:'private',caseId:'different-draft-domain',expectedOutcome:'draft',before,actual:{...before,clean:false,workingHead},
  artifact:{passed:true,semanticReviewInput:{actual:{state:workingHead}}},journal}};
 return {test:{prompt:goal},result:{passed:false},execution:{diagnostics:{openGoalVerification:proof,natlangTeacher:{entry,report,captureErrors:[],rootCallIds:['root'],calls:[{callId:'root',record:{outcome:'done',output:bind(report),effects:journal.map(row=>({...row,args:bind(row.args),result:bind(row.result)}))}}]}}}};
}
test('reassesses the completed private draft without prescribing its final disposition or masking honest returns',()=>{
 const result=reassessGoalReturnRoles(draft());assert.equal(Object.values(result.checks).every(Boolean),true);
 assert.equal(result.checks.correctOutcomeReport,undefined);assert.equal(result.outcomeDimensions.returnedValue.required,false);
 assert.deepEqual(result.outcomeDimensions.incidentalValues,{passed:false,checks:{correctOutcomeReport:false},incorrectReturns:[]});
});
test('preserves every independent failure and requires actual native evidence',()=>{
 const mutations=[p=>p.execution.diagnostics.openGoalVerification.checks.exactArtifactOrReadOnlyHold=false,
  p=>p.execution.diagnostics.openGoalVerification.preparedContentProject.artifact.passed=false,
  p=>p.execution.diagnostics.openGoalVerification.preparedContentProject.actual.mainEventId='changed',
  p=>p.execution.diagnostics.natlangTeacher.calls[0].record.effects.pop(),
  p=>p.execution.diagnostics.natlangTeacher.calls[0].record.output.hash='forged',
  p=>p.execution.diagnostics.natlangTeacher.executionError='failed',
  p=>p.test.prompt='Another user goal'];
 for(const mutate of mutations){const packet=draft();mutate(packet);assert.throws(()=>reassessGoalReturnRoles(packet));}
});
test('rejects native commits and writes omitted from the checker journal',()=>{
 for(const [service,method] of [['contentProjectCommit','commit'],['contentProjectDraft','write']]){
  const packet=draft();
  packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.push({service,method,args:bind([]),result:bind({unexpected:true})});
  assert.throws(()=>reassessGoalReturnRoles(packet),/journal omits original native effects/);
 }
});
test('binds the retained snapshot to both the write receipt and independent working head',()=>{
 for(const kind of ['write','readback']){
  const packet=draft(),diagnostic=packet.execution.diagnostics;
  const state=diagnostic.openGoalVerification.preparedContentProject,native=diagnostic.natlangTeacher;
  const snapshot={head:JSON.stringify({kind:'application',applicationId:'another-draft'})};
  native.report.draft.snapshot=snapshot;native.report.final.candidate=snapshot;
  if(kind==='readback'){
   state.journal.at(-1).result=snapshot;
   native.calls[0].record.effects.at(-1).result=bind(snapshot);
  }
  native.calls[0].record.output=bind(native.report);
  assert.throws(()=>reassessGoalReturnRoles(packet),/snapshot differs/);
 }
});
