import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {originalOutcomeObservations,validateReassessmentCheckRoles} from '../scripts/observed-workspace-evidence.mjs';
test('promotes an already recorded answer with exact old projection and new bound roles',async()=>{
 const root=await mkdtemp(join(tmpdir(),'TEST-only-reassessment-promotion-'));
 try{
  const original={checks:{native:true},observedChecks:{native:true,answer:true},outcomeContract:{effects:{checks:['native'],rationale:'Native state'},returnedValue:{checks:['answer'],required:false,rationale:'Old optional finding'}},outcomeDimensions:{effects:{passed:true,checks:{native:true}},returnedValue:{required:false,passed:true,checks:{answer:true}}}};
  const observations=originalOutcomeObservations(original),contract=structuredClone(original.outcomeContract);contract.returnedValue.required=true;
  const raw=JSON.stringify(contract);await writeFile(join(root,'contract.json'),raw);
  const proof={checks:{native:true,answer:true},outcomeContract:contract,outcomeDimensions:{effects:{passed:true,checks:{native:true}},returnedValue:{required:true,passed:true,checks:{answer:true}}},checker_sources:[{path:'contract.json',sha256:createHash('sha256').update(raw).digest('hex'),role:'outcome-contract'}]};
  await validateReassessmentCheckRoles(proof,original.checks,root,observations);
  for(const mutate of [p=>p.checks.native=false,p=>p.checks.invented=true,p=>p.checker_sources=[]]){const bad=structuredClone(proof);mutate(bad);await assert.rejects(validateReassessmentCheckRoles(bad,original.checks,root,observations));}
  for(const mutate of [p=>p.observedChecks.native=false,p=>p.observedChecks.invented=true,p=>delete p.outcomeContract,p=>p.outcomeDimensions.returnedValue.checks.answer=false,p=>p.outcomeDimensions.returnedValue.passed=false,p=>p.outcomeContract.effects.checks.push('answer'),p=>p.outcomeContract.returnedValue.required=true]){const bad=structuredClone(original);mutate(bad);assert.throws(()=>originalOutcomeObservations(bad));}
 }finally{await rm(root,{recursive:true,force:true});}
});

test('historical bound projection repartition preserves already optional original observations',async()=>{
 const {validateReassessmentOutcomeRoles}=await import('../scripts/observed-workspace-evidence.mjs');
 const root=await mkdtemp(join(tmpdir(),'TEST-only-historical-projection-'));
 try{
  const original={checks:{native:true,oldLabel:false},observedChecks:{native:true,oldLabel:false,receipt:true},outcomeContract:{effects:{checks:['native'],rationale:'Actual native state'},returnedValue:{checks:['oldLabel'],required:true,rationale:'Original required label'},incidentalValues:{checks:['receipt'],rationale:'Already optional native receipt echo'}},outcomeDimensions:{effects:{passed:true,checks:{native:true}},returnedValue:{required:true,passed:false,checks:{oldLabel:false}},incidentalValues:{passed:true,checks:{receipt:true}}}};
  const contract={effects:{checks:['native'],rationale:'Actual native state remains required'},returnedValue:{checks:[],required:false,rationale:'Literal goal requires the actual effect'},incidentalValues:{checks:['oldLabel'],rationale:'Original unrequested label retained diagnostically'}};
  const raw=JSON.stringify(contract);await writeFile(join(root,'contract.json'),raw);
  const proof={checks:{native:true},outcomeContract:contract,outcomeDimensions:{effects:{passed:true,checks:{native:true}},returnedValue:{required:false,passed:true,checks:{}},incidentalValues:{passed:false,checks:{oldLabel:false}}},checker_sources:[{path:'contract.json',sha256:createHash('sha256').update(raw).digest('hex'),role:'outcome-contract'}]};
  await validateReassessmentOutcomeRoles(proof,original,root);
  for(const mutate of [p=>p.checks.native=false,p=>p.checks.receipt=true,p=>p.outcomeDimensions.incidentalValues.checks.oldLabel=true,p=>p.checker_sources=[]]){const bad=structuredClone(proof);mutate(bad);await assert.rejects(validateReassessmentOutcomeRoles(bad,original,root));}
  const corrupt=structuredClone(original);corrupt.outcomeDimensions.incidentalValues.checks.receipt=false;await assert.rejects(validateReassessmentOutcomeRoles(proof,corrupt,root));
 }finally{await rm(root,{recursive:true,force:true});}
});
