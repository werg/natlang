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
