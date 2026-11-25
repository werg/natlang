/** TEST-only certificate fixtures; no model calls or dataset registration. */
import test from 'node:test';
import assert from 'node:assert/strict';
import{mkdtemp,writeFile,rm,readFile}from'node:fs/promises';
import{tmpdir}from'node:os';import{join}from'node:path';import{createHash}from'node:crypto';
import{verifyNativeInvocationCertificate,verifyObservedWorkspaceEvidence}from'../scripts/observed-workspace-evidence.mjs';
import{materializeNativeRows}from'../dist/teacher/native-materializer.js';
const digest=text=>createHash('sha256').update(text).digest('hex');
const ref=value=>{const text=JSON.stringify(value);return{complete:true,hash:digest(text),bytes:Buffer.byteLength(text)};};
async function fixture(root){
 const bind=async(name,value,raw=false)=>{const text=raw?value:JSON.stringify(value)+'\n';await writeFile(join(root,name),text);return{path:name,sha256:digest(text)};};
 const goal='Review the actual current question.',history={complete:true,messages:[]},answer={status:'no-question',message:null};
 const body='Return the actual native question judgment.\n',source='---\nreturns: DurableQuestionJudgment\n---\n'+body;
 const definition={id:'nl:reviewDurableQuestion/reviewQuestion/interpretQuestion.nl',source:'reviewDurableQuestion/reviewQuestion/interpretQuestion.nl',returns:'DurableQuestionJudgment',instructions:ref(body)};
 const record={call_id:'question',parent_call_id:null,definition,outcome:'done',effects:[],inputs:{goal:ref(goal),history:ref(history)},output:ref(answer)};
 const records=new Map([['question',record]]),proof={goal,diagnostics:{history},actualResult:{question:answer},checks:{exactIndividualPublications:true,exactQuestionDisposition:true,exactParticipantDisposition:false}};
 const outcome={passed:false,reason:'Incorrect independent participant result'};
 const raw={result:outcome,execution:{diagnostics:{natlangTeacher:{calls:[{callId:'question',record}]},openGoalVerification:proof}}};
 const certificate={schema:'natlang.independent-native-return/1',contract:'gad-question-judgment/1',definition,inputs:record.inputs,output:record.output,expectedInputs:{goal,history},expectedInputJson:{goal:JSON.stringify(goal),history:JSON.stringify(history)},expectedReturn:answer,expectedReturnJson:JSON.stringify(answer),instructionsJson:JSON.stringify(body),checker:await bind('contract.nl',source,true),originalPacket:await bind('original.json',raw),nativeBinding:{returnPath:['actualResult','question'],inputPaths:{goal:['goal'],history:['diagnostics','history']},checkNames:['exactIndividualPublications','exactQuestionDisposition']}};
 const scope={original_outcome:await bind('outcome.json',outcome),assessor:await bind('assessor.mjs','source-bound native assessor',true)};
 const invocation={call_id:'question',certificate:await bind('certificate.json',certificate)};
 return{bind,records,scope,invocation,certificate,raw};
}
for(const mutation of [null,'native-return','native-input','unknown-contract','remove-required-check','failed-check','wrong-contract-source','contract-bytes','original-passed','original-call','effects','assessor-missing'])test('native complete-result certificate '+(mutation??'acceptance'),async()=>{
 const root=await mkdtemp(join(tmpdir(),'native-invocation-certificate-'));
 try{
  const f=await fixture(root),{certificate:c,records,scope,invocation,raw,bind}=f;
  if(mutation==='native-return')c.expectedReturn.status='current';
  if(mutation==='native-input')c.expectedInputs.goal='Other goal';
  if(mutation==='unknown-contract')c.contract='arbitrary-successful-call';
  if(mutation==='remove-required-check')c.nativeBinding.checkNames=['exactIndividualPublications'];
  if(mutation==='failed-check')raw.execution.diagnostics.openGoalVerification.checks.exactQuestionDisposition=false;
  if(mutation==='wrong-contract-source')c.contract='gad-participant-judgment/1';
  if(mutation==='contract-bytes')c.checker=await bind('different.nl','---\nreturns: string\n---\nSay something else.\n',true);
  if(mutation==='original-passed')raw.result.passed=true;
  if(mutation==='original-call')raw.execution.diagnostics.natlangTeacher.calls[0].record={...records.get('question'),outcome:'error'};
  if(mutation==='effects')records.get('question').effects=[{service:'unexpected',method:'write'}];
  if(mutation==='assessor-missing')delete scope.assessor;
  c.originalPacket=await bind('original.json',raw);invocation.certificate=await bind('certificate.json',c);
  if(mutation)await assert.rejects(verifyNativeInvocationCertificate(invocation,records,scope,root));
  else await verifyNativeInvocationCertificate(invocation,records,scope,root);
 }finally{await rm(root,{recursive:true,force:true});}
});
test('original failed goal retains only independently native-certified decisions',{skip:!process.env.NATLANG_TEST_PARTIAL_NATIVE_IR},async()=>{
 const native=JSON.parse(await readFile(process.env.NATLANG_TEST_PARTIAL_NATIVE_IR,'utf8'));
 await verifyObservedWorkspaceEvidence(native);
 const rows=materializeNativeRows([native]).turns;
 assert.equal(native.outcome.status,'partial');assert.equal(native.outcome.task_completed,false);
 assert.equal(native.outcome.invocation_ledger.length,3);assert.equal(rows.length,1);
 assert.equal(rows[0].decision.function,'interpretQuestion');
 assert.equal(native.trajectory.filter(d=>d.training_admission.approved).length,1);
});
