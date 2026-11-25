import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {reassessNativeReview} from './reassess_native_review.mjs';
const sha=value=>createHash('sha256').update(value).digest('hex');
const ref=value=>({complete:true,bytes:Buffer.byteLength(JSON.stringify(value)),hash:sha(JSON.stringify(value))});
function fixture(){
 const goal='Check this original worker and leave a private diagnostic.',entry={program:'chat',callable:'chat',arguments:{goal}};
 const source=Buffer.from('if(agent.participantId&&agent.participantId!==receipt.participantId) retainedWorkerRelationship(independentlyRead.events,newWorker.participantId) oldWorker.participantId===newWorker.participantId');
 const index={openingWorker:{objectKey:'original',participantId:'native-original'}},frames=[{channelId:'chat',worker:{revision:1}},{channelId:'chat',worker:{revision:3}}],report={disposition:{status:'ready'}};
 const input={actual:{goal,observations:{index,frames,evidence:{native:true}},artifacts:{artifact:'private diagnostic'},requestedReturn:report},expected:{goal,requestedReturnRequired:true},rubric:'Old review.'};
 const prior={caseId:'chat-continuity-case',goal,checks:{native:true},actualResult:report,index,frames,evidence:{native:true},artifact:'private diagnostic',semanticReviewRequired:true,semanticReviewInput:input,outcomeContract:{returnedValue:{required:true}}};
 const effects=[['chatContinuityIndex','list',[],index],...frames.map(frame=>['chatContinuityState','read',['chat'],frame])].map(([service,method,args,result])=>({service,method,by:'agent',args:ref(args),result:ref(result)}));
 const native={entry,report,rootCallIds:['root'],calls:[{callId:'root',record:{outcome:'done',output:ref(report),effects}}]};
 const packet={test:{name:'open-goal-chat'},result:{passed:true},execution:{diagnostics:{openGoalVerification:prior,natlangTeacher:native}}};
 const review={schema:'natlang.live-workflow-review/1',input,verdict:{accepted:false},exchange:{request:{},reply:{}}};
 const pins={teacher:'openai-codex:gpt-6-luna',repositories:{base:{dirty:false}},validator_sha256:sha(source)},task={scenario:packet.test.name,split:'train',interaction:'open-user-goal',entry,goalSha256:sha(JSON.stringify(goal))};
 const checker={observedChatWorkerIdentity:()=>({sameObjectKey:true,sameParticipantId:true}),chatContinuityReviewGuidance:'Native identity guidance',observedContentReview:(proof,contract,value,keys,artifacts)=>({semanticReviewRequired:true,semanticReviewInput:{actual:{goal:proof.goal,observations:Object.fromEntries(keys.map(key=>[key,proof[key]])),artifacts:Object.fromEntries(artifacts.map(key=>[key,proof[key]])),requestedReturn:value},expected:{goal:proof.goal,requestedReturnRequired:contract.returnedValue.required},rubric:'Judge actual mode. '}})};
 const bindInput=proof=>{assert.deepEqual(proof.semanticReviewInput.actual.observations,{index:proof.index,frames:proof.frames,evidence:proof.evidence});return proof.semanticReviewInput;};return {packet,pins,task,review,source,checker,bindInput};
}
const run=s=>reassessNativeReview(s.packet,s.pins,s.task,s.review,s.source,'chat-continuity',s.checker,entry=>entry,s.bindInput);
test('keeps all native checks and immutable original rejection while binding new source-derived facts',()=>{const s=fixture(),before=structuredClone(s.packet),r=run(s);assert.deepEqual(r.projected.checks,{native:true});assert.deepEqual(r.reviewInput.actual.observations.evidence.nativeWorkerIdentity,{sameObjectKey:true,sameParticipantId:true});assert.deepEqual(r.reviewInput.actual.requestedReturn,before.execution.diagnostics.natlangTeacher.report);assert.deepEqual(s.packet,before);assert.equal(s.review.verdict.accepted,false);});
for(const[label,change]of[
 ['native failure',s=>{s.packet.execution.diagnostics.openGoalVerification.checks.native=false;}],
 ['unbound review',s=>{s.review={...s.review,input:{actual:{goal:'different'}}};}],
 ['missing original rejection exchange',s=>{delete s.review.exchange;}],
 ['unbound original actor effect',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.pop();}],
 ['unfinished original',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.outcome='error';}],
 ['altered source',s=>{s.source=Buffer.from('other');}],
 ['altered root output',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.output.hash='0'.repeat(64);}],
 ['held split',s=>{s.task.split='test';}],
 ['review waiver',s=>{s.checker.observedContentReview=()=>({semanticReviewRequired:false});}],
])test(`refuses ${label}`,()=>{const s=fixture();change(s);assert.throws(()=>run(s));});
