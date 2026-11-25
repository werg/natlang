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

function authorityFixture(){
 const s=fixture(), prior=s.packet.execution.diagnostics.openGoalVerification, native=s.packet.execution.diagnostics.natlangTeacher;
 s.source=Buffer.from('binding.services.assignedAuthorityRead.preflight(operationId) snapshots.push(observation)');s.pins.validator_sha256=sha(s.source);
 const observation={operationId:'site',decision:'acquirable',leaves:[{capability:'external.open',resourceKey:'https://review.example.invalid',tier:'gated',status:'acquirable'}]};
 prior.caseId='route-authority-case';prior.offer={request:{operations:[{id:'site'}],ceiling:[{capability:'external.open',resourceKey:'https://review.example.invalid'}]}};prior.snapshots=[observation];
 prior.authority={status:'needs-acquisition',protectedLeaves:prior.offer.request.ceiling,held:[],reviews:[{observation,outside:[],disposition:'needs-acquisition'}]};prior.choice=null;prior.readback=null;prior.artifact=null;
 const report={outcome:{status:'needs-approval',summary:'Native approval remains acquirable; no visit occurred.'},readback:null,handoff:null};prior.actualResult=native.report=report;native.calls[0].record.output=ref(report);
 native.calls[0].record.effects=[['routeEvidenceRead','discover',[],prior.offer],['assignedAuthorityRead','preflight',['site'],observation]].map(([service,method,args,result])=>({service,method,by:'crisp',args:ref(args),result:ref(result)}));
 s.checker.observedRouteAuthority=(rows,ceiling)=>({status:'needs-acquisition',protectedLeaves:ceiling,held:[],reviews:rows.map(observation=>({observation,outside:[],disposition:'needs-acquisition'}))});s.checker.routeAuthorityReviewGuidance='Exact acquirable preflight requires approval; do not visit to report it.';
 const keys=['offer','snapshots','authority','choice','readback'];
 prior.semanticReviewInput=s.checker.observedContentReview(prior,prior.outcomeContract,report,keys,['artifact']).semanticReviewInput;s.review.input=structuredClone(prior.semanticReviewInput);
 s.bindInput=proof=>{assert.deepEqual(proof.semanticReviewInput.actual.observations,Object.fromEntries(keys.map(key=>[key,proof[key]])));return proof.semanticReviewInput;};return s;
}
const authorityRun=s=>reassessNativeReview(s.packet,s.pins,s.task,s.review,s.source,'route-authority',s.checker,entry=>entry,s.bindInput);
test('binds an approval-state query to actual native offer, exact preflights and unchanged answer without a visit',()=>{const s=authorityFixture(),before=structuredClone(s.packet),r=authorityRun(s);assert.deepEqual(s.packet,before);assert.deepEqual(r.projected.checks, before.execution.diagnostics.openGoalVerification.checks);assert.deepEqual(r.reviewInput.actual.requestedReturn,s.packet.execution.diagnostics.natlangTeacher.report);assert.match(r.reviewInput.rubric,/do not visit/);assert.equal(s.review.verdict.accepted,false);});
for(const[label,change]of[
 ['missing native offer',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.shift();}],
 ['missing native preflight',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.pop();}],
 ['unobserved approval',s=>{s.packet.execution.diagnostics.openGoalVerification.snapshots[0].decision='allowed';}],
 ['forged aggregate authority',s=>{s.packet.execution.diagnostics.openGoalVerification.authority.status='eligible';}],
 ['duplicate preflight',s=>{const p=s.packet.execution.diagnostics.openGoalVerification;p.snapshots.push(structuredClone(p.snapshots[0]));}],
 ['foreign operation',s=>{s.packet.execution.diagnostics.openGoalVerification.offer.request.operations=[];}],
 ['missing domain guidance',s=>{delete s.checker.routeAuthorityReviewGuidance;}],
])test(`rejects approval reassessment with ${label}`,()=>{const s=authorityFixture();change(s);assert.throws(()=>authorityRun(s));});
