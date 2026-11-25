import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {reassessImageAnswer} from './reassess_image_answer.mjs';
const sha=value=>createHash('sha256').update(value).digest('hex');
const ref=value=>({complete:true,bytes:Buffer.byteLength(JSON.stringify(value)),hash:sha(JSON.stringify(value))});
function fixture(){
 const goal='Is this original WebP? Verify the retained data.',entry={program:'image',callable:'image',arguments:{goal}};
 const source=Buffer.from('const independent=await Promise.all(queue.map(id=>inspect(latest,id))); sameEvidenceMembers(reads,independent) isDeepStrictEqual(independent.map(read=>read.status),observed) groundedImageAssessment:isDeepStrictEqual');
 const index={channelId:'channel',briefs:[{id:'brief'}]},snapshot={channelId:'channel',through:1,rows:[]},read={candidateId:'candidate',rowId:'row',status:'mismatch',original:{mime:'image/png',width:1,height:1}};
 const report={candidates:[{id:'candidate',eligible:true,reasons:[]}],originals:[read],decision:{action:'hold',candidateId:null,explanation:'Actual PNG is not WebP.'},handoff:null};
 const checks=Object.fromEntries(['exactImageFrontiers','exactImageBrief','independentlyReadImageOriginals','verifiedPrivateImageHandoff','preservedImageWorkspace','rejectedForeignImage','returnedImageOriginals','groundedImageDecision'].map(name=>[name,true]));checks.groundedImageAssessment=false;
 const effects=[['imageCandidateBriefs','list',[],index],['imageCandidateJournal','read',['channel'],snapshot],['imageCandidateOriginal','read',['brief','candidate'],read]].map(([service,method,args,result])=>({service,method,by:'crisp',args:ref(args),result:ref(result)}));
 const prior={goal,checks,actualResult:report,index,snapshots:[snapshot],reads:[read],plan:{briefId:'brief'},decision:report.decision,handoff:null,artifact:null,outcomeContract:{returnedValue:{required:true}},outcomeDimensions:{effects:{passed:true}},semanticReviewInput:{actual:{goal,observations:{index,snapshots:[snapshot],reads:[read]},artifacts:{artifact:null},requestedReturn:report},expected:{goal,requestedReturnRequired:true},rubric:'Judge actual meaning.'}};
 const native={entry,report,rootCallIds:['root'],calls:[{callId:'root',record:{definition:{name:'image',source:'image.ts'},outcome:'done',output:ref(report),effects}}]};
 const packet={test:{name:'open-goal-image'},result:{passed:false},execution:{diagnostics:{openGoalVerification:prior,natlangTeacher:native}}};
 const pins={teacher:'openai-codex:gpt-6-luna',repositories:{system:{dirty:false}},validator_sha256:sha(source)};
 const task={scenario:packet.test.name,split:'train',interaction:'open-user-goal',entry,goalSha256:sha(JSON.stringify(goal))};return {packet,pins,task,source};
}
const run=s=>reassessImageAnswer(s.packet,s.pins,s.task,s.source,entry=>entry);
test('preserves actual answer and false diagnostic without a gold bookkeeping mask',()=>{const s=fixture(),before=structuredClone(s.packet),r=run(s);assert.equal(r.checks.groundedImageAssessment,undefined);assert.equal(r.outcomeDimensions.incidentalValues.checks.groundedImageAssessment,false);assert.deepEqual(r.outcomeDimensions.incidentalValues.incorrectReturns,[]);assert.deepEqual(r.reviewInput.actual.requestedReturn,before.execution.diagnostics.natlangTeacher.report);assert.deepEqual(s.packet,before);});
for(const[label,change]of[
 ['other native failure',s=>{s.packet.execution.diagnostics.openGoalVerification.checks.preservedImageWorkspace=false;}],
 ['missing requested answer check',s=>{delete s.packet.execution.diagnostics.openGoalVerification.checks.groundedImageDecision;}],
 ['different root bytes',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.output.hash='0'.repeat(64);}],
 ['unknown original effect',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.pop();}],
 ['altered source',s=>{s.source=Buffer.from('other');}],
 ['different native original',s=>{s.packet.execution.diagnostics.openGoalVerification.reads=[{candidateId:'invented'}];}],
 ['unbound review',s=>{s.packet.execution.diagnostics.openGoalVerification.semanticReviewInput.actual.goal='other';}],
 ['unrelated return mask',s=>{s.packet.execution.diagnostics.openGoalVerification.incorrectReturnValues=[{contribution:{checks:['other']}}];}],
 ['unfinished root',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.outcome='error';}],
 ['held split',s=>{s.task.split='test';}],
 ['different teacher',s=>{s.pins.teacher='openai-codex:gpt-6-sol';}],
])test(`refuses ${label}`,()=>{const s=fixture();change(s);assert.throws(()=>run(s));});
