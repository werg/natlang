import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {reassessLocalPresentation,validatePresentationReview} from './reassess_local_presentation.mjs';
import {modelOracleJudge} from '../../ts-host/dist/teacher/model-judge.js';
const sha=x=>createHash('sha256').update(x).digest('hex');
const ref=value=>({complete:true,hash:sha(JSON.stringify(value)),bytes:Buffer.byteLength(JSON.stringify(value))});
function fixture(){
 const source=Buffer.from('const display = await ports.observe(\n!display.passed ||\ndisplay.panelAttempt !== owned.attemptId\ndisplay.buildKey !== owned.buildKey\nif (after.revision !== revision)\nreturn finish({ status: "visible", receipt });');
 const goal='Show the two confirmed rooms here without reserving anything.';
 const entry={program:'skills/open-user-goal/review.ts',callable:'review',arguments:{goal}};
 const report={views:[{id:'actual-layout',mdx:'<Choices id="actual" />'}]};
 const receipt={source:'actual-source',revision:'actual-revision',viewIds:['actual-layout'],mdxDigests:[sha(report.views[0].mdx)]};
 const current={id:receipt.source,revision:receipt.revision,sections:[{data:'Confirmed rooms: Cedar; Willow.'}]};
 const preview={service:'presentationRecords',method:'preview',args:[current.id,current.revision,report],result:{status:'visible',receipt}};
 const prior={goal,entry,current,report,preview:receipt,display:null,journal:[preview],outcomeContract:{returnedValue:{required:false}},
  checks:{independentlyVerifiedPresentationDom:false,protectedPresentationHeadUnchanged:true,joinedCurrentPresentationRevision:true}};
 const native={entry,report:{notes:'Incidental return is not a user requirement'},rootCallIds:['actual-root'],calls:[]};
 native.calls=[{callId:'actual-root',record:{outcome:'done',output:ref(native.report),effects:[{service:preview.service,method:preview.method,args:ref(preview.args),result:ref(preview.result),by:'crisp'}]}}];
 const packet={test:{name:'open-goal-rooms'},result:{passed:false},execution:{diagnostics:{openGoalVerification:prior,natlangTeacher:native}}};
 const pins={teacher:'openai-codex:gpt-6-luna',validator_sha256:sha(source),repositories:{'system-testing':{dirty:false}}};
 const task={scenario:packet.test.name,split:'train',entry,interaction:'open-user-goal',goalSha256:sha(JSON.stringify(goal))};
 const project=actual=>actual.views.map(view=>({id:view.id,mdx:view.mdx}));
 return {packet,pins,task,source,project};
}
const assess=f=>reassessLocalPresentation(f.packet,f.pins,f.task,f.source,f.project,entry=>entry);
test('native preview effect certifies original rendering without fabricating uncaptured final DOM or requiring optional notes',()=>{
 const f=fixture(),before=JSON.stringify(f),r=assess(f);
 assert.equal(r.checks.independentlyVerifiedPresentationDom,true);
 assert.equal(r.witness.finalDomCaptured,false);
 assert.equal(r.reviewInput.actual.artifacts.display,null);
 assert.deepEqual(r.reviewInput.actual.artifacts.report,f.packet.execution.diagnostics.openGoalVerification.report);
 assert.equal(r.reviewInput.actual.requestedReturn,null);
 assert.equal(JSON.stringify(f),before);
});
for(const [name,alter]of [
 ['another-failed-check',f=>f.packet.execution.diagnostics.openGoalVerification.checks.protectedPresentationHeadUnchanged=false],
 ['source-hash',f=>f.pins.validator_sha256='0'.repeat(64)],
 ['missing-dom-guard',f=>{f.source=Buffer.from('unreviewed implementation');f.pins.validator_sha256=sha(f.source);}],
 ['changed-mdx',f=>f.packet.execution.diagnostics.openGoalVerification.report.views[0].mdx='<Changed />'],
 ['effect-result',f=>f.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects[0].result.hash='0'.repeat(64)],
 ['missing-effect',f=>f.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects=[]],
 ['required-return',f=>f.packet.execution.diagnostics.openGoalVerification.outcomeContract.returnedValue.required=true],
 ['changed-root-value',f=>f.packet.execution.diagnostics.natlangTeacher.report={notes:'changed bytes'}],
 ['another-goal',f=>f.task.goalSha256=sha(JSON.stringify('Different goal'))],
 ['held-task',f=>f.task.split='test'],
])test(`reassessment refuses ${name}`,()=>{const f=fixture();alter(f);assert.throws(()=>assess(f));});

test('meaning review must bind the actual source/artifact and exact independently captured grade',async()=>{
 const input=assess(fixture()).reviewInput,teacher='openai-codex:gpt-6-luna',grade={accepted:true,verdict:'The actual native choices match the confirmed rooms and no reservation occurred.'};
 const reply={calls:[['grade',grade]],raw_calls:[{function:{name:'grade',arguments:JSON.stringify(grade)}}],raw_response:{provider:'openai-codex',model:'gpt-5.6-terra'}};
 let request;const verdict=await modelOracleJudge(async actual=>{request=actual;return reply;})(input);
 const review={schema:'natlang.live-workflow-review/1',model:'openai-codex:gpt-5.6-terra',teacher,input,exchange:{request,reply},verdict};
 await validatePresentationReview(review,input,teacher,modelOracleJudge);
 for(const alter of [r=>r.input.actual.artifacts.report={views:[]},r=>r.exchange.request.messages[0].content='Different rubric',
  r=>r.exchange.reply.raw_calls[0].function.arguments='{}',r=>r.model=teacher,r=>r.verdict.accepted=false]){
  const changed=structuredClone(review);alter(changed);await assert.rejects(validatePresentationReview(changed,input,teacher,modelOracleJudge));
 }
});
