import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {reassessNotificationAnswer} from './reassess_notification_answer.mjs';
const sha=value=>createHash('sha256').update(value).digest('hex');
const ref=value=>({complete:true,bytes:Buffer.byteLength(JSON.stringify(value)),hash:sha(JSON.stringify(value))});
function fixture(){
 const goal='Check whether the original read entry survives and retain the current note.',entry={program:'reviewNotificationState.ts',callable:'reviewNotificationState',arguments:{goal}};
 const source=Buffer.from('decision.finding.id === (expected.target?.[0] ?? null) decision.finding.revision === (expected.target?.[1] ?? null) isDeepStrictEqual(raw.rows, expectedStore(world, lanes)) expected.action === "acknowledge" ? ack?.status === "acknowledged"');
 const assigned={id:'owned',label:'Registration',generation:1},initial={lane:assigned,retained:[{id:'old'}]},refresh={lane:assigned,producerReceipt:'removed'},current={lane:assigned,retained:[{id:'access'}]};
 const report={action:'hold',finding:{state:'missing',id:'old',revision:1},reason:'The original was removed; the accessibility note remains.'};
 const journal=[{method:'read',args:[assigned],result:initial},{method:'refresh',args:[assigned],result:refresh},{method:'read',args:[assigned],result:current}];
 const checks={exactNotificationReadScope:true,verifiedNotificationRefresh:true,independentNotificationReadback:true,returnedNotificationResolution:false};
 const masks=[['root','root'],['projection','leaf']].map(([kind,id])=>({definition:{name:id,source:id+'.ts'},valueJson:JSON.stringify(report),contribution:{kind,checks:['returnedNotificationResolution'],root_call_id:'root',producer_call_id:id,root_value_json:JSON.stringify(report),root_sha256:sha(JSON.stringify(report)),root_path:[],producer_path:[],lineage:id==='root'?['root']:['root','leaf']}}));
 const prior={caseId:'notification-state-removed',goal,checks,journal,assigned,initial,refresh,current,ack:null,final:current,durable:{rows:[{id:'access'}]},actualResult:report,outcomeContract:{returnedValue:{required:true}},outcomeDimensions:{effects:{passed:true},incidentalValues:{incorrectReturns:masks}},incorrectReturnValues:masks,unboundReturnContributions:[]};
 const calls=['root','leaf'].map(id=>({callId:id,record:{outcome:'done',parent_call_id:id==='root'?null:'root',definition:{name:id,source:id+'.ts'},output:ref(report),effects:id==='leaf'?journal.map(row=>({service:'notificationState',method:row.method,args:ref(row.args),result:ref(row.result),by:'agent'})):[]}}));
 const native={entry,report,rootCallIds:['root'],calls};const packet={test:{name:'open-goal-notification-removed'},result:{passed:false},execution:{diagnostics:{openGoalVerification:prior,natlangTeacher:native}}};
 const pins={teacher:'openai-codex:gpt-6-luna',repositories:{base:{dirty:false}},validator_sha256:sha(source)};
 const task={scenario:packet.test.name,split:'train',interaction:'open-user-goal',entry,goalSha256:sha(JSON.stringify(goal))};
 const world={kind:'notificationState',id:prior.caseId,goal,expected:{action:'hold'}};
 const checker={observedContentReview:(proof,contract,value,keys)=>({semanticReviewRequired:true,semanticReviewInput:{actual:{goal:proof.goal,requestedReturn:value,observations:Object.fromEntries(keys.map(key=>[key,proof[key]??null])),artifacts:{}},expected:{goal:proof.goal,requestedReturnRequired:contract.returnedValue.required},rubric:'Judge actual meaning.'}})};
 return{packet,pins,task,source,world,checker};
}
const run=s=>reassessNotificationAnswer(s.packet,s.pins,s.task,s.source,s.world,s.checker,entry=>entry);
test('preserves original false gate and masks while binding unchanged typed answer/native observations for mandatory review',()=>{const s=fixture(),original=structuredClone(s.packet),r=run(s);assert.deepEqual(s.packet,original);assert.deepEqual(r.checks,{exactNotificationReadScope:true,verifiedNotificationRefresh:true,independentNotificationReadback:true});assert.deepEqual(r.outcomeDimensions.incidentalValues.checks,{returnedNotificationResolution:false});assert.equal(r.outcomeContract.returnedValue.required,true);assert.deepEqual(r.reviewInput.actual.requestedReturn,original.execution.diagnostics.natlangTeacher.report);assert.equal(r.masks.length,4);assert.match(r.reviewInput.rubric,/Distinguish current visible projections/);});
for(const[label,change]of[
 ['native state failure',s=>{s.packet.execution.diagnostics.openGoalVerification.checks.independentNotificationReadback=false;}],
 ['empty required original answer',s=>{const n=s.packet.execution.diagnostics.natlangTeacher;n.report.reason='';n.calls[0].record.output=ref(n.report);}],
 ['changed original output',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.output.hash='0'.repeat(64);}],
 ['held split',s=>{s.task.split='test';}],
 ['dirty source',s=>{s.pins.repositories.base.dirty=true;}],
 ['changed validator source',s=>{s.source=Buffer.from('other');}],
 ['missing exact native effect',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[1].record.effects.pop();}],
 ['forged actor current view',s=>{s.packet.execution.diagnostics.openGoalVerification.current={retained:[]};}],
 ['requested acknowledgment branch',s=>{s.world.expected.action='acknowledge';}],
 ['actual native acknowledgment',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[1].record.effects.push({service:'notificationState',method:'acknowledge'});}],
 ['unbound return contribution',s=>{s.packet.execution.diagnostics.openGoalVerification.unboundReturnContributions=[{}];}],
 ['unrelated mask check',s=>{s.packet.execution.diagnostics.openGoalVerification.incorrectReturnValues[0].contribution.checks=['other'];}],
 ['forged mask definition',s=>{s.packet.execution.diagnostics.openGoalVerification.incorrectReturnValues[0].definition.source='other';}],
 ['false causal projection ancestry',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[1].record.parent_call_id='other';}],
 ['review waiver',s=>{s.checker.observedContentReview=()=>({semanticReviewRequired:false});}],
])test(`rejects ${label}`,()=>{const s=fixture();change(s);assert.throws(()=>run(s));});
