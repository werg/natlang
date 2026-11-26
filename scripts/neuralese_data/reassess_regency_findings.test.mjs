import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {reassessRegencyFindings} from './reassess_regency_findings.mjs';
const sha=value=>createHash('sha256').update(value).digest('hex');
const ref=value=>({complete:true,bytes:Buffer.byteLength(JSON.stringify(value)),hash:sha(JSON.stringify(value))});
function fixture(){
 const goal='Compare actual repair progress with two forecast months and whether the crossing reopens.',entry={program:'skills/review.ts',callable:'review',resultType:'RegencyResult',arguments:{goal}};
 const source=Buffer.from('if(revision(current)!==scope.revision||!isDeepStrictEqual(observation.basis,current.game.world))\nisDeepStrictEqual(forecasts,nativeVerification)\npreservedRegencyWorkspace:status.mainEventId===before.mainEventId\nawait services.regencySaved.read(other)===null');
 const index={realm:'owned-realm'},accounts={id:'saved',month:1,routes:[{id:'crossing',progress:0,condition:.12}]},frontier={realm:index.realm,revision:'basis',accounts};
 const forecast={realm:index.realm,basis:'basis',months:2,accounts:{...accounts,id:'forecast',month:3}};
 const finding={claim:{kind:'compare',from:'opening',to:2,check:{kind:'route-works-progressed',route:'crossing'},assertCommitted:false},review:{decision:'contradicted',finding:'Native progress remains0→0; the crossing is not reopened.',evidenceIds:['saved','forecast']}};
 const report={forecasts:[forecast],findings:[finding],decision:{action:'report',explanation:'No repair progress and no full reopening.'},advance:null,note:null};
 const nativeChecks=['exactRegencyForecasts','verifiedRegencyAdvance','exactRegencyAssignment','independentlyReadRegencyMemo','exactSavedRegencyFrontiers','verifiedPrivateRegencyNote','preservedRegencyWorkspace','rejectedRegencyRetarget','returnedRegencyForecasts','groundedRegencyAction','returnedRegencyAdvance'];
 const checks=Object.fromEntries(nativeChecks.map(name=>[name,true]));Object.assign(checks,{exactInstalledRegencyFindings:false,exactRegencyFinding0:false,exactRegencyFinding1:false});
 const effect=(service,method,args,result)=>({by:'crisp',service,method,args:ref(args),result:ref(result)});
 const native={entry,report,applicationResult:report,rootCallIds:['root'],calls:[{callId:'root',record:{outcome:'done',output:ref(report),definition:{name:'review',source:'review.ts'},effects:[effect('regencyAssignment','list',[],index),effect('regencySaved','read',[index.realm],frontier),effect('regencySaved','read',[index.realm],frontier),effect('regencyForecast','preview',[index.realm,'basis',2],forecast)]}},{callId:'leaf',record:{outcome:'done',output:ref(finding),definition:{name:'qualifyRegencyClaim',source:'review/qualifyRegencyClaim.nl'}}}]};
 const prior={caseId:'query',goal,index,frontiers:[frontier,structuredClone(frontier)],forecasts:[forecast],decision:report.decision,advance:null,note:null,artifact:null,actualResult:report,checks,outcomeContract:{returnedValue:{required:true}},outcomeDimensions:{effects:{passed:true}},incorrectReturnValues:[],unboundReturnContributions:[{checks:['exactRegencyFinding1'],reason:'no-verified-actual-producer'}]};
 const packet={test:{name:'query'},result:{passed:false},execution:{diagnostics:{openGoalVerification:prior,natlangTeacher:native}}};
 const task={taskId:'task',scenario:'query',split:'train',interaction:'open-user-goal',entry,goalSha256:sha(JSON.stringify(goal))},pins={teacher:'openai-codex:gpt-6-luna',repositories:{'system-testing':{dirty:false}},validator_sha256:sha(source)};
 const world={kind:'regency-briefing',id:'query',goal,plan:{advance:null,localNote:false},action:'report'};
 // Custody seam: the actual archived replay bundles the unchanged installed numeric oracle.
 const checker={exampleReference:({review})=>({decision:review.after.routes[0].progress>review.before.routes[0].progress?'supported':'contradicted',finding:'works-progress',evidenceIds:[review.before.id,review.after.id]}),matchesNativeRegencyFinding:(actual,expected)=>actual.review.decision===expected.review.decision&&JSON.stringify(actual.review.evidenceIds)===JSON.stringify(expected.review.evidenceIds),observedContentReview:(proof,_contract,value)=>({semanticReviewInput:{actual:{goal:proof.goal,observations:proof,requestedReturn:value},expected:{goal:proof.goal,requestedReturnRequired:true},rubric:'Judge requested meaning against actual native accounts.'}})};
 return{packet,task,pins,source,world,checker};
}
const run=s=>reassessRegencyFindings(s.packet,s.pins,s.task,s.source,s.world,s.checker,entry=>entry);
test('preserves false legacy reference observations while retaining genuine query requirements and immutable originals',()=>{
 const s=fixture(),before=structuredClone(s.packet),r=run(s);
 assert.deepEqual(r.outcomeDimensions.incidentalValues.checks,{exactInstalledRegencyFindings:false,exactRegencyFinding0:false,exactRegencyFinding1:false});
 assert(r.outcomeContract.returnedValue.required);assert(Object.values(r.checks).every(Boolean));assert.deepEqual(r.reviewInput.actual.requestedReturn,s.packet.execution.diagnostics.natlangTeacher.report);assert.deepEqual(s.packet,before);
});
test('retires only the exact native-grounded producer exclusions, including legacy uncertain projection for a missing reference slot',()=>{
 const s=fixture(),{natlangTeacher:native,openGoalVerification:prior}=s.packet.execution.diagnostics;
 for(const kind of ['projection','uncertain-projection'])prior.incorrectReturnValues.push({definition:{name:'qualifyRegencyClaim',source:'review/qualifyRegencyClaim.nl'},value:native.report.findings[0],contribution:{kind,checks:['exactRegencyFinding1'],root_call_id:'root',producer_call_id:'leaf',root_value_json:JSON.stringify(native.report),root_sha256:sha(JSON.stringify(native.report))}});
 assert.equal(run(s).masks.length,2);
});
for(const [label,change]of [
 ['held task',s=>s.task.split='test'],['dirty source',s=>s.pins.repositories['system-testing'].dirty=true],['different entry',s=>s.task.entry={...s.task.entry,resultType:'Other'}],['changed checker',s=>s.source=Buffer.from('other')],['requested native advance',s=>s.world.plan.advance=1],['other native failure',s=>s.packet.execution.diagnostics.openGoalVerification.checks.preservedRegencyWorkspace=false],['missing root return',s=>delete s.packet.execution.diagnostics.openGoalVerification.actualResult],['wrong root output',s=>s.packet.execution.diagnostics.natlangTeacher.calls[0].record.output.hash='0'.repeat(64)],['reused native read effect',s=>s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.splice(1,1)],['unbound actual native forecast',s=>s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.at(-1).result.hash='0'.repeat(64)],['incorrect numerical finding',s=>s.checker.exampleReference=()=>({decision:'supported',finding:'works-progress',evidenceIds:['saved','forecast']})],['foreign citation',s=>s.checker.exampleReference=()=>({decision:'contradicted',finding:'works-progress',evidenceIds:['foreign']})],['unrelated unresolved producer',s=>s.packet.execution.diagnostics.openGoalVerification.unboundReturnContributions[0].checks=['another-check']],['unrelated return exclusion',s=>s.packet.execution.diagnostics.openGoalVerification.incorrectReturnValues=[{contribution:{checks:['another-check']}}]],
])test(`refuses ${label}`,()=>{const s=fixture();change(s);assert.throws(()=>run(s));});
