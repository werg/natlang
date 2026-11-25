import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {reassessAdventureFindings} from './reassess_adventure_findings.mjs';
const sha=value=>createHash('sha256').update(value).digest('hex');
const ref=value=>({complete:true,bytes:Buffer.byteLength(JSON.stringify(value)),hash:sha(JSON.stringify(value))});
function fixture(){
 const goal='Compare Front Steps and Counter without changing them.',entry={program:'skills/journey.nl',callable:'journey',arguments:{goal},resultType:'JourneyOutcome'};
 const source=Buffer.from('const native = await Promise.all(saves.map(save => save.native.read()));\npreservedNativeAdventureScene: !final.seated\nisDeepStrictEqual(nativeSummary(native[index]), nativeSummary(save.before))\nverifiedAdventureJournal: isDeepStrictEqual(final.world.journal, expectedJournal) && retiredInputIsIdempotent');
 const assigned=[{id:'counter-id',label:'Counter'},{id:'front-id',label:'Front Steps'}];
 const views=assigned.map((save,index)=>({save,observation:'snapshot-'+index,place:{id:index?'landing':'customs'},tick:0,scene:{signature:'scene-'+index,artwork:'missing'},pending:null}));
 const report={save:assigned[1].id,compared:[assigned[0].id],observation:views[1].observation,disposition:'preserved',evidence:[{topic:'same-time',subject:'Front Steps and Counter',verdict:'confirmed'}],pending:null};
 const checks=Object.fromEntries(['discoveredAssignedAdventureSave','readComparedAdventureSaves','verifiedExactAdventureTurn','verifiedAdventureJournal','preservedNativeAdventureScene','isolatedSiblingAdventureSaves','matchedAdventureHandoff'].map(key=>[key,true]));checks.attestedAdventurePlayerEvidence=false;
 const prior={caseId:'query',goal,entry,actualResult:report,checks,selected:assigned[1].id,final:views[1],catalogs:[assigned],readbacks:views,actions:[],outcomeContract:{returnedValue:{required:true}},outcomeDimensions:{effects:{passed:true},returnedValue:{required:true}}};
 const native={entry,report,rootCallIds:['root'],calls:[{callId:'root',record:{outcome:'done',output:ref(report)}}]};
 const packet={test:{name:'open-goal-query'},result:{passed:false},execution:{diagnostics:{openGoalVerification:prior,natlangTeacher:native}}};
 const pins={teacher:'openai-codex:gpt-6-luna',repositories:{examples:{dirty:false}},validator_sha256:sha(source)};
 const world={id:'query',kind:'adventure-journey',goal,saves:assigned.map(({label})=>({label})),expected:{save:1,compared:[0],turn:'preserved',evidence:[{topic:'same-time',subject:0,verdict:'confirmed'}]}};
 const task={taskId:'task',scenario:packet.test.name,split:'train',interaction:'open-user-goal',entry,goalSha256:sha(JSON.stringify(goal))};
 // This seam tests archive custody; actual domain semantics are replayed against
 // the source-bundled Examples checker in the real archived-packet validation.
 const checker={validJourneyEvidence:value=>Array.isArray(value)&&value.length>0,verifyJourneyFact:(_fact,selected,views)=>selected.tick===views[0].tick?'confirmed':'refuted',createJourneyMeaningReview:(goal,observations,requestedReturn)=>({actual:{goal,observations,requestedReturn},expected:{goal,requestedReturnRequired:true},rubric:'Judge actual native findings.'})};
 return {packet,pins,task,source,world,checker};
}
const run=s=>reassessAdventureFindings(s.packet,s.pins,s.task,s.source,s.world,s.checker,entry=>entry);
test('corrects only the bound legacy subject rejection and preserves old API and actual claims',()=>{const s=fixture(),old=structuredClone(s.packet);const r=run(s);assert.equal(r.checks.attestedAdventurePlayerEvidence,true);assert.deepEqual(Object.keys(r.checks).sort(),Object.keys(old.execution.diagnostics.openGoalVerification.checks).sort());assert.deepEqual(r.reviewInput.actual.requestedReturn,old.execution.diagnostics.natlangTeacher.report);assert.equal(r.witness.nativeInitialSnapshotCaptured,false);assert.equal(r.reviewInput.actual.observations.kind,'archived-player-readbacks');assert.deepEqual(s.packet,old);});
for(const [label,change]of [
 ['changed source',s=>{s.source=Buffer.from('different source');}],
 ['other failed native check',s=>{s.packet.execution.diagnostics.openGoalVerification.checks.isolatedSiblingAdventureSaves=false;}],
 ['missing native guarantee',s=>{delete s.packet.execution.diagnostics.openGoalVerification.checks.verifiedAdventureJournal;}],
 ['unrequested mutation goal',s=>{s.world.expected.turn='replaced';}],
 ['uncompared scope',s=>{s.world.expected.compared=[];}],
 ['ambiguous labels',s=>{s.world.saves[0].label=s.world.saves[1].label;}],
 ['different original entry',s=>{s.task.entry={...s.task.entry,resultType:'JourneyFindings'};}],
 ['missing required return',s=>{s.packet.execution.diagnostics.natlangTeacher.report={};}],
 ['unfinished actual root',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.outcome='error';}],
 ['unbound root bytes',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.output.hash='0'.repeat(64);}],
 ['missing comparison read',s=>{s.packet.execution.diagnostics.openGoalVerification.readbacks.shift();}],
 ['changed native snapshot',s=>{const v=structuredClone(s.packet.execution.diagnostics.openGoalVerification.readbacks[0]);v.tick=3;s.packet.execution.diagnostics.openGoalVerification.readbacks.push(v);}],
 ['physical facts disagree',s=>{s.packet.execution.diagnostics.openGoalVerification.readbacks[0].tick=2;}],
 ['actual mutation',s=>{s.packet.execution.diagnostics.openGoalVerification.actions.push({args:['front-id'],result:{status:'withdrawn'}});}],
 ['other return exclusion',s=>{s.packet.execution.diagnostics.openGoalVerification.incorrectReturnValues=[{contribution:{checks:['another failure']}}];}],
 ['held task',s=>{s.task.split='test';}],
 ['different teacher',s=>{s.pins.teacher='openai-codex:gpt-6-sol';}],
 ['unclean source',s=>{s.pins.repositories.examples.dirty=true;}],
])test(`refuses ${label}`,()=>{const s=fixture();change(s);assert.throws(()=>run(s));});
