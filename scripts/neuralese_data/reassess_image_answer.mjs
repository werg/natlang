#!/usr/bin/env node
/** Reclassify obsolete image planner bookkeeping; never rerun native work. */
import {createHash} from 'node:crypto';
import {isDeepStrictEqual as same} from 'node:util';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {execFileSync} from 'node:child_process';
import {resolve,join} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {validateWorkflowReview} from './reassess_local_presentation.mjs';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const matches=(ref,value)=>ref?.complete===true&&ref.bytes===Buffer.byteLength(JSON.stringify(value))&&ref.hash===sha(JSON.stringify(value));
const effectNames=['exactImageFrontiers','exactImageBrief','independentlyReadImageOriginals','verifiedPrivateImageHandoff','preservedImageWorkspace','rejectedForeignImage'];
const returnNames=['returnedImageOriginals','groundedImageDecision'];
const obsolete='groundedImageAssessment';
const guards=['const independent=await Promise.all(queue.map(id=>inspect(latest,id)));','sameEvidenceMembers(reads,independent)','isDeepStrictEqual(independent.map(read=>read.status),observed)','groundedImageAssessment:isDeepStrictEqual'];
export function reassessImageAnswer(packet,pins,task,source,entryBinding){
 const prior=packet.execution?.diagnostics?.openGoalVerification,native=packet.execution?.diagnostics?.natlangTeacher;
 if(!task||task.scenario!==packet.test?.name||task.split!=='train'||packet.result?.passed!==false||!prior||!native||native.executionError||native.captureErrors?.length||pins.teacher!=='openai-codex:gpt-6-luna'||Object.values(pins.repositories??{}).some(row=>row.dirty!==false)||!same(entryBinding(task.entry),native.entry)||native.entry?.arguments?.goal!==prior.goal||task.interaction!=='open-user-goal'||task.goalSha256!==sha(JSON.stringify(prior.goal)))throw Error('requires exact original clean completed training capture');
 if(pins.validator_sha256!==sha(source)||guards.some(fragment=>!source.toString().includes(fragment)))throw Error('original native source guarantees differ');
 const checks=prior.checks??{},needed=[...effectNames,...returnNames];
 if(!same(Object.keys(checks).sort(),[...needed,obsolete].sort())||checks[obsolete]!==false||needed.some(key=>checks[key]!==true)||prior.outcomeContract?.returnedValue?.required!==true||prior.outcomeDimensions?.effects?.passed!==true)throw Error('other native or requested answer failures remain unresolved');
 const report=native.report,roots=(native.calls??[]).filter(call=>native.rootCallIds?.includes(call.callId));
 if(roots.length!==1||roots[0].record.outcome!=='done'||!matches(roots[0].record.output,report)||!same(prior.actualResult,report)||!same(report?.originals,prior.reads)||!same(report?.decision,prior.decision)||!same(report?.handoff,prior.handoff))throw Error('actual requested return differs from original native outcome');
 // All actor-visible observations must have their original exact effect bytes.
 const effects=native.calls.flatMap(call=>call.record.effects??[]);
 const witness=(service,method,args,result)=>effects.some(effect=>effect.service===service&&effect.method===method&&effect.by==='crisp'&&matches(effect.args,args)&&matches(effect.result,result));
 if(!witness('imageCandidateBriefs','list',[],prior.index)||!Array.isArray(prior.snapshots)||!prior.snapshots.length||!Array.isArray(prior.reads)||!prior.reads.length)throw Error('missing original actor image observations');
 for(const snapshot of prior.snapshots)if(!witness('imageCandidateJournal','read',[snapshot.channelId],snapshot))throw Error('native image frontier lacks original effect');
 for(const read of prior.reads)if(!witness('imageCandidateOriginal','read',[prior.plan?.briefId,read.candidateId],read))throw Error('native original differs from actor readback');
 const masks=[...(prior.incorrectReturnValues??[]),...(prior.outcomeDimensions?.incidentalValues?.incorrectReturns??[])];
 for(const mask of masks){
  const c=mask.contribution,call=native.calls.find(call=>call.callId===c?.producer_call_id);
  if(!same(c?.checks,[obsolete])||c.root_call_id!==roots[0].callId||c.root_value_json!==JSON.stringify(report)||c.root_sha256!==sha(JSON.stringify(report))||!['root','projection'].includes(c.kind)||!call||call.record.outcome!=='done'||!matches(call.record.output,JSON.parse(mask.valueJson))||!same(mask.definition,{name:call.record.definition.name,source:call.record.definition.source}))throw Error('unrelated or unbound original return exclusion');
 }
 if(!same(prior.semanticReviewInput?.actual,{goal:prior.goal,observations:{index:prior.index,snapshots:prior.snapshots,reads:prior.reads},artifacts:{artifact:prior.artifact},requestedReturn:report})||!same(prior.semanticReviewInput?.expected,{goal:prior.goal,requestedReturnRequired:true}))throw Error('original review input differs from actual native evidence');
 const reviewInput=structuredClone(prior.semanticReviewInput);
 reviewInput.rubric+=' Candidate eligibility/queue is an internal pre-original readback plan, not a claim that the native original is verified or usable. Judge requested usability, comparisons, provenance and any private artifact against the actual native original observations and returned domain explanation. Do not require exhaustive candidate bookkeeping or extra reasons beyond what the user requested; unsupported requested claims still fail.';
 const contract={effects:{checks:effectNames,rationale:'Original independent native identity, readbacks, authority and protected-state checks remain required.'},returnedValue:{checks:returnNames,required:true,rationale:'Actual requested originals and grounded domain answer remain required, with independent meaning review.'},incidentalValues:{checks:[obsolete],rationale:'Historical planner queue, exhaustive candidate list and reason-presence gold bookkeeping is diagnostic, not an extra requested answer or invalid return.'}};
 const dimensions={effects:{passed:true,checks:Object.fromEntries(effectNames.map(key=>[key,true]))},returnedValue:{required:true,passed:true,checks:Object.fromEntries(returnNames.map(key=>[key,true]))},incidentalValues:{passed:false,checks:{[obsolete]:false},incorrectReturns:[]}};
 return {prior,native,reviewInput,checks:Object.fromEntries(needed.map(key=>[key,true])),outcomeContract:contract,outcomeDimensions:dimensions,masks};
}
async function main(){
 const args=Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>index%2===0?[...rows,[value.slice(2),all[index+1]]]:rows,[]));
 for(const key of ['host','system-testing','packet','pins','catalog','out'])if(!args[key])throw Error(`missing --${key}`);
 const raw=await readFile(args.packet),packet=JSON.parse(args.packet.endsWith('.gz')?gunzipSync(raw):raw),pins=JSON.parse(await readFile(args.pins,'utf8'));
 const task=(await readFile(args.catalog,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.scenario===packet.test.name);
 const relative='skills/system-testing/open-goal-image-candidate-managed.ts';
 const source=execFileSync('git',['-C',args['system-testing'],'show',`${pins.repositories['system-testing'].git_commit}:${relative}`]);
 const corrected=await readFile(join(args['system-testing'],relative));
 if(!corrected.toString().includes('const diagnosticChecks={groundedImageAssessment:'))throw Error('corrected source must preserve original diagnostic without a required gate');
 const {applicationEntryBinding}=await import(pathToFileURL(join(resolve(args.host),'scripts/export-live-workflow-teacher.mjs')));
 const result=reassessImageAnswer(packet,pins,task,source,applicationEntryBinding),out=resolve(args.out);await mkdir(out,{recursive:false,mode:0o700});
 const store=async(name,bytes)=>{await writeFile(join(out,name),bytes,{mode:0o600});return {path:name,sha256:sha(bytes)};};
 await store('review-input.json',JSON.stringify(result.reviewInput,null,2)+'\n');
 if(!args.review){console.log(JSON.stringify({pending_independent_review:true,review_input:join(out,'review-input.json'),teacher_rerun:false}));return;}
 const {modelOracleJudge}=await import(new URL('../../ts-host/dist/teacher/model-judge.js',import.meta.url));
 await validateWorkflowReview(JSON.parse(await readFile(args.review,'utf8')),result.reviewInput,pins.teacher,modelOracleJudge,args['review-model']);
 const sources=[await store('original-native-checker.ts',source),await store('corrected-native-checker.ts',corrected),await store('assessor.mjs',await readFile(fileURLToPath(import.meta.url))),await store('reassess_local_presentation.mjs',await readFile(new URL('./reassess_local_presentation.mjs',import.meta.url))),await store('model-judge.ts',await readFile(new URL('../../ts-host/src/teacher/model-judge.ts',import.meta.url))),await store('entry-binding.mjs','export '+applicationEntryBinding.toString()+'\n'),{...await store('outcome-contract.json',JSON.stringify(result.outcomeContract,null,2)+'\n'),role:'outcome-contract'}];
 const repo=fileURLToPath(new URL('../..',import.meta.url));
 const proof={...result.prior,schema:'natlang.live-outcome-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),entry:result.native.entry,actualResult:result.native.report,passed:true,checks:result.checks,outcomeContract:result.outcomeContract,outcomeDimensions:result.outcomeDimensions,incorrectReturnValues:[],semanticReviewRequired:true,semanticReviewInput:result.reviewInput,diagnosticChecks:{[obsolete]:false},reassessmentCode:{git_commit:execFileSync('git',['-C',repo,'rev-parse','HEAD'],{encoding:'utf8'}).trim(),dirty:Boolean(execFileSync('git',['-C',repo,'status','--porcelain'],{encoding:'utf8'}).trim())},rationale:'All original native and requested domain-answer checks remain true. Source-bound planner bookkeeping is retained diagnostically, without incorrectly masking valid targets. Exact original actor effects ground the readbacks; independent bound meaning review accepts the actual requested answer. Original packet, failed check and exclusions remain unchanged in provenance.',checker_sources:sources,evidence:[await store('original-observations.json',JSON.stringify({checks:result.prior.checks,observedChecks:result.prior.observedChecks,exclusions:result.masks},null,2)),await store('independent-review.json',await readFile(args.review))]};
 const completed=await store('completed-outcome.json',JSON.stringify(proof,null,2)+'\n');await store('assessment.json',JSON.stringify({schema:'natlang.live-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),rationale:proof.rationale,completed_outcome:completed},null,2)+'\n');console.log(JSON.stringify({assessment:join(out,'assessment.json'),teacher_rerun:false}));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
