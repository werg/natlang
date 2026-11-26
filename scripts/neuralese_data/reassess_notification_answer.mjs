#!/usr/bin/env node
/** Reassess completed notification answers against original native rows, never rerun work. */
import {createHash} from 'node:crypto';
import {isDeepStrictEqual as same} from 'node:util';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {execFileSync} from 'node:child_process';
import {createRequire} from 'node:module';
import {resolve,join,basename} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {validateWorkflowReview} from './reassess_local_presentation.mjs';
const sha=value=>createHash('sha256').update(value).digest('hex');
const matches=(ref,value)=>ref?.complete===true&&ref.bytes===Buffer.byteLength(JSON.stringify(value))&&ref.hash===sha(JSON.stringify(value));
const nativeChecks=['exactNotificationReadScope','verifiedNotificationRefresh','independentNotificationReadback'];
const obsolete='returnedNotificationResolution';
const keys=['journal','assigned','initial','refresh','current','ack','final','durable'];
const guards=['decision.finding.id === (expected.target?.[0] ?? null)','decision.finding.revision === (expected.target?.[1] ?? null)','isDeepStrictEqual(raw.rows, expectedStore(world, lanes))','expected.action === "acknowledge" ? ack?.status === "acknowledged"'];
export function reassessNotificationAnswer(packet,pins,task,source,world,checker,entryBinding){
 const prior=packet.execution?.diagnostics?.openGoalVerification,native=packet.execution?.diagnostics?.natlangTeacher;
 if(!task||task.scenario!==packet.test?.name||task.split!=='train'||packet.result?.passed!==false||!prior||!native||native.executionError||native.captureErrors?.length||pins.teacher!=='openai-codex:gpt-6-luna'||Object.values(pins.repositories??{}).some(row=>row.dirty!==false)||!same(entryBinding(task.entry),native.entry)||native.entry?.arguments?.goal!==prior.goal||task.interaction!=='open-user-goal'||task.goalSha256!==sha(JSON.stringify(prior.goal)))throw Error('requires exact original clean completed training capture');
 if(pins.validator_sha256!==sha(source)||guards.some(text=>!source.toString().includes(text)))throw Error('original notification guarantees differ');
 if(world?.kind!=='notificationState'||world.id!==prior.caseId||world.goal!==prior.goal||world.expected.action==='acknowledge'||prior.ack!=null||prior.outcomeContract?.returnedValue?.required!==true)throw Error('only original read-only requested answers are covered');
 const checks=prior.checks??{};
 if(!same(Object.keys(checks).sort(),[...nativeChecks,obsolete].sort())||checks[obsolete]!==false||nativeChecks.some(key=>checks[key]!==true)||prior.outcomeDimensions?.effects?.passed!==true)throw Error('other native failures remain unresolved');
 const report=native.report,roots=(native.calls??[]).filter(call=>native.rootCallIds?.includes(call.callId));
 if(roots.length!==1||roots[0].record.outcome!=='done'||!matches(roots[0].record.output,report)||!same(prior.actualResult,report)||typeof report?.reason!=='string'||!report.reason.trim()||!report.finding||!['hold','retain'].includes(report.action))throw Error('actual required original typed answer is missing or differs');
 const effects=native.calls.flatMap(call=>call.record.effects??[]);
 if(effects.some(effect=>effect.service==='notificationState'&&effect.method==='acknowledge'))throw Error('not a read-only native observation');
 const witness=(method,args,result)=>effects.some(effect=>['agent','crisp'].includes(effect.by)&&effect.service==='notificationState'&&effect.method===method&&matches(effect.args,args)&&matches(effect.result,result));
 if(!Array.isArray(prior.journal)||!prior.journal.length||!prior.initial||!prior.refresh||!prior.durable||!prior.assigned)throw Error('original native observations missing');
 for(const row of prior.journal)if(!['discover','read','refresh'].includes(row.method)||!witness(row.method,row.args,row.result))throw Error('journal observation lacks its exact original native effect');
 if(!witness('read',[prior.assigned],prior.initial)||!witness('refresh',[prior.assigned],prior.refresh)||prior.refresh.lane&&!witness('read',[prior.refresh.lane],prior.current))throw Error('native lifecycle differs from actual actor readback');
 const masks=[...(prior.incorrectReturnValues??[]),...(prior.outcomeDimensions?.incidentalValues?.incorrectReturns??[])];
 if(prior.unboundReturnContributions?.length)throw Error('unbound historical return contribution');
 for(const mask of masks){
  const c=mask.contribution,call=native.calls.find(call=>call.callId===c?.producer_call_id);
  if(!same(c?.checks,[obsolete])||c.root_call_id!==roots[0].callId||c.root_value_json!==JSON.stringify(report)||c.root_sha256!==sha(JSON.stringify(report))||!['root','projection'].includes(c.kind)||!same(c.root_path,[])||!same(c.producer_path,[])||!call||call.record.outcome!=='done'||!matches(call.record.output,JSON.parse(mask.valueJson))||!same(mask.definition,{name:call.record.definition.name,source:call.record.definition.source})||!same(JSON.parse(mask.valueJson),report))throw Error('unrelated or unbound original return exclusion');
  if(!Array.isArray(c.lineage)||c.lineage[0]!==roots[0].callId||c.lineage.at(-1)!==call.callId||new Set(c.lineage).size!==c.lineage.length||c.lineage.slice(1).some((id,index)=>native.calls.find(row=>row.callId===id)?.record.parent_call_id!==c.lineage[index]))throw Error('return exclusion has false causal lineage');
 }
 const outcomeContract={effects:{checks:nativeChecks,rationale:'All original independently read native rows, refresh ownership, scope and protected state checks remain required.'},returnedValue:{checks:[],required:true,rationale:'The exact original nonempty typed answer and completed root remain bound. Independent mandatory meaning review certifies the requested explanation, not hidden reference labels or IDs.'},incidentalValues:{checks:[obsolete],rationale:'The old combined reference action/state/id/revision comparison is diagnostic for these read-only branches; it is not an additional user-requested literal answer.'}};
 const selected=Object.fromEntries(nativeChecks.map(key=>[key,true]));
 const outcomeDimensions={effects:{passed:true,checks:selected},returnedValue:{required:true,passed:true,checks:{}},incidentalValues:{passed:false,checks:{[obsolete]:false},incorrectReturns:[]}};
 const review=checker.observedContentReview(prior,outcomeContract,report,keys,[]);
 if(review.semanticReviewRequired!==true||!same(review.semanticReviewInput?.actual?.requestedReturn,report)||!same(review.semanticReviewInput?.expected,{goal:prior.goal,requestedReturnRequired:true})||!same(review.semanticReviewInput.actual.observations,Object.fromEntries(keys.map(key=>[key,prior[key]??null]))))throw Error('meaning review must bind original requested answer and native facts');
 review.semanticReviewInput.rubric+=' This is the exact original typed answer with reason/finding/action fields; do not require the later explanation-field spelling or a private expected finding ID/state/action. Judge the actual requested explanation, scope and completeness, including which event explains a changed view when asked. Distinguish current visible projections from retained SQLite storage; an item absent from a visible projection is not thereby deleted or acknowledged. The captured initial, refreshed and final views and durable rows are authoritative. Do not accept merely because native custody passed; reject false, misleading or incomplete requested explanations.';
 return{prior,native,reviewInput:review.semanticReviewInput,checks:selected,outcomeContract,outcomeDimensions,masks};
}
async function main(){
 const args=Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>index%2===0?[...rows,[value.slice(2),all[index+1]]]:rows,[]));
 for(const key of ['host','base','system-testing','packet','pins','catalog','out'])if(!args[key])throw Error(`missing --${key}`);
 const raw=await readFile(args.packet),packet=JSON.parse(args.packet.endsWith('.gz')?gunzipSync(raw):raw),pins=JSON.parse(await readFile(args.pins,'utf8'));
 const task=(await readFile(args.catalog,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.scenario===packet.test.name);
 const host=resolve(args.host),base=resolve(args.base),sst=resolve(args['system-testing']),out=resolve(args.out);
 const relative='skills/system-testing/open-goal-notification-state-managed.ts';
 const source=execFileSync('git',['-C',sst,'show',`${pins.repositories['system-testing'].git_commit}:${relative}`]);
 const worldSource=execFileSync('git',['-C',sst,'show',`${pins.repositories['system-testing'].git_commit}:skills/system-testing/open-goal-notification-state-worlds.ts`]);
 const corrected=await readFile(join(sst,relative));
 if(!corrected.toString().includes('bindObservedContentReview')||corrected.toString().includes('returnedNotificationResolution:'))throw Error('corrected notification source must bind mandatory meaning review without gold answer equality');
 const {build}=createRequire(join(host,'package.json'))('esbuild');
 await mkdir(out,{recursive:false,mode:0o700});const store=async(name,bytes)=>{await writeFile(join(out,name),bytes,{mode:0o600});return{path:name,sha256:sha(bytes)};};
 const bundled=await build({stdin:{contents:`export {observedContentReview} from ${JSON.stringify(join(base,'packages/application-outcomes/observed-content-review.ts'))};`,resolveDir:host},bundle:true,platform:'node',format:'esm',write:false,metafile:true,alias:{'@vibestudio/content-addressing':join(host,'packages/content-addressing/src/index.ts')}});
 const bundle=await store('checker.bundle.mjs',bundled.outputFiles[0].contents),checker=await import(pathToFileURL(join(out,bundle.path)));
 const worldBundle=await build({stdin:{contents:worldSource.toString(),loader:'ts',sourcefile:'original-notification-worlds.ts',resolveDir:sst},bundle:true,platform:'node',format:'esm',write:false});await store('original-worlds.bundle.mjs',worldBundle.outputFiles[0].contents);
 const {openNotificationStateWorlds}=await import(pathToFileURL(join(out,'original-worlds.bundle.mjs'))),world=openNotificationStateWorlds.find(row=>row.id===packet.execution.diagnostics.openGoalVerification.caseId);
 const {applicationEntryBinding}=await import(pathToFileURL(join(host,'scripts/export-live-workflow-teacher.mjs')));
 const result=reassessNotificationAnswer(packet,pins,task,source,world,checker,applicationEntryBinding);
 await store('review-input.json',JSON.stringify(result.reviewInput,null,2)+'\n');
 if(!args.review){console.log(JSON.stringify({pending_independent_review:true,review_input:join(out,'review-input.json'),teacher_rerun:false}));return;}
 const {modelOracleJudge}=await import(new URL('../../ts-host/dist/teacher/model-judge.js',import.meta.url));
 await validateWorkflowReview(JSON.parse(await readFile(args.review,'utf8')),result.reviewInput,pins.teacher,modelOracleJudge,args['review-model']);
 const sources=[await store('original-native-checker.ts',source),await store('corrected-native-checker.ts',corrected),await store('original-worlds.ts',worldSource),bundle,await store('assessor.mjs',await readFile(fileURLToPath(import.meta.url))),await store('reassess_local_presentation.mjs',await readFile(new URL('./reassess_local_presentation.mjs',import.meta.url))),await store('model-judge.ts',await readFile(new URL('../../ts-host/src/teacher/model-judge.ts',import.meta.url))),await store('entry-binding.mjs','export '+applicationEntryBinding.toString()+'\n'),{...await store('outcome-contract.json',JSON.stringify(result.outcomeContract,null,2)+'\n'),role:'outcome-contract'}];
 for(const[index,path]of Object.keys(bundled.metafile.inputs).filter(path=>path!=='<stdin>').entries())sources.push(await store(`checker-source-${index}-${basename(path)}`,await readFile(resolve(path))));
 const repo=fileURLToPath(new URL('../..',import.meta.url));
 const proof={...result.prior,schema:'natlang.live-outcome-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),entry:result.native.entry,actualResult:result.native.report,passed:true,checks:result.checks,observedChecks:result.prior.checks,outcomeContract:result.outcomeContract,outcomeDimensions:result.outcomeDimensions,incorrectReturnValues:[],unboundReturnContributions:[],semanticReviewRequired:true,semanticReviewInput:result.reviewInput,diagnosticChecks:{[obsolete]:false},reassessmentCode:{git_commit:execFileSync('git',['-C',repo,'rev-parse','HEAD'],{encoding:'utf8'}).trim(),dirty:Boolean(execFileSync('git',['-C',repo,'status','--porcelain'],{encoding:'utf8'}).trim())},rationale:'All original independently read native scope, refresh ownership and retained/protected row checks remain required and true. Exact original actor effects and root/projection output bytes ground the actual requested answer. A genuine independent bound meaning review certifies its requested explanation without private reference enum/ID/revision equality. The original false check and masks remain unchanged in preserved raw observations; no output, effect, causal edge or task identity is rewritten.',checker_sources:sources,evidence:[await store('original-observations.json',JSON.stringify({checks:result.prior.checks,observedChecks:result.prior.observedChecks,exclusions:result.masks,unboundReturnContributions:result.prior.unboundReturnContributions},null,2)),await store('independent-review.json',await readFile(args.review))]};
 const completed=await store('completed-outcome.json',JSON.stringify(proof,null,2)+'\n');await store('assessment.json',JSON.stringify({schema:'natlang.live-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),rationale:proof.rationale,completed_outcome:completed},null,2)+'\n');console.log(JSON.stringify({assessment:join(out,'assessment.json'),teacher_rerun:false}));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
