#!/usr/bin/env node
/** Reassess rejected meaning reviews with exact native API semantics; no execution. */
import {createHash} from 'node:crypto';
import {isDeepStrictEqual as same} from 'node:util';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {execFileSync} from 'node:child_process';
import {createRequire} from 'node:module';
import {resolve,join,basename} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {validateWorkflowReview} from './reassess_local_presentation.mjs';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const matches=(ref,value)=>ref?.complete===true&&ref.bytes===Buffer.byteLength(JSON.stringify(value))&&ref.hash===sha(JSON.stringify(value));
const fields={"chat-continuity":[['index','frames','evidence'],['artifact']],"owned-task":[['effects','current','original','delivery','nativeDeliveryError','raceInjected'],[]],"route-authority":[['offer','snapshots','authority','choice','readback'],['artifact']]};
export function reassessNativeReview(packet,pins,task,originalReview,source,kind,checker,entryBinding,bindInput){
 const prior=packet.execution?.diagnostics?.openGoalVerification,native=packet.execution?.diagnostics?.natlangTeacher;
 if(!fields[kind]||!task||task.scenario!==packet.test?.name||task.split!=='train'||packet.result?.passed!==true||!prior||!native||native.executionError||native.captureErrors?.length||pins.teacher!=='openai-codex:gpt-6-luna'||Object.values(pins.repositories??{}).some(row=>row.dirty!==false)||!same(entryBinding(task.entry),native.entry)||native.entry?.arguments?.goal!==prior.goal||task.interaction!=='open-user-goal'||task.goalSha256!==sha(JSON.stringify(prior.goal))||pins.validator_sha256!==sha(source))throw Error('requires exact original clean native-passed training capture');
 if(!prior.caseId?.startsWith(kind+'-')||!prior.checks||!Object.keys(prior.checks).length||Object.values(prior.checks).some(value=>value!==true)||prior.semanticReviewRequired!==true||!same(prior.actualResult,native.report))throw Error('native or required answer failures remain unresolved');
 const roots=(native.calls??[]).filter(call=>native.rootCallIds?.includes(call.callId));
 if(roots.length!==1||roots[0].record.outcome!=='done'||!matches(roots[0].record.output,native.report))throw Error('actual root completion or output differs');
 const oldInput=bindInput(prior,{kind,goal:prior.goal});
 if(originalReview?.schema!=='natlang.live-workflow-review/1'||originalReview.verdict?.accepted!==false||!originalReview.exchange?.request||!originalReview.exchange?.reply||!same(originalReview.input,oldInput))throw Error('requires the exact captured rejected original review');
 const effects=native.calls.flatMap(call=>call.record.effects??[]);
 const witness=(service,method,args,result)=>effects.some(effect=>['agent','crisp'].includes(effect.by)&&effect.service===service&&effect.method===method&&matches(effect.args,args)&&matches(effect.result,result));
 let projected={...prior};
 if(kind==='chat-continuity'){
  for(const guard of ['if(agent.participantId&&agent.participantId!==receipt.participantId)','retainedWorkerRelationship(independentlyRead.events,newWorker.participantId)','oldWorker.participantId===newWorker.participantId'])if(!source.toString().includes(guard))throw Error('original source lacks exact native worker lifecycle guarantees');
  if(!witness('chatContinuityIndex','list',[],prior.index)||!Array.isArray(prior.frames)||prior.frames.length!==2)throw Error('missing original native chat index or observations');
  for(const frame of prior.frames)if(!witness('chatContinuityState','read',[frame.channelId],frame))throw Error('unbound actor native chat readback');
  projected.evidence={...prior.evidence,nativeWorkerIdentity:checker.observedChatWorkerIdentity(prior.index,prior.frames)};
 }else if(kind==='route-authority'){
  for(const guard of ['binding.services.assignedAuthorityRead.preflight(operationId)','snapshots.push(observation)'])if(!source.toString().includes(guard))throw Error('original source lacks exact native preflight observations');
  if(!witness('routeEvidenceRead','discover',[],prior.offer)||!Array.isArray(prior.snapshots)||!prior.snapshots.length||!Array.isArray(prior.offer?.request?.operations)||!Array.isArray(prior.offer.request.ceiling))throw Error('missing original native route offer or preflights');
  const operationIds=prior.offer.request.operations.map(row=>row.id);
  if(new Set(prior.snapshots.map(row=>row.operationId)).size!==prior.snapshots.length)throw Error('duplicate native preflight observation');
  for(const observation of prior.snapshots)if(!operationIds.includes(observation.operationId)||!witness('assignedAuthorityRead','preflight',[observation.operationId],observation))throw Error('unbound original native preflight');
  if(!same(checker.observedRouteAuthority(prior.snapshots,prior.offer.request.ceiling),prior.authority))throw Error('reported authority differs from exact native preflights');
 }else{
  for(const guard of ['owner: view.resolution?.kind === "ref" ? view.resolution.ref : null','lane.lane.maySend && lane.owner'])if(!source.toString().includes(guard))throw Error('original source lacks exact recipient lookup and send authorization guarantees');
  if(!Array.isArray(prior.effects)||!prior.effects.length||!prior.effects.some(row=>row.service==='ownedTaskAttempts'&&row.method==='inspect'))throw Error('missing actual recipient and task observations');
  for(const effect of prior.effects)if(!witness(effect.service,effect.method,effect.args,effect.result))throw Error('task observation differs from original native effect');
 }
 const [observations,artifacts]=fields[kind];
 const review=checker.observedContentReview(projected,prior.outcomeContract,native.report,observations,artifacts);
 if(review.semanticReviewRequired!==true)throw Error('cannot waive required meaning review');
 const guidance={"chat-continuity":checker.chatContinuityReviewGuidance,"owned-task":checker.ownedTaskReviewGuidance,"route-authority":checker.routeAuthorityReviewGuidance}[kind];
 if(typeof guidance!=='string'||!guidance.trim())throw Error('missing source-derived domain guidance');
 review.semanticReviewInput.rubric+=guidance;
 projected={...projected,...review};bindInput(projected,{kind,goal:prior.goal});
 return {prior,native,projected,reviewInput:review.semanticReviewInput};
}
async function main(){
 const args=Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>index%2===0?[...rows,[value.slice(2),all[index+1]]]:rows,[]));
 for(const key of ['host','base','system-testing','packet','pins','catalog','original-review','kind','out'])if(!args[key])throw Error(`missing --${key}`);
 const raw=await readFile(args.packet),packet=JSON.parse(args.packet.endsWith('.gz')?gunzipSync(raw):raw),pins=JSON.parse(await readFile(args.pins,'utf8'));
 const originalReviewBytes=await readFile(args['original-review']),originalReview=JSON.parse(originalReviewBytes);
 const task=(await readFile(args.catalog,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.scenario===packet.test.name);
 const relative='skills/system-testing/open-goal-'+args.kind+'-managed.ts';
 const source=execFileSync('git',['-C',args['system-testing'],'show',`${pins.repositories['system-testing'].git_commit}:${relative}`]);
 const host=resolve(args.host),sst=resolve(args['system-testing']),base=resolve(args.base),out=resolve(args.out);
 const require=createRequire(join(host,'package.json')),{build}=require('esbuild');
 const bundled=await build({stdin:{contents:`export * from ${JSON.stringify(join(sst,'skills/system-testing/native-domain-review.ts'))};export {observedContentReview} from ${JSON.stringify(join(base,'packages/application-outcomes/observed-content-review.ts'))};`,resolveDir:host},bundle:true,platform:'node',format:'esm',write:false,metafile:true,alias:{'@vibestudio/content-addressing':join(host,'packages/content-addressing/src/index.ts')}});
 await mkdir(out,{recursive:false,mode:0o700});const store=async(name,bytes)=>{await writeFile(join(out,name),bytes,{mode:0o600});return {path:name,sha256:sha(bytes)};};
 const bundle=await store('checker.bundle.mjs',bundled.outputFiles[0].contents),checker=await import(pathToFileURL(join(out,bundle.path)));
 const {applicationEntryBinding}=await import(pathToFileURL(join(host,'scripts/export-live-workflow-teacher.mjs')));
 const {observedContentReviewInput}=await import(pathToFileURL(join(host,'scripts/lib/observed-content-admission.mjs')));
 const result=reassessNativeReview(packet,pins,task,originalReview,source,args.kind,checker,applicationEntryBinding,observedContentReviewInput);
 await store('review-input.json',JSON.stringify(result.reviewInput,null,2)+'\n');
 if(!args.review){console.log(JSON.stringify({pending_independent_review:true,review_input:join(out,'review-input.json'),teacher_rerun:false}));return;}
 const {modelOracleJudge}=await import(new URL('../../ts-host/dist/teacher/model-judge.js',import.meta.url));
 await validateWorkflowReview(JSON.parse(await readFile(args.review,'utf8')),result.reviewInput,pins.teacher,modelOracleJudge,args['review-model']);
 const sources=[await store('original-native-checker.ts',source),bundle,await store('assessor.mjs',await readFile(fileURLToPath(import.meta.url))),await store('reassess_local_presentation.mjs',await readFile(new URL('./reassess_local_presentation.mjs',import.meta.url))),await store('model-judge.ts',await readFile(new URL('../../ts-host/src/teacher/model-judge.ts',import.meta.url))),await store('entry-binding.mjs','export '+applicationEntryBinding.toString()+'\n')];
 for(const[index,path]of Object.keys(bundled.metafile.inputs).filter(path=>path!=='<stdin>').entries())sources.push(await store(`checker-source-${index}-${basename(path)}`,await readFile(resolve(path))));
 const repo=fileURLToPath(new URL('../..',import.meta.url));
 const proof={...result.projected,schema:'natlang.live-outcome-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),entry:result.native.entry,actualResult:result.native.report,passed:true,reassessmentCode:{git_commit:execFileSync('git',['-C',repo,'rev-parse','HEAD'],{encoding:'utf8'}).trim(),dirty:Boolean(execFileSync('git',['-C',repo,'status','--porcelain'],{encoding:'utf8'}).trim())},rationale:'Every original native and requested-answer check remains true. Exact original actor effects ground native observations, identity, lifecycle, recipient resolution, permissions, preflight state and retained artifacts. Source-bound API semantics and the actual requested diagnostic/conditional mode guide a new independent bound meaning review. The original rejection, raw packet and all requested obligations remain intact; no native outcome or return value is rewritten.',checker_sources:sources,evidence:[{...await store('original-rejected-review.json',originalReviewBytes),role:'original-rejected-review'},await store('independent-review.json',await readFile(args.review))]};
 const completed=await store('completed-outcome.json',JSON.stringify(proof,null,2)+'\n');await store('assessment.json',JSON.stringify({schema:'natlang.live-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),rationale:proof.rationale,completed_outcome:completed},null,2)+'\n');console.log(JSON.stringify({assessment:join(out,'assessment.json'),teacher_rerun:false}));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
