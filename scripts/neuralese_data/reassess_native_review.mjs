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
const fields={"installed-package":[["effects","assignment","observed","plan","repair","authorization"],[]],"system-release":[["assignment","runs","cases","effects","handoffProtocol"],["handoff","handoffText"]],"chat-continuity":[['index','frames','evidence'],['artifact']],"owned-task":[['effects','current','original','delivery','nativeDeliveryError','raceInjected'],[]],"route-authority":[['offer','snapshots','authority','choice','readback'],['artifact']],"regency-briefing":[['index','frontiers','forecasts','witnesses','advance','nativeFacts'],['artifact']],"grimoire-scene":[['docket','view','current','scope','gate','readback','journal','nativeFacts'],['body']]};
export function reassessNativeReview(packet,pins,task,originalReview,source,kind,checker,entryBinding,bindInput,originalBindInput=bindInput){
 const prior=packet.execution?.diagnostics?.openGoalVerification,native=packet.execution?.diagnostics?.natlangTeacher;
 if(!fields[kind]||!task||task.scenario!==packet.test?.name||task.split!=='train'||packet.result?.passed!==true||!prior||!native||native.executionError||native.captureErrors?.length||pins.teacher!=='openai-codex:gpt-6-luna'||Object.values(pins.repositories??{}).some(row=>row.dirty!==false)||!same(entryBinding(task.entry),native.entry)||native.entry?.arguments?.goal!==prior.goal||task.interaction!=='open-user-goal'||task.goalSha256!==sha(JSON.stringify(prior.goal))||pins.validator_sha256!==sha(source))throw Error('requires exact original clean native-passed training capture');
 if(!prior.caseId?.startsWith(kind+'-')||!prior.checks||!Object.keys(prior.checks).length||Object.values(prior.checks).some(value=>value!==true)||prior.semanticReviewRequired!==true||!same(prior.actualResult,native.report))throw Error('native or required answer failures remain unresolved');
 const roots=(native.calls??[]).filter(call=>native.rootCallIds?.includes(call.callId));
 if(roots.length!==1||roots[0].record.outcome!=='done'||!matches(roots[0].record.output,native.report))throw Error('actual root completion or output differs');
 const worldKind=kind==='grimoire-scene'?'grimoireScene':kind;
 const oldInput=originalBindInput(prior,{kind:worldKind,goal:prior.goal});
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
 }else if(kind==='regency-briefing'){
  for(const guard of ['Regency forecast basis changed during observation','verifiedRegencyAdvance:','verifiedPrivateRegencyNote:','preservedRegencyWorkspace:'])if(!source.toString().includes(guard))throw Error('original source lacks native Regency authority and persistence checks');
  if(!witness('regencyAssignment','list',[],prior.index)||!Array.isArray(prior.frontiers)||!prior.frontiers.length||!Array.isArray(prior.forecasts))throw Error('missing actual native council observations');
  for(const scope of prior.frontiers)if(scope.realm!==prior.index.realm||!witness('regencySaved','read',[prior.index.realm],scope))throw Error('unbound saved council observation');
  for(const row of prior.forecasts)if(row.realm!==prior.index.realm||!prior.frontiers.some(scope=>scope.revision===row.basis)||!witness('regencyForecast','preview',[row.realm,row.basis,row.months],row))throw Error('unbound council forecast');
  let current=prior.frontiers.at(-1);
  if(prior.advance){
   const a=prior.advance;
   if(a.realm!==prior.index.realm||!prior.frontiers.some(scope=>scope.revision===a.basis)||!witness('regencyAdvance','commit',[a.realm,a.basis,a.months],a))throw Error('unbound actual council advance');
   current=a.committed;
  }
  if(prior.note){
   if(typeof prior.artifact!=='string'||prior.note.digest!==sha(prior.artifact)||prior.note.realm!==prior.index.realm||prior.note.revision!==current.revision||!effects.some(effect=>effect.service==='regencyNote'&&effect.method==='stage'&&['agent','crisp'].includes(effect.by)&&matches(effect.result,prior.note)))throw Error('unbound retained council note or bytes');
   const body=JSON.parse(prior.artifact);
   if(body.realm!==prior.index.realm||!same(body.advance,prior.advance)||!prior.frontiers.some(scope=>same(scope,body.opening))||!prior.frontiers.some(scope=>same(scope,body.current)))throw Error('retained note has foreign observations or committed ledger');
  }else if(prior.artifact!==null)throw Error('council artifact has no retained receipt');
  projected.nativeFacts=checker.observedRegencyEvidence(current,prior.note,prior.artifact);
 }else if(kind==='installed-package'){
  for(const guard of ['scope.workingRevision === workingRevision','scope.workingDigest === local.digest','decisionFor(current.observation) !== "restore-approved-field"','verifiedLocalManifestEffect:','preservedProtectedAndNeighbors:'])if(!source.toString().includes(guard))throw Error('original source lacks exact pre-action authority, CAS and protected-state checks');
  if(!prior.assignment||!prior.observed||!prior.repair||prior.repair.status!=='restored'||prior.assignment.operationId!==prior.observed.operationId||prior.assignment.scope?.grant!=='restore'||prior.observed.scopeMatches!==true||!witness('installedPackageRead','discover',[],prior.assignment)||!witness('installedPackageRead','inspect',[prior.assignment.operationId],prior.observed)||!witness('installedPackageChange','restore',[prior.assignment.operationId,prior.observed.workingRevision,prior.observed.workingDigest],prior.repair))throw Error('missing exact originally authorized restore witnesses');
  const scope=prior.assignment.scope,after=prior.repair.observation;
  if(scope.source!==prior.observed.source||scope.field!==prior.assignment.field||scope.key!==prior.assignment.key||scope.workingRevision!==prior.observed.workingRevision||scope.workingDigest!==prior.observed.workingDigest||scope.baselineDigest!==prior.observed.baselineDigest||after.workingDigest===prior.observed.workingDigest||after.baselineDigest!==prior.observed.baselineDigest||after.source!==prior.observed.source||after.targetFieldMatches!==true)throw Error('restore scope or post-state identity differs');
  const world=checker.openInstalledPackageWorlds?.find(row=>row.id===prior.caseId&&row.goal===prior.goal);
  if(!world)throw Error('missing exact authored package return obligation');
  if(!prior.plan||!after.suiteRequest||!witness('testSuitePlanRead','resolve',[after.suiteRequest.operationId],prior.plan)||!same(native.report.verification?.plan,prior.plan)||typeof native.report.finding?.reason!=='string'||!native.report.finding.reason.trim())throw Error('missing actual requested suite or finding');
  projected.outcomeContract=checker.installedPackageGoalContract(world);
  projected.authorization=checker.installedPackageAuthorization({operationId:prior.assignment.operationId,scope,before:prior.observed,action:prior.repair,after});
 }else if(kind==='system-release'){
  for(const guard of ['contextId !== ports.contextId','directory: () => fs.mkdtemp("system-release-evidence")','decode.decode(raw) === expectedBytes','receipt.sha256 === hash(expectedBytes)','sameEvidenceMembers(cases, reads)'])if(!source.toString().includes(guard))throw Error('original source lacks context-scoped exact artifact and native evidence checks');
  if(!prior.assignment||prior.assignment.contextId!==prior.contextId||!Array.isArray(prior.cases)||!witness('systemReleaseEvidence','discover',[],prior.assignment)||!witness('systemReleaseEvidence','compare',[prior.assignment.id],prior.runs))throw Error('unbound original release assignment or run pair');
  const receipt=prior.handoff,text=prior.handoffText;
  if(!receipt||typeof text!=='string'||Buffer.byteLength(text)!==receipt.bytes||sha(text)!==receipt.sha256)throw Error('missing actual retained release artifact bytes');
  const body=JSON.parse(text);
  const stages=(prior.effects??[]).filter(row=>row.service==='systemReleaseHandoff'&&row.method==='stage'&&same(row.result,receipt));
  if(stages.length!==1)throw Error('missing unique original native stage');
  const stage=stages[0],cases=stage.args?.[3];
  if(body.contextId!==prior.contextId||body.protocol!=='vibestudio-local-system-release-review.v1'||!same(body.cases,prior.cases)||stage.args.length!==4||stage.args[0]!==prior.assignment.id||stage.args[1]!==body.runId||stage.args[2]!==body.disposition||!Array.isArray(cases)||cases.length!==prior.cases.length||new Set(cases.map(row=>row.name)).size!==cases.length||cases.some(row=>!prior.cases.some(actual=>same(row,actual)))||!witness('systemReleaseHandoff','stage',stage.args,receipt))throw Error('release artifact differs from actual native stage');
  for(const row of prior.cases)if(!witness('systemReleaseEvidence','inspectCase',[prior.assignment.id,body.runId,row.name],row))throw Error('unbound original retained case');
  const world=checker.openSystemReleaseWorlds?.find(row=>row.id===prior.caseId&&row.goal===prior.goal);
  if(!world)throw Error('missing exact authored release return obligation');
  if(world.requestedReturnRequired&&(typeof native.report.assessment?.reason!=='string'||!native.report.assessment.reason.trim()))throw Error('missing required actual release finding');
  projected.outcomeContract=checker.systemReleaseGoalContract(world);
  projected.handoffProtocol=checker.systemReleaseHandoffProtocol({contextId:prior.contextId,verifiedPrivateHandoff:prior.observedChecks?.verifiedPrivateHandoff===true||prior.checks.verifiedPrivateHandoff===true,receipt,text});
  if(!projected.handoffProtocol)throw Error('cannot establish native local artifact protocol');
 }else if(kind==='grimoire-scene'){
  for(const guard of ['boundOwnedScene:','keptCurrentAnchor:','independentlyReadPrivateDraft:','preservedNativeWorld:'])if(!source.toString().includes(guard))throw Error('original source lacks native caretaker authority and persistence checks');
  if(!Array.isArray(prior.journal)||!prior.journal.length||prior.journal.some(row=>!witness(row.service,row.method,row.args,row.result)))throw Error('unbound native caretaker journal');
  if(!witness('grimoireScene','readScope',[prior.docket.id],prior.scope)||!witness('grimoireScene','view',[prior.docket.id],prior.view)||!prior.current||prior.scope.ownerId!==prior.current.ownerId||prior.receipt!==null||prior.body!==null)throw Error('requires actual scope-bound caretaker hold without fabricated artifacts');
  if(prior.gate&&!prior.journal.some(row=>row.service==='grimoireScene'&&row.method==='confirm'&&same(row.result,prior.gate)))throw Error('unbound caretaker confirmation');
  projected.nativeFacts=checker.observedGrimoireAuthority(prior.scope,prior.view,prior.current,prior.gate);
 }else{
  for(const guard of ['owner: view.resolution?.kind === "ref" ? view.resolution.ref : null','lane.lane.maySend && lane.owner'])if(!source.toString().includes(guard))throw Error('original source lacks exact recipient lookup and send authorization guarantees');
  if(!Array.isArray(prior.effects)||!prior.effects.length||!prior.effects.some(row=>row.service==='ownedTaskAttempts'&&row.method==='inspect'))throw Error('missing actual recipient and task observations');
  for(const effect of prior.effects)if(!witness(effect.service,effect.method,effect.args,effect.result))throw Error('task observation differs from original native effect');
 }
 if(['installed-package','system-release'].includes(kind)){
  projected=checker.assessSemanticOutcome(projected,projected.outcomeContract);
  if(projected.passed!==true)throw Error('new requested answer or original native requirement remains unsatisfied');
 }
 const [observations,artifacts]=fields[kind];
 const review=checker.observedContentReview(projected,projected.outcomeContract,native.report,observations,artifacts);
 if(review.semanticReviewRequired!==true)throw Error('cannot waive required meaning review');
 const guidance={"chat-continuity":checker.chatContinuityReviewGuidance,"owned-task":checker.ownedTaskReviewGuidance,"route-authority":checker.routeAuthorityReviewGuidance,"regency-briefing":checker.regencyReviewGuidance,"grimoire-scene":checker.grimoireReviewGuidance}[kind];
 if(!['installed-package','system-release'].includes(kind)){if(typeof guidance!=='string'||!guidance.trim())throw Error('missing source-derived domain guidance');review.semanticReviewInput.rubric+=guidance;}
 projected={...projected,...review};bindInput(projected,{kind:worldKind,goal:prior.goal});
 return {prior,native,projected,reviewInput:review.semanticReviewInput};
}
async function main(){
 const args=Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>index%2===0?[...rows,[value.slice(2),all[index+1]]]:rows,[]));
 for(const key of ['host','base','system-testing','packet','pins','catalog','original-review','kind','out'])if(!args[key])throw Error(`missing --${key}`);
 const raw=await readFile(args.packet),packet=JSON.parse(args.packet.endsWith('.gz')?gunzipSync(raw):raw),pins=JSON.parse(await readFile(args.pins,'utf8'));
 const originalReviewBytes=await readFile(args['original-review']),originalReview=JSON.parse(originalReviewBytes);
 const task=(await readFile(args.catalog,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.scenario===packet.test.name);
 const relative='skills/system-testing/open-goal-'+(args.kind==='regency-briefing'?'regency':args.kind)+'-managed.ts';
 const source=execFileSync('git',['-C',args['system-testing'],'show',`${pins.repositories['system-testing'].git_commit}:${relative}`]);
 const host=resolve(args.host),sst=resolve(args['system-testing']),base=resolve(args.base),out=resolve(args.out);
 const require=createRequire(join(host,'package.json')),{build}=require('esbuild');
 const premise= ['installed-package','system-release'].includes(args.kind);
 const exports=[`export * from ${JSON.stringify(join(sst,'skills/system-testing/native-domain-review.ts'))};export {observedContentReview} from ${JSON.stringify(join(base,'packages/application-outcomes/observed-content-review.ts'))};`];
 if(premise)exports.push(`export {assessSemanticOutcome} from ${JSON.stringify(join(base,'packages/application-outcomes/semantic-outcome-contract.ts'))};export {installedPackageGoalContract,systemReleaseGoalContract} from ${JSON.stringify(join(sst,'skills/system-testing/open-goal-review-contracts.ts'))};`);
 if(args.kind==='installed-package')exports.push(`export {installedPackageAuthorization} from ${JSON.stringify(join(sst,'skills/system-testing/installed-package-authorization-review.ts'))};export {openInstalledPackageWorlds} from ${JSON.stringify(join(sst,'skills/system-testing/open-goal-installed-package-worlds.ts'))};`);
 if(args.kind==='system-release')exports.push(`export {systemReleaseHandoffProtocol} from ${JSON.stringify(join(sst,'skills/system-testing/system-release-handoff-protocol.ts'))};export {openSystemReleaseWorlds} from ${JSON.stringify(join(sst,'skills/system-testing/open-goal-system-release-worlds.ts'))};`);
 const bundled=await build({stdin:{contents:exports.join('\n'),resolveDir:host},bundle:true,platform:'node',format:'esm',write:false,metafile:true,alias:{'@vibestudio/content-addressing':join(host,'packages/content-addressing/src/index.ts'),'@workspace-skills/scratch-handoff-bindings':join(base,'skills/scratch-handoff-bindings/index.ts')}});
 await mkdir(out,{recursive:false,mode:0o700});const store=async(name,bytes)=>{await writeFile(join(out,name),bytes,{mode:0o600});return {path:name,sha256:sha(bytes)};};
 const bundle=await store('checker.bundle.mjs',bundled.outputFiles[0].contents),checker=await import(pathToFileURL(join(out,bundle.path)));
 const {applicationEntryBinding}=await import(pathToFileURL(join(host,'scripts/export-live-workflow-teacher.mjs')));
 const {observedContentReviewInput}=await import(pathToFileURL(join(host,'scripts/lib/observed-content-admission.mjs')));
 const {observedContentReviewInput:originalBindInput}=await import(pathToFileURL(join(resolve(args['original-host']??host),'scripts/lib/observed-content-admission.mjs')));
 const result=reassessNativeReview(packet,pins,task,originalReview,source,args.kind,checker,applicationEntryBinding,observedContentReviewInput,originalBindInput);
 await store('review-input.json',JSON.stringify(result.reviewInput,null,2)+'\n');
 if(!args.review){console.log(JSON.stringify({pending_independent_review:true,review_input:join(out,'review-input.json'),teacher_rerun:false}));return;}
 const {modelOracleJudge}=await import(new URL('../../ts-host/dist/teacher/model-judge.js',import.meta.url));
 await validateWorkflowReview(JSON.parse(await readFile(args.review,'utf8')),result.reviewInput,pins.teacher,modelOracleJudge,args['review-model']);
 const sources=[await store('original-native-checker.ts',source),bundle,await store('assessor.mjs',await readFile(fileURLToPath(import.meta.url))),await store('reassess_local_presentation.mjs',await readFile(new URL('./reassess_local_presentation.mjs',import.meta.url))),await store('model-judge.ts',await readFile(new URL('../../ts-host/src/teacher/model-judge.ts',import.meta.url))),await store('entry-binding.mjs','export '+applicationEntryBinding.toString()+'\n')];
 for(const[index,path]of Object.keys(bundled.metafile.inputs).filter(path=>path!=='<stdin>').entries())sources.push(await store(`checker-source-${index}-${basename(path)}`,await readFile(resolve(path))));
 if(['installed-package','system-release'].includes(args.kind))sources.push({...await store('outcome-contract.json',JSON.stringify(result.projected.outcomeContract,null,2)+'\n'),role:'outcome-contract'});
 const repo=fileURLToPath(new URL('../..',import.meta.url));
 const proof={...result.projected,schema:'natlang.live-outcome-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),entry:result.native.entry,actualResult:result.native.report,passed:true,reassessmentCode:{git_commit:execFileSync('git',['-C',repo,'rev-parse','HEAD'],{encoding:'utf8'}).trim(),dirty:Boolean(execFileSync('git',['-C',repo,'status','--porcelain'],{encoding:'utf8'}).trim())},rationale:'Every original native and requested-answer check remains true. Exact original actor effects ground native observations, identity, lifecycle, recipient resolution, permissions, preflight state and retained artifacts. Source-bound API semantics and the actual requested diagnostic/conditional mode guide a new independent bound meaning review. The original rejection, raw packet and all requested obligations remain intact; no native outcome or return value is rewritten.',checker_sources:sources,evidence:[{...await store('original-rejected-review.json',originalReviewBytes),role:'original-rejected-review'},await store('independent-review.json',await readFile(args.review))]};
 const completed=await store('completed-outcome.json',JSON.stringify(proof,null,2)+'\n');await store('assessment.json',JSON.stringify({schema:'natlang.live-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),rationale:proof.rationale,completed_outcome:completed},null,2)+'\n');console.log(JSON.stringify({assessment:join(out,'assessment.json'),teacher_rerun:false}));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
