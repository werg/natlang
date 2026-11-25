#!/usr/bin/env node
/** Review original Adventure query findings; never executes a teacher or retired workspace. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {gunzipSync} from 'node:zlib';
import {resolve,join,basename} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {isDeepStrictEqual as same} from 'node:util';
import {execFileSync} from 'node:child_process';
import {validateWorkflowReview} from './reassess_local_presentation.mjs';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const matches=(ref,value)=>ref?.complete===true&&ref.bytes===Buffer.byteLength(JSON.stringify(value))&&ref.hash===sha(JSON.stringify(value));
const requiredNativeChecks=['discoveredAssignedAdventureSave','readComparedAdventureSaves','verifiedExactAdventureTurn','verifiedAdventureJournal','preservedNativeAdventureScene','isolatedSiblingAdventureSaves','matchedAdventureHandoff'];
const nativeGuards=['const native = await Promise.all(saves.map(save => save.native.read()));','preservedNativeAdventureScene: !final.seated','isDeepStrictEqual(nativeSummary(native[index]), nativeSummary(save.before))','verifiedAdventureJournal: isDeepStrictEqual(final.world.journal, expectedJournal) && retiredInputIsIdempotent'];

/** Reuse actual captured player projections; missing native initial snapshots stay absent. */
export function reassessAdventureFindings(packet,pins,task,source,world,checker,entryBinding){
 const prior=packet.execution?.diagnostics?.openGoalVerification,native=packet.execution?.diagnostics?.natlangTeacher;
 if(!task||task.scenario!==packet.test?.name||task.split!=='train'||packet.result?.passed!==false||
  !prior||!native||native.executionError||native.captureErrors?.length||pins.teacher!=='openai-codex:gpt-6-luna'||
  Object.values(pins.repositories??{}).some(row=>row.dirty!==false)||
  !same(entryBinding(task.entry),native.entry)||native.entry?.arguments?.goal!==prior.goal||
  task.interaction!=='open-user-goal'||task.goalSha256!==sha(JSON.stringify(prior.goal)))
  throw Error('requires the exact original rejected training query with clean completed execution');
 if(pins.validator_sha256!==sha(source)||nativeGuards.some(fragment=>!source.toString().includes(fragment)))
  throw Error('original source lacks native readback, journal and sibling-state guarantees');
 if(world?.kind!=='adventure-journey'||world.id!==prior.caseId||world.goal!==prior.goal||world.expected.turn!=='preserved'||
  world.expected.compared?.length!==world.saves.length-1||new Set(world.saves.map(save=>save.label)).size!==world.saves.length)
  throw Error('only unchanged explicitly compared queries with unambiguous captured save labels are covered');
 const checks=prior.checks??{};
 if(!same(Object.entries(checks).filter(([,value])=>value!==true).map(([key])=>key),['attestedAdventurePlayerEvidence'])||
  requiredNativeChecks.some(key=>checks[key]!==true)||prior.outcomeContract?.returnedValue?.required!==true||
  prior.outcomeDimensions?.effects?.passed!==true||prior.outcomeDimensions?.returnedValue?.required!==true)
  throw Error('other native failures or missing required findings remain unresolved');
 const report=native.report;
 if(!checker.validJourneyEvidence(report?.evidence)||!same(prior.actualResult,report))throw Error('requested findings differ from the actual captured return');
 const roots=(native.calls??[]).filter(call=>native.rootCallIds?.includes(call.callId));
 if(roots.length!==1||roots[0].record.outcome!=='done'||!matches(roots[0].record.output,report))throw Error('actual root completion or returned bytes differ');
 const reads=prior.readbacks;
 if(!Array.isArray(reads)||!reads.length||!Array.isArray(prior.catalogs)||!prior.catalogs.length||!Array.isArray(prior.actions))throw Error('missing actual actor readbacks');
 const catalog=prior.catalogs[0];
 if(!Array.isArray(catalog)||catalog.length!==world.saves.length||!prior.catalogs.every(row=>same(row,catalog))||
  new Set(catalog.map(save=>save.id)).size!==catalog.length)throw Error('assigned save catalog changed or is ambiguous');
 const assigned=world.saves.map(seed=>{const rows=catalog.filter(save=>save.label===seed.label);if(rows.length!==1)throw Error('missing unique native save label');return rows[0];});
 const views=assigned.map(save=>{const rows=reads.filter(view=>same(view.save,save));if(!rows.length||rows.some(view=>!same(view,rows[0])))throw Error('native comparison changed or lacks its original readback');return rows[0];});
 const final=views[world.expected.save];
 if(!same(final,prior.final)||prior.selected!==final.save.id||report.save!==final.save.id||report.observation!==final.observation||
  report.disposition!=='preserved'||!same(report.pending,final.pending)||
  !same([...report.compared].sort(),world.expected.compared.map(index=>assigned[index].id).sort()))
  throw Error('original selected save, comparison scope or exact native final readback differs');
 for(const action of prior.actions){
  if(!['stale','unavailable'].includes(action.result?.status)||action.args?.[0]!==final.save.id||!same(action.result?.view,final))
   throw Error('a native mutation or unbound action remains unresolved');
 }
 if(!world.expected.evidence.every(fact=>checker.verifyJourneyFact(fact,final,views)===fact.verdict))throw Error('independently observed native facts do not satisfy the authored query');
 const masks=[...(prior.incorrectReturnValues??[]),...(prior.outcomeDimensions?.incidentalValues?.incorrectReturns??[])];
 if(masks.some(item=>!same(item.contribution?.checks,['attestedAdventurePlayerEvidence'])||item.contribution.kind!=='root'||
   item.contribution.root_call_id!==roots[0].callId||item.contribution.producer_call_id!==roots[0].callId||
   item.contribution.root_value_json!==JSON.stringify(report)||item.contribution.root_sha256!==sha(JSON.stringify(report))))
  throw Error('an unrelated or unbound original return exclusion cannot be corrected');
 const witness={kind:'source-verified-unchanged-adventure-query',originalValidatorSha256:sha(source),
  nativeChecks:Object.fromEntries(requiredNativeChecks.map(key=>[key,true])),
  guarantees:['native-scene-time-revision-unchanged','native-pending-and-journal-unchanged','sibling-saves-unchanged','actual-current-player-readback'],
  nativeInitialSnapshotCaptured:false,playerSnapshotsOrigin:'original actor readbacks and independently attested final projection'};
 const observations={kind:'archived-player-readbacks',assigned,readbacks:reads,actions:prior.actions,final:prior.final,
  selected:prior.selected,comparison:world.expected.compared.map(index=>assigned[index].id),nativeStateWitness:witness};
 return {prior,native,world,witness,reviewInput:checker.createJourneyMeaningReview(prior.goal,observations,report),
  correctedMasks:masks,checks:{...checks,attestedAdventurePlayerEvidence:true}};
}

async function main(){
 const args=Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>index%2===0?[...rows,[value.slice(2),all[index+1]]]:rows,[]));
 for(const key of ['host','examples','system-testing','packet','pins','catalog','out'])if(!args[key])throw Error(`missing --${key}`);
 const out=resolve(args.out),host=resolve(args.host),examples=resolve(args.examples),sst=resolve(args['system-testing']);
 const raw=await readFile(args.packet),packet=JSON.parse(args.packet.endsWith('.gz')?gunzipSync(raw):raw),pins=JSON.parse(await readFile(args.pins,'utf8'));
 const task=(await readFile(args.catalog,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.scenario===packet.test.name);
 const source=execFileSync('git',['-C',examples,'show',`${pins.repositories.examples.git_commit}:skills/open-adventure-journey-runtime/index.ts`]);
 const originalWorlds=execFileSync('git',['-C',sst,'show',`${pins.repositories['system-testing'].git_commit}:skills/system-testing/open-goal-adventure-worlds.ts`]);
 const currentWorlds=await readFile(join(sst,'skills/system-testing/open-goal-adventure-worlds.ts'));
 if(!same(originalWorlds,currentWorlds))throw Error('query goals or their assigned native setup changed from the actual run');
 const require=createRequire(join(host,'package.json')),{build}=require('esbuild');
 const facts=join(examples,'skills/open-adventure-journey-runtime/findings.ts');
 const bundled=await build({stdin:{contents:`export * from ${JSON.stringify(facts)};`,resolveDir:host},bundle:true,platform:'node',format:'esm',write:false,metafile:true});
 await mkdir(out,{recursive:false,mode:0o700});
 const store=async(name,bytes)=>{await writeFile(join(out,name),bytes,{mode:0o600});return {path:name,sha256:sha(bytes)};};
 const bundle=await store('checker.bundle.mjs',bundled.outputFiles[0].contents),checker=await import(pathToFileURL(join(out,bundle.path)));
 const {openAdventureJourneyWorlds}=await import(pathToFileURL(join(sst,'skills/system-testing/open-goal-adventure-worlds.ts')));
 const world=openAdventureJourneyWorlds.find(world=>world.id===packet.execution.diagnostics.openGoalVerification.caseId);
 const {applicationEntryBinding}=await import(pathToFileURL(join(host,'scripts/export-live-workflow-teacher.mjs')));
 const result=reassessAdventureFindings(packet,pins,task,source,world,checker,applicationEntryBinding);
 await store('review-input.json',JSON.stringify(result.reviewInput,null,2)+'\n');
 if(!args.review){console.log(JSON.stringify({task_id:task.taskId,pending_independent_review:true,review_input:join(out,'review-input.json'),teacher_rerun:false}));return;}
 const review=JSON.parse(await readFile(args.review,'utf8'));
 const {modelOracleJudge}=await import(new URL('../../ts-host/dist/teacher/model-judge.js',import.meta.url));
 await validateWorkflowReview(review,result.reviewInput,pins.teacher,modelOracleJudge,args['review-model']);
 const sources=[await store('original-native-checker.ts',source),await store('original-query-worlds.ts',originalWorlds),bundle,
  await store('entry-binding.mjs','export '+applicationEntryBinding.toString()+'\n'),await store('assessor.mjs',await readFile(fileURLToPath(import.meta.url)))];
 for(const [index,path]of Object.keys(bundled.metafile.inputs).filter(path=>path!=='<stdin>').entries())sources.push(await store(`checker-source-${index}-${basename(path)}`,await readFile(resolve(path))));
 sources.push(await store('reassess_local_presentation.mjs',await readFile(new URL('./reassess_local_presentation.mjs',import.meta.url))));
 sources.push(await store('model-judge.ts',await readFile(new URL('../../ts-host/src/teacher/model-judge.ts',import.meta.url))));
 const dimensions=structuredClone(result.prior.outcomeDimensions);
 dimensions.returnedValue.checks={...dimensions.returnedValue.checks,attestedAdventurePlayerEvidence:true};dimensions.returnedValue.passed=true;
 dimensions.incidentalValues.incorrectReturns=[];
 const repo=fileURLToPath(new URL('../..',import.meta.url));
 const reassessmentCode={git_commit:execFileSync('git',['-C',repo,'rev-parse','HEAD'],{encoding:'utf8'}).trim(),dirty:Boolean(execFileSync('git',['-C',repo,'status','--porcelain'],{encoding:'utf8'}).trim())};
 const proof={...result.prior,schema:'natlang.live-outcome-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),reassessmentCode,
  entry:result.native.entry,actualResult:result.native.report,passed:true,checks:result.checks,outcomeDimensions:dimensions,incorrectReturnValues:[],
  semanticReviewRequired:true,semanticReviewInput:result.reviewInput,
  rationale:'All original native state, journal, scope and old-API handoff checks remain true. Canonical player projections independently establish the requested query facts. A bound independent reviewer judges the actual returned meaning instead of hidden opaque subject spelling or exact count. Only the sole source-identified mechanical claim-match rejection and its exact root exclusion are corrected; all original bytes and exclusions remain in the bound original capture and correction evidence. No initial native snapshot or changed-world success is invented.',
  checker_sources:sources,evidence:[await store('native-query-witness.json',JSON.stringify(result.witness,null,2)),await store('original-return-exclusions.json',JSON.stringify(result.correctedMasks,null,2)),await store('independent-review.json',await readFile(args.review))]};
 const completed=await store('completed-outcome.json',JSON.stringify(proof,null,2)+'\n');
 await store('assessment.json',JSON.stringify({schema:'natlang.live-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),rationale:proof.rationale,completed_outcome:completed},null,2)+'\n');
 console.log(JSON.stringify({task_id:task.taskId,completed_outcome:completed,assessment:join(out,'assessment.json'),teacher_rerun:false}));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
