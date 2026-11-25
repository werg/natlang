#!/usr/bin/env node
/** Reassess archived Regency query findings against their actual native accounts; no execution. */
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
const effects=['exactRegencyForecasts','verifiedRegencyAdvance','exactRegencyAssignment','independentlyReadRegencyMemo','exactSavedRegencyFrontiers','verifiedPrivateRegencyNote','preservedRegencyWorkspace','rejectedRegencyRetarget'];
const returns=['returnedRegencyForecasts','groundedRegencyAction','returnedRegencyAdvance'];
const obsolete=name=>name==='exactInstalledRegencyFindings'||/^exactRegencyFinding\d+$/.test(name);
const guards=['if(revision(current)!==scope.revision||!isDeepStrictEqual(observation.basis,current.game.world))','isDeepStrictEqual(forecasts,nativeVerification)','preservedRegencyWorkspace:status.mainEventId===before.mainEventId','await services.regencySaved.read(other)===null'];
export function reassessRegencyFindings(packet,pins,task,source,world,checker,entryBinding){
 const prior=packet.execution?.diagnostics?.openGoalVerification,native=packet.execution?.diagnostics?.natlangTeacher;
 if(!task||task.scenario!==packet.test?.name||task.split!=='train'||packet.result?.passed!==false||!prior||!native||native.executionError||native.captureErrors?.length||pins.teacher!=='openai-codex:gpt-6-luna'||Object.values(pins.repositories??{}).some(row=>row.dirty!==false)||!same(entryBinding(task.entry),native.entry)||native.entry?.arguments?.goal!==prior.goal||task.interaction!=='open-user-goal'||task.goalSha256!==sha(JSON.stringify(prior.goal)))throw Error('requires an exact original clean completed training query');
 if(pins.validator_sha256!==sha(source)||guards.some(text=>!source.toString().includes(text)))throw Error('original native source guarantees differ');
 if(world?.kind!=='regency-briefing'||world.id!==prior.caseId||world.goal!==prior.goal||world.plan.advance!==null||world.plan.localNote||!['report','hold'].includes(world.action))throw Error('only original read-only queries without a requested native artifact/advance are covered');
 const original=prior.checks??{},legacy=Object.keys(original).filter(obsolete),needed=[...effects,...returns];
 if(!legacy.length||!same(Object.keys(original).sort(),[...needed,...legacy].sort())||needed.some(name=>original[name]!==true)||Object.entries(original).filter(([,value])=>value!==true).some(([name,value])=>!obsolete(name)||value!==false)||prior.outcomeContract?.returnedValue?.required!==true||prior.outcomeDimensions?.effects?.passed!==true)throw Error('other native or requested answer failures remain unresolved');
 const report=native.report,roots=(native.calls??[]).filter(call=>native.rootCallIds?.includes(call.callId));
 if(roots.length!==1||roots[0].record.outcome!=='done'||!matches(roots[0].record.output,report)||!same(prior.actualResult,report)||!same(native.applicationResult,report)||report?.advance!==null||report?.note!==null||prior.advance!==null||prior.note!==null||!same(report.forecasts,prior.forecasts)||!same(report.decision,prior.decision)||!report.decision?.explanation?.trim())throw Error('actual original query return or read-only state differs');
 const captured=native.calls.flatMap(call=>call.record.effects??[]);
 const usedEffects=new Set();
 const witness=(service,method,args,result)=>{const index=captured.findIndex((effect,index)=>!usedEffects.has(index)&&effect.service===service&&effect.method===method&&effect.by==='crisp'&&matches(effect.args,args)&&matches(effect.result,result));if(index<0)return false;usedEffects.add(index);return true;};
 if(!witness('regencyAssignment','list',[],prior.index)||!Array.isArray(prior.frontiers)||prior.frontiers.length!==2||!Array.isArray(prior.forecasts)||!same(prior.artifact,null))throw Error('missing original native query observations');
 for(const frontier of prior.frontiers)if(frontier.realm!==prior.index.realm||!witness('regencySaved','read',[prior.index.realm],frontier))throw Error('native saved account lacks its exact original read effect');
 const [opening,current]=prior.frontiers;
 for(const row of prior.forecasts)if(row.realm!==prior.index.realm||row.basis!==current.revision||!witness('regencyForecast','preview',[row.realm,row.basis,row.months],row))throw Error('native forecast lacks its exact original basis and effect');
 const account=source=>source==='opening'?opening.accounts:source==='current'?current.accounts:prior.forecasts.find(row=>row.months===source)?.accounts;
 const request=claim=>{
  if(claim.kind==='settlement'){const current=account(claim.source);return current?{kind:'settlement',check:claim.check,current}:null;}
  if(claim.kind!=='compare')throw Error('unknown claim kind');
  const before=account(claim.from),after=account(claim.to);
  return before&&after?{kind:'compare',check:claim.check,before,after,assertCommitted:claim.assertCommitted,basis:typeof claim.from==='number'||typeof claim.to==='number'?'forecast':'observed'}:null;
 };
 if(!Array.isArray(report.findings)||!report.findings.length)throw Error('missing actual requested findings');
 const reviewed=report.findings.map(actual=>{const input=request(actual.claim),expected={claim:actual.claim,review:input?checker.exampleReference({kind:'realm',goal:'Qualify exact installed Regency accounts',review:input}):{decision:'not-observed',finding:'missing-source',evidenceIds:[]}};
  if(!checker.matchesNativeRegencyFinding(actual,expected))throw Error('actual finding contradicts native accounts or cites foreign evidence');
  return {claim:actual.claim,request:input,result:expected};});
 if((prior.unboundReturnContributions??[]).some(item=>!item.checks?.length||item.checks.some(name=>!legacy.includes(name))||item.reason!=="no-verified-actual-producer"))throw Error("unrelated unresolved return contribution");
 const masks=[...(prior.incorrectReturnValues??[]),...(prior.outcomeDimensions?.incidentalValues?.incorrectReturns??[])];
 for(const mask of masks){
  const c=mask.contribution,call=native.calls.find(call=>call.callId===c?.producer_call_id),value=mask.value??JSON.parse(mask.valueJson??'null');
  if(!c?.checks?.length||c.checks.some(name=>!legacy.includes(name))||c.root_call_id!==roots[0].callId||c.root_value_json!==JSON.stringify(report)||c.root_sha256!==sha(JSON.stringify(report))||!['root','projection','uncertain-projection'].includes(c.kind)||!call||call.record.outcome!=='done'||!matches(call.record.output,value)||!same(mask.definition,{name:call.record.definition.name,source:call.record.definition.source}))throw Error('unrelated or unbound original return exclusion');
  if(c.kind==='root'?!same(value,report):!report.findings.some(finding=>same(value,finding)))throw Error('excluded producer does not return an actual native-grounded finding');
 }
 const outcomeContract={effects:{checks:effects,rationale:'Every original independent native realm, account, forecast, protected-state and negative-scope check remains required.'},returnedValue:{checks:returns,required:true,rationale:'The original capture-bound query result remains required; its requested factual meaning is independently reviewed against exact native accounts.'},incidentalValues:{checks:legacy,rationale:'Legacy private reference claim source, position, array count and exact finding-token conformance are retained as diagnostic observations, not requested semantic values.'}};
 const select=names=>Object.fromEntries(names.map(name=>[name,original[name]]));
 const outcomeDimensions={effects:{passed:true,checks:select(effects)},returnedValue:{required:true,passed:true,checks:select(returns)},incidentalValues:{passed:legacy.every(name=>original[name]===true),checks:select(legacy),incorrectReturns:[]}};
 const projected={...prior,witnesses:reviewed};
 const review=checker.observedContentReview(projected,outcomeContract,report,['index','frontiers','forecasts','witnesses','advance'],['artifact']);
 return {prior,native,reviewInput:review.semanticReviewInput,checks:select(needed),outcomeContract,outcomeDimensions,masks,legacy,witnesses:reviewed};
}
async function main(){
 const args=Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>index%2===0?[...rows,[value.slice(2),all[index+1]]]:rows,[]));
 for(const key of ['host','base','examples','system-testing','packet','pins','catalog','out'])if(!args[key])throw Error(`missing --${key}`);
 const host=resolve(args.host),base=resolve(args.base),examples=resolve(args.examples),sst=resolve(args['system-testing']),out=resolve(args.out);
 const raw=await readFile(args.packet),packet=JSON.parse(args.packet.endsWith('.gz')?gunzipSync(raw):raw),pins=JSON.parse(await readFile(args.pins,'utf8'));
 const task=(await readFile(args.catalog,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.scenario===packet.test.name);
 const relative='skills/system-testing/open-goal-regency-managed.ts';
 const source=execFileSync('git',['-C',sst,'show',`${pins.repositories['system-testing'].git_commit}:${relative}`]);
 const oraclePath='skills/examples-review-fixtures/oracle.ts',oracle=await readFile(join(examples,oraclePath));
 if(!same(oracle,execFileSync('git',['-C',examples,'show',`${pins.repositories.examples.git_commit}:${oraclePath}`])))throw Error('installed numeric realm reviewer changed from the actual capture');
 const worldsPath='skills/system-testing/open-goal-regency-worlds.ts',originalWorlds=execFileSync('git',['-C',sst,'show',`${pins.repositories['system-testing'].git_commit}:${worldsPath}`]);
 const require=createRequire(join(host,'package.json')),{build}=require('esbuild');
 await mkdir(out,{recursive:false,mode:0o700});const store=async(name,bytes)=>{await writeFile(join(out,name),bytes,{mode:0o600});return {path:name,sha256:sha(bytes)};};
 const bundled=await build({stdin:{contents:`export {exampleReference} from ${JSON.stringify(join(examples,oraclePath))};export {matchesNativeRegencyFinding} from ${JSON.stringify(join(sst,'skills/system-testing/regency-finding-evidence.ts'))};export {observedContentReview} from ${JSON.stringify(join(base,'packages/application-outcomes/observed-content-review.ts'))};`,resolveDir:host},bundle:true,platform:'node',format:'esm',write:false,metafile:true,alias:{'@vibestudio/content-addressing':join(host,'packages/content-addressing/src/index.ts')}});
 const bundle=await store('checker.bundle.mjs',bundled.outputFiles[0].contents),checker=await import(pathToFileURL(join(out,bundle.path)));
 const worldBundle=await build({stdin:{contents:originalWorlds.toString(),resolveDir:sst},bundle:true,platform:'node',format:'esm',write:false});
 await store('original-worlds.bundle.mjs',worldBundle.outputFiles[0].contents);
 const {openRegencyWorlds}=await import(pathToFileURL(join(out,'original-worlds.bundle.mjs'))),world=openRegencyWorlds.find(world=>world.id===packet.execution.diagnostics.openGoalVerification.caseId);
 const {applicationEntryBinding}=await import(pathToFileURL(join(host,'scripts/export-live-workflow-teacher.mjs')));
 const result=reassessRegencyFindings(packet,pins,task,source,world,checker,applicationEntryBinding);
 await store('review-input.json',JSON.stringify(result.reviewInput,null,2)+'\n');
 if(!args.review){console.log(JSON.stringify({pending_independent_review:true,review_input:join(out,'review-input.json'),teacher_rerun:false,original_sha256:sha(raw)}));return;}
 const {modelOracleJudge}=await import(new URL('../../ts-host/dist/teacher/model-judge.js',import.meta.url));
 await validateWorkflowReview(JSON.parse(await readFile(args.review,'utf8')),result.reviewInput,pins.teacher,modelOracleJudge,args['review-model']);
 const sources=[await store('original-native-checker.ts',source),await store('corrected-native-checker.ts',await readFile(join(sst,relative))),await store('original-worlds.ts',originalWorlds),bundle,await store('assessor.mjs',await readFile(fileURLToPath(import.meta.url))),await store('reassess_local_presentation.mjs',await readFile(new URL('./reassess_local_presentation.mjs',import.meta.url))),await store('model-judge.ts',await readFile(new URL('../../ts-host/src/teacher/model-judge.ts',import.meta.url))),await store('entry-binding.mjs','export '+applicationEntryBinding.toString()+'\n'),{...await store('outcome-contract.json',JSON.stringify(result.outcomeContract,null,2)+'\n'),role:'outcome-contract'}];
 for(const[index,path]of Object.keys(bundled.metafile.inputs).filter(path=>path!=='<stdin>').entries())sources.push(await store(`checker-source-${index}-${basename(path)}`,await readFile(resolve(path))));
 const repo=fileURLToPath(new URL('../..',import.meta.url));
 const proof={...result.prior,schema:'natlang.live-outcome-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),entry:result.native.entry,actualResult:result.native.report,passed:true,checks:result.checks,observedChecks:result.prior.checks,outcomeContract:result.outcomeContract,outcomeDimensions:result.outcomeDimensions,incorrectReturnValues:[],unboundReturnContributions:[],semanticReviewRequired:true,semanticReviewInput:result.reviewInput,witnesses:result.witnesses,diagnosticChecks:Object.fromEntries(result.legacy.map(name=>[name,result.prior.checks[name]])),reassessmentCode:{git_commit:execFileSync('git',['-C',repo,'rev-parse','HEAD'],{encoding:'utf8'}).trim(),dirty:Boolean(execFileSync('git',['-C',repo,'status','--porcelain'],{encoding:'utf8'}).trim())},rationale:'All original independent native query state, account, forecast, protected-state and negative-scope checks remain required and true. Actual supplied findings are recomputed through the unchanged installed numeric realm reviewer and bound to original crisp read effects and actual returned bytes. Independent bound meaning review judges the requested answer without private claim order, count, source-label alias or explanatory token requirements. Legacy reference-conformance observations remain false diagnostically; no missing array element or new execution is invented. Only source-identified masks on actually native-grounded original values are retired; the original capture, all observations and exclusions remain immutable in provenance.',checker_sources:sources,evidence:[await store('original-observations.json',JSON.stringify({checks:result.prior.checks,observedChecks:result.prior.observedChecks,exclusions:result.masks,unboundReturnContributions:result.prior.unboundReturnContributions},null,2)),await store('actual-native-claim-grounding.json',JSON.stringify(result.witnesses,null,2)),await store('independent-review.json',await readFile(args.review))]};
 const completed=await store('completed-outcome.json',JSON.stringify(proof,null,2)+'\n');await store('assessment.json',JSON.stringify({schema:'natlang.live-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),rationale:proof.rationale,completed_outcome:completed},null,2)+'\n');console.log(JSON.stringify({assessment:join(out,'assessment.json'),teacher_rerun:false}));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
