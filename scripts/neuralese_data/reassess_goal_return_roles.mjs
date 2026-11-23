#!/usr/bin/env node
/** Reassess original native evidence under goal-specific return roles; no execution. */
import {createHash} from 'node:crypto';
import {isDeepStrictEqual as same} from 'node:util';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {execFileSync} from 'node:child_process';
import {resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const bytes=value=>Buffer.from(JSON.stringify(value));
const bindingMatches=(binding,value)=>binding?.complete===true&&binding.hash===sha(bytes(value))&&binding.bytes===bytes(value).length;
export function reassessGoalReturnRoles(packet){
 const prior=packet.execution?.diagnostics?.openGoalVerification,native=packet.execution?.diagnostics?.natlangTeacher;
 if(packet.result?.passed!==false||!prior||!native||native.executionError||native.captureErrors?.length||!same(prior.actualResult,native.report))throw Error('requires original completed native execution and exact actual result');
 const roots=native.calls.filter(call=>native.rootCallIds.includes(call.callId));
 if(roots.length!==1||roots[0].record.outcome!=='done'||!bindingMatches(roots[0].record.output,native.report))throw Error('original application output identity differs');
 const effects=native.calls.flatMap(call=>call.record.effects??[]);
 const witnessed=(service,method,args,result)=>effects.some(effect=>effect.service===service&&effect.method===method&&bindingMatches(effect.args,args)&&bindingMatches(effect.result,result));
 const failed=Object.entries(prior.checks).filter(([,value])=>value!==true).map(([name])=>name);
 if(prior.preparedContentProject){
  const state=prior.preparedContentProject,report=native.report;
  const nativeNames=['exactOwnedSourceDiscovery','genuineEvidenceDisposition','exactScopedAuthorization','exactArtifactOrReadOnlyHold','protectedPrivateBoundary'];
  if(!same(Object.keys(prior.checks).sort(),[...nativeNames,'correctOutcomeReport'].sort())||
     !same(failed,['correctOutcomeReport'])||state.expectedOutcome!=='draft'||
     prior.goal!==packet.test?.prompt||prior.goal!==native.entry?.arguments?.goal||
     native.entry?.callable!=='prepareContentProject'||prior.contextId!==state.contextId||prior.caseId!==state.caseId||
     report.final?.outcome!=='review-needed'||report.final?.event!==null||report.decision?.action!=='draft'||
     state.actual?.clean!==false||state.actual.workingHead?.kind!=='application'||
     state.actual.mainEventId!==state.before?.mainEventId||!same(state.actual.committed,state.before?.committed)||
     state.artifact?.passed!==true||!same(state.artifact.semanticReviewInput?.actual?.state,state.actual.workingHead)||
     !same(report.final.candidate,report.draft?.snapshot)||report.draft?.review?.ready!==false||
     !report.draft.review.unsupportedClaims?.length||typeof report.explanation!=='string'||!report.explanation.trim())
    throw Error('prepared project lacks an independently completed private draft with an honest review hold');
  for(const row of state.journal??[])if(!witnessed(row.service,row.method,row.args,row.result))
    throw Error('prepared project journal differs from exact original native effects');
  const projectServices=new Set(['contentProjectSources','contentProjectScope','contentProjectDraft','contentProjectCommit']);
  if(effects.some(effect=>projectServices.has(effect.service)&&!state.journal?.some(row=>
     row.service===effect.service&&row.method===effect.method&&bindingMatches(effect.args,row.args)&&bindingMatches(effect.result,row.result))))
    throw Error('prepared project journal omits original native effects');
  if(!['contentProjectSources.observe','contentProjectScope.confirm','contentProjectDraft.prepare','contentProjectDraft.write'].every(key=>
     state.journal?.some(row=>row.service+'.'+row.method===key))||state.journal.some(row=>row.service==='contentProjectCommit'))
    throw Error('prepared project lacks its native draft authorization and write witnesses');
  const snapshot=report.draft.snapshot;
  if(!snapshot||snapshot.head!==JSON.stringify(state.actual.workingHead)||
     !state.journal.some(row=>row.service==='contentProjectDraft'&&row.method==='write'&&same(row.result,snapshot)))
    throw Error('prepared project snapshot differs from its native write or independent readback');
  return {checks:Object.fromEntries(nativeNames.map(name=>[name,true])),
   outcomeContract:{effects:{checks:nativeNames,rationale:'Independent native project, authorization and private-state observations.'},
    returnedValue:{checks:[],required:false,rationale:'The goal requests retained private state without a requested answer.'},
    incidentalValues:{checks:['correctOutcomeReport'],rationale:'The old exact disposition is a response convention; review-needed can be honest without an incorrect return.'}},
   outcomeDimensions:{effects:{passed:true,checks:Object.fromEntries(nativeNames.map(name=>[name,true]))},
    returnedValue:{required:false,passed:true,checks:{}},
    incidentalValues:{passed:false,checks:{correctOutcomeReport:false},incorrectReturns:[]}},
   rationale:'The requested private draft is retained and independently read back under exact source authorization, with main and committed state unchanged. The final value honestly flags a malformed citation for further local review. Its review-needed disposition is not a failed project effect and remains intact as application data; no return target is excluded or fabricated.'};
 }
 let repaired,incidental;
 if(prior.caseId.startsWith('desktop-readiness-')){
  repaired='exactDesktopClientSource';
  if(!same(failed,[repaired]))throw Error('other requested desktop outcomes remain unproved');
  const admission=native.report.evidence?.admission,expected=prior.exactAdmission,observation=expected?.source?.observation;
  if(!admission||!expected||!observation||!same(admission.source,expected.source)||!same(admission.host,expected.host)||
   observation.contextId!==prior.contextId||observation.operationId!==prior.desk.operationId||observation.host.kind!=='desktop'||
   observation.published.id!=='browser'||observation.published.runtime!=='web'||observation.published.abi!=='web-host-1'||
   !witnessed('clientAdmissionRead','observe',[prior.desk.operationId],observation)||
   !prior.journal.some(row=>row.service==='clientAdmissionRead'&&row.method==='observe'&&same(row.args,[prior.desk.operationId])&&same(row.result,observation)))throw Error('original desktop source/host has no exact native witness');
  if(admission.decision===expected.decision)throw Error('expected only an incorrect incidental recommendation');
  const root=roots[0].record;
  const values=[
   {required:false,definition:{name:'chooseNext',source:'reviewDesktopReadiness/clientDevelopment/reviewClientAdmission/chooseNext.nl'},value:admission.decision},
   {required:false,definition:{name:'reviewClientAdmission',source:'reviewDesktopReadiness/clientDevelopment/reviewClientAdmission.nl'},value:admission},
   {required:true,definition:{name:root.definition.name,source:root.definition.source},value:native.report},
  ];
  incidental={passed:false,checks:{incidentalDesktopAdmissionClassification:false},incorrectReturns:values.map(({value,...item})=>({...item,valueJson:JSON.stringify(value),sha256:sha(bytes(value))}))};
 }else if(prior.caseId.startsWith('intent-discovery-')){
  repaired='installedIntentDiscovery';
  if(!same(failed,[repaired]))throw Error('other requested intent outcomes remain unproved');
  const search=native.report.search;
  const actualSearchBindings=[];
  for(const query of search?.queries??[]){
   const effect=effects.find(effect=>effect.service==='nativeIntentDiscovery'&&effect.method==='search'&&bindingMatches(effect.args,[query]));
   if(!effect||effect.result?.complete!==true)throw Error('claimed query has no original native witness');
   actualSearchBindings.push(effect.result);
  }
  if(!same(search,prior.search)||!search||!Array.isArray(search.queries)||!Array.isArray(search.hits)||search.queries.length!==search.hits.length||!Array.isArray(search.described)||
   !search.described.every(value=>value&&typeof value.id==='string'&&witnessed('nativeIntentDiscovery','describe',[value.id],value)))throw Error('claimed discovery contract has no exact original native witness');
  if(!search.hits.every((hits,index)=>bindingMatches(actualSearchBindings[index],hits))){
   const root=roots[0].record;
   const values=[{required:false,definition:{name:'searchRelevantIntent',source:'discoverScopedIntent/searchRelevantIntent.nl'},value:search},
    {required:true,definition:{name:root.definition.name,source:root.definition.source},value:native.report}];
   incidental={passed:false,checks:{incidentalDiscoveryTranscript:false},incorrectReturns:values.map(({value,...item})=>({...item,valueJson:JSON.stringify(value),sha256:sha(bytes(value))}))};
  }
 }else throw Error('unsupported goal return-role reassessment');
 return {checks:{...prior.checks,[repaired]:true},...(incidental?{outcomeDimensions:{incidentalValues:incidental}}:{}),
  rationale:repaired==='installedIntentDiscovery'?'Chosen discovery queries and installed descriptions match exact original native effect bytes. Requested scoped observations and answer passed their original independent checks. Full native search output bindings differ from the unused mirrored hit arrays; those noncanonical transcript values and their producer/root return generations are excluded from targets. Original native result bytes are not reconstructed. The obsolete transcript requirement is removed.':'Exact original native client observation proves source, manifest and host values; all requested layout, build distinction, answer and protected-state checks passed. The unused incorrect admission recommendation is excluded from supervised returns at its actual producers.'};
}
async function main(){
 const args=Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>index%2===0?[...rows,[value.slice(2),all[index+1]]]:rows,[]));
 for(const key of ['packet','pins','catalog','checker-root','out'])if(!args[key])throw Error(`missing --${key}`);
 const raw=await readFile(args.packet),packet=JSON.parse(args.packet.endsWith('.gz')?gunzipSync(raw):raw),pins=JSON.parse(await readFile(args.pins,'utf8'));
 if(Object.values(pins.repositories).some(row=>row.dirty!==false))throw Error('requires clean original source provenance');
 const task=(await readFile(args.catalog,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.scenario===packet.test.name);
 if(!task||task.split!=='train')throw Error('requires catalogued training task');
 const result=reassessGoalReturnRoles(packet),out=resolve(args.out);await mkdir(out,{recursive:false});
 const store=async(name,value)=>{await writeFile(join(out,name),value);return {path:name,sha256:sha(value)}};
 const relative='skills/system-testing/open-goal-'+(packet.execution?.diagnostics?.openGoalVerification?.preparedContentProject?'prepared-content-project':packet.test.name.includes('desktop-readiness')?'desktop-readiness':'intent-discovery')+'-managed.ts';
 const original=execFileSync('git',['-C',args['checker-root'],'show',`${pins.repositories['system-testing'].git_commit}:${relative}`]);
 const corrected=await readFile(join(args['checker-root'],relative));
 const checkerSources=[await store('original-checker.ts',original),await store('corrected-checker.ts',corrected),await store('assessor.mjs',await readFile(fileURLToPath(import.meta.url)))];
 if(relative.includes('intent-discovery'))checkerSources.push(await store('intent-capability-evidence.ts',await readFile(join(args['checker-root'],'skills/system-testing/intent-capability-evidence.ts'))));
 if(result.outcomeContract)checkerSources.push({...await store('outcome-contract.json',JSON.stringify(result.outcomeContract,null,2)+'\n'),role:'outcome-contract'});
 if(relative.includes('prepared-content-project'))checkerSources.push(await store('prepared-content-project-outcome.ts',await readFile(join(args['checker-root'],'skills/system-testing/prepared-content-project-outcome.ts'))));
 const diagnostic=packet.execution.diagnostics,prior=diagnostic.openGoalVerification,native=diagnostic.natlangTeacher;
 const proof={schema:'natlang.live-outcome-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),goal:prior.goal,entry:native.entry,actualResult:native.report,passed:true,...result,
  checker_sources:checkerSources,evidence:[await store('native-witnesses.json',JSON.stringify({originalChecks:prior.checks,prior,actualResult:native.report,effects:native.calls.flatMap(call=>call.record.effects??[])}))]};
 const completed=await store('completed-outcome.json',JSON.stringify(proof,null,2)+'\n');
 await store('assessment.json',JSON.stringify({schema:'natlang.live-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),rationale:result.rationale,completed_outcome:completed},null,2)+'\n');
 console.log(JSON.stringify({task_id:task.taskId,assessment:join(out,'assessment.json'),teacher_rerun:false}));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
