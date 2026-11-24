#!/usr/bin/env node
/** Reassess immutable workspace-edit observations; never execute a teacher or edit a raw packet. */
import {readFile,writeFile,mkdir,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {resolve,join,basename} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {isDeepStrictEqual as same} from 'node:util';
const sha=value=>createHash('sha256').update(value).digest('hex');
const encoded=value=>Buffer.from(JSON.stringify(value));
const matches=(binding,value)=>binding?.complete===true&&binding.hash===sha(encoded(value))&&binding.bytes===encoded(value).length;
const load=async path=>{const bytes=await readFile(path);return {bytes,value:JSON.parse(path.endsWith('.gz')?gunzipSync(bytes):bytes)}};

export function reassessWorkspaceEdit(packet,world,check){
 const prior=packet.execution?.diagnostics?.openGoalVerification,native=packet.execution?.diagnostics?.natlangTeacher;
 if(!world||!prior||!native||native.captureErrors?.length)throw Error('Needs an original intact capture from an affected workspace-edit world');
 const roots=native.calls.filter(call=>native.rootCallIds.includes(call.callId));
 if(roots.length!==1||(roots[0].record.output?.complete? !matches(roots[0].record.output,native.report):native.report!=null)||native.entry?.arguments?.goal!==world.goal||prior.goal!==world.goal)throw Error('Original application value or goal identity differs');
 const observations=prior.document??prior;
 if(!observations.contents||!observations.reads||!observations.attempts)throw Error('Original independent retained-byte observations are absent');
 const corrected=check(world,observations),effects=native.calls.flatMap(call=>call.record.effects??[]);
 const witness=(method,args,result)=>effects.some(effect=>effect.service==='goalWorkspace'&&effect.method===method&&matches(effect.args,args)&&matches(effect.result,result));
 const texts=new Map(Object.values(world.files).map(text=>[sha(text),text]));
 texts.set(sha(world.files[world.destination]+world.concurrentAddition),world.files[world.destination]+world.concurrentAddition);
 for(const attempt of observations.attempts)if(attempt.status==='written')texts.set(sha(attempt.text),attempt.text);
 for(const read of observations.reads){const text=texts.get(read.version);if(text===undefined||!witness('read',[read.path],{path:read.path,version:read.version,text}))throw Error('A captured read lacks its exact original native effect');}
 for(const attempt of observations.attempts){const result=attempt.status==='written'?{status:'written',version:sha(attempt.text)}:{status:attempt.status};
  if(!witness('revise',[attempt.path,attempt.version,attempt.text],result))throw Error('A captured revision attempt lacks its exact original native effect');}
 const checks={...prior.checks};
 for(const [name,value]of Object.entries(corrected.checks)){if(!Object.hasOwn(checks,name))throw Error('Original outcome did not expose the affected check');checks[name]=value;}
 return {checks,corrected,prior,native,observations,eligibleForContentAudit:Object.values(checks).every(value=>value===true)};
}

function validateAudit(audit,result,packetSha){
 const edit=result.corrected.edit;
 if(audit?.schema!=='natlang.workspace-edit-content-audit/1'||audit.original_sha256!==packetSha||audit.retained_sha256!==edit.retainedSha256||
   audit.goal!==result.prior.goal||audit.accepted!==true||!audit.auditor||!audit.rationale||audit.unsupported_additions!==false||!audit.coverage?.length)
  throw Error('Independent content audit must bind this exact original goal and retained text');
 const sources=[{path:edit.authority.path,text:edit.authority.text},...(result.prior.semanticReviewInput?.actual?.sources??[])];
 for(const item of audit.coverage){
  const source=sources.find(source=>(source.path??source.url)===item.source);
  if(source?.url&&sha(source.text)!==source.sha256)throw Error('Content audit public source bytes differ from the original captured digest');
  if(!source||typeof item.source_quote!=='string'||!item.source_quote||!source.text.includes(item.source_quote)||
    typeof item.retained_quote!=='string'||!item.retained_quote||!edit.retainedText.includes(item.retained_quote)||!item.rationale)
   throw Error('Content audit coverage differs from exact retained bytes or authoritative source');
 }
}

async function main(){
 const args=Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>index%2===0?[...rows,[value.slice(2),all[index+1]]]:rows,[]));
 if(!args.checker)throw Error('Requires --checker compiled shared source artifact');
 const checkerPath=resolve(args.checker),{worlds,checkWorkspaceEditEvidence}=await import(pathToFileURL(checkerPath));
 if(args.archive){
  const rows=[];for(const wave of (await readdir(args.archive)).filter(name=>/^\d{4}$/.test(name)).sort()){
   const cases=join(args.archive,wave,'capture/cases');let scenarios;try{scenarios=await readdir(cases)}catch(error){if(error.code==='ENOENT')continue;throw error;}
   for(const scenario of scenarios){const id=scenario.replace(/^open-goal-/,'');if(!worlds[id])continue;
    for(const entry of await readdir(join(cases,scenario),{withFileTypes:true})){
     if(!entry.isDirectory())continue;const run=entry.name;
     for(const file of await readdir(join(cases,scenario,run))){
     if(!file.endsWith('-full.json.gz'))continue;const path=join(cases,scenario,run,file),{bytes,value}=await load(path);
     let result;try{const r=reassessWorkspaceEdit(value,worlds[id],checkWorkspaceEditEvidence);result={eligibleForContentAudit:r.eligibleForContentAudit,originalChecks:r.prior.checks,correctedChecks:r.checks,retainedSha256:r.corrected.edit.retainedSha256,retainedText:r.corrected.edit.retainedText};}catch(error){result={eligibleForContentAudit:false,reason:error.message};}
     rows.push({wave,scenario,path,sha256:sha(bytes),originalPassed:value.result?.passed,...result});
    }
    }
   }
  }
  const out={schema:'natlang.workspace-edit-cohort/1',checker_sha256:sha(await readFile(checkerPath)),rows};
  await writeFile(args.out,JSON.stringify(out,null,2)+'\n');console.log(JSON.stringify({out:args.out,captures:rows.length,candidates:rows.filter(row=>row.eligibleForContentAudit&&row.originalPassed===false).length}));return;
 }
 for(const key of ['packet','audit','catalog','out'])if(!args[key])throw Error('Missing --'+key);
 const {bytes:raw,value:packet}=await load(args.packet),id=packet.test.name.replace(/^open-goal-/,''),result=reassessWorkspaceEdit(packet,worlds[id],checkWorkspaceEditEvidence);
 if(packet.result?.passed!==false||!result.eligibleForContentAudit)throw Error('Unresolved independent goal checks remain');
 const audit=(await load(args.audit)).value;validateAudit(audit,result,sha(raw));
 const task=(await readFile(args.catalog,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.scenario===packet.test.name);
 if(!task||task.split!=='train'||!task.entry||task.entry.callable!==result.native.entry.callable||
   task.entry.program!==result.native.entry.program||!same(task.entry.arguments,result.native.entry.arguments))throw Error('Requires the exact bound catalog training task and original entry arguments');
 await mkdir(args.out,{recursive:false});
 const store=async(name,bytes)=>{await writeFile(join(args.out,name),bytes);return {path:name,sha256:sha(bytes)}};
 const checkerSources=[await store('shared-workspace-edit-checker.mjs',await readFile(checkerPath)),await store('assessor.mjs',await readFile(fileURLToPath(import.meta.url)))];
 const evidence=[await store('independent-content-audit.json',await readFile(args.audit)),await store('native-workspace-edit-witness.json',JSON.stringify({originalChecks:result.prior.checks,checks:result.checks,originalObservations:result.observations,corrected:result.corrected}))];
 const proof={schema:'natlang.live-outcome-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),goal:result.prior.goal,entry:result.native.entry,actualResult:result.native.report,passed:true,checks:result.checks,
  rationale:'Exact original native read/write effects and independent retained bytes pass the corrected shared workspace-edit scope/revision checker. A source-bound independent audit establishes the requested corrected prose and rejects unsupported additions. The previous authored-example equality was not a literal user requirement; original packet and validator outcome remain intact.',checker_sources:checkerSources,evidence};
 const completed=await store('completed-outcome.json',JSON.stringify(proof,null,2)+'\n');
 await store('assessment.json',JSON.stringify({schema:'natlang.live-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),rationale:proof.rationale,completed_outcome:completed},null,2)+'\n');
 console.log(JSON.stringify({task_id:task.taskId,assessment:join(args.out,'assessment.json'),teacher_rerun:false,reviewer_rerun:false}));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
