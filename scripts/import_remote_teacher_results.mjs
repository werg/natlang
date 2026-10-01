#!/usr/bin/env node
/** Admit remote assignment identities before exposing raw results to automatic training discovery. */
import {readFile,readdir,mkdir,writeFile,rename,appendFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';import {resolve,join} from 'node:path';import {pathToFileURL} from 'node:url';
const [campaignArg]=process.argv.slice(2);if(!campaignArg)throw Error('usage: import_remote_teacher_results CAMPAIGN');
const campaign=resolve(campaignArg);const load=p=>import(pathToFileURL(resolve(p)));
const {recordDigest}=await load(join(campaign,'runtime-v26/dist/teacher/collector.js'));
const {sourceConversionDigest}=await load(join(campaign,'runtime-v26/dist/teacher/source-conversion.js'));
const {admitRow}=await load('ts-host/dist/teacher/curriculum.js');
const irs=(await readFile(join(campaign,'bundle/cases.ir.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);
const assignment=JSON.parse(await readFile(join(campaign,'assignment.json'),'utf8'));
const allowed=new Set(irs.map(recordDigest)),expectedControls=JSON.parse(await readFile(join(campaign,'bundle/chat-request-config.json'),'utf8'));
const sha=x=>createHash('sha256').update(x).digest('hex');
const staging=join(campaign,'runtime-import-staging'), published=join(campaign,'imports');await mkdir(published,{recursive:true});
const prior=new Set();try{for(const line of (await readFile(join(campaign,'import-ledger.jsonl'),'utf8')).trim().split('\n'))if(line)prior.add(JSON.parse(line).sha256);}catch(e){if(e.code!=='ENOENT')throw e;}
async function* files(dir){let entries;try{entries=await readdir(dir,{withFileTypes:true});}catch(e){if(e.code==='ENOENT')return;throw e;}for(const e of entries){const p=join(dir,e.name);if(e.isDirectory())yield*files(p);else if(e.isFile()&&e.name.endsWith('.result.json'))yield p;}}
let newRows=0,newAdmitted=0,newRejected=0,newHeld=0;
for await(const file of files(join(staging,'jobs'))){const bytes=await readFile(file),hash=sha(bytes);if(prior.has(hash))continue;let row,reasons=[];
try{row=JSON.parse(bytes);if(!allowed.has(recordDigest(row.task?.program_ir)))reasons.push('outside_assigned_ir');if(row.provenance?.model!==assignment.model)reasons.push('unexpected_teacher_model');if(row.provenance?.collection_role!=='teacher')reasons.push('not_explicit_teacher');
if(sourceConversionDigest(row.provenance?.chat_request_controls)!==sourceConversionDigest(expectedControls))reasons.push('unexpected_chat_request_controls');}catch(e){reasons.push('invalid_result_or_program');}
let admitted=false,admissionReasons=[];if(!reasons.length){try{const a=admitRow(row);admitted=a.admitted;admissionReasons=a.reasons;}catch(e){admissionReasons=['admission_error'];}}
const folder=join(published,reasons.length?'held':'jobs',hash);await mkdir(folder,{recursive:true});const destination=join(folder,reasons.length?'result.raw.json':'result.result.json');const tmp=destination+'.tmp';await writeFile(tmp,bytes);await rename(tmp,destination);
const item={imported_at:new Date().toISOString(),host:assignment.host,model:assignment.model,model_revision:assignment.model_revision,sha256:hash,id:row?.id,program_id:row?.task?.program_ir?.id,assignment_sha256:sha(await readFile(join(campaign,'assignment.json'))),disposition:reasons.length?'assignment_held':admitted?'admitted':'rejected',reasons:reasons.length?reasons:admissionReasons,path:destination};await appendFile(join(campaign,'import-ledger.jsonl'),JSON.stringify(item)+'\n');prior.add(hash);newRows++;if(reasons.length)newHeld++;else if(admitted)newAdmitted++;else newRejected++;
}
const report={checked_at:new Date().toISOString(),host:assignment.host,model:assignment.model,new_rows:newRows,new_admitted:newAdmitted,new_rejected:newRejected,new_assignment_held:newHeld,total_unique_artifacts:prior.size,training_publication:'Eligible for next generated snapshot; no snapshot or final dataset published by this importer'};
await writeFile(join(campaign,'import-status.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
