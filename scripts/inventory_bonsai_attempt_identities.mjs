#!/usr/bin/env node
// Content-based, streaming inventory of saved Bonsai teacher attempt identities.
// Unlike path-name filters, this scans every result artifact and requires model provenance.
import { createReadStream } from 'node:fs';
import { readdir, stat, readlink, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createGunzip } from 'node:zlib';
import { PassThrough, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { createInterface } from 'node:readline';
import path from 'node:path';
import process from 'node:process';

function parseArgs(argv) {
  const out={};
  for(let i=2;i<argv.length;i++) { const k=argv[i]; if(!k.startsWith('--')) throw new Error(`unexpected_argument:${k}`); const v=argv[++i]; if(!v||v.startsWith('--')) throw new Error(`missing_value:${k}`); const name=k.slice(2); if(name==='source-catalog'||name==='additional-root'){(out[name]??=[]).push(v);}else out[name]=v; }
  if(!out['runs-dir']||!out.output) throw new Error('usage: --runs-dir PATH --output PATH [--prior-inventory PATH] [--additional-root PATH ...] [--source-catalog PATH ...]');
  return out;
}
const args=parseArgs(process.argv);
const scanStart=Date.now();
const root=process.cwd();
const runsDir=path.resolve(args['runs-dir']);
const scanRoots=[runsDir,...(args['additional-root']??[]).map(x=>path.resolve(x))];
const outputPath=path.resolve(args.output);
const hashBytes=bytes=>createHash('sha256').update(bytes).digest('hex');
function hashedInput(file,{gzip=false}={}){
 const contentHash=createHash('sha256');
 const raw=createReadStream(file);
 const hashStream=new Transform({transform(chunk,encoding,callback){contentHash.update(chunk);callback(null,chunk);}});
 const decoded=gzip?createGunzip():new PassThrough();
 const output=new PassThrough();
 const streams=gzip?[raw,hashStream,decoded,output]:[raw,hashStream,output];
 const done=pipeline(...streams);
 // Attach a handler immediately; the reader may surface the same stream error
 // before finish() awaits this promise. Awaiting done still rethrows the failure.
 done.catch(()=>{});
 return {input:output,finish:async()=>{await done;return contentHash.digest('hex');},cleanup:async()=>{for(const stream of streams)if(!stream.destroyed)stream.destroy();await done.catch(()=>{});}};
}
const scanErrors=[], skippedSymlinks=[];
async function* walk(dir){
 let entries;
 try{entries=await readdir(dir,{withFileTypes:true});}catch(e){scanErrors.push({path:path.relative(root,dir),error:`directory_read_failed:${String(e?.message??e)}`});return;}
 for(const ent of entries){
  const file=path.join(dir,ent.name);
  if(ent.isDirectory()){yield*walk(file);continue;}
  if(ent.isFile()&&(/\.result\.json$|\.partial\.json$|\.results\.jsonl(?:\.gz)?$/).test(ent.name))yield file;
  else if(ent.isSymbolicLink()){
   let target=null,kind='unknown';
   try{target=await readlink(file);const targetStat=await stat(file);kind=targetStat.isDirectory()?'directory':'file';}
   catch(e){scanErrors.push({path:path.relative(root,file),error:`symlink_target_unavailable:${String(e?.message??e)}`});continue;}
   const artifactName=/\.result\.json$|\.partial\.json$|\.results\.jsonl(?:\.gz)?$/.test(ent.name);
   const item={path:path.relative(root,file),target,kind};
   if(kind==='file'&&artifactName)scanErrors.push({...item,error:'artifact_symlink_not_followed'});
   else if(kind==='directory'&&!['node_modules','vendor'].includes(ent.name))scanErrors.push({...item,error:'unreviewed_symlink_directory_not_followed'});
   else skippedSymlinks.push({...item,classification:kind==='directory'?'known_dependency_directory':'non_artifact_file'});
  }
 }
}
const savedProgramIds=new Set(), savedSourceIds=new Set(), queuedProgramIds=new Set(), queuedSourceIds=new Set();
let priorInventorySha256=null;
if(args['prior-inventory']){
 const priorBytes=await readFile(path.resolve(args['prior-inventory'])); priorInventorySha256=hashBytes(priorBytes);
 const prior=JSON.parse(priorBytes.toString('utf8'));
 const addPrior=(set,values,label)=>{if(values===undefined)return;if(!Array.isArray(values)||values.some(id=>typeof id!=='string'||!id))throw new Error(`invalid_prior_inventory_${label}`);for(const id of values)set.add(id);};
 addPrior(savedProgramIds,prior.saved_bonsai_program_ids??prior.program_ids,'saved_program_ids');
 addPrior(savedSourceIds,prior.saved_bonsai_source_ids??prior.source_ids,'saved_source_ids');
 addPrior(queuedProgramIds,prior.queued_program_ids??(prior.program_ids===undefined?prior.live_queue_program_ids:undefined),'queued_program_ids');
 addPrior(queuedSourceIds,prior.queued_source_ids??(prior.source_ids===undefined?prior.live_queue_source_ids:undefined),'queued_source_ids');
}
const artifacts=[], malformed=[], vanished=[], partials=[], modelCounts={}, programCounts=new Map(),sourceCounts=new Map();
const metaRootByFile=new Map();
const partialProgramIds=new Set(), resolvedPartialPrograms=new Map();
let scannedFiles=0,scannedBytes=0,bonsaiRows=0;
function consume(row,file,meta,recordNo){
 if(!row||typeof row!=='object'||Array.isArray(row)){malformed.push({path:path.relative(root,file),record:recordNo,error:'record_is_not_object'});return;}
 const model=typeof row.provenance?.model==='string'?row.provenance.model:'';
 if(!/bonsai/i.test(model))return;
 const ir=row.task?.program_ir;
 if(!ir||typeof ir!=='object'||typeof ir.id!=='string'||!ir.id){
  if(file.endsWith('.partial.json')&&typeof row.program_id==='string'&&row.program_id){
   savedProgramIds.add(row.program_id);programCounts.set(row.program_id,(programCounts.get(row.program_id)??0)+1);
   const partial={path:path.relative(root,file),program_id:row.program_id,program_ir_sha256:row.provenance.program_ir_sha256??null,model};partials.push(partial);partialProgramIds.add(row.program_id);meta.partial_records++;meta.partial_program_ids.push(row.program_id);meta.models[model]=(meta.models[model]??0)+1;modelCounts[model]=(modelCounts[model]??0)+1;return;
  }
  malformed.push({path:path.relative(root,file),record:recordNo,error:'Bonsai provenance present without task.program_ir.id or partial program_id'});return;
 }
 bonsaiRows++;meta.bonsai_rows++;meta.models[model]=(meta.models[model]??0)+1;modelCounts[model]=(modelCounts[model]??0)+1;
 savedProgramIds.add(ir.id);programCounts.set(ir.id,(programCounts.get(ir.id)??0)+1);
 if(ir.source_ids!==undefined&&(!Array.isArray(ir.source_ids)||ir.source_ids.some(x=>typeof x!=='string'||!x)))malformed.push({path:path.relative(root,file),record:recordNo,error:'invalid_task_program_source_ids'});
 if(ir.external_source?.source_id!==undefined&&(typeof ir.external_source.source_id!=='string'||!ir.external_source.source_id))malformed.push({path:path.relative(root,file),record:recordNo,error:'invalid_external_source_id'});
 const rowSources=new Set([...(Array.isArray(ir.source_ids)?ir.source_ids:[]),ir.external_source?.source_id].filter(x=>typeof x==='string'&&x));
 for(const id of rowSources){savedSourceIds.add(id);sourceCounts.set(id,(sourceCounts.get(id)??0)+1);}
}
const visitedFiles=new Set();
for(const scanRoot of scanRoots) for await(const file of walk(scanRoot)){
 if(visitedFiles.has(file))continue;visitedFiles.add(file);
 scannedFiles++;
 metaRootByFile.set(file,scanRoot);
 let before;
 try{before=await stat(file);}catch(e){if(e.code==='ENOENT'){vanished.push(path.relative(root,file));continue;}scanErrors.push({path:path.relative(root,file),error:`stat_failed:${String(e?.message??e)}`});continue;}
 scannedBytes+=before.size;
 const meta={path:path.relative(root,file),encoding:file.endsWith('.gz')?'gzip':'identity',size_bytes:before.size,sha256:null,bonsai_rows:0,partial_records:0,partial_program_ids:[],models:{}};
 try{
  if(file.endsWith('.results.jsonl')||file.endsWith('.results.jsonl.gz')){
   const reader=hashedInput(file,{gzip:file.endsWith('.gz')});
   try{
    let recordNo=0;
    for await(const line of createInterface({input:reader.input,crlfDelay:Infinity})){
     recordNo++;if(!line.trim())continue;
     try{consume(JSON.parse(line),file,meta,recordNo);}catch(e){malformed.push({path:path.relative(root,file),record:recordNo,error:String(e?.message??e)});}
    }
    meta.sha256=await reader.finish();
   }finally{await reader.cleanup();}
  }else{
   const bytes=await readFile(file);meta.sha256=hashBytes(bytes);const obj=JSON.parse(bytes.toString('utf8'));consume(obj,file,meta,1);
  }
 }catch(e){malformed.push({path:path.relative(root,file),record:null,error:String(e?.message??e)});}
 let after;
 try{after=await stat(file);}catch(e){if(e.code==='ENOENT'){vanished.push(path.relative(root,file));continue;}scanErrors.push({path:path.relative(root,file),error:`postscan_stat_failed:${String(e?.message??e)}`});continue;}
 if(after.size!==before.size||after.mtimeMs!==before.mtimeMs||after.ctimeMs!==before.ctimeMs||after.ino!==before.ino||after.dev!==before.dev){malformed.push({path:path.relative(root,file),record:null,error:'artifact_changed_during_scan'});continue;}
 if(meta.bonsai_rows||meta.partial_records){if(!meta.sha256){malformed.push({path:path.relative(root,file),record:null,error:'artifact_content_hash_missing'});continue;}meta.scan_root=path.relative(root,metaRootByFile.get(file));artifacts.push(meta);}
}
const catalogSummaries=[];
for(const catalogArg of args['source-catalog']??[]){
 const file=path.resolve(catalogArg);let before;
 try{before=await stat(file);}catch(e){scanErrors.push({path:path.relative(root,file),error:`source_catalog_stat_failed:${String(e?.message??e)}`});continue;}
 const summary={path:path.relative(root,file),size_bytes:before.size,sha256:null,partial_matches:0};
 const reader=hashedInput(file);
 try{for await(const line of createInterface({input:reader.input,crlfDelay:Infinity})){
  if(!line.trim())continue;let row;try{row=JSON.parse(line);}catch(e){malformed.push({path:path.relative(root,file),error:`malformed_source_catalog:${String(e?.message??e)}`});continue;}
  if(!row||typeof row!=='object'||Array.isArray(row)||!row.task?.program_ir||typeof row.task.program_ir!=='object'||typeof row.task.program_ir.id!=='string'||!row.task.program_ir.id){malformed.push({path:path.relative(root,file),error:'invalid_source_catalog_row_shape'});continue;}
  const ir=row?.task?.program_ir;if(!ir||!partialProgramIds.has(ir.id))continue;
  if(ir.source_ids!==undefined&&(!Array.isArray(ir.source_ids)||ir.source_ids.some(x=>typeof x!=='string'||!x))) { malformed.push({path:path.relative(root,file),error:'invalid_partial_catalog_source_ids'});continue; }
  if(ir.external_source?.source_id!==undefined&&(typeof ir.external_source.source_id!=='string'||!ir.external_source.source_id)){malformed.push({path:path.relative(root,file),error:'invalid_partial_catalog_external_source_id'});continue;}
  const ids=[...new Set([...(ir.source_ids??[]),ir.external_source?.source_id].filter(x=>typeof x==='string'&&x))];
  const prior=resolvedPartialPrograms.get(ir.id)??{program_id:ir.id,catalog_rows:[],source_ids:new Set()};
  prior.catalog_rows.push({path:path.relative(root,file),source:ir.source,record_id:row.id??null});for(const id of ids){prior.source_ids.add(id);savedSourceIds.add(id);sourceCounts.set(id,(sourceCounts.get(id)??0)+1);}resolvedPartialPrograms.set(ir.id,prior);summary.partial_matches++;
 }}catch(e){scanErrors.push({path:path.relative(root,file),error:`source_catalog_read_failed:${String(e?.message??e)}`});}
 finally{try{summary.sha256=await reader.finish();}catch(e){scanErrors.push({path:path.relative(root,file),error:`source_catalog_stream_failed:${String(e?.message??e)}`});}await reader.cleanup();}
 try{const after=await stat(file);if(after.size!==before.size||after.mtimeMs!==before.mtimeMs||after.ctimeMs!==before.ctimeMs||after.ino!==before.ino||after.dev!==before.dev)malformed.push({path:path.relative(root,file),error:'source_catalog_changed_during_scan'});}catch(e){if(e.code==='ENOENT')vanished.push(path.relative(root,file));else scanErrors.push({path:path.relative(root,file),error:`source_catalog_postscan_stat_failed:${String(e?.message??e)}`});}
 catalogSummaries.push(summary);
}
const unresolvedPartialProgramIds=[...partialProgramIds].filter(id=>!resolvedPartialPrograms.has(id)).sort();
const resolvedPartialProgramRows=[...resolvedPartialPrograms.values()].map(x=>({program_id:x.program_id,catalog_rows:x.catalog_rows,source_ids:[...x.source_ids].sort()})).sort((a,b)=>a.program_id.localeCompare(b.program_id));
const repeated=(map)=>[...map].filter(([,n])=>n>1).map(([id,count])=>({id,count})).sort((a,b)=>a.id.localeCompare(b.id));
const inventory={
 version:'bonsai-attempt-identity-inventory/4',
 scan:{status:malformed.length||vanished.length||scanErrors.length?'incomplete':'complete_with_partial_ids',started_at:startedAt(),finished_at:new Date().toISOString(),runs_root:runsDir,additional_roots:scanRoots.slice(1),
  method:'Walk regular files under each root matching *.result.json, *.partial.json, *.results.jsonl, or *.results.jsonl.gz; stream JSONL (and gzip) while hashing the same bytes; include records only when provenance.model matches /bonsai/i; inspect record content, not path names. Do not follow symlinks; report non-artifact file links and node_modules/vendor directory links separately, and fail on artifact-named file links, other directory links, or inaccessible link targets.',
  files_scanned:scannedFiles,bytes_scanned:scannedBytes,bonsai_rows:bonsaiRows,matched_artifacts:artifacts.length,malformed_count:malformed.length,vanished_count:vanished.length,scan_error_count:scanErrors.length,skipped_symlink_count:skippedSymlinks.length,
  prior_inventory_sha256:priorInventorySha256,model_counts:modelCounts,source_catalogs:catalogSummaries,partial_program_ids:partialProgramIds.size,partial_program_ids_resolved:resolvedPartialProgramRows.length,partial_program_ids_unresolved:unresolvedPartialProgramIds.length},
 saved_bonsai_program_ids:[...savedProgramIds].sort(),saved_bonsai_source_ids:[...savedSourceIds].sort(),queued_program_ids:[...queuedProgramIds].sort(),queued_source_ids:[...queuedSourceIds].sort(),saved_bonsai_partials:partials.sort((a,b)=>a.path.localeCompare(b.path)),resolved_partial_programs:resolvedPartialProgramRows,unresolved_partial_program_ids:unresolvedPartialProgramIds,
 repeated_bonsai_program_ids:repeated(programCounts),repeated_bonsai_source_ids:repeated(sourceCounts),
 scanned_artifacts:artifacts.sort((a,b)=>a.path.localeCompare(b.path)),malformed,vanished,scan_errors:scanErrors,skipped_symlinks:skippedSymlinks.sort((a,b)=>a.path.localeCompare(b.path))
};
function startedAt(){return new Date(scanStart).toISOString();}
await writeFile(outputPath,JSON.stringify(inventory,null,2)+'\n');
console.log(JSON.stringify({output:outputPath,scan:inventory.scan,saved_program_ids:inventory.saved_bonsai_program_ids.length,saved_source_ids:inventory.saved_bonsai_source_ids.length,queued_program_ids:inventory.queued_program_ids.length,queued_source_ids:inventory.queued_source_ids.length,malformed:malformed.length,vanished:vanished.length},null,2));
if(malformed.length||vanished.length||scanErrors.length)process.exitCode=2;
