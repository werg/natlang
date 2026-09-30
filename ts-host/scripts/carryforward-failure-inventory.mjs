#!/usr/bin/env node
// Copy an immutable failure-candidate inventory into a recipe run and write a hash-bound receipt.
import {readFile, mkdir, open, link, unlink} from 'node:fs/promises';
import {createReadStream, createWriteStream} from 'node:fs';
import {Transform} from 'node:stream';
import {pipeline} from 'node:stream/promises';
import {dirname, resolve} from 'node:path';
import {createHash, randomUUID} from 'node:crypto';
import {parseArgs} from 'node:util';

const {values,positionals}=parseArgs({allowPositionals:true,options:{}});
const [manifestArg,outputArg]=positionals;
if(!manifestArg||!outputArg)throw Error('usage: carryforward-failure-inventory.mjs FAILURE_MANIFEST OUTPUT_GZIP');
const sourceManifestPath=resolve(manifestArg),output=resolve(outputArg);
const sourceManifestBytes=await readFile(sourceManifestPath);
const sourceManifestSha256=createHash('sha256').update(sourceManifestBytes).digest('hex');
const sourceManifest=JSON.parse(sourceManifestBytes.toString('utf8'));
if(sourceManifest.version!=='natlang.teacher_failure_inventory/1'||sourceManifest.negative_labels_assigned!==false||sourceManifest.automatic_pair_generation!==false)
  throw Error('Unsupported or unsafe failure inventory manifest');
const source=resolve(sourceManifest.artifact.path),digestHash=createHash('sha256');
await mkdir(dirname(output),{recursive:true});
const artifactStaging=output+'.'+randomUUID()+'.tmp';
const digestStream=new Transform({transform(chunk,encoding,callback){digestHash.update(chunk);callback(null,chunk);}});
try{await pipeline(createReadStream(source),digestStream,createWriteStream(artifactStaging,{flags:'wx'}));}
catch(error){await unlink(artifactStaging).catch(()=>{});throw error;}
const digest=digestHash.digest('hex');
if(digest!==sourceManifest.artifact.sha256){await unlink(artifactStaging);throw Error('Failure inventory artifact hash mismatch');}
const completed=await open(artifactStaging,'r+');try{await completed.sync();}finally{await completed.close();}
try{await link(artifactStaging,output);}catch(error){
  if(error.code!=='EEXIST'){await unlink(artifactStaging);throw error;}
  const existingHash=createHash('sha256');for await(const chunk of createReadStream(output))existingHash.update(chunk);
  if(existingHash.digest('hex')!==digest){await unlink(artifactStaging);throw Error('Existing run failure inventory has different bytes');}
}
await unlink(artifactStaging);
const receipt={version:'natlang.teacher_failure_inventory_carryforward/1',
  source_manifest:{path:sourceManifestPath,sha256:sourceManifestSha256},
  artifact:{path:output,sha256:digest,candidates:sourceManifest.candidates,content_sha256:sourceManifest.artifact.content_sha256},
  positive_snapshot:sourceManifest.positive_snapshot,source_ledger:sourceManifest.source_ledger,
  policy_identity:sourceManifest.policy_identity,negative_labels_assigned:false,automatic_pair_generation:false};
const receiptPath=output+'.manifest.json',receiptStaging=receiptPath+'.'+randomUUID()+'.tmp',handle=await open(receiptStaging,'wx');
try{await handle.writeFile(JSON.stringify(receipt,null,2)+'\n');await handle.sync();}finally{await handle.close();}
try{await link(receiptStaging,receiptPath);}catch(error){
  if(error.code!=='EEXIST')throw error;
  const currentBytes=await readFile(receiptPath,'utf8'),current=JSON.parse(currentBytes);
  if(JSON.stringify(current)!==JSON.stringify(receipt))throw Error('Existing carryforward receipt conflicts');
}finally{await unlink(receiptStaging);}
console.log(JSON.stringify(receipt));
