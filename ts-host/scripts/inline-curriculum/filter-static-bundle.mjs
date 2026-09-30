#!/usr/bin/env node
/** Preserve a published bundle while applying newly adjudicated source/evaluation holds. */
import {open, readFile, writeFile, rename} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {dirname, resolve, join} from 'node:path';
import {parseArgs} from 'node:util';
import {jsonlRows, fileDigest} from '../jsonl-stream.mjs';
import {digest, safePath} from './directory-sources.mjs';
import {admitRow} from '../../dist/teacher/curriculum.js';
import {materializeNativeRows} from '../../dist/teacher/native-materializer.js';
import {staticBundleInput} from './static-bundle-input.mjs';

const {values} = parseArgs({options:{manifest:{type:'string'},revision:{type:'string'},publish:{type:'boolean'}}});
if (!values.manifest || !/^[a-z0-9-]+$/.test(values.revision ?? ''))
  throw Error('usage: filter-static-bundle.mjs --manifest FILE --revision quality-v3 [--publish]');
const manifestPath=resolve(values.manifest), root=dirname(manifestPath), revision=values.revision;
const priorBytes=await readFile(manifestPath,'utf8'), prior=JSON.parse(priorBytes);
if(prior.version!=='natlang.source_static_bundle/1'||prior.results?.rows!==prior.cases)throw Error('unsupported_static_manifest');
const priorHash=await fileDigest(manifestPath);
const paths=Object.fromEntries(['ir','results','turns'].map(key=>[key,join(root,safePath(prior[key].path))]));
for(const key of ['ir','results','turns'])if(await fileDigest(paths[key])!==prior[key].sha256)throw Error('input_checksum_mismatch:'+key);
const streams={};
for(const [key,name] of Object.entries({ir:'train.ir.jsonl',results:'static.results.jsonl',turns:'static.turns.jsonl',ledger:'holds.jsonl'})){
  const path=join(root,`${revision}.${name}`);
  streams[key]={path,handle:await open(path,'wx'),hash:createHash('sha256'),rows:0};
}
const emit=async(key,row)=>{const bytes=JSON.stringify(row)+'\n', stream=streams[key];await stream.handle.writeFile(bytes);stream.hash.update(bytes);stream.rows++;};
const records=jsonlRows(paths.ir), bySource={}, reasons={}, held=[];
let scanned=0, approvedDecisions=0, heldDecisions=0;
try {
  for await(const row of jsonlRows(paths.results)){
    const ir=await records.next();if(ir.done||digest(ir.value)!==digest(row.task?.program_ir))throw Error('ir_result_mismatch');
    scanned++;
    const admission=admitRow(row);
    if(!admission.admitted){
      if(!admission.reasons.length||admission.reasons.some(reason=>!['source_review_pending','held_out_reserved_curriculum'].includes(reason)))
        throw Error('unreviewed_admission_change:'+JSON.stringify(admission));
      const item={trajectory_id:row.id,program_id:ir.value.id,source_ids:ir.value.source_ids,source_groups:ir.value.source_groups,
        reasons:admission.reasons,disposition:'preserved_source_or_evaluation_hold',dpo_negative_eligible:false};
      held.push(item);await emit('ledger',item);
      for(const reason of admission.reasons)reasons[reason]=(reasons[reason]??0)+1;
      continue;
    }
    const materialized=materializeNativeRows([row]);
    if(materialized.rejectedRows||materialized.unlinked.length)
      throw Error('materialization_not_fully_linked:'+row.id);
    await emit('ir',ir.value);await emit('results',row);
    for(const turn of materialized.turns){
      if(turn.training_admission.approved)approvedDecisions++;else heldDecisions++;
      await emit('turns',turn);
    }
    bySource[ir.value.source]=(bySource[ir.value.source]??0)+1;
  }
  if(!(await records.next()).done||scanned!==prior.cases||scanned!==prior.ir.rows)throw Error('row_count_mismatch');
} finally {
  await records.return();
  for(const stream of Object.values(streams)){await stream.handle.sync();await stream.handle.close();}
}
if(!held.length)throw Error('no_holds_require_publication');
const identity=key=>({path:streams[key].path.slice(root.length+1),sha256:streams[key].hash.digest('hex'),rows:streams[key].rows});
const next={...prior,current_publication_revision:revision,cases:streams.ir.rows,by_source:bySource,ir:identity('ir'),results:identity('results'),turns:identity('turns'),
  training_decisions:approvedDecisions,held_decisions:heldDecisions,
  quality_filter:{revision,prior_manifest_sha256:priorHash,prior_cases:prior.cases,held_cases:held.length,reasons,ledger:identity('ledger'),
    policy:'Only explicit source-review/reserved-evaluation holds; original inputs, golds and trajectories unchanged; no DPO negatives.'},
  publication:{...prior.publication,version:revision,previous_manifest:`${revision}.prior.manifest.json`,supersedes_manifest_sha256:priorHash,
    final_student_audit:'pending'}};
const candidate=join(root,`${revision}.static.manifest.json`);
await writeFile(candidate,JSON.stringify(next,null,2)+'\n',{flag:'wx'});
await staticBundleInput(candidate);
await writeFile(join(root,`${revision}.prior.manifest.json`),priorBytes,{flag:'wx'});
if(values.publish){
  if(await fileDigest(manifestPath)!==priorHash)throw Error('canonical_manifest_changed');
  const staged=manifestPath+`.${revision}.pending`;
  await writeFile(staged,JSON.stringify(next,null,2)+'\n',{flag:'wx'});await rename(staged,manifestPath);
}
console.log(JSON.stringify({manifest:manifestPath,published:!!values.publish,scanned,kept:next.cases,held:held.length,decisions:next.training_decisions,reasons}));
