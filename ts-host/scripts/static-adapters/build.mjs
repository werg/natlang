#!/usr/bin/env node
/** Conservative, model-free legacy recovery. Converted libraries are not admitted results. */
import {open,mkdir,readFile,writeFile,readdir,unlink} from 'node:fs/promises';
import {resolve,join,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {jsonLines,fileHash,sha,requireValue,VERSION} from './common.mjs';
import {decisionLeaf,decisionBatch,decisionSchema} from './decisions.mjs';
import {finqa} from './numeric.mjs';
import {clevr} from './scene.mjs';
import {stateSequence} from './state.mjs';
import {synthetic,leafDefinitions,reviewedLeaf} from './synthetic.mjs';
import {referenceRow} from '../inline-curriculum/references.mjs';
import {admitRow,validateCurriculum,renderOpening} from '../../dist/teacher/curriculum.js';
import {defaultToolSurfaceHash,recordDigest} from '../../dist/teacher/collector.js';
import {materializeNativeRows} from '../../dist/teacher/native-materializer.js';
import {sourceConversionDigest} from '../../dist/teacher/source-conversion.js';
import {TOOLS_PROMPT} from '../../dist/native/prompt.js';

export async function buildRecovered({source,out,joins,pilot=16}) {
  requireValue(Number.isSafeInteger(pilot)&&pilot>=0,'invalid_pilot_limit');
  const adapterDirectory=dirname(fileURLToPath(import.meta.url)), adapterFiles={};
  for(const name of (await readdir(adapterDirectory)).filter(name=>name.endsWith('.mjs')).sort())adapterFiles[name]=await fileHash(join(adapterDirectory,name));
  await mkdir(out,{recursive:true});
  // Exclusive output: preserve evidence from earlier attempts instead of silently replacing it.
  const handles=new Map(), counts={}, rejected={}, sources={}, heldGroups=new Set(), heldInputs=new Set();
  const put=async(name,row)=> {
    if(!handles.has(name)) handles.set(name,await open(join(out,name),'wx'));
    await handles.get(name).write(JSON.stringify(row)+'\n'); counts[name]=(counts[name]??0)+1;
  };
  const reject=async(original,info,error)=> {
    const reason=error.message;rejected[reason]=(rejected[reason]??0)+1;
    await put('conversion-rejections.jsonl',{id:original.id??original.key??null,source:info.source,source_path:info.path,source_sha256:info.sha256,reason,original});
  };
  const joinManifest=JSON.parse(await readFile(join(joins,'source-joins.manifest.json'),'utf8'));
  requireValue(joinManifest.version==='natlang.recovered_source_joins/1','invalid_source_join_manifest');
  for(const entry of [joinManifest.ledger,joinManifest.schemas]) requireValue(await fileHash(join(joins,entry.path))===entry.sha256,'source_join_checksum_mismatch');
  // Recheck all pinned raw inputs. A cached annotation join is invalid after source changes.
  for(const [path,hash] of Object.entries({...joinManifest.inputs,...joinManifest.raw_sources})) requireValue(await fileHash(path)===hash,'raw_source_changed_after_audit');
  const verified=new Map();for await(const row of jsonLines(join(joins,joinManifest.ledger.path))) verified.set(row.id,row.verified);
  const schemas=JSON.parse(await readFile(join(joins,joinManifest.schemas.path),'utf8'));
  const infoFor=async(name,label,license)=> {
    const path=resolve(source,name), hash=await fileHash(path);
    let manifest;try{manifest=JSON.parse(await readFile(path+'.manifest.json','utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
    const expected=manifest?.ir_sha256??manifest?.output_sha256;
    if(expected)requireValue(expected===hash,'preserved_source_checksum_mismatch');
    sources[path]={sha256:hash,source:label,license};return {path,sha256:hash,source:label,license};
  };
  const save=async(record)=> {
    record=JSON.parse(JSON.stringify(record));validateCurriculum(record);
    requireValue(['train','test'].includes(record.split),'invalid_converted_split');
    if(record.split!=='train'){
      for(const group of record.source_groups)heldGroups.add(group);
      heldInputs.add(sha(record.semantics.inputs));
    }
    await put('converted.candidates.jsonl',record);
  };
  try {
    for(const [name,label] of [['nanojev-stage1-curated-tasks.jsonl','NanoJev'],['typed-tasks.jsonl','TypedDecisions'],['sales-spread-labeled.jsonl','Sales']]) {
      const info=await infoFor(name,label,'source-record-required'),batches=new Map();
      for await(const row of jsonLines(info.path)) {
        try {
          const leaf=decisionLeaf(row,info);await save(leaf);
          if(leaf.external_source.quality.status==='eligible') {
            const key=decisionSchema(row),group=batches.get(key)??[];group.push(row);
            if(group.length===4){await save(decisionBatch(group,info));batches.delete(key);}else batches.set(key,group);
          }
        }catch(e){await reject(row,info,e);}
      }
      for(const group of batches.values()) if(group.length>=2)try{await save(decisionBatch(group,info));}catch(e){await reject(group[0],info,e);}
    }
    for(const split of ['train','dev']) {
      const info=await infoFor(`finqa/dataset/${split}.json`,'FinQA','CC-BY-4.0');
      for(const row of JSON.parse(await readFile(info.path,'utf8')))try{await save(finqa(row,info,split));}catch(e){await reject(row,info,e);}
    }
    for(const [name,label] of [['scone-v1.ir.jsonl','SCONE'],['scone-dev-v1.ir.jsonl','SCONE'],['sgd-v2.ir.jsonl','SGD'],['sgd-dev-v2.ir.jsonl','SGD'],['clevr-v1.ir.jsonl','CLEVR'],['clevr-val-v1.ir.jsonl','CLEVR']]) {
      const info=await infoFor(name,label,'source-record-required');
      for await(const row of jsonLines(info.path))try{
        const sourceVerified=verified.get(row.id)===true;
        requireValue(sourceVerified,'raw_annotation_join_failed');
        const record=label==='CLEVR'?clevr(row,info):stateSequence(row,info,{sourceVerified,schema:schemas[row.split]?.[row.semantics.domain]});
        record.external_source.quality.checks.push('pinned_raw_annotation_join');await save(record);
      }catch(e){await reject(row,info,e);}
    }
    const synthInfo=await infoFor('synthetic-all-complete-702a03e8.ir.jsonl','LegacySynthetic','project-generated');
    const syntheticRows=[];
    for await(const row of jsonLines(synthInfo.path)){syntheticRows.push(row);try{await save(synthetic(row,synthInfo));}catch(e){await reject(row,synthInfo,e);}}
    const definitions=leafDefinitions(syntheticRows),leafPath=resolve(source,'..','leaf_references.jsonl');
    const leafInfo={path:leafPath,source:'LegacyLeafBank',sha256:await fileHash(leafPath),license:'project-generated'};sources[leafPath]=leafInfo;
    for await(const row of jsonLines(leafPath))try{await save(reviewedLeaf(row,leafInfo,definitions));}catch(e){await reject(row,leafInfo,e);}
    await handles.get('converted.candidates.jsonl')?.close();handles.delete('converted.candidates.jsonl');
    const options={modelId:'recovered-static-reference',rootSeed:928,systemPrompt:TOOLS_PROMPT,contextTokens:16384,maxTurns:128,
      toolSurfaceSha256:await defaultToolSurfaceHash(),collectionRole:'reference'};
    const sampled={},admitted=[],rows=[],qualityCounts={},bySource={};
    for await(const record of jsonLines(join(out,'converted.candidates.jsonl'))) {
      const q=record.external_source.quality;
      if(record.split==='train'&&(record.source_groups.some(g=>heldGroups.has(g))||heldInputs.has(sha(record.semantics.inputs)))) {
        q.status='held';q.checks.push('source_group_or_input_overlaps_holdout');
      }
      const destination=q.status!=='eligible'?'held.ir.jsonl':record.split==='train'?'converted.train.ir.jsonl':'converted.test.ir.jsonl';
      await put(destination,record);qualityCounts[destination]=(qualityCounts[destination]??0)+1;
      if(destination!=='converted.train.ir.jsonl')continue;
      // Sample independently by shape/family so short leaf answers cannot exhaust the composed pilot.
      const key=record.source+':'+record.curriculum.family;
      if((sampled[key]??0)>=pilot)continue;sampled[key]=(sampled[key]??0)+1;
      try {
        const opening=await renderOpening(record,TOOLS_PROMPT);
        requireValue(!record.curriculum.decisive.some(item=>opening.includes(item.marker)),'private_reference_in_opening');
        const row=JSON.parse(JSON.stringify(await referenceRow(record,rows.length,options)));
        const admission=admitRow(row);requireValue(admission.admitted,`native_admission:${admission.reasons.join(',')}`);
        requireValue(!(row.outcome.action_ledger??[]).some(e=>['error','refused','rejected'].includes(e.outcome)),'native_action_failed');
        row.provenance.source_conversion={version:'natlang.source_static_conversion/1',adapter:VERSION,source:record.source,
          source_ids:record.source_ids,source_revisions:record.source_revisions,license:record.license,source_snapshot_sha256:record.external_source.snapshot_sha256,
          program_ir_sha256:recordDigest(record),native_replay_accepted:true,native_outcome_sha256:sourceConversionDigest(row.outcome),
          native_trajectory_sha256:sourceConversionDigest(row.trajectory),conversion_scope:'source_task_reference',source_success:null,whole_issue_replayed:false,
          original_trajectory_id:null,original_row_sha256:record.external_source.original_row_sha256};
        requireValue(admitRow(row).admitted,'sealed_source_admission_failed');
        const materialized=materializeNativeRows([row]);requireValue(!materialized.unlinked.length,'unlinked_training_decision');
        // Leaf answers remain reviewable evidence; the default reasoning lane admits no fabricated derivation.
        requireValue(materialized.turns.some(t=>t.training_admission.approved),'no_approved_training_decisions');
        rows.push(row);admitted.push(record);bySource[record.source]=(bySource[record.source]??0)+1;
      }catch(e){await put('replay-rejections.jsonl',{id:record.id,source:record.source,reason:e.message});}
    }
    await unlink(join(out,'converted.candidates.jsonl'));
    for(const record of admitted)await put('train.ir.jsonl',record);
    for(const row of rows)await put('static.results.jsonl',row);
    const materialized=materializeNativeRows(rows);for(const turn of materialized.turns)await put('static.turns.jsonl',turn);
    for(const h of handles.values())await h.close();handles.clear();
    const entry=async(name)=>({path:name,rows:counts[name]??0,sha256:await fileHash(join(out,name))});
    // Keep empty result files valid as audit evidence, but publish no ready bundle without admitted cases.
    for(const name of ['train.ir.jsonl','static.results.jsonl'])if(!counts[name])await writeFile(join(out,name),'',{flag:'wx'});
    for(const [name,hash] of Object.entries(adapterFiles))requireValue(await fileHash(join(adapterDirectory,name))===hash,'adapter_changed_during_build');
    for(const [path,entry] of Object.entries(sources))requireValue(await fileHash(path)===entry.sha256,'source_changed_during_build');
    const report={version:'natlang.recovered_source_build/1',adapter_files_sha256:adapterFiles,adapter:VERSION,model_calls:0,source_inputs:sources,
      annotation_join_manifest_sha256:await fileHash(join(joins,'source-joins.manifest.json')),converted:qualityCounts,rejection_counts:rejected,pilot_attempts:sampled,
      cases:rows.length,by_source:bySource,training_decisions:materialized.turns.filter(t=>t.training_admission.approved).length,
      held_decisions:materialized.turns.filter(t=>!t.training_admission.approved).length,ir:await entry('train.ir.jsonl'),results:await entry('static.results.jsonl')};
    await writeFile(join(out,'recovery.manifest.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
    if(rows.length)await writeFile(join(out,'static.manifest.json'),JSON.stringify({...report,version:'natlang.source_static_bundle/1'},null,2)+'\n',{flag:'wx'});
    return report;
  }finally{for(const h of handles.values())await h.close();}
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  const {values}=parseArgs({options:{source:{type:'string',default:'../data/external_pilot'},out:{type:'string',default:'../data/teacher/recovered'},joins:{type:'string'},pilot:{type:'string',default:'16'}}});
  requireValue(values.joins,'--joins is required');console.log(JSON.stringify(await buildRecovered({source:resolve(values.source),out:resolve(values.out),joins:resolve(values.joins),pilot:Number(values.pilot)}),null,2));
}
