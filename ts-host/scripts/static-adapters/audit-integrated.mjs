#!/usr/bin/env node
/** Replay integrated static references, with private metadata removed from the model opening. */
import {open,mkdir,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {jsonLines,sha,fileHash} from './common.mjs';
import {referenceRow} from '../inline-curriculum/references.mjs';
import {admitRow,renderOpening} from '../../dist/teacher/curriculum.js';
import {defaultToolSurfaceHash} from '../../dist/teacher/collector.js';
import {materializeNativeRows} from '../../dist/teacher/native-materializer.js';
import {TOOLS_PROMPT} from '../../dist/native/prompt.js';
const [outPath,...inputs]=process.argv.slice(2);
if(!outPath||!inputs.length)throw Error('usage: audit-integrated.mjs OUT RESULTS...');
const out=resolve(outPath);await mkdir(out,{recursive:true});
const ledger=await open(join(out,'audit.jsonl'),'wx'),approved=await open(join(out,'audited.results.jsonl'),'wx');
const options={modelId:'integrated-static-audit',rootSeed:928,systemPrompt:TOOLS_PROMPT,contextTokens:16384,maxTurns:128,
  toolSurfaceSha256:await defaultToolSurfaceHash(),collectionRole:'reference'};
const summary={version:'natlang.integrated_static_audit/1',model_calls:0,inputs:{},rows:0,admitted:0,held:0,reasons:{},approved_decisions:0,held_decisions:0};
try {
  for(const path of inputs) {
    summary.inputs[resolve(path)]=await fileHash(path);
    for await(const old of jsonLines(path)) {
      summary.rows++;const record=old.task?.program_ir;let reason=null;
      try {
        if(!record||record.version!=='natlang.program/2')throw Error('missing_current_ir');
        const admission=admitRow(old);if(!admission.admitted)throw Error('old_admission:'+admission.reasons.join(','));
        const opening=await renderOpening(record,TOOLS_PROMPT);
        if(record.curriculum.decisive.some(d=>opening.includes(d.marker)))throw Error('private_decisive_marker_in_opening');
        const fresh=await referenceRow(record,summary.rows-1,options);
        const current=admitRow(fresh);if(!current.admitted)throw Error('fresh_replay:'+current.reasons.join(','));
        const materialized=materializeNativeRows([fresh]);
        if(materialized.unlinked.length)throw Error('unlinked_current_training_decision');
        summary.approved_decisions+=materialized.turns.filter(t=>t.training_admission.approved).length;
        summary.held_decisions+=materialized.turns.filter(t=>!t.training_admission.approved).length;
        // Raw historical traces remain immutable; rebuilt results train only current API decisions.
        await approved.write(JSON.stringify(fresh)+'\n');summary.admitted++;
      }catch(e){reason=e.message;summary.held++;summary.reasons[reason]=(summary.reasons[reason]??0)+1;}
      await ledger.write(JSON.stringify({id:record?.id??old.id,source:resolve(path),old_row_sha256:sha(old),admitted:reason===null,reason})+'\n');
      if(summary.rows%100===0)console.error(JSON.stringify({rows:summary.rows,admitted:summary.admitted,held:summary.held}));
    }
  }
}finally{await ledger.close();await approved.close();}
summary.audited_results_sha256=await fileHash(join(out,'audited.results.jsonl'));
await writeFile(join(out,'audit.manifest.json'),JSON.stringify(summary,null,2)+'\n');console.log(JSON.stringify(summary,null,2));
