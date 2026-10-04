import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, writeFile, cp, mkdtemp, rm, access} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve, join} from 'node:path';
import {staticBundleInput} from '../scripts/inline-curriculum/static-bundle-input.mjs';

const base=resolve('runs/self-improvement-expansion-20261004');
const pilot=join(base,'nllb-seed-native-pilot-v1');
const sha=body=>createHash('sha256').update(body).digest('hex');

test('pinned NLLB bundle rejects substituted answers, inputs, identities and missing policy after checksums are updated', async t=>{
  try {await access(join(pilot,'static.manifest.json'));}
  catch {t.skip('pinned source-backed pilot is not in this development cache');return;}
  const variants=[
    ['answer',row=>{row.task.program_ir.semantics.expected='substituted answer';}],
    ['input',row=>{row.task.program_ir.semantics.inputs.source_text='substituted input';}],
    ['identity',row=>{row.task.program_ir.source_ids=['unrelated-source-task'];}],
    ['visibility',row=>{for(const step of row.trajectory)for(const message of step.context)
      if(message.role==='user'||message.role==='tool')message.content='source input removed';}],
    ['policy',null],
  ];
  for(const [name,mutate] of variants)await t.test(name,async()=>{
    const out=await mkdtemp(join(base,'nllb-admission-negative-'));
    try {
      await cp(pilot,out,{recursive:true});
      const manifest=JSON.parse(await readFile(join(out,'static.manifest.json'),'utf8'));
      if(mutate){
        const rows=(await readFile(join(out,manifest.results.path),'utf8')).trim().split('\n').map(JSON.parse);
        mutate(rows[0]);
        const results=rows.map(JSON.stringify).join('\n')+'\n';
        const ir=rows.map(row=>JSON.stringify(row.task.program_ir)).join('\n')+'\n';
        await writeFile(join(out,manifest.results.path),results);
        await writeFile(join(out,manifest.ir.path),ir);
        manifest.results.sha256=sha(results);manifest.ir.sha256=sha(ir);
      }else{delete manifest.nllb_reference_policy;delete manifest.source_answer_policy;}
      await writeFile(join(out,'static.manifest.json'),JSON.stringify(manifest));
      await assert.rejects(staticBundleInput(join(out,'static.manifest.json')),/nllb_(static_reference_binding_failed|reference_policy_missing)/);
    }finally{await rm(out,{recursive:true,force:true});}
  });
});
