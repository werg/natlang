import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm,readdir} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {gunzipSync} from 'node:zlib';
import {spawnSync} from 'node:child_process';
const root=process.env.NATLANG_ARCHIVED_REASSESSMENT_ROOT;
const repo=resolve(import.meta.dirname,'../..');
async function archived(kind){
 const name=kind==='presentation'?'open-goal-present-local-tool-station-equipment':'open-goal-public-brief-print-example-without-domain-purchase';
 const directory=join(root,'live-wave0027/cases',name);
 const run=(await readdir(directory)).find(item=>item.startsWith('st_'));
 return {directory,path:join(directory,run,`trajectory-${name}-full.json.gz`)};
}
for(const kind of ['presentation','publicbrief'])test(`actual archived ${kind} reassessment, no teacher rerun`,{skip:!root},async()=>{
 const work=await mkdtemp(join(tmpdir(),'native-outcome-reassessment-'));
 try{
  const actual=await archived(kind);
  const command=[join(repo,`scripts/neuralese_data/reassess_${kind==='presentation'?'local_presentation':'public_brief'}.mjs`),
   '--audit-root',root,'--packet',actual.path,'--pins',join(actual.directory,'source.json'),
   '--catalog',join(root,'../catalog.jsonl'),'--out',join(work,'assessment')];
  if(kind==='publicbrief')command.push('--review',join(root,'reassessment-wave27-publicbrief/review.json'));
  const valid=spawnSync(process.execPath,command,{encoding:'utf8'});assert.equal(valid.status,0,valid.stderr);
  assert.equal(JSON.parse(valid.stdout).teacher_rerun,false);
  const packet=JSON.parse(gunzipSync(await readFile(actual.path)));
  if(kind==='presentation')packet.execution.diagnostics.openGoalVerification.report.views[0].rendered.mdx='forged MDX';
  else packet.execution.diagnostics.openGoalVerification.assignedPublicBrief.shared.reads[0].page.head+=' invented source bytes';
  const altered=join(work,'forged.json');await writeFile(altered,JSON.stringify(packet));
  command[command.indexOf('--packet')+1]=altered;command[command.indexOf('--out')+1]=join(work,'rejected');
  const rejected=spawnSync(process.execPath,command,{encoding:'utf8'});assert.notEqual(rejected.status,0);
  assert.match(rejected.stderr,kind==='presentation'?/rejects actual presentation/:/hash-mismatched/);
 }finally{await rm(work,{recursive:true,force:true})}
});

for(const [suffix,expected]of [['asset-integrity-deleted-legal-fixture',18],['tendril-route-capture-supply-label',3]])
 test(`retained ${suffix} read subtrees require native result evidence, not done status`,{skip:!root},async()=>{
  const work=await mkdtemp(join(tmpdir(),'native-prefix-reassessment-'));
  try{
   const directory=join(root,'live-wave0027/cases','open-goal-'+suffix);
   const run=(await readdir(directory)).find(item=>item.startsWith('st_'));
   const original=join(directory,run,`trajectory-open-goal-${suffix}-full.json.gz`);
   const command=[join(repo,'scripts/neuralese_data/assess_readonly_subtrees.mjs'),'--packet',original,
    '--catalog',join(root,'../catalog.jsonl'),'--out',join(work,'valid')];
   const valid=spawnSync(process.execPath,command,{encoding:'utf8'});assert.equal(valid.status,0,valid.stderr);
   assert.equal(JSON.parse(valid.stdout).subtrees,expected);
   const packet=JSON.parse(gunzipSync(await readFile(original)));
   const native=packet.execution.diagnostics.natlangTeacher;
   const assessment=JSON.parse(await readFile(join(work,'valid/assessment.json'),'utf8'));
   const target=native.calls.find(call=>call.callId===assessment.invocations[0].call_id);
   assert.notEqual(target.record.parent_call_id,null);
   target.record.output.hash='0'.repeat(64);assert.equal(target.record.outcome,'done');
   const altered=join(work,'forged.json');await writeFile(altered,JSON.stringify(packet));
   command[command.indexOf('--packet')+1]=altered;command[command.indexOf('--out')+1]=join(work,'unproved');
   const rejected=spawnSync(process.execPath,command,{encoding:'utf8'});assert.equal(rejected.status,0,rejected.stderr);
   const remaining=JSON.parse(await readFile(join(work,'unproved/assessment.json'),'utf8'));
   assert.ok(!remaining.invocations.some(row=>row.call_id===target.callId));
  }finally{await rm(work,{recursive:true,force:true})}
 });
