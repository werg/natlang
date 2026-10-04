import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { exactObjectiveBounds } from '../dist/skills/objective.js';
import { createHash } from 'node:crypto';

function episode(id='a') {
 return {version:'natlang.skill-episode/1',id,family:'fixture',split:'train',license:'project-generated',
 source_groups:[id+'-s',id+'-q'],target:{kind:'improvement-case',entry:'solve.nl',source:{schema:'fixture',id},
 files:{'solve.nl':'---\nargs:\n  value: number\nreturns: number\n---\nIncrement the value.\n'}},
 library:{kind:'empty',skills:{}},support:{cases:[{id:id+'-s',group:id+'-s',args:[1],expected:2}]},
 query:{cases:[{id:id+'-q',group:id+'-q',args:[3],expected:4}]},operations:['create'],limits:{maxSteps:1},provenance:{}};
}
function audit(rows) {
 const dir=mkdtempSync(join(tmpdir(),'skill-audit-'));
 try {
  const input=join(dir,'episodes.jsonl');writeFileSync(input,rows.map(JSON.stringify).join('\n')+'\n');
  const run=spawnSync(process.execPath,['ts-host/scripts/skills/audit-episodes.mjs',input,'--out',join(dir,'report.json')],{encoding:'utf8'});
  assert.equal(run.signal,null);return {status:run.status,report:JSON.parse(readFileSync(join(dir,'report.json'),'utf8'))};
 } finally {rmSync(dir,{recursive:true,force:true});}
}
test('audit preserves role-consistent configuration variants and rejects cross-role input aliases',()=>{
 const a=episode(),b=episode('b');
 assert.equal(audit([a,b]).status,0);
 b.query.cases[0].args=[1];
 const bad=audit([a,b]);assert.equal(bad.status,1);
 assert.ok(bad.report.errors.some(error=>error.code==='input_alias_boundary_collision'));
});
test('audit reports schema and duplicate IDs without dereferencing invalid episode fields',()=>{
 const malformed=episode('nested-malformed');malformed.support.cases=[null];
 const bad=audit([null,{id:'malformed'},malformed,episode(),episode()]);assert.equal(bad.status,1);
 assert.ok(bad.report.errors.some(error=>error.code==='episode-version'));
 assert.ok(bad.report.errors.some(error=>error.code==='duplicate_episode'));
 assert.ok(bad.report.errors.some(error=>error.code==='episode-validator-error'));
});
test('audit independently rejects incorrect optimization reference bounds',()=>{
 const a=episode();a.provenance.metric={schema:'natlang.skill-objective/1',kind:'knapsack'};
 for(const [role,capacity] of [['support',1],['query',2]]) {
  const instance={capacity,items:[{id:'x',weight:1,value:2},{id:'y',weight:2,value:3}]};
  a[role].cases[0].args=[JSON.stringify(instance)];a[role].cases[0].expected=exactObjectiveBounds('knapsack',instance);
 }
 assert.equal(audit([a]).report.independent_objective_references_checked,2);
 a.query.cases[0].expected.best=999;
 const bad=audit([a]);assert.equal(bad.status,1);
 assert.ok(bad.report.errors.some(error=>error.code==='independent_reference_error'));
});

function auditWith(rows, manifest) {
 const dir=mkdtempSync(join(tmpdir(),'skill-gate-'));
 try {
  const input=join(dir,'episodes.jsonl'),body=rows.map(JSON.stringify).join('\n')+'\n';writeFileSync(input,body);
  if(manifest) writeFileSync(join(dir,'episodes.manifest.json'),JSON.stringify(manifest(body)));
  const run=spawnSync(process.execPath,['ts-host/scripts/skills/audit-episodes.mjs',input,'--out',join(dir,'report.json')],{encoding:'utf8'});
  return {status:run.status,report:JSON.parse(readFileSync(join(dir,'report.json'),'utf8'))};
 } finally {rmSync(dir,{recursive:true,force:true});}
}
function choiceEpisode(id, answers) {
 const e=episode(id), metric={schema:'natlang.skill-graded/1',kind:'choice-brier'};
 e.target.files['solve.nl']='---\nargs: { question: string }\nreturns: Record<string, number>\n---\nAnswer.\n';
 const row=(role,i,answer)=>({id:`${id}-${role}${i}`,group:`${id}-${role}${i}`,args:[`${role} question ${i}`],
   expected:{kind:'choice',answer,options:['A','B','C']}});
 e.support.cases=[row('s',0,'A'),row('s',1,'B')];e.query.cases=answers.map((answer,i)=>row('q',i,answer));
 e.provenance={metric};return e;
}
test('the gate checks manifests, target loading and that gold outputs reach the best score',()=>{
 const good=auditWith([episode()],body=>({episodes:1,sha256:createHash('sha256').update(body).digest('hex')}));
 assert.equal(good.status,0);assert.equal(good.report.targets_loaded,1);
 const stale=auditWith([episode()],()=>({episodes:2,sha256:'0'.repeat(64)}));
 assert.ok(stale.report.errors.some(e=>e.code==='manifest_sha_mismatch'));
 assert.ok(stale.report.errors.some(e=>e.code==='manifest_count_mismatch'));
 const broken=episode();broken.target.files['solve.nl']='---\nargs: { value: Missing }\nreturns: number\n---\nx\n';
 assert.ok(auditWith([broken]).report.errors.some(e=>e.code==='target_load_error'));
 const choice=choiceEpisode('c',['B','A']);
 const passed=auditWith([choice]);assert.equal(passed.status,0);assert.equal(passed.report.gold_outputs_checked,2);
 choice.query.cases[0].expected.answer='Z';  // an answer outside the options: the reference cannot be scored
 assert.ok(auditWith([choice]).report.errors.some(e=>e.code==='gold_not_best'));
});
test('the gate warns when a constant answer from support already scores on the query',()=>{
 const constant=auditWith([choiceEpisode('k',['A','A','A'])]);
 assert.equal(constant.status,0);
 assert.ok(constant.report.warnings.some(w=>w.code==='constant_answer_baseline'&&w.quality===1));
 assert.ok(!auditWith([choiceEpisode('v',['A','B','C'])]).report.warnings.some(w=>w.code==='constant_answer_baseline'));
});
