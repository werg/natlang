import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { exactObjectiveBounds } from '../dist/skills/objective.js';

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
 const bad=audit([{id:'malformed'},episode(),episode()]);assert.equal(bad.status,1);
 assert.ok(bad.report.errors.some(error=>error.code==='episode-version'));
 assert.ok(bad.report.errors.some(error=>error.code==='duplicate_episode'));
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
