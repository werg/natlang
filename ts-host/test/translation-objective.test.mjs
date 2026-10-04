import test from 'node:test';
import assert from 'node:assert/strict';
import {evaluateExpression,scoreSpecifiedTranslation} from '../dist/skills/translation-objective.js';
import {createTranslationEpisodes} from '../scripts/skills/build-translation-episodes.mjs';
test('translation preserves exact arithmetic and rejects invalid nodes',()=>{
 const language={operators:{minus:'subtract'}};
 const task={schema:'natlang.skill-translation/1',id:'t',revision:'1',language,tests:[{input:{x:2,y:3},expected:-1}]};
 assert.equal(scoreSpecifiedTranslation(task,{op:'minus',args:[{variable:'x'},{variable:'y'}]}).quality,1);
 assert.equal(scoreSpecifiedTranslation(task,{op:'minus',args:[{variable:'y'},{variable:'x'}]}).quality,0);
 assert.equal(scoreSpecifiedTranslation(task,{constant:-1,extra:true}).quality,0);
 assert.throws(()=>scoreSpecifiedTranslation({...task,language:{operators:{a:'invented'}}},{constant:0}),/invalid language/);
 assert.throws(()=>evaluateExpression({constant:1001},{x:0,y:0},language));
});
test('translation public JS source agrees independently with every reference input',()=>{
 const episodes=createTranslationEpisodes();assert.equal(episodes.length,54);
 const roles=new Map();let checks=0;
 for(const e of episodes)for(const role of ['support','query'])for(const row of e[role].cases){
  if(roles.has(row.group))assert.equal(roles.get(row.group),role);else roles.set(row.group,role);
  const publicTask=JSON.parse(row.args[0]);
  if(publicTask.source_language==='javascript'){
   const solve=Function(publicTask.source_program+';return solve;')();
   for(const {input,expected} of row.expected.tests){assert.equal(solve(input.x,input.y),expected);checks++;}
  }
 }
 assert.equal(checks,3969);
});

test('public Python source agrees with reference behavior independently',async()=>{
 const {spawnSync}=await import('node:child_process');
 const rows=createTranslationEpisodes({replicas:1,variants:['empty']}).filter(e=>e.family.includes('python')).flatMap(e=>[...e.support.cases,...e.query.cases]);
 // These are project-authored fixtures, never model-produced source.
 const script='import json,sys\nfor row in json.load(sys.stdin):\n ns={}\n exec(row["source"],ns)\n for t in row["tests"]:\n  assert ns["solve"](**t["input"])==t["expected"]\nprint("passed")\n';
 const result=spawnSync('python3',['-I','-c',script],{input:JSON.stringify(rows.map(row=>({source:JSON.parse(row.args[0]).source_program,tests:row.expected.tests}))),encoding:'utf8'});
 assert.equal(result.status,0,result.stderr);assert.equal(result.stdout.trim(),'passed');
});
