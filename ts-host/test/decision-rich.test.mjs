import assert from 'node:assert/strict';
import {test} from 'node:test';
import {decisionSkillCatalog,decisionExtractChain} from '../scripts/inline-curriculum/decision-rich.mjs';
import {verifyCases} from '../dist/teacher/curriculum.js';
import {referenceRow} from '../scripts/inline-curriculum/references.mjs';
import {materializeNativeRows} from '../dist/teacher/native-materializer.js';
import {TOOLS_PROMPT} from '../dist/native/prompt.js';
test('rich decision fixtures isolate held-out facts and retain real catalog bodies',()=>{
 for(const build of [decisionSkillCatalog,decisionExtractChain]){
  const train=new Set(Array.from({length:72},(_,i)=>build(7202,i,'train')[0].source_groups[0]));
  const held=new Set(Array.from({length:48},(_,i)=>build(7203,i,'test')[0].source_groups[0]));
  assert.equal([...held].some(g=>train.has(g)),false);
 }
 const record=decisionSkillCatalog(7202,0)[0];
 assert.equal(record.generation.skill_catalog.paths.length,3);
 assert.equal(Object.keys(record.generation.skill_catalog.sha256).length,3);
});
test('deep three-branch extraction runs in real scope and exports observed parents',async()=>{
 const [record]=decisionExtractChain(7202,71);
 const [check]=await verifyCases([record],TOOLS_PROMPT);assert.deepEqual(check.problems,[]);
 const row=await referenceRow(record,0,{modelId:'static-proof',rootSeed:7202,systemPrompt:TOOLS_PROMPT,contextTokens:65536,maxTurns:60,collectionRole:'reference',authoredActionPlans:true,followCutoffPages:true,followEvalCutoffPages:true});
 assert.equal(row.outcome.accepted,true);
 const native=materializeNativeRows([row],{directAnswers:true});assert.equal(native.unlinked.length,0);
 const root=row.outcome.invocation_ledger.find(x=>x.parent_invocation_id===null);
 assert.equal(row.outcome.invocation_ledger.filter(x=>x.parent_invocation_id===null).length,1);
 assert.ok(row.outcome.invocation_ledger.some(x=>x.parent_invocation_id===root.invocation_id));
 assert.ok(native.turns.some(x=>x.source_ref.parent_invocation_id));
 assert.ok(native.turns.every(x=>x.training_admission.approved));
});
