import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { collectBatch } from '../dist/teacher/collector.js';
import { generationHoldReason, quarantineReason } from '../dist/teacher/curriculum-policy.js';
const record = {version:'natlang.program/2',id:'qasper-fixture',kind:'lambda_source',source:'qasper',split:'train',
  source_ids:['paper:q'],source_groups:['paper'],license:'fixture',semantics:{root:'main.nl',files:{'main.nl':'---\nargs: {}\nreturns: string\n---\nRead.\n'},inputs:{},expected:'answer'}};
test('QASPER live generation pauses while static references and prior valid samples remain eligible', async () => {
  assert.equal(generationHoldReason(record), 'awaiting_extractive_equivalence_oracle');
  assert.equal(quarantineReason(record), undefined);
  assert.equal(generationHoldReason({...record,source:'commitpack'}), undefined);
  const dir=await mkdtemp(join(tmpdir(),'generation-hold-'));
  const config={jobs:join(dir,'jobs'),output:join(dir,'results.jsonl'),workers:1,modelId:'fixture',rootSeed:1,systemPrompt:'fixture',contextTokens:16384,toolSurfaceSha256:'fixture'};
  let calls=0;
  const runner=async (item,provenance)=>{calls++;return {task:{program_ir:item.record},provenance,outcome:{accepted:true}};};
  assert.deepEqual((await collectBatch([{index:0,record}],config,runner)).missing,[0]);
  assert.equal(calls,0);
  assert.equal(JSON.parse(await readFile(join(config.jobs,'000000.error.json'),'utf8')).generation_hold,'awaiting_extractive_equivalence_oracle');
  assert.equal(await readFile(config.output,'utf8'),'');
  assert.equal((await collectBatch([{index:0,record}],{...config,collectionRole:'reference'},runner)).completed,1);
  assert.equal(calls,1);
});

test('iteration guidance explains the initial stopping check and complete pagination', async () => {
  const {TOOLS_PROMPT,TOOLS_PROMPT_AT_NL_DEPTH_LIMIT}=await import('../dist/native/prompt.js');
  for (const prompt of [TOOLS_PROMPT,TOOLS_PROMPT_AT_NL_DEPTH_LIMIT]) {
    assert.match(prompt,/stopping check runs on the initial state before any step/);
    assert.match(prompt,/For pagination, start with more: true/);
  }
});

test('semantic sweep holds disputed claims while retaining valid negatives and second-document evidence', () => {
  for (const id of [79,104,160,164,165,168,169,200,258,331,372,397,564,584,747]) {
    assert.equal(quarantineReason({source:'scifact',source_ids:[`claim:${id}`],semantics:{}}), 'source_review_pending');
  }
  for (const id of [47,71,121,122,245,246,334,347,394,542,563,668,669,709]) {
    assert.equal(quarantineReason({source:'scifact',source_ids:[`claim:${id}`],semantics:{}}), undefined);
  }
});
