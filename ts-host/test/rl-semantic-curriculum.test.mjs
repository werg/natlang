import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { terminalJudgments } from '../../scripts/neuralese_data/build_rl_semantic_curriculum.mjs';
import { buildRecords } from '../scripts/inline-curriculum/families.mjs';
import { verifyCases } from '../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

const parent=()=>buildRecords({seed:810123,shapes:1,families:['workbench_tickets'],split:'train',hints:false})[0];

test('terminal semantic tasks retain their complete parent custody and execute through the actual task verifier', async()=>{
 const source=parent(),before=JSON.stringify(source),rows=terminalJudgments(source);
 assert.ok(rows.length>3);
 const digest=createHash('sha256').update(before).digest('hex');
 for(const row of rows){
  assert.deepEqual(row.source_groups,source.source_groups);
  assert.ok(row.source_ids.includes(source.id));
  assert.equal(row.generation.parent_record_sha256,digest);
  assert.equal(row.generation.teacher_generation,false);
  assert.equal(row.curriculum.rl.level,1);
  assert.equal(row.semantics.inputs.ticket.id,row.generation.ticket_id);
 }
 assert.equal(JSON.stringify(source),before);
 assert.equal(new Set(rows.map(row=>row.id)).size,rows.length);
 const verification=await verifyCases(rows.slice(0,3),TOOLS_PROMPT);
 assert.deepEqual(verification.filter(row=>!row.ok),[]);
});

test('held inputs and incomplete source oracle evidence cannot become train terminal judgments',()=>{
 const source=parent();
 assert.throws(()=>terminalJudgments({...source,split:'test'}),/train parents/);
 const incomplete=structuredClone(source);
 incomplete.curriculum.evidence.world.pop();
 assert.throws(()=>terminalJudgments(incomplete),/membership differs/);
 const corrupt=structuredClone(source);
 corrupt.curriculum.evidence.world[0]=corrupt.curriculum.evidence.world[0].replace(/"churn":(true|false)/,'"churn":"yes"');
 assert.throws(()=>terminalJudgments(corrupt),/invalid exact ticket label/);
 assert.deepEqual(terminalJudgments({...source,id:source.id+':hinted'}),[]);
});
