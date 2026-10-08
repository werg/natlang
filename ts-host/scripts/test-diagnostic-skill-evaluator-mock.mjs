#!/usr/bin/env node
/** Local-only integration proof for the held diagnostic evaluator. No network/provider call is made. */
import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {join, resolve} from 'node:path';

const hash = value => createHash('sha256').update(value).digest('hex');
const base = resolve('runs/criterion-grounded-skill-candidate-20261008-v1/diagnostic-runner-v1');
const sourcePlan = JSON.parse(await readFile(join(base,'plan-root-review-v3.json'),'utf8'));
const proofId = 'mock-provider-proof-v11';
const output = join(base,proofId);
const planPath = join(base,`${proofId}-plan.json`);
try { await readFile(join(output,'summary.json')); throw new Error('mock output already exists; preserve it and use a new proof ID'); }
catch (error) { if (error.message.includes('already exists')) throw error; if (error.code !== 'ENOENT') throw error; }
const expected = ['eligible','ineligible','unresolved','ineligible'];
const plan={...sourcePlan,root_approved:true,model:'mock-luna',endpoint:'http://127.0.0.1:65533',provider:null,
  local_fake_provider:true,pi_options:{},output:resolve(output),job_wall_cap_ms:30000,min_free_bytes:0,
  text_neuralese_emulation:false,execution_plans:true,max_requests_per_execution:4,max_total_requests:32};
const bytes=Buffer.from(JSON.stringify(plan,null,2)+'\n'); await writeFile(planPath,bytes,{flag:'wx'});
const runner=resolve('ts-host/scripts/diagnostic-evaluate-student-skills.mjs');
const exchangeLog=join(base,`${proofId}-provider-exchanges.jsonl`);
const hook=resolve('ts-host/scripts/test-diagnostic-skill-fake-provider-hook.mjs');
const result=spawnSync(process.execPath,[runner,planPath,'--execute',hash(bytes)],{encoding:'utf8',timeout:30000,
  env:{...process.env,NODE_OPTIONS:`--import=${hook}`,CRITERION_FAKE_EXPECTED:JSON.stringify(expected),CRITERION_FAKE_LOG:exchangeLog}});
if (result.error) throw result.error;
assert.equal(result.status,0,`diagnostic runner failed: ${result.stderr}\n${result.stdout}`);
const summary=JSON.parse(await readFile(join(output,'summary.json'),'utf8'));
for (const arm of ['discovery','instructed']) {
  assert.equal(summary.arms[arm].cases,4);
  assert.equal(summary.arms[arm].answer_accepted,4);
  assert.equal(summary.arms[arm].exact_value_matches,4);
  assert.equal(summary.arms[arm].collector_oracle_accepted,4);
  assert.equal(summary.arms[arm].offered,4);
  assert.equal(summary.arms[arm].body_reads,4);
  assert.equal(summary.arms[arm].failures,0);
  for (let index=0;index<4;index++) {
    const row=JSON.parse(await readFile(join(output,arm,String(index).padStart(2,'0'),'result.json'),'utf8'));
    assert.equal(row.actual,expected[index]);
    assert.equal(row.answer_accepted,true);
    assert.equal(row.collector_answer_check,true);
    assert.equal(row.collector_oracle_accepted,true);
    assert.equal(row.error,undefined);
    assert(row.skill_events.some(event=>event.phase==='offered'));
    assert(row.skill_events.some(event=>event.phase==='body_read'));
    assert(row.skill_reads.some(read=>read.target==='skills.judge-against-criteria'));
    assert.equal(row.split,'diagnostic');
    assert.equal(row.training_admission,false);
  }
}
console.log(JSON.stringify({status:'passed',network:'in-process endpoint interceptor; no socket opened',provider_requests:summary.arms.discovery.cases*4+summary.arms.instructed.cases*4,
  executions:8,arms:['discovery','instructed'],per_arm:{answer_accepted:4,offered:4,body_read:4},training_admission:false,
  output,output_sha256:hash(await readFile(join(output,'summary.json')))}));
