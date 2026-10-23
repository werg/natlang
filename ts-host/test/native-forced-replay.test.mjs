import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { createVerifiedQwenTeacherForcedDriver, executeVerifiedQwenTeacherReplay, loadVerifiedQwenSource } from '../dist/teacher/native-forced-replay.js';
import { executeProgram } from '../dist/teacher/collector.js';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const canonical = value => value === null || typeof value !== 'object' ? JSON.stringify(value) : Array.isArray(value) ?
  `[${value.map(canonical).join(',')}]` : `{${Object.entries(value).sort(([a],[b])=>a.localeCompare(b))
    .map(([key,item])=>`${JSON.stringify(key)}:${canonical(item)}`).join(',')}}`;
const occurrenceBase={schema:'natlang.execution-occurrence/1',definition_id:'task',revision:'1',context:'ctx:files',
  signature:'() => string',parent_occurrence_sha256:null,sibling_ordinal:0,argument_binding_sha256:'f'.repeat(64)};
const occurrence={...occurrenceBase,occurrence_sha256:hash(canonical(occurrenceBase))};

function teacherRow() {
  const evalBody='writer body';
  return { version:'natlang.teacher_trajectory.native/1', id:'qwen-row', task:{program_ir:{id:'exact'}},
    provenance:{model:'qwen3.6-35b-a3b'},
    outcome:{accepted:true,execution_graph:[
      {kind:'invocation',phase:'start',seq:1,call_id:'source-run/execution-source',definition:{id:'task',revision:'1'},
        context:'ctx:files',signature:'() => string',native_occurrence:occurrence},
      {kind:'model_turn',call_id:'source-run/execution-source',turn:1,node:'source#turn1'},
      {kind:'block_write',call_id:'source-run/execution-source',turn:'source#turn1',marker_context:'eval-code',
        result_type:'Neuralese<string>',source_kind:'typed-text-result',producer:'text-marker-emulation',
        emulation_version:'text-marker-standin/2',text_body_sha256:hash(evalBody),truncated:false}]},
    trajectory:[{phase:'action',invocation_id:'source-run/execution-source',context:[{role:'user',content:'privileged hint here'}],
      tools_offered:[],assistant:{content:'',calls:[{tool:'eval',arguments:{
        code:`const body = "<|neuralese|>${evalBody}<|/neuralese|>";`,
        comment:'ordinary literal <|neuralese|>writer body<|/neuralese|>'}}]}}]};
}

async function sourceFixture(rowValue=teacherRow()) {
  const root=await mkdtemp(join(tmpdir(),'native-qwen-source-')), authority=join(root,'authority'); await mkdir(authority);
  const source=Buffer.from(JSON.stringify(rowValue)+'\n'), sourcePath=join(root,'native-trajectories.jsonl'); await writeFile(sourcePath,source);
  const artifact={sha256:'d'.repeat(64)}, actor={schema:'natlang.selected-teacher-actor-source/1',
    model:'nvidia/Qwen3.6-35B-A3B-NVFP4',revision:'1355db6a052410cfd62085d94b58866fd0f2c3c5',download:artifact};
  const actorBytes=Buffer.from(JSON.stringify(actor)), actorPath=join(authority,'teacher-serving-actor.json'); await writeFile(actorPath,actorBytes);
  const execution={schema:'natlang.self-hosted-teacher-execution-identity/1',model_id:'qwen3.6-35b-a3b',actor:{sha256:hash(actorBytes)}};
  const identityBytes=Buffer.from(JSON.stringify(execution)), identityPath=join(authority,'teacher-execution-identity.json'); await writeFile(identityPath,identityBytes);
  const transfer={schema:'natlang.verified-teacher-trajectory-transfer/1',teacher:{model:actor.model,revision:actor.revision,artifact},files:[
    {path:actorPath,bytes:actorBytes.length,sha256:hash(actorBytes)},
    {path:identityPath,bytes:identityBytes.length,sha256:hash(identityBytes)}]};
  await writeFile(join(root,'transfer-receipt.json'),JSON.stringify(transfer));
  await writeFile(join(root,'canonical-source-verification.json'),JSON.stringify({
    schema:'natlang.teacher-live-tranche-canonical-source-verification/1',status:'passed',training_admission:false,
    source:{path:sourcePath,bytes:source.length,rows:1,sha256:hash(source)},
    actor:{sha256:hash(actorBytes)},execution_identity:{sha256:hash(identityBytes)}}));
  return {root,sourcePath};
}

test('Qwen forced replay matches fresh execution IDs by exact graph occurrence and forces only verified typed spans', async () => {
  const fixture=await sourceFixture();
  try {
    const received=[]; const {driver,runId}=await createVerifiedQwenTeacherForcedDriver(fixture.root,'qwen-row','privileged-hint',
      async request=>(received.push(request),{text:''}));
    assert.match(runId,/^qwen-forced-/);
    const request={invocation_id:'student-run/execution-fresh',native_execution_occurrence:occurrence,
      messages:[{role:'user',content:'original task only'}],tools:[]};
    await driver(request);
    assert.deepEqual(received[0].messages,request.messages);
    const assistant=received[0].native_teacher_forced_turn.assistant;
    const args=JSON.parse(assistant.tool_calls[0].function.arguments);
    assert.match(args.code,/^const body = "NATLANGQWENSPAN[0-9a-f]+END";$/);
    assert.deepEqual(received[0].native_teacher_forced_turn.native_text_spans,[{path:['tool_calls',0,'arguments','code'],
      marker:args.code.match(/NATLANGQWENSPAN[0-9a-f]+END/)[0],text:'writer body'}]);
    assert.equal(args.comment,'ordinary literal <|neuralese|>writer body<|/neuralese|>');
    assert.equal(received[0].native_teacher_forced_turn.source_receipt.source_view,'privileged-hint');
    assert.equal(received[0].native_teacher_forced_turn.source_receipt.invocation_id,'source-run/execution-source');
    await assert.rejects(()=>driver({...request,native_execution_occurrence:{...occurrence,occurrence_sha256:'0'.repeat(64)}}),
      /authenticated structural execution occurrence/);
    const wrongArgument={...occurrence,argument_binding_sha256:'e'.repeat(64)};
    delete wrongArgument.occurrence_sha256; wrongArgument.occurrence_sha256=hash(canonical(wrongArgument));
    await assert.rejects(()=>driver({...request,native_execution_occurrence:wrongArgument}),
      /Qwen source argument binding differs/);
    const legacy=teacherRow(); delete legacy.outcome.execution_graph[0].native_occurrence;
    const legacyFixture=await sourceFixture(legacy);
    try {
      const legacyDriver=(await createVerifiedQwenTeacherForcedDriver(legacyFixture.root,'qwen-row','privileged-hint',
        async target=>(received.push(target),{text:''}))).driver;
      await legacyDriver(request);
      assert.equal(received.at(-1).native_teacher_forced_turn.source_receipt.invocation_id,'source-run/execution-source');
    } finally { await rm(legacyFixture.root,{recursive:true,force:true}); }
  } finally { await rm(fixture.root,{recursive:true,force:true}); }
});

test('Qwen source loader detects source and actor proof tampering', async () => {
  const fixture=await sourceFixture();
  try {
    const loaded=await loadVerifiedQwenSource(fixture.root,'qwen-row','privileged-hint');
    assert.equal(loaded.row.id,'qwen-row'); assert.equal(loaded.binding.sourceFileSha256,hash(await import('node:fs/promises').then(fs=>fs.readFile(fixture.sourcePath))));
    await writeFile(fixture.sourcePath,Buffer.from('{}\n'));
    await assert.rejects(()=>loadVerifiedQwenSource(fixture.root,'qwen-row','privileged-hint'),/source bytes do not match/);
  } finally { await rm(fixture.root,{recursive:true,force:true}); }
});

test('executeProgram maps a fresh invocation UUID to the exact source graph occurrence without hint history', async () => {
  const record={version:'natlang.program/2',id:'forced-replay-exact-task',kind:'lambda_source',split:'train',
    semantics:{root:'solve.nl',files:{'solve.nl':'---\nargs: {}\nreturns: number\n---\nCompute two plus two.\n'},inputs:{},expected:4}};
  const options={systemPrompt:'Original unhinted system prompt.',contextTokens:8192,maxTurns:3,rootSeed:17};
  let sourceRequest,sourceReply;
  const original=await executeProgram(record,async request=>{sourceRequest=structuredClone(request);
    sourceReply={calls:[['eval',{code:'2 + 2',finish:true}]]};return sourceReply;},{...options,runId:'teacher-source-run'});
  assert.equal(original.outcome.accepted,true);
  assert.ok(sourceRequest.native_execution_occurrence?.occurrence_sha256);
  const sourceInvocation=original.outcome.execution_graph.find(event=>event.kind==='invocation'&&event.phase==='start');
  const rowValue={version:'natlang.teacher_trajectory.native/1',id:'fresh-execution-row',task:{program_ir:record},
    provenance:{model:'qwen3.6-35b-a3b'},outcome:{accepted:true,execution_graph:original.outcome.execution_graph},
    trajectory:[{phase:'action',invocation_id:sourceRequest.invocation_id,
      context:[{role:'user',content:'privileged repair hint visible to Qwen only'}],tools_offered:sourceRequest.tools,
      assistant:{content:'',calls:sourceReply.calls.map(([tool,args])=>({tool,arguments:args}))}}]};
  const fixture=await sourceFixture(rowValue);
  try {
    let replayRequest;
    const repaired=await executeVerifiedQwenTeacherReplay(fixture.root,rowValue.id,'privileged-hint',record,options,
      async request=>{replayRequest=structuredClone(request);return sourceReply;});
    assert.equal(repaired.outcome.accepted,true);
    assert.notEqual(replayRequest.invocation_id,sourceRequest.invocation_id,'each replay has a fresh execution UUID');
    assert.deepEqual(replayRequest.native_execution_occurrence,sourceRequest.native_execution_occurrence);
    assert.doesNotMatch(JSON.stringify(replayRequest.messages),/privileged repair hint/);
    assert.equal(replayRequest.native_teacher_forced_turn.source_receipt.invocation_id,sourceInvocation.call_id);
  } finally { await rm(fixture.root,{recursive:true,force:true}); }
});
