/** Execute an authenticated Qwen trajectory as a real no-hint task on a native student endpoint. */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chatCompletionModelTurn, httpChatTransport } from '../model/chat-completion.js';
import { HttpNeuraleseStore } from '../model/neuralese-server.js';
import { nativeStateRuntimeOptions } from '../model/native-state-client.js';
import { canonical } from '../native/canonical-data.js';
import { executeVerifiedQwenTeacherReplayCapture } from './native-forced-replay.js';
import type { ProgramRecord } from './collector.js';

type Spec={schema:'natlang.qwen-teacher-repair-job/1';source_tranche_directory:string;source_row_id:string;
  source_view:'unhinted'|'privileged-hint';failed_run_dir:string;task_id:string;failed_sample:number;
  target_endpoint:string;model:string;headers?:Record<string,string>;system_prompt:string;context_tokens:number;
  root_seed:number;max_turns:number;output_capture_path:string};
function requireSpec(value:unknown):Spec {
  if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('replay spec must be a JSON object');
  const spec=value as Record<string,unknown>;
  if(spec.schema!=='natlang.qwen-teacher-repair-job/1'||
    !['source_tranche_directory','source_row_id','failed_run_dir','task_id','target_endpoint','model',
      'system_prompt','output_capture_path'].every(key=>typeof spec[key]==='string'&&spec[key])||
    !['unhinted','privileged-hint'].includes(String(spec.source_view))||
    !['failed_sample','context_tokens','root_seed','max_turns'].every(key=>Number.isSafeInteger(spec[key])&&Number(spec[key])>=0)||
    (spec.headers!==undefined&&(!spec.headers||typeof spec.headers!=='object'||Array.isArray(spec.headers))))
    throw new Error('replay spec is missing source, endpoint, exact prompt, task seed, or output bindings');
  return spec as unknown as Spec;
}
async function json(path:string):Promise<any>{return JSON.parse(await readFile(path,'utf8'));}
async function jsonl(path:string):Promise<any[]>{return (await readFile(path,'utf8')).split(/\r?\n/).filter(line=>line.trim()).map(line=>JSON.parse(line));}

export async function executeQwenTeacherRepairJob(specValue:unknown) {
  const spec=requireSpec(specValue),failedDir=resolve(spec.failed_run_dir);
  const [binding,preparation,packet,queue]=await Promise.all([
    json(resolve(failedDir,'collection-binding.json')),json(resolve(failedDir,'preparation.json')),
    jsonl(resolve(failedDir,'packet.jsonl')),jsonl(resolve(failedDir,'queue.jsonl'))]);
  const candidates=packet.map((task,index)=>({task,index})).filter(item=>item.task.id===spec.task_id);
  if(candidates.length!==1)throw new Error('failed run packet task id is missing or ambiguous');
  const {task,index}=candidates[0]!;
  if(task.split!=='train'||!task.program_ir)throw new Error('Qwen repair target must be the exact original train task packet');
  const queueRows=queue.filter(item=>item.index===index&&item.sample===spec.failed_sample);
  if(queueRows.length!==1)throw new Error('failed run does not uniquely bind the selected task sample');
  const entry=queueRows[0]!;const failedResult=await jsonl(resolve(entry.output));
  if(failedResult.length!==1||failedResult[0].task?.program_ir===undefined||
      JSON.stringify(failedResult[0].task.program_ir)!==JSON.stringify(task.program_ir))
    throw new Error('failed result row does not bind the exact selected task packet');
  const firstTurn=(failedResult[0].trajectory??[]).find((turn:any)=>turn.phase==='action');
  const originalSystem=(firstTurn?.context??[]).find((message:any)=>message.role==='system')?.content;
  if(typeof originalSystem!=='string'||originalSystem!==spec.system_prompt)
    throw new Error('repair system prompt differs from the exact original unhinted first-turn view');
  if(preparation.schema!=='natlang.task-rollout-preparation/1'||
      preparation.policy_checkpoint_sha256!==binding.policy_id||
      binding.schema!=='natlang.rl-collection-binding/1')
    throw new Error('failed run preparation and exact native actor binding disagree');
  const infoResponse=await fetch(spec.target_endpoint.replace(/\/$/,'')+'/v1/neuralese/info',{headers:spec.headers});
  if(!infoResponse.ok)throw new Error(`native actor info request returned HTTP ${infoResponse.status}`);
  const info=await infoResponse.json() as Record<string,any>,actor=info.rl_policy_binding??{};
  const expected=binding.expected_native_actor??{};
  if(!actor.native_identity||actor.native_identity.architecture!==expected.native_identity?.architecture||
      actor.native_identity.tokenizer_fingerprint!==expected.native_identity?.tokenizer_fingerprint)
    throw new Error('live repair target must preserve the failed model architecture and tokenizer family');
  if(!actor.python_runtime||actor.python_runtime.schema!=='natlang.python-runtime/1')
    throw new Error('live repair target has no authenticated Python runtime inventory');
  const store=new HttpNeuraleseStore(spec.target_endpoint,spec.headers);
  const neuralese=await nativeStateRuntimeOptions({endpoint:spec.target_endpoint,headers:spec.headers,store},{write:'required'});
  const send=chatCompletionModelTurn(httpChatTransport({endpoint:spec.target_endpoint,model:spec.model,
    headers:spec.headers,stream:false}));
  const result=await executeVerifiedQwenTeacherReplayCapture(spec.source_tranche_directory,spec.source_row_id,
    spec.source_view,task.program_ir as ProgramRecord,{systemPrompt:spec.system_prompt,contextTokens:spec.context_tokens,
      rootSeed:spec.root_seed,maxTurns:spec.max_turns,temperature:0,neuralese},send);
  const destination=resolve(spec.output_capture_path);await mkdir(dirname(destination),{recursive:true});
  const targetActor={endpoint:spec.target_endpoint,native_identity:actor.native_identity,python_runtime:actor.python_runtime,
    binding_sha256:createHash('sha256').update(canonical(actor)).digest('hex')};
  await writeFile(destination,JSON.stringify({...result,target_actor:targetActor})+'\n',{flag:'wx'});
  return {capture_path:destination,run_id:result.run_id,task_id:spec.task_id,accepted:result.run.outcome.accepted,
    turns:result.trajectory.length,teacher_source_row_id:spec.source_row_id};
}

async function main(argv=process.argv.slice(2)) {
  if(argv.length!==2||argv[0]!=='--spec')throw new Error('usage: node native-forced-replay-cli.js --spec JOB.json');
  const spec=await json(resolve(argv[1]!));const result=await executeQwenTeacherRepairJob(spec);process.stdout.write(JSON.stringify(result)+'\n');
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)
  main().catch(error=>{process.stderr.write(`${error instanceof Error?error.stack:String(error)}\n`);process.exitCode=1;});
