import { readdir,readFile,writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import {Folder} from '../../dist/index.js';
const [directory,output]=process.argv.slice(2);if(!output)throw Error('usage: export-improvement.mjs COLLECTION_DIR OUTPUT_JSONL');
const turns=[];
const verification=JSON.parse(await readFile(join(directory,'verification.json'),'utf8'));
const verified=new Set(verification.verified);
for(const file of await readdir(directory)) {
  if(!file.endsWith('.result.json'))continue;
  const row=JSON.parse(await readFile(join(directory,file),'utf8'));if(!row.accepted || !verified.has(row.id))continue;
  if(!row.authored?.files || !row.authored?.identity)throw Error(`accepted trajectory ${row.id} lacks frozen authored source`);
  const digest=createHash('sha256').update(JSON.stringify(row)).digest('hex');
  for(const [index,exchange] of row.exchanges.entries()) {
    const response=exchange.wireResponse?.choices?.[0]?.message;if(!response)continue;
    const target=structuredClone(response);delete target.reasoning_content;
    const next=row.exchanges.slice(index+1).flatMap(item=>item.wireRequest?.messages??[]);
    const calls=response.tool_calls??[];
    const outcomes=calls.map(call=>next.find(message=>message.role==='tool' && message.tool_call_id===call.id));
    const clean=calls.length===0 || outcomes.length===calls.length && outcomes.every(message=>message && !/^(error|rejected)|Nothing else from this eval was kept/.test(String(message.content)));
    if(!clean)continue;
    turns.push({version:'natlang.teacher_training_turn.native/1',id:row.id+':'+index,teacher_trajectory_id:row.id,teacher_trajectory_digest:digest,owner:'improver',
      program_id:'program-improver:'+row.id,source_groups:['improvement-target:'+row.id,...(row.sourceGroups??[])],split:'train',family:'native-program-improvement',task_family:'native-program-improvement',task_kind:'directory-reducer',task_modality:'program-editing',
      task:{kind:'whole_program',program_ir:{version:'natlang.program/2',id:'program-improver:'+row.id,kind:'lambda_source',family:'native-program-improvement',split:'train',source_groups:['improvement-target:'+row.id,...(row.sourceGroups??[])],source_ids:row.incidents.map(item=>item.id),source:'generated-failure-corpus',license:'project-generated',semantics:{root:'improveStep.nl',files:row.authored.files,inputs:{state:{iteration:0,done:false,incumbent:Folder.fromFiles(row.caseDefinition.files).snapshot().digest,quality:0,population:[],history:[],stopReason:''},policy:row.caseDefinition.policy},folder_files:row.caseDefinition.files,evaluation_fixture:{kind:'flat-program-evaluator',caseDefinition:row.caseDefinition},expected:row.state,expected_files:row.source}}},
      source_ref:{trajectory_id:row.id,source_row_sha256:digest},source:row.state.incumbent,
      messages:exchange.wireRequest.messages,tools:exchange.wireRequest.tools??[],target,teacher_reasoning:response.reasoning_content??null,teacher_reasoning_trained:true,
      provenance:{collection_role:'teacher',owner:'improver',runtime_api:'folder-revisions-v4'},outcome:{accepted:true,targetResolved:verification.targetOutcomes?.find(item=>item.id===row.id)?.resolved??null},
      training_admission:{kind:'exact-native-runtime-oracle',approved:clean,reason:clean?'source-state-and-native-evaluation-verified':'recovered failed action retained as context'},
      trace_admission:{admitted:clean},evidence:{state:row.state,ledger:row.ledger,actionOutcomes:outcomes},sourceIncidents:row.incidents.map(item=>item.id)});
  }
}
await writeFile(output,turns.map(JSON.stringify).join('\n')+'\n');console.log(JSON.stringify({turns:turns.length}));
