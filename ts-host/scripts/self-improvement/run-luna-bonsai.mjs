/** Separate optimizer/executor study; historical test groups remain closed during development. */
import {mkdir,readFile,writeFile,appendFile,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {Folder,improveProgram,SourceEvaluator,OperationJournal,openAICompatibleModelTurn} from '../../dist/index.js';
import {UsageGateway} from '../../dist/evaluation/usage.js';
import {createPiModelBackend} from '../../dist/model/pi-provider.js';
const [output,command='develop',only]=process.argv.slice(2);
if(!output||!['develop','confirm'].includes(command))throw Error('usage: run-luna-bonsai.mjs OUTPUT [develop|confirm] [FAMILY]');
await mkdir(output,{recursive:true});
const protocolPath=join(output,'protocol.json');
let protocol;
try{protocol=JSON.parse(await readFile(protocolPath,'utf8'));}catch(error){
 if(error.code!=='ENOENT')throw error;
 const originals=(await readFile(new URL('../../../runs/self-improvement-data/folder-api-v1/original-failure-cases-v5.jsonl',import.meta.url),'utf8')).trim().split('\n').map(JSON.parse);
 const families=['curriculum_simple_deadline','curriculum_relational_late_argmax','curriculum_simple_count_positive'];
 const cases=families.map(family=>{
  const row=structuredClone(originals.find(row=>row.family===family));if(!row)throw Error('historical task missing');
  row.cases=['train','validation','test'].flatMap(split=>row.cases.filter(c=>c.split===split).slice(0,split==='train'?2:2));
  row.policy={...row.policy,maxExperiments:2,maxPopulation:3,mode:'instruction',goal:family==='curriculum_simple_deadline'?'Preserve urgency semantics and deadline arithmetic; improve reliable execution by the Bonsai interpreter.':family==='curriculum_relational_late_argmax'?'Preserve the Q3 late-shipment count and alphabetical tie rule; improve reliable complete pagination and aggregation by Bonsai.':'Preserve semantic judgment of whether each review recommends the product and exact counting; improve reliable execution by Bonsai.'};
  delete row.policy.outerLesson;
  return row;
 });
 protocol={schema:'natlang.luna-bonsai-study/1',optimizer:{provider:'openai-codex',model:'gpt-6-luna'},executor:{endpoint:'http://127.0.0.1:8081',model:'/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'},cases,
  developmentBudget:{maxModelCalls:400,maxRollouts:40,maxProposals:4,maxElapsedMs:900000},confirmationBudget:{maxModelCalls:200,maxRollouts:8,maxProposals:0,maxElapsedMs:600000},
  design:'Three historical source tasks; sequential shared-server target requests. Train gold is visible; validation scores only. No student weight training. Freeze selected sources and runtime before one paired test evaluation. No revised candidates after confirmation.'};
 await writeFile(protocolPath,JSON.stringify(protocol,null,2)+'\n');
}
const backend=createPiModelBackend(protocol.optimizer.provider,protocol.optimizer.model);
let current;
const log=async(role,request,turn)=>{
 const record={at:new Date().toISOString(),task:current,role,request,turn};await appendFile(join(output,'exchanges.ndjson'),JSON.stringify(record)+'\n');
 console.log(JSON.stringify({task:current,role,calls:turn.calls?.map(call=>call[0]),text:turn.text?.slice(0,100),tokens:turn.usage?.outputTokens}));
};
const optimizer=async(request,signal)=>{const turn=await backend.turn(request,signal);await log('optimizer',request,turn);return turn;};
const rawStudent=openAICompatibleModelTurn({...protocol.executor,request:{temperature:0.2}});
const student=async(request,signal)=>{const turn=await rawStudent(request,signal);await log('student',request,turn);return turn;};
try{
 if(command==='develop')for(const row of protocol.cases){
  if(only&&row.family!==only)continue;current=row.family;
  const folder=Folder.fromFiles(row.files),directory=join(output,'development',row.family);await mkdir(directory,{recursive:true});
  const resultPath=join(directory,'result.json');try{await readFile(resultPath);continue;}catch(error){if(error.code!=='ENOENT')throw error;}
  console.log(JSON.stringify({task:current,status:'starting',optimizer:protocol.optimizer.model,student:protocol.executor.model}));
  const result=await improveProgram({folder,contract:row.contract,cases:row.cases,policy:row.policy,improver:optimizer,executor:student,executorId:protocol.executor.model,executorTimeoutMs:protocol.executorTimeoutMs,budget:protocol.developmentBudget,directory:join(directory,'journal'),signal:AbortSignal.timeout(protocol.developmentBudget.maxElapsedMs),trace:trace=>{void appendFile(join(directory,'traces.ndjson'),JSON.stringify(trace)+'\n');}});
  const {evaluator,folder:returned,...portable}=result;
  await writeFile(resultPath,JSON.stringify({...portable,source:returned.digest},null,2)+'\n');
  console.log(JSON.stringify({task:current,status:'developed',disposition:result.disposition,baseline:result.baseline?.quality,selected:result.validation?.quality,experiments:result.state.iteration,history:result.state.history,usage:result.ledger.roles,error:result.error}));
 }
 else {
  const freezePath=join(output,'confirmation-freeze.json');let freeze;
  const sources=[];for(const row of protocol.cases){const path=join(output,'development',row.family,'result.json');const result=JSON.parse(await readFile(path,'utf8'));sources.push({family:row.family,source:result.source,files:result.sourceManifest.files,baseline:row.files});}
  const runtime={};async function walk(path){for(const entry of await readdir(path,{withFileTypes:true})){const name=join(path,entry.name);if(entry.isDirectory())await walk(name);else if(entry.name.endsWith('.js'))runtime[name]=createHash('sha256').update(await readFile(name)).digest('hex');}}await walk(new URL('../../dist/',import.meta.url).pathname);
  const proposed={protocol,sources,runtime};try{freeze=JSON.parse(await readFile(freezePath,'utf8'));if(JSON.stringify(freeze)!==JSON.stringify(proposed))throw Error('frozen confirmation changed');}catch(error){if(error.code!=='ENOENT')throw error;freeze=proposed;await writeFile(freezePath,JSON.stringify(freeze)+'\n');}
  const experiment=createHash('sha256').update(JSON.stringify(freeze)).digest('hex'),results=[];
  for(const row of protocol.cases){current=row.family;const source=sources.find(source=>source.family===current),directory=join(output,'confirmation',current);await mkdir(directory,{recursive:true});
   const gateway=new UsageGateway(protocol.confirmationBudget),evaluator=new SourceEvaluator(row.contract,row.cases,student,gateway,{executorId:protocol.executor.model,timeoutMs:protocol.executorTimeoutMs,journal:new OperationJournal(join(directory,'journal')),signal:AbortSignal.timeout(protocol.confirmationBudget.maxElapsedMs)});
   const report=await evaluator.confirmPair(Folder.fromFiles(source.baseline).snapshot(),Folder.fromFiles(source.files).snapshot(),experiment);
   const result={family:current,...report,ledger:gateway.snapshot()};results.push(result);await writeFile(join(directory,'result.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({task:current,status:'confirmed',baseline:report.baseline.quality,selected:report.selected.quality,effect:report.effect,pValue:report.pValue,supported:report.supported}));
  }
  await writeFile(join(output,'confirmation-results.json'),JSON.stringify({experiment,results},null,2)+'\n');
 }
}finally{backend.close();}
