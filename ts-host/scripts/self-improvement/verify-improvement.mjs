/** Validate saved source/state/evidence links. Replay commands retain their original source case. */
import { readdir,readFile,writeFile } from 'node:fs/promises';
import {createHash} from 'node:crypto';
import { join,dirname } from 'node:path';
import { Folder,validateSourceEdit,SourceEvaluator,openAICompatibleModelTurn,AssignmentBudget,OperationJournal } from '../../dist/index.js';
import {UsageGateway} from '../../dist/evaluation/usage.js';
import {createPiModelBackend} from '../../dist/model/pi-provider.js';
const [casesPath,directory,onlyIds]=process.argv.slice(2);if(!directory)throw Error('usage: verify-improvement.mjs CASES_JSONL COLLECTION_DIR');
const cases=new Map((await readFile(casesPath,'utf8')).trim().split('\n').filter(Boolean).map(line=>{const row=JSON.parse(line);return [row.id,row];}));
const selected=onlyIds?new Set(onlyIds.split(',')):null;
const previous=selected?JSON.parse(await readFile(join(directory,'verification.json'),'utf8')):null;
const verified=previous?previous.verified.filter(id=>!selected.has(id)):[],rejected=previous?previous.rejected.filter(row=>!selected.has(row.id)):[],targetOutcomes=previous?(previous.targetOutcomes??[]).filter(row=>!selected.has(row.id)):[];
const allocation=new AssignmentBudget({collection:120,modelCalls:12480,caseExecutions:4800,trainingJobs:6,trainingUpdates:1200,confirmation:1,cost:100},{trainingJobs:4,trainingUpdates:800,confirmation:1},3,new OperationJournal(join(dirname(directory),'assignment-allocation')));
const files=await readdir(directory),seen=new Set();
for(const file of files) {
 if(!file.endsWith('.result.json'))continue;
 const row=JSON.parse(await readFile(join(directory,file),'utf8')),sourceCase=cases.get(row.id),errors=[];
 if(selected&&!selected.has(row.id))continue;
 seen.add(row.id);
 if(!sourceCase)errors.push('missing source case');
 if(Folder.fromFiles(row.source).snapshot().digest!==row.state.incumbent)errors.push('source/state mismatch');
 if(sourceCase)errors.push(...validateSourceEdit(sourceCase.files,row.source,sourceCase.policy.mode,sourceCase.policy.allowedFiles));
 if(!row.exchanges?.length)errors.push('missing actual improver exchanges');
 if(row.ledger?.usage?.modelCalls>(sourceCase?.budget?.maxModelCalls??104) || row.ledger?.roles?.reflection?.modelCalls>(sourceCase?.budget?.maxModelCalls??104) || row.ledger?.rollouts>40 || row.ledger?.proposals>2*(sourceCase?.policy.maxExperiments??0))errors.push('case allocation exceeded');
 if(sourceCase && !errors.length && row.accepted) {
  const executor=row.executor??{endpoint:'http://127.0.0.1:8081',model:'/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'};
  const backend=executor.endpoint.startsWith('pi:')?createPiModelBackend(executor.endpoint.slice(3),executor.model):undefined;
  const driver=backend?backend.turn.bind(backend):openAICompatibleModelTurn({endpoint:executor.endpoint,model:executor.model});
  try {
   const modelAllowance=sourceCase.executorModelCalls??sourceCase.budget?.maxModelCalls??30;allocation.allocateOnce('verify:'+createHash('sha256').update(JSON.stringify({row,modelAllowance})).digest('hex'),{modelCalls:modelAllowance,caseExecutions:20});
   const evaluator=new SourceEvaluator(sourceCase.contract,sourceCase.cases,driver,new UsageGateway({maxModelCalls:modelAllowance,maxRollouts:20,maxProposals:0}),{executorId:executor.model,signal:AbortSignal.timeout(120000)});
   const check=await evaluator.check(Folder.fromFiles(row.source).snapshot());
   if(!check.valid)errors.push(...check.diagnostics);
   else {const training=await evaluator.evaluate(Folder.fromFiles(row.source).snapshot(),{split:'train'});targetOutcomes.push({id:row.id,resolved:training.quality===1&&training.gatesPassed,trainingQuality:training.quality,gatesPassed:training.gatesPassed});if(training.quality!==1||!training.gatesPassed)errors.push('selected source fails declared training cases or gates');const report=await evaluator.evaluate(Folder.fromFiles(row.source).snapshot(),{split:'validation'});if(report.quality!==1||!report.gatesPassed)errors.push('selected source fails declared validation cases or gates');if(Math.abs(report.quality-row.state.quality)>1e-12)errors.push('fresh selected-source reproduction differs from reported quality');}
  }catch(error){errors.push('independent reproduction failed: '+String(error));}finally{backend?.close();}
 }
 if(errors.length || !row.accepted)rejected.push({id:row.id,errors});else verified.push(row.id);
}
for(const [id]of cases){if((selected&&!selected.has(id))||seen.has(id))continue;const errorFile=id+'.error.json';const detail=files.includes(errorFile)?JSON.parse(await readFile(join(directory,errorFile),'utf8')).error:'no collected result';rejected.push({id,errors:[detail]});}
await writeFile(join(directory,'verification.json'),JSON.stringify({version:'natlang.improvement-verification/1',verified,rejected,targetOutcomes,allocation:allocation.snapshot()},null,2));
console.log(JSON.stringify({verified:verified.length,rejected:rejected.length}));
