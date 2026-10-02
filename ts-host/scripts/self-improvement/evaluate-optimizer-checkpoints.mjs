/** One frozen weight-learning comparison: Sharp optimizes source; Bonsai still executes it. */
import {mkdir,readFile,writeFile,appendFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {pinReplayRuntime,runtimeModule,digest} from './replay-runtime.mjs';
const [input,mapPath,endpoint='http://127.0.0.1:18084']=process.argv.slice(2);
if(!mapPath)throw Error('usage: evaluate-optimizer-checkpoints.mjs CAMPAIGN CHECKPOINT_MAP [ENDPOINT]');
const root=resolve(input),output=join(root,'sharp-training','evaluation');await mkdir(output,{recursive:true});
const runtime=await pinReplayRuntime(join(root,'sdk'));
const {Folder,improveProgram,openAICompatibleModelTurn,OperationJournal}=await runtimeModule(runtime,'index.js');
const source=JSON.parse(await readFile(join(root,'optimizer-source.json'),'utf8'));
const training=JSON.parse(await readFile(join(root,'sharp-training','prepared','manifest.json'),'utf8'));
const checkpoints=JSON.parse(await readFile(mapPath,'utf8'));
if(JSON.stringify(Object.keys(checkpoints))!==JSON.stringify(['starting','trained']))throw Error('The comparison requires the two predeclared checkpoints.');
const definitions=[];for(const id of ['directory-eligibility-appeals','directory-confirmed-meetings'])definitions.push(JSON.parse(await readFile(join(root,id,'protocol.json'),'utf8')));
const freeze={schema:'natlang.optimizer-weight-confirmation/1',runtime:runtime.identity,source:digest(JSON.stringify(source)),training:training.sha256,checkpoints,definitions:digest(JSON.stringify(definitions)),endpoint,optimizerBudget:{maxModelCalls:150,maxRollouts:32,maxProposals:3,maxElapsedMs:3600000},assignment:{modelCalls:600,caseExecutions:128,elapsedMs:14400000},interpretation:'One descriptive comparison, two transfer families, no cohort replacements or tuning after evaluation. Primary: correct frozen selected source on independently labelled sealed cases. Secondary: valid completion, false promotion, full optimizer/student request count.'};
const journal=new OperationJournal(join(output,'journal'));
const previous=journal.read('freeze')?.value;if(previous&&digest(JSON.stringify(previous))!==digest(JSON.stringify(freeze)))throw Error('Weight confirmation inputs changed');journal.record('freeze',freeze);
const started=journal.read('started')?.value??Date.now();journal.record('started',started);
const results=[];
for(const checkpoint of Object.keys(checkpoints)){
 const selection=await fetch(endpoint+'/select',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({checkpoint})});if(!selection.ok)throw Error('Checkpoint selection failed: '+selection.status);
 for(const definition of definitions){
  const key=checkpoint+'--'+definition.id;
  const done=journal.read(key);if(done?.status==='done'){results.push(done.value);continue;}
  const elapsed=Date.now()-started;if(elapsed>=freeze.assignment.elapsedMs){results.push({checkpoint,id:definition.id,disposition:'assignment-exhausted',primary:false});continue;}
  const directory=join(output,key);await mkdir(directory,{recursive:true});
  const log=role=>async(request,turn)=>{await appendFile(join(directory,'exchanges.ndjson'),JSON.stringify({role,request,turn})+'\n');};
  const optimizerDriver=openAICompatibleModelTurn({endpoint,model:checkpoint,request:{temperature:0}});
  const optimizer=async(request,signal)=>{const turn=await optimizerDriver(request,signal);await log('optimizer')(request,turn);return turn;};
  const studentDriver=openAICompatibleModelTurn({...definition.executor,request:{temperature:0.2}});
  const student=async(request,signal)=>{const turn=await studentDriver(request,signal);await log('student')(request,turn);return turn;};
  const signal=AbortSignal.timeout(Math.min(freeze.optimizerBudget.maxElapsedMs,freeze.assignment.elapsedMs-elapsed));
  try{
  const result=await improveProgram({folder:Folder.fromFiles(definition.files),contract:definition.contract,cases:definition.cases,policy:definition.policy,improverSource:Folder.fromFiles(source).snapshot(),improver:optimizer,executor:student,executorId:definition.executor.model,executorTimeoutMs:definition.executorTimeoutMs,budget:freeze.optimizerBudget,signal,directory,trace:trace=>{void appendFile(join(directory,'traces.ndjson'),JSON.stringify(trace)+'\n');}});
  let confirmation=null,confirmationError=null;try{if(result.validation)confirmation=await result.evaluator.confirm(result.folder,{source:result.folder.digest,experiment:digest(JSON.stringify(freeze))+':'+key});}catch(error){confirmationError=String(error);}
  const {folder,evaluator,...portable}=result;
  const record={checkpoint,id:definition.id,primary:!!confirmation&&confirmation.quality===1&&result.state.done,completed:!!result.validation&&result.state.done,sourceAgree:result.state.incumbent===result.sourceManifest.source,falsePromotion:result.sourceDiff.length>0&&(result.validation?.quality??0)<(result.baseline?.quality??0),...portable,confirmation,confirmationError,ledger:evaluator.gateway.snapshot()};
  journal.record(key,record);results.push(record);await writeFile(join(output,'results.json'),JSON.stringify({freeze,results},null,2));console.log(JSON.stringify({checkpoint,id:definition.id,primary:record.primary,quality:confirmation?.quality,requests:record.ledger.usage.modelCalls}));
  }catch(error){
   const ledger=new OperationJournal(directory).read('ledger')?.value;
   const record={checkpoint,id:definition.id,primary:false,completed:false,disposition:'failed',error:String(error),ledger};
   journal.record(key,record);results.push(record);await writeFile(join(output,'results.json'),JSON.stringify({freeze,results},null,2));
   console.error(JSON.stringify({checkpoint,id:definition.id,error:record.error}));
  }
 }
}
await writeFile(join(output,'results.json'),JSON.stringify({freeze,results,summary:Object.keys(checkpoints).map(checkpoint=>{const rows=results.filter(row=>row.checkpoint===checkpoint);return {checkpoint,success:rows.filter(row=>row.primary).length,total:rows.length,requests:rows.reduce((sum,row)=>sum+(row.ledger?.usage.modelCalls??0),0)};})},null,2));
