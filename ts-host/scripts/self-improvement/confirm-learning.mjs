/** One frozen, resumable confirmation block. Never revise its cohort, runtime or checkpoints after opening it. */
import {readFile,writeFile,readdir,mkdir} from 'node:fs/promises';
import {join,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {Folder,improveProgram,openAICompatibleModelTurn,AssignmentBudget,OperationJournal,SourceEvaluator,improverExecution} from '../../dist/index.js';
import {UsageGateway} from '../../dist/evaluation/usage.js';
import {createPiModelBackend} from '../../dist/model/pi-provider.js';
import {AUTHORED_IMPROVER} from '../../dist/improvement/authored-source.js';
import {NATLANG_COMPILE_VERSION} from '../../dist/compiler/intrinsics.js';
import {SOURCE_EVALUATION_VERSION} from '../../dist/improvement/host.js';
const [directory,mapPath,endpoint='http://127.0.0.1:8082']=process.argv.slice(2);
if(!mapPath)throw Error('usage: confirm-learning.mjs PROTOCOL_DIR CHECKPOINT_MAP [ENDPOINT]');
const protocol=JSON.parse(await readFile(join(directory,'protocol.json'),'utf8')),map=JSON.parse(await readFile(mapPath,'utf8'));
const hash=value=>createHash('sha256').update(typeof value==='string'?value:JSON.stringify(value)).digest('hex');
const runtime={};
async function inventory(directory,prefix=''){
 for(const entry of (await readdir(directory,{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))){
  const relative=prefix+entry.name;if(entry.isDirectory())await inventory(join(directory,entry.name),relative+'/');
  else if(entry.name.endsWith('.js'))runtime[relative]=hash(await readFile(join(directory,entry.name),'utf8'));
 }
}
await inventory(fileURLToPath(new URL('../../dist/',import.meta.url)));
const meta=JSON.parse(await readFile(join(directory,'../../self-improvement-data/folder-api-v1/frozen-improver-live/result.json'),'utf8'));
const checkpoints={};
for(const name of protocol.checkpoints){
 if(!(name in map))throw Error('missing frozen checkpoint '+name);
 const files={};if(map[name])for(const file of (await readdir(map[name])).sort())files[file]=hash(await readFile(join(map[name],file)).then(buffer=>buffer.toString('base64')));
 checkpoints[name]={path:map[name],files};
}
const freeze={protocol:hash(protocol),checkpoints,runtime,authored:hash(AUTHORED_IMPROVER),meta:{protocol:hash(meta.protocol),source:meta.sourceManifest.source,base:meta.sourceManifest.base},compiler:NATLANG_COMPILE_VERSION,evaluation:SOURCE_EVALUATION_VERSION,endpoint};
await mkdir(directory,{recursive:true});const journal=new OperationJournal(join(directory,'operations'));
const previous=journal.read('freeze')?.value;if(previous&&hash(previous)!==hash(freeze))throw Error('confirmation is frozen; changed inputs cannot open another attempt');journal.record('freeze',freeze);
const allocation=new AssignmentBudget({collection:120,modelCalls:12480,caseExecutions:4800,trainingJobs:6,trainingUpdates:1200,confirmation:1,cost:100},{trainingJobs:4,trainingUpdates:800,confirmation:1},3,new OperationJournal(join(dirname(directory),'..','self-improvement-data','folder-api-v1','assignment-allocation')));
// The assignment journal owns the one confirmation allocation; the experiment journal reconciles crashes.
allocation.allocateOnce('student-confirmation:'+hash(freeze),{confirmation:1,modelCalls:protocol.checkpoints.length*protocol.cases.length*protocol.allowance.maxModelCalls+200,caseExecutions:protocol.checkpoints.length*protocol.cases.length*(protocol.allowance.maxRollouts+4)+120},true);
const results=[];
for(const name of protocol.checkpoints){
 const response=await fetch(endpoint+'/select',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({checkpoint:name})});if(!response.ok)throw Error('checkpoint selection failed: '+response.status);
 for(const row of protocol.cases){
  const key=name+':'+row.id;
  const execute=async()=>{
   const exchanges=[],start=Date.now();
   const driver=openAICompatibleModelTurn({endpoint,model:name,request:{stream:false,temperature:0},onExchange:exchange=>exchanges.push(exchange)});
   const result=await improveProgram({folder:Folder.fromFiles(row.files),contract:row.contract,cases:row.cases,policy:row.policy,improver:driver,executor:()=>{throw Error('confirmation targets are exact source');},executorId:'exact-ts-final',budget:protocol.allowance,directory:join(directory,'runs',name,row.id),signal:AbortSignal.timeout(protocol.allowance.maxElapsedMs)});
   let heldout=null,error='error'in result?result.error:null;
   if(result.validation){try{heldout=await result.evaluator.confirm(result.folder,{source:result.folder.digest,experiment:hash(freeze)+':'+key});}catch(failure){error=String(failure);}}
   const sourceAgree=result.state.incumbent===result.folder.digest;
   const completed=!!result.validation&&result.state.done&&sourceAgree;
   const primary=completed&&!!heldout?.gatesPassed&&heldout.quality===1;
   const record={checkpoint:name,id:row.id,family:row.family,primary,completed,sourceAgree,falsePromotion:result.sourceDiff.length>0&&(!result.validation?.gatesPassed||result.validation.quality<(result.baseline?.quality??0)),heldoutQuality:heldout?.quality??null,disposition:result.disposition,error,state:result.state,sourceManifest:result.sourceManifest,sourceDiff:result.sourceDiff,validation:result.validation,baseline:result.baseline,heldout,ledger:result.evaluator.gateway.snapshot(),wallMs:Date.now()-start,exchanges};
   await writeFile(join(directory,name+'--'+row.id+'.json'),JSON.stringify(record));return record;
  };
  const result=await journal.run(key,execute,execute);results.push(result);console.log(JSON.stringify({checkpoint:name,family:row.family,primary:result.primary,completed:result.completed,heldoutQuality:result.heldoutQuality}));
 }
}
const metaConfirmation=await journal.run('meta-confirmation',async()=>{
 const backend=createPiModelBackend(meta.protocol.provider,meta.protocol.model);
 try{
  const gateway=new UsageGateway({maxModelCalls:200,maxRollouts:120,maxProposals:12,maxElapsedMs:1200000});
  const executeCase=improverExecution(meta.protocol.targets,{improver:backend.turn.bind(backend),executor:()=>{throw Error('exact targets');},improverId:meta.protocol.provider+'/'+meta.protocol.model,executorId:'exact-source'});
  const evaluator=new SourceEvaluator({entry:'improveStep/diagnose.nl',exportName:'default',programId:'frozen-improver-final'},meta.protocol.cases,backend.turn.bind(backend),gateway,{executorId:meta.protocol.provider+'/'+meta.protocol.model,evaluationLevel:2,executeCase,journal:new OperationJournal(join(directory,'meta-journal'))});
  return await evaluator.confirmPair(Folder.fromFiles(meta.sourceManifest.baseFiles).snapshot(),Folder.fromFiles(meta.sourceManifest.files).snapshot(),hash(freeze)+':meta');
 }finally{backend.close();}
});
const summary={schema:'natlang.student-confirmation-result/1',freeze,metaConfirmation,analysis:protocol.analysis,allocation:allocation.snapshot(),results:results.map(({exchanges,...row})=>row),checkpoints:protocol.checkpoints.map(checkpoint=>{const rows=results.filter(row=>row.checkpoint===checkpoint);return {checkpoint,families:rows.length,success:rows.filter(row=>row.primary).length,complete:rows.filter(row=>row.completed).length,falsePromotions:rows.filter(row=>row.falsePromotion).length,modelCalls:rows.reduce((sum,row)=>sum+(row.ledger.usage.modelCalls??0),0),wallMs:rows.reduce((sum,row)=>sum+row.wallMs,0)};})};
await writeFile(join(directory,'results.json'),JSON.stringify(summary,null,2));console.log(JSON.stringify(summary.checkpoints));
