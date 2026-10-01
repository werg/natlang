/** Collect student failures only from the train pool, preserving actual wire actions and observations. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join,dirname} from 'node:path';
import {Folder,improveProgram,openAICompatibleModelTurn,AssignmentBudget,OperationJournal} from '../../dist/index.js';
const [casesPath,output,endpoint='http://127.0.0.1:8082',model='round-1',count='2']=process.argv.slice(2);
await mkdir(output,{recursive:true});
const rows=(await readFile(casesPath,'utf8')).trim().split('\n').map(JSON.parse).filter(row=>!row.control).slice(0,Number(count));
const allocation=new AssignmentBudget({collection:120,modelCalls:12480,caseExecutions:4800,trainingJobs:6,trainingUpdates:1200,confirmation:1,cost:100},{trainingJobs:4,trainingUpdates:800,confirmation:1},3,new OperationJournal(join(dirname(output),'assignment-allocation')));
const results=[];
for(const row of rows){
 allocation.allocate({modelCalls:24,caseExecutions:40});const exchanges=[],traces=[];
 const driver=openAICompatibleModelTurn({endpoint,model,stream:false,request:{max_tokens:384},onExchange:exchange=>exchanges.push(structuredClone(exchange))});
 const result=await improveProgram({folder:Folder.fromFiles(row.files),contract:row.contract,cases:row.cases,policy:row.policy,improver:driver,executor:()=>{throw Error('pure source fixtures use no executor model');},executorId:'exact-native',budget:{maxModelCalls:24,maxRollouts:40,maxProposals:3,maxElapsedMs:180000},signal:AbortSignal.timeout(180000),trace:trace=>traces.push(trace),directory:join(output,row.id+'.journal')});
 const accepted=!!result.validation&&result.state.done&&result.state.incumbent===result.folder.digest;
 const artifact={id:row.id,role:'student',checkpoint:model,accepted,caseDefinition:row,state:result.state,sourceManifest:result.sourceManifest,validation:result.validation,disposition:result.disposition,error:'error'in result?result.error:null,ledger:result.ledger,exchanges,traces};
 await writeFile(join(output,row.id+'.json'),JSON.stringify(artifact));results.push({id:row.id,accepted,disposition:result.disposition});console.log(JSON.stringify(results.at(-1)));
}
await writeFile(join(output,'summary.json'),JSON.stringify({results,allocation:allocation.snapshot()},null,2));
