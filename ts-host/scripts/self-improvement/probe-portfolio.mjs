/** A bounded deployment-only routing example; training/validation evidence, not a confirmation claim. */
import {mkdir,writeFile} from 'node:fs/promises';
import {join,dirname} from 'node:path';
import {Folder,composePortfolio,SourceEvaluator,AssignmentBudget,OperationJournal} from '../../dist/index.js';
import {UsageGateway} from '../../dist/evaluation/usage.js';
import {createPiModelBackend} from '../../dist/model/pi-provider.js';
const output=process.argv[2];if(!output)throw Error('usage: probe-portfolio.mjs OUTPUT_DIR');await mkdir(output,{recursive:true});
const allocation=new AssignmentBudget({collection:120,modelCalls:12480,caseExecutions:4800,trainingJobs:6,trainingUpdates:1200,confirmation:1,cost:100},{trainingJobs:4,trainingUpdates:800,confirmation:1},3,new OperationJournal(join(dirname(output),'assignment-allocation')));
allocation.allocateOnce('portfolio-deployment-example',{collection:1,modelCalls:20,caseExecutions:20});
const contract={entry:'main.ts',exportName:'solve',programId:'portfolio-example',signature:'solve(value:number):Promise<number>'};
const positive=Folder.fromFiles({'main.ts':'export async function solve(value:number):Promise<number>{return value*2;}'}).snapshot();
const negative=Folder.fromFiles({'main.ts':'export async function solve(value:number):Promise<number>{return value*3;}'}).snapshot();
const portfolio=composePortfolio([{name:'double_positive',source:positive,contract},{name:'triple_negative',source:negative,contract}],{parameters:'value:number',arguments:['value'],returns:'number'},'double_positive');
const values=[-2,1,0,3,-4],cases=values.map((value,index)=>({id:'routing-'+index,group:'routing-'+index,split:index<3?'train':'validation',args:[value],expected:value>=0?value*2:value*3}));
const backend=createPiModelBackend('openai-codex','gpt-6-luna'),exchanges=[];
const driver=async(request,signal)=>{const turn=await backend.turn(request,signal);exchanges.push({request,turn});return turn;};
const gateway=new UsageGateway({maxModelCalls:20,maxRollouts:20,maxProposals:0,maxElapsedMs:300000});
try{
 const evaluator=new SourceEvaluator(contract,cases,driver,gateway,{executorId:'openai-codex/gpt-6-luna',signal:AbortSignal.timeout(300000)});
 const baseline={train:await evaluator.evaluate(positive,{split:'train'}),validation:await evaluator.evaluate(positive,{split:'validation'})};
 const selected={train:await evaluator.evaluate(portfolio,{split:'train'}),validation:await evaluator.evaluate(portfolio,{split:'validation'})};
 const result={schema:'natlang.portfolio-development-example/1',source:portfolio.digest,files:Object.fromEntries(portfolio.filePaths().map(path=>[path,new TextDecoder().decode(portfolio.readBytesSync(path))])),contract,cases,baseline,selected,ledger:gateway.snapshot(),exchanges,claim:'Deployment-only composition example; source and output are checked on declared finite cases. Routing incurs model calls; the single specialist does not. No held-out or statistical benefit claim.'};
 await writeFile(join(output,'result.json'),JSON.stringify(result));console.log(JSON.stringify({baseline:baseline.validation.quality,portfolio:selected.validation.quality,modelCalls:result.ledger.usage.modelCalls}));
}finally{backend.close();}
