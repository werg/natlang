/** Independently execute reconstructed original failures before requesting source edits. */
import {readFile,mkdir,writeFile,readdir} from 'node:fs/promises';
import {join,dirname} from 'node:path';
import {Folder,SourceEvaluator,openAICompatibleModelTurn,AssignmentBudget,OperationJournal} from '../../dist/index.js';
import {UsageGateway} from '../../dist/evaluation/usage.js';
import {createPiModelBackend} from '../../dist/model/pi-provider.js';
const [input,output,endpoint='http://127.0.0.1:8081',model='/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf']=process.argv.slice(2);
if(!output)throw Error('usage: probe-targets.mjs TARGETS_JSONL OUTPUT_DIR [ENDPOINT MODEL]');
await mkdir(output,{recursive:true});
const allocation=new AssignmentBudget({collection:120,modelCalls:12480,caseExecutions:4800,trainingJobs:6,trainingUpdates:1200,confirmation:1,cost:100},{trainingJobs:4,trainingUpdates:800,confirmation:1},3,new OperationJournal(join(dirname(output),'assignment-allocation')));
const targets=(await readFile(input,'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse),rows=[];
const backend=endpoint.startsWith('pi:')?createPiModelBackend(endpoint.slice(3),model):undefined;
const present=new Set(await readdir(output));
for(const target of targets){
 const path=join(output,target.id+'.json');if(present.has(target.id+'.json')){rows.push(JSON.parse(await readFile(path,'utf8')));continue;}
 const gateway=new UsageGateway({maxModelCalls:12,maxRollouts:1,maxProposals:0,maxElapsedMs:120000});
 const driver=backend?backend.turn.bind(backend):openAICompatibleModelTurn({endpoint,model});
 const evaluator=new SourceEvaluator(target.contract,target.cases,driver,gateway,{executorId:model,signal:AbortSignal.timeout(120000)});
 const source=Folder.fromFiles(target.files).snapshot();const check=await evaluator.check(source);
 let report,error;
 if(check.valid){allocation.allocate({modelCalls:12,caseExecutions:1});try{report=await evaluator.evaluate(source,{split:'train',caseIds:[target.cases.find(row=>row.split==='train').id]});}catch(failure){error=String(failure);}}
 const row={id:target.id,family:target.family,source:source.digest,check,report,error,ledger:gateway.snapshot(),disposition:!check.valid?'source-reconstruction-invalid':error?'probe-infrastructure-failure':report.quality<1?'reproducible-failure':'baseline-passed'};
 await writeFile(path,JSON.stringify(row));rows.push(row);await writeFile(join(output,'summary.json'),JSON.stringify({rows,allocation:allocation.snapshot()},null,2));console.log(JSON.stringify({id:row.id,disposition:row.disposition}));
}
backend?.close();
