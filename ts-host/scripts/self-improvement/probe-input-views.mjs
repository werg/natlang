/** Matched semantic inputs with the preceding and current shared value renderer. No source optimization. */
import {mkdir,readFile,writeFile,appendFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {pathToFileURL} from 'node:url';
import * as current from '../../dist/index.js';
import {UsageGateway} from '../../dist/evaluation/usage.js';
const [previousRuntime,output]=process.argv.slice(2);
if(!output)throw Error('usage: probe-input-views.mjs PREVIOUS_FROZEN_RUNTIME OUTPUT');
await mkdir(output,{recursive:true});
const previous=await import(pathToFileURL(resolve(previousRuntime,'index.js')).href);
const files={'recommendation.nl':'---\nargs: {bundle: ReviewBundle}\nreturns: boolean\n---\nDoes bundle.review recommend the product overall? Judge that review here by its meaning, including contrast, negation and sarcasm. bundle.background and bundle.diagnostics are unrelated logs, not reviews. Return the boolean with eval finish:true.\n','types.ts':'export type ReviewBundle={background:string;diagnostics:{note:string}[];review:string};\n'};
const texts=[['train','Setup nearly made me return it. Now I recommend it to everyone.',true],['train','What a triumph: it failed twice in a week. I cannot recommend buying this.',false],['validation','The finish is ugly, but I would choose this again for how well it works.',true],['validation','At least the packaging was nice. I advise everyone to avoid the actual product.',false],['test','I was sceptical. After a year of dependable service I bought another for my mother.',true],['test','A flawless dust collector: that is all this broken machine is good for. Do not buy it.',false]];
const cases=texts.map(([split,review,expected],index)=>({id:'input-overview-'+index,group:'input-overview-'+index,split,args:[{background:'Unrelated diagnostic archive. '.repeat(600),diagnostics:[{note:'Old logs without task evidence. '.repeat(150)}],review}],expected}));
const contract={entry:'recommendation.nl',exportName:'default',programId:'shared-input-overview'};
const executor={endpoint:'http://127.0.0.1:8081',model:'/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'};
const budget={maxModelCalls:80,maxRollouts:4,maxProposals:0,maxElapsedMs:1800000};
const protocol={files,cases,contract,executor,budget,previousRuntime:resolve(previousRuntime),design:'Same source, train inputs, executor and seed. Compare shared native rendering; reserved validation/test examples stay unopened. Descriptive two-case language usability probe.'};
const protocolPath=join(output,'protocol.json');
try{const old=JSON.parse(await readFile(protocolPath,'utf8'));if(JSON.stringify(old)!==JSON.stringify(protocol))throw Error('Input-view protocol changed.');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(protocolPath,JSON.stringify(protocol,null,2)+'\n');}
const allocation=new current.OperationJournal(join(output,'allocation'));
const gateway=new UsageGateway(budget,allocation.read('ledger')?.value);gateway.onUpdate=ledger=>allocation.record('ledger',ledger);
const signal=AbortSignal.timeout(Math.max(1,budget.maxElapsedMs-gateway.ledger.elapsedMs));
const result={arms:{},protocol:protocolPath};
for(const [arm,sdk]of [['previous',previous],['overview',current]]){
 const cached=allocation.read('arm:'+arm);if(cached?.status==='done'){result.arms[arm]=cached.value;continue;}
 const raw=current.openAICompatibleModelTurn({...executor,request:{temperature:0.2}});
 const driver=async(request,signal)=>{const turn=await raw(request,signal);await appendFile(join(output,'exchanges.ndjson'),JSON.stringify({arm,request,turn})+'\n');return turn;};
 const evaluator=new sdk.SourceEvaluator(contract,cases,driver,gateway,{executorId:executor.model,timeoutMs:600000,signal,journal:new sdk.OperationJournal(join(output,arm+'-journal'))});
 const report=await evaluator.evaluate(sdk.Folder.fromFiles(files).snapshot(),{split:'train',seed:0});
 allocation.record('arm:'+arm,report);result.arms[arm]=report;
 await writeFile(join(output,'progress.json'),JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({arm,quality:report.quality,modelCalls:report.modelCalls,gatesPassed:report.gatesPassed}));
}
result.ledger=gateway.snapshot();
await writeFile(join(output,'result.json'),JSON.stringify(result,null,2)+'\n');
const row={version:'natlang.improvement-case/1',id:'observed-large-input-overview',family:'large-argument-semantic-judgment',files,contract,cases,policy:{maxExperiments:3,maxPopulation:3,mode:'structural',strategy:'adaptive',objective:'model-calls',goal:'Preserve semantic judgment of the actual review while eliminating unnecessary input discovery, paging and repeated inspection.',allowedFiles:Object.keys(files)},budget:{maxModelCalls:100,maxRollouts:30,maxProposals:3,maxElapsedMs:1200000},incidents:[{id:'observed-luna-repeated-context-inspection',cluster:'large-input-discovery',split:'train',route:'structural-repair',reason:'Actual Luna optimizer traces repeatedly redisplayed paged request/context objects; exercise the general input presentation problem outside the optimizer.',source:{file:'runs/luna-bonsai-prepared-optimizer-20261001/exchanges.ndjson'}}],sourceGroups:['large-argument-semantic-judgment'],provenance:{kind:'general-native-input-usability',positiveSFT:false,oracle:'Fresh independently declared semantic labels',disposition:'requires-native-collection-and-independent-replay'}};
await writeFile(join(output,'cases.jsonl'),JSON.stringify(row)+'\n');
