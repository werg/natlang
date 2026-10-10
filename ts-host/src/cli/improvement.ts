import {readFileSync,mkdirSync,writeFileSync,readdirSync,existsSync} from 'node:fs';
import {resolve,join,dirname} from 'node:path';
import {Folder} from '../native/scoped-fs.js';
import {improveProgram} from '../improvement/program.js';
import {SourceEvaluator} from '../improvement/host.js';
import {adoptSource,rollbackSource,recoverAdoption,type SourceManifest,type AdoptionRecord} from '../improvement/adoption.js';
import {fingerprint} from '../adaptation/identity.js';
import {UsageGateway} from '../evaluation/usage.js';
import {validateBudget} from '../evaluation/suite.js';
import {buildProject} from '../compiler/node-project.js';
import {openAICompatibleModelTurn} from '../model/openai-compatible.js';
import {createPiModelBackend} from '../model/pi-provider.js';
import type {ModelTurnOptions,ModelTurnRequest} from '../contracts.js';
const read=(path:string)=>JSON.parse(readFileSync(path,'utf8'));
const write=(path:string,value:unknown)=>{mkdirSync(dirname(path),{recursive:true});writeFileSync(path,JSON.stringify(value,null,2)+'\n');};
export async function improvementCommand(args:string[]):Promise<number>{
 const [command,...rest]=args;
 if(!command)throw Error('usage: natlang improve CASE.json --out DIR --model MODEL; or init|resume|step|inspect|export|adopt|rollback');
 const flags=rest;
 const option=(name:string,fallback?:string)=>{const index=flags.indexOf(name);if(index<0)return fallback;const value=flags[index+1];if(!value||value.startsWith('--'))throw Error('missing '+name+' value');return value;};
 if(command==='init'){
  const directory=resolve(rest[0]??'.'),entry=option('--entry','main.ts')!,out=resolve(option('--out',join(directory,'improvement.case.json'))!);
  const build=buildProject({project:directory,programId:option('--program-id','improvement-scaffold'),write:false,constrained:true});
  if(build.diagnostics.some(row=>row.severity==='error'))throw Error(build.diagnostics.map(row=>row.message).join('\n'));
  write(out,{files:build.manifest.adaptation!.sources,contract:{entry,exportName:option('--export','solve'),programId:build.manifest.adaptation!.id},cases:[],policy:{maxExperiments:3,maxPopulation:3,mode:'structural',strategy:'adaptive',goal:'AUTHOR: declare the intended behavior',allowedFiles:Object.keys(build.manifest.adaptation!.sources)},budget:{maxModelCalls:104,maxRollouts:40,maxProposals:6,maxElapsedMs:600000},requiredAuthorInput:['independent train/validation/test cases with expected outcomes','objective and editable files']});
  console.log(out);return 0;
 }
 if(command==='inspect'){
  const directory=resolve(rest[0]??'');console.log(JSON.stringify({run:read(join(directory,'run.json')),result:existsSync(join(directory,'result.json'))?read(join(directory,'result.json')):null,
   operations:existsSync(join(directory,'journal'))?readdirSync(join(directory,'journal')).filter(path=>path.endsWith('.json')).map(path=>({path,record:read(join(directory,'journal',path))})):[]},null,2));return 0;
 }
 if(command==='export'){const directory=resolve(rest[0]??''),out=resolve(option('--out')??'selected-source.json');write(out,read(join(directory,'source.json')));console.log(out);return 0;}
 if(command==='recover'){recoverAdoption(resolve(rest[0]??''));return 0;}
 if(command==='rollback'){rollbackSource(read(resolve(rest[0]??'')) as AdoptionRecord);return 0;}
 if(command==='adopt'){
  const directory=resolve(rest[0]??''),checkout=option('--into');if(!checkout)throw Error('adopt requires --into CHECKOUT');
  const manifest=read(join(directory,'source.json')) as SourceManifest;
  const result=read(join(directory,'result.json'));
  if(result.source!==manifest.source || !result.state?.done || !result.validation?.gatesPassed || result.validation.quality < result.baseline?.quality || result.transformation?.eligible===false)throw Error('adoption requires a completed independently evaluated eligible source');
  const evaluator=new SourceEvaluator(manifest.contract as never,[],()=>{throw Error('build verification uses no inference');},new UsageGateway({maxModelCalls:0,maxProposals:0,maxRollouts:0}),{executorId:'compile-only'});
  const record=await adoptSource(checkout,manifest,source=>evaluator.check(source),join(directory,'adoption.json'));console.log(JSON.stringify({changed:record.changed,source:record.afterDigest}));return 0;
 }
 const resume=command==='resume'||command==='step';
 const output=resume?resolve(rest[0]??''):resolve(option('--out')??'');
 if(!resume&&!option('--out')&&!flags.includes('--dry-run'))throw Error('--out is required');
 const recorded=resume?read(join(output,'run.json')):null;
 const fixture=resume?recorded.fixture:read(resolve(command));
 const provider=option('--provider',recorded?.provider),model=option('--model',recorded?.model);
 const server=option('--server',recorded?.server??'http://127.0.0.1:8081');
 const executorModel=option('--executor-model',recorded?.executorModel??model),executorProvider=option('--executor-provider',recorded?.executorProvider??provider),executorServer=option('--executor-server',recorded?.executorServer??server);
 const executorTimeoutMs=Number(option('--executor-timeout-ms',String(recorded?.executorTimeoutMs??fixture.executorTimeoutMs??120000)));
 const budget=fixture.budget??{maxModelCalls:104,maxRollouts:40,maxProposals:6,maxElapsedMs:600000};validateBudget(budget);
 if(flags.includes('--dry-run')){
  const evaluator=new SourceEvaluator(fixture.contract,fixture.cases,()=>{throw Error('dry run');},new UsageGateway(budget),{executorId:'preflight'});
  const checked=await evaluator.check(Folder.fromFiles(fixture.files).snapshot());console.log(JSON.stringify({checked,cases:fixture.cases.length,budget},null,2));return checked.valid&&fixture.cases.length?0:2;
 }
 if(!model||!executorModel)throw Error('--model is required');
 const profile={fixture,provider:provider??null,model,server,executorProvider:executorProvider??null,executorModel,executorServer,executorTimeoutMs};
 if(recorded&&fingerprint(recorded)!==fingerprint(profile))throw Error('resume requires the recorded model configuration and fixture');
 const cancellation=new AbortController();const stop=()=>cancellation.abort(new Error('user interrupted improvement'));
 process.on('SIGINT',stop);process.on('SIGTERM',stop);
 const improverBackend=provider?createPiModelBackend(provider,model):undefined;
 const executorBackend=executorProvider?createPiModelBackend(executorProvider,executorModel):undefined;
 try{
  await improverBackend?.prepare();await executorBackend?.prepare();
  const improver=improverBackend?(request:ModelTurnRequest,signal?:AbortSignal,options?:ModelTurnOptions)=>improverBackend.turn(request,signal,options):openAICompatibleModelTurn({endpoint:server!,model});
  const executor=executorBackend?(request:ModelTurnRequest,signal?:AbortSignal,options?:ModelTurnOptions)=>executorBackend.turn(request,signal,options):openAICompatibleModelTurn({endpoint:executorServer!,model:executorModel});
  if(!recorded)write(join(output,'run.json'),profile);
  const result=await improveProgram({folder:Folder.fromFiles(fixture.files),contract:fixture.contract,cases:fixture.cases,policy:fixture.policy,improver,executor,executorTimeoutMs,executorId:`${executorProvider??executorServer}/${executorModel}`,budget,signal:cancellation.signal,directory:join(output,'journal'),transformation:fixture.transformation,singleStep:command==='step',trace:trace=>write(join(output,'traces',encodeURIComponent(trace.callId)+'.json'),trace)});
  write(join(output,'source.json'),result.sourceManifest);
  write(join(output,'result.json'),{source:result.folder.digest,state:result.state,validation:result.validation,baseline:result.baseline,disposition:result.disposition,sourceDiff:result.sourceDiff,ledger:result.ledger,transformation:'transformation'in result?result.transformation:undefined,...('error'in result?{error:result.error}:{})});
  console.log(JSON.stringify({directory:output,source:result.folder.digest,disposition:result.disposition,quality:result.validation?.quality??null}));
  return result.disposition==='infrastructure-failure'?3:result.disposition==='interrupted'?4:result.disposition==='incomplete-search'||result.disposition==='no-eligible-promotion'?2:0;
 }finally{process.off('SIGINT',stop);process.off('SIGTERM',stop);improverBackend?.close();executorBackend?.close();}
}
