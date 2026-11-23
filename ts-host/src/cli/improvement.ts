import {readFileSync,mkdirSync,writeFileSync,readdirSync,existsSync,statSync} from 'node:fs';
import {resolve,join,dirname} from 'node:path';
import {Folder} from '../native/scoped-fs.js';
import {buildDirectoryCasePack,evaluateDirectorySplitWithModel,improveDirectory,improveProgram,protectedProgramPaths,type DirectoryProposal} from '../improvement/program.js';
import {proposeFromRuns,type FunctionDecision} from '../improvement/propose-from-runs.js';
import {createNatlangRuntime} from '../runtime/node.js';
import {SourceEvaluator} from '../improvement/host.js';
import {adoptSource,rollbackSource,recoverAdoption,type SourceManifest,type AdoptionRecord} from '../improvement/adoption.js';
import {adoptCall,type AdoptionVerdict} from '../improvement/adopt-call.js';
import {readExamples} from '../calls/examples.js';
import {machineStoreRoot,openMachineStore} from '../machine-store.js';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
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
 if(!command)throw Error('usage: natlang improve CASE.json --out DIR --model MODEL; or init|resume|step|inspect|export|adopt|adopt-call|rollback');
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
 if(command==='adopt-call')return adoptCallCommand(rest,option,flags);
 if(command==='adopt'){
  const directory=resolve(rest[0]??''),checkout=option('--into');if(!checkout)throw Error('adopt requires --into CHECKOUT');
  const manifest=read(join(directory,'source.json')) as SourceManifest;
  const result=read(join(directory,'result.json'));
  if(result.source!==manifest.source || !result.state?.done || !result.validation?.gatesPassed || result.validation.quality < result.baseline?.quality || result.transformation?.eligible===false)throw Error('adoption requires a completed independently evaluated eligible source');
  const evaluator=new SourceEvaluator(manifest.contract as never,[],()=>{throw Error('build verification uses no inference');},new UsageGateway({maxModelCalls:0,maxProposals:0,maxRollouts:0}),{executorId:'compile-only'});
  const record=await adoptSource(checkout,manifest,source=>evaluator.check(source),join(directory,'adoption.json'));console.log(JSON.stringify({changed:record.changed,source:record.afterDigest}));return 0;
 }
 const resume=command==='resume'||command==='step';
 if(!resume){
  const target=resolve(command);
  if(existsSync(target)&&statSync(target).isDirectory())return improveDirectoryCommand(target,option,flags);
 }
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

const CLI=fileURLToPath(new URL('./main.js',import.meta.url));
/** Run this natlang CLI and parse its --json report (natlang check / natlang test print one). */
function cliJson(argv:string[]):Promise<{code:number;report:Record<string,unknown>|null;stderr:string}>{
 return new Promise((done,fail)=>{
  const child=spawn(process.execPath,[CLI,...argv],{stdio:['ignore','pipe','pipe'],env:{...process.env,NO_COLOR:'1',FORCE_COLOR:'0'}});
  let out='',err='';child.stdout.on('data',chunk=>{out+=chunk;});child.stderr.on('data',chunk=>{err+=chunk;});
  child.on('error',fail);
  child.on('close',code=>{let report:Record<string,unknown>|null=null;try{report=JSON.parse(out);}catch{report=null;}done({code:code??1,report,stderr:err.trim()});});
 });
}
/**
 * natlang improve adopt-call CALL [--into DIR] [--store DIR] [--record FILE] [--dry-run] [--untested] [--profile P]
 * [--provider ID] [--model ID] [--yes] [--json]: check, test and install the code a successful call fixed (WS1 L3).
 */
async function adoptCallCommand(rest:string[],option:(name:string,fallback?:string)=>string|undefined,flags:string[]):Promise<number>{
 const call=rest[0];
 if(!call||call.startsWith('--'))throw Error('usage: natlang improve adopt-call CALL [--into DIR] [--store DIR] [--record FILE] [--dry-run] [--untested] [--json]');
 const json=flags.includes('--json'),untested=flags.includes('--untested');
 const selection=['--profile','--provider','--model'].flatMap(name=>{const value=option(name);return value?[name,value]:[];});
 const storeRoot=option('--store')??machineStoreRoot();
 if(!storeRoot)throw Error('the call store is off on this machine (NATLANG_CALL_STORE=off); pass --store DIR to read one');
 const store=openMachineStore(storeRoot);
 try{
  const {resolveCall}=await import('./calls.js');
  const id=resolveCall(store,call,'natlang improve adopt-call CALL');
  const verify=async(directory:string):Promise<AdoptionVerdict>=>{
   const checked=await cliJson(['check',directory,'--json']);
   const check={ok:checked.code===0&&checked.report?.ok===true,diagnostics:((checked.report?.diagnostics as {message?:string}[]|undefined)??[]).map(row=>row.message??JSON.stringify(row)).concat(checked.report?[]:[checked.stderr])};
   if(!check.ok)return{check,test:{ok:false,skipped:'natlang check failed'}};
   const tested=readExamples(directory).filter(example=>Object.hasOwn(example,'expected'));
   if(!tested.length)return{check,test:untested?{ok:true,skipped:'no saved examples (--untested)'}:{ok:false,detail:'the program has no saved examples to test the fix against; save one with natlang traces fix CALL --expected JSON, or pass --untested to adopt on natlang check alone'}};
   const ran=await cliJson(['test',directory,'--json',...selection,...(flags.includes('--yes')?['--yes']:[])]);
   const report=ran.report??{};
   return{check,test:{ok:ran.code===0&&report.ok===true,passed:Number(report.passed??0),failed:Number(report.failed??0),
    ...(ran.code===0&&report.ok===true?{}:{detail:report.failed!==undefined?`${report.failed} of ${tested.length} saved examples fail with the fix`:ran.stderr.split('\n').slice(-3).join(' ')||'natlang test did not report'})}};
  };
  const into=option('--into'),recordPath=option('--record');
  const result=await adoptCall(store,id,{verify,dryRun:flags.includes('--dry-run'),...(into?{into}:{}),...(recordPath?{recordPath}:{})});
  if(json)console.log(JSON.stringify({status:result.status,call:result.call,directory:result.directory,changed:result.changed,reasons:result.reasons,verdict:result.verdict??null,record:result.recordPath??null},null,2));
  else console.log(result.status==='accepted'?`adopted ${result.changed.join(', ')} into ${result.directory}\nnatlang improve rollback ${JSON.stringify(result.recordPath)} restores the previous source`:
   result.status==='checked'?`the fix to ${result.changed.join(', ')} passes natlang check and natlang test; run without --dry-run to install it`:
   `not adopted: ${result.reasons.join('; ')}`);
  return result.status==='rejected'?1:0;
 }finally{store.close();}
}

/**
 * `natlang improve DIR [--proposal FILE] [--model MODEL] [--store DIR] [--seed N] [--max-functions N] [--record FILE] [--dry-run] [--json]`
 * (WS8 N4): builds the deterministic support/dev/confirmation case pack from `<DIR>/examples.json` + recorded calls in
 * the CallStore. Without --proposal, the natural-language improver (propose-from-runs.ts) reads the program's recorded
 * failures, escalations, repairs and feedback plus the support examples it answers wrongly, and proposes a change set
 * on MODEL. Either change set is verified through a temp-copy check + split evaluation transaction and installed via
 * adoptSource.
 */
async function improveDirectoryCommand(directory:string,option:(name:string,fallback?:string)=>string|undefined,flags:string[]):Promise<number>{
 const json=flags.includes('--json'),dryRun=flags.includes('--dry-run');
 const seedRaw=option('--seed','0')!,seed=Number(seedRaw);
 if(!Number.isSafeInteger(seed))throw Error(`--seed must be a safe integer, got ${seedRaw}`);
 const storeOpt=option('--store');
 const defaultStore=machineStoreRoot();
 const storeRoot=storeOpt?resolve(storeOpt):(defaultStore&&existsSync(join(defaultStore,'calls.sqlite'))?defaultStore:null);
 const store=storeRoot?openMachineStore(storeRoot):undefined;
 const proposalPath=option('--proposal'),recordPath=option('--record');
 const provider=option('--provider'),model=option('--model'),server=option('--server','http://127.0.0.1:8081');
 const backend=provider&&model?createPiModelBackend(provider,model):undefined;
 try{
  // --dry-run without a change set or a model to propose one: show the case splits the gate would use.
  if(!proposalPath&&!model&&dryRun){
   const casePack=buildDirectoryCasePack(directory,{store,seed});
   const checked=buildProject({project:directory,write:false,emit:false,writeDeclarations:false});
   const report={directory,seed,ok:checked.ok,diagnostics:checked.diagnostics.filter(row=>row.severity==='error').map(row=>row.message),
    cases:casePack.cases.length,splits:{support:casePack.splits.support.length,dev:casePack.splits.dev.length,confirmation:casePack.splits.confirmation.length},sources:casePack.sources};
   if(json)console.log(JSON.stringify(report,null,2));
   else console.log(`${directory}: ${report.cases} cases (support=${report.splits.support}, dev=${report.splits.dev}, confirmation=${report.splits.confirmation}); check=${report.ok?'ok':'failed'}`);
   return report.ok?0:1;
  }
  let proposal:DirectoryProposal|undefined;
  if(proposalPath){
   const raw=read(resolve(proposalPath)) as Record<string,unknown>;
   proposal=raw&&typeof raw==='object'&&'files'in raw&&raw.files&&typeof raw.files==='object'?
    {files:raw.files as Record<string,string|null>,...(typeof raw.summary==='string'?{summary:raw.summary}:{})}:
    {files:raw as Record<string,string|null>};
  }
  if(!proposal&&!model)throw Error('natlang improve DIR proposes a change from the program\'s recorded runs on a model: pass --model MODEL '+
   '(with --server URL or --provider ID), or --proposal FILE ({ "files": { "PATH": "new text" or null } }) to check and adopt a change set you wrote');
  const maxRaw=option('--max-functions','3')!,maxFunctions=Number(maxRaw);
  if(!Number.isSafeInteger(maxFunctions)||maxFunctions<1)throw Error(`--max-functions must be a positive whole number, got ${maxRaw}`);
  await backend?.prepare();
  const driver=model?(backend?(request:ModelTurnRequest,signal?:AbortSignal,options?:ModelTurnOptions)=>backend.turn(request,signal,options):openAICompatibleModelTurn({endpoint:server!,model})):undefined;
  let decisions:FunctionDecision[]=[];
  const propose=proposal?undefined:async(context:Parameters<NonNullable<Parameters<typeof improveDirectory>[0]['propose']>>[0])=>{
   // The improver's calls are ordinary recorded calls (training data for its stages), on the same model.
   const runtime=createNatlangRuntime({model:driver!});
   const proposed=await runtime.run(()=>proposeFromRuns(context,{store:store as never,maxFunctions,protectedPaths:protectedProgramPaths(directory),
    evaluateSupport:cases=>evaluateDirectorySplitWithModel(directory,cases,driver!)}));
   decisions=proposed?.decisions??[];
   return proposed;
  };
  const result=await improveDirectory({directory,store,seed,...(proposal?{proposal}:{propose}),model:driver,recordPath,dryRun});
  if(json){
   console.log(JSON.stringify({status:result.status,directory:result.directory,changed:result.changed,reasons:result.reasons,summary:result.summary??null,
    splits:{support:result.casePack.splits.support.length,dev:result.casePack.splits.dev.length,confirmation:result.casePack.splits.confirmation.length},
    sources:result.casePack.sources,check:result.check??null,baseline:result.baseline??null,proposed:result.proposed??null,record:result.recordPath??null,
    ...(proposal?{}:{improver:decisions})},null,2));
   return result.status==='rejected'?1:0;
  }
  for(const decision of decisions)console.log(`${decision.path}: ${decision.changed?'revised':decision.problem?`not revised (${decision.problem})`:'left as is'} - ${decision.diagnosis}`);
  if(result.status==='accepted'){
   console.log(`adopted ${result.changed.join(', ')} into ${result.directory}\nnatlang improve rollback ${JSON.stringify(result.recordPath)} restores the previous source`);
  }else if(result.status==='checked'){
   console.log(`verified ${result.changed.join(', ')} (${result.casePack.cases.length} cases); run without --dry-run to install`);
  }else if(result.status==='unchanged'){
   console.log(`unchanged: ${result.reasons.join('; ')}`);
  }else{
   console.log(`rejected: ${result.reasons.join('; ')}`);
  }
  return result.status==='rejected'?1:0;
 }finally{store?.close();backend?.close();}
}
