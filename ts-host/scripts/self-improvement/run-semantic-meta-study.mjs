/** Optimize authored optimizer source using finite, independent Luna/Bonsai tasks. */
import {mkdir,cp,symlink,readdir,readFile,writeFile,appendFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {followupCases} from './build-followup-cases.mjs';
const output=resolve(process.argv[2]??'../runs/native-next-six-20261001/meta');await mkdir(output,{recursive:true});
const runtimePath=join(output,'runtime');
try{await readFile(join(runtimePath,'identity.json'));}catch(error){if(error.code!=='ENOENT')throw error;await cp(new URL('../../dist/',import.meta.url),runtimePath,{recursive:true});await symlink(new URL('../../node_modules/',import.meta.url).pathname,join(runtimePath,'node_modules'));const hashes={};async function walk(dir){for(const e of await readdir(dir,{withFileTypes:true})){const path=join(dir,e.name);if(e.isDirectory())await walk(path);else if(e.name.endsWith('.js'))hashes[path.slice(runtimePath.length+1)]=createHash('sha256').update(await readFile(path)).digest('hex');}}await walk(runtimePath);await writeFile(join(runtimePath,'identity.json'),JSON.stringify(hashes));await cp(new URL('../../prelude.js',import.meta.url),join(output,'prelude.js'));}
const imp=path=>import(pathToFileURL(join(runtimePath,path)).href);
const {Folder,improveProgram,improverExecution,SourceEvaluator,OperationJournal,openAICompatibleModelTurn}=await imp('index.js');
const {AUTHORED_IMPROVER}=await imp('improvement/authored-source.js');
const {createPiModelBackend}=await imp('model/pi-provider.js');const {UsageGateway}=await imp('evaluation/usage.js');
const priorPath=process.argv[3]?resolve(process.argv[3]):undefined;
const prior=priorPath?JSON.parse(await readFile(priorPath,'utf8')):undefined;
const budget={maxModelCalls:400-(prior?.ledger.usage.modelCalls??0),maxRollouts:100-(prior?.ledger.rollouts??0),maxProposals:20-(prior?.ledger.proposals??0),maxElapsedMs:7200000-(prior?.ledger.elapsedMs??0)};
if(Object.values(budget).some(value=>value<=0))throw Error('Meta assignment allocation exhausted');
const sourceCases=followupCases();
const definitions=[
 ['followup-priority','train',[
  {args:['The archived checklist says urgent. It is obsolete; handle this whenever convenient.'],expected:'routine'},
  {args:['Please act now: the live service is down and customers cannot sign in.'],expected:'urgent'},
  {args:['Do not escalate this; next week is fine despite the old urgent label.'],expected:'routine'}]],
 ['followup-resolution','train',[
  {args:['I do not want a refund. Please repair the broken latch.'],expected:'repair'},
  {args:['A repair will not solve this for me. Please return my payment.'],expected:'refund'},
  {args:['The refund offer is declined; fixing the hinge is my final request.'],expected:'repair'}]],
 ['followup-quantity','validation',[
  {args:['We considered seven lamps. The final committed order is three lamps.'],expected:{item:'lamps',quantity:3}},
  {args:['An estimate mentioned nine desks. Final order: two desks.'],expected:{item:'desks',quantity:2}},
  {args:['Six tables was only a proposal. We committed to four tables.'],expected:{item:'tables',quantity:4}}]],
 ['followup-confirmed-events','test',[
  {args:['Oak was suggested only. Pine is currently confirmed. Elm was cancelled.'],expected:['Pine']},
  {args:['Aster is confirmed; Birch is cancelled; Cedar is also confirmed.'],expected:['Aster','Cedar']},
  {args:['Maple is unconfirmed; Rowan was cancelled; Willow remains confirmed.'],expected:['Willow']}]]
];
const targets=definitions.map(([id,_split,rows])=>{const base=sourceCases.find(row=>row.id===id);return {id:'meta-'+id,files:base.files,contract:{...base.contract,programId:'meta-'+id},policy:{...base.policy,maxExperiments:1},cases:rows.map((row,i)=>({...row,id:'meta-'+id+'-'+i,group:'meta-independent-'+id+'-'+i,split:['train','validation','test'][i]}))};});
const cases=targets.map((target,i)=>({id:target.id,group:'meta-target-family-'+definitions[i][0],split:definitions[i][1],args:[],expected:null}));
const protocol={schema:'natlang.semantic-meta-study/1',optimizer:{provider:'openai-codex',model:'gpt-6-luna'},student:{endpoint:'http://127.0.0.1:8081',model:'/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'},targets,cases,source:AUTHORED_IMPROVER,budget,...(priorPath?{priorAttempt:{path:priorPath,sha256:createHash('sha256').update(JSON.stringify(prior)).digest('hex'),disposition:prior.disposition,ledger:prior.ledger}}:{}),policy:{maxExperiments:1,maxPopulation:3,strategy:'adaptive',mode:'instruction',objective:'model-calls',allowedFiles:['improveStep.nl','improveStep/rewriteProgram.nl'],goal:'Improve this natlang-native optimizer, especially excess inspection, source reprinting, invalid eval quoting, and redundant actions. Edit its authored instructions so Luna diagnoses and edits with normal file tools cleanly and finishes promptly. Preserve finite iteration, source/state ownership, independent evaluation, rejection and selection. Inner request counts include the complete Luna optimizer plus Bonsai student run; preserve independently verified task quality while lowering those counts.'},design:'One outer source-edit experiment. Fixed independent development targets; one held-out target family. No answer or trace from the outer test target enters development. No weight training. No special runtime prompt or tool filtering.'};
const protocolPath=join(output,'protocol.json');try{if(JSON.stringify(JSON.parse(await readFile(protocolPath,'utf8')))!==JSON.stringify(protocol))throw Error('Frozen meta protocol changed');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(protocolPath,JSON.stringify(protocol,null,2));}
try{await readFile(join(output,'result.json'));console.log('Meta study already completed');process.exit(0);}catch(error){if(error.code!=='ENOENT')throw error;}
const backend=createPiModelBackend(protocol.optimizer.provider,protocol.optimizer.model),journal=new OperationJournal(join(output,'journal'));
const gateway=new UsageGateway(protocol.budget,journal.read('ledger')?.value);gateway.onUpdate=ledger=>journal.record('ledger',ledger);
const signal=AbortSignal.timeout(Math.max(1,protocol.budget.maxElapsedMs-gateway.ledger.elapsedMs));
const log=async(role,request,turn)=>{await appendFile(join(output,'exchanges.ndjson'),JSON.stringify({role,request,turn})+'\n');console.log(JSON.stringify({role,calls:turn.calls?.map(call=>call[0])}));return turn;};
const optimizer=role=>async(request,abort)=>log(role,request,await backend.turn(request,abort));
const rawStudent=openAICompatibleModelTurn({...protocol.student,request:{temperature:0.2}});
const student=async(request,abort)=>log('student',request,await rawStudent(request,abort));
try{
 const execution=improverExecution(targets,{improver:optimizer('inner-optimizer'),executor:student,improverId:'openai-codex/gpt-6-luna',executorId:protocol.student.model},{signal,executorTimeoutMs:300000});
 const result=await improveProgram({folder:Folder.fromFiles(protocol.source),contract:{entry:'improveStep.nl',exportName:'default',programId:'semantic-meta-improver'},cases,policy:protocol.policy,improver:optimizer('outer-optimizer'),executor:student,executorId:protocol.student.model,executeCase:execution,evaluationLevel:2,budget:protocol.budget,gateway,signal,directory:join(output,'journal'),trace:trace=>{void appendFile(join(output,'traces.ndjson'),JSON.stringify(trace)+'\n');}});
 const {folder,evaluator,...portable}=result;await writeFile(join(output,'result.json'),JSON.stringify(portable,null,2));
 // Freeze once. Reuse development cache, never use the final report to edit this candidate.
 if(result.validation&&result.state.done){const confirmation=await evaluator.confirmPair(Folder.fromFiles(protocol.source).snapshot(),folder,'semantic-meta-final-v1');await writeFile(join(output,'confirmation.json'),JSON.stringify({confirmation,ledger:gateway.snapshot()},null,2));}
 console.log(JSON.stringify({done:true,disposition:result.disposition,quality:result.validation?.quality,calls:gateway.ledger.usage.modelCalls}));
}finally{backend.close();}
