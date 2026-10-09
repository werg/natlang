/** Real two-level development experiment: the editable source is the improver itself. */
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {join,dirname} from 'node:path';
import {Folder,improveProgram,improverExecution,FrozenImprover,AssignmentBudget,OperationJournal} from '../../dist/index.js';
import {AUTHORED_IMPROVER} from '../../dist/improvement/authored-source.js';
import {createPiModelBackend} from '../../dist/model/pi-provider.js';
const [output,provider='openai-codex',model='gpt-6-luna']=process.argv.slice(2);if(!output)throw Error('usage: improve-frozen-improver.mjs OUTPUT_DIR [PROVIDER MODEL]');
await mkdir(output,{recursive:true});
const definitions=[
 ['absolute','number','number','return value;',[1,-2,-3,4,-5,6,-7],v=>Math.abs(v),'Return the absolute value.'],
 ['uppercase','string','string','return value;',['a','b','c','d','e','f','g'],v=>v.toUpperCase(),'Return the upper-case string.'],
 ['double','number','number','return value+2;',[0,1,3,4,5,6,7],v=>v*2,'Return twice the number.'],
 ['prefix','string','boolean','return false;',['ax','by','az','aq','bt','au','cv'],v=>v.startsWith('a'),'Return whether the string starts with a.'],
 ['minimum','number[]','number','return value[0];',[[3,1],[5,2],[-1,-3],[8,4],[0,-2],[9,6],[-5,-8]],v=>Math.min(...v),'Return the smallest number in the nonempty list.'],
 ['length','string','number','return 0;',['ab','xyz','','hello','a','four','example'],v=>v.length,'Return the string length.'],
 ['reverse','string[]','string[]','return value;',[['a','b'],['c','d','e'],[],['f','g'],['h','i'],['j','k'],['l','m','n']],v=>[...v].reverse(),'Return a reversed copy of the list.']
];
const targets=definitions.map(([family,type,returns,body,inputs,oracle,goal])=>{const id='meta-target-'+family;return {id,files:{'main.ts':`export function solve(value:${type}):${returns}{${body}}`},contract:{entry:'main.ts',exportName:'solve',programId:id},cases:inputs.map((input,index)=>({id:id+'-'+index,group:id+'-'+index,split:index<3?'train':index<5?'validation':'test',args:[input],expected:oracle(input)})),policy:{maxExperiments:1,maxPopulation:3,strategy:'adaptive',mode:'structural',goal,allowedFiles:['main.ts']}};});
const cases=targets.map((target,index)=>({id:target.id,group:target.id,split:index<3?'train':index<5?'validation':'test',args:[],expected:null}));
const protocol={schema:'natlang.frozen-improver-experiment/1',targets,cases,provider,model,budget:{maxModelCalls:504,maxRollouts:240,maxProposals:32,maxElapsedMs:1800000},policy:{maxExperiments:1,maxPopulation:3,strategy:'adaptive',mode:'instruction',goal:'Refine the improver instructions for a weak coding model. Make the diagnose, hypothesize and edit stages clearer and shorter, especially helper editing authority and honest stopping. Preserve typed contracts, independent evaluation and finite iteration. Improve completion on the independently executed lower-level repair tasks.',allowedFiles:['improveStep/diagnose.nl','improveStep/hypothesize.nl','improveStep/editSource.nl','improveStep/editSourceStructural.nl']}};
const protocolPath=join(output,'protocol.json'),text=JSON.stringify(protocol,null,2);try{if(await readFile(protocolPath,'utf8')!==text)throw Error('meta protocol is frozen');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(protocolPath,text);}
const assignment=new AssignmentBudget({collection:120,modelCalls:12480,caseExecutions:4800,trainingJobs:6,trainingUpdates:1200,confirmation:1,cost:100},{trainingJobs:4,trainingUpdates:800,confirmation:1},3,new OperationJournal(join(dirname(output),'assignment-allocation')));
await new OperationJournal(join(output,'operations')).run('allocate',async()=>{assignment.attempt('frozen-improver-development');assignment.allocate({collection:1,modelCalls:504,caseExecutions:240});return assignment.snapshot();});
const backend=createPiModelBackend(provider,model),exchanges=[],traces=[];
const driver=async(request,signal)=>{const response=await backend.turn(request,signal);exchanges.push({request,response});return response;};
try{
 const execution=improverExecution(targets,{improver:driver,executor:()=>{throw Error('meta targets are exact TypeScript');},improverId:provider+'/'+model,executorId:'exact-source'});
 const baseline=Folder.fromFiles(AUTHORED_IMPROVER),frozen=new FrozenImprover(baseline.snapshot());
 const result=await improveProgram({folder:baseline,contract:{entry:'improveStep/diagnose.nl',exportName:'default',programId:'frozen-improver'},cases,policy:protocol.policy,improver:driver,executor:driver,executorId:provider+'/'+model,budget:protocol.budget,evaluationLevel:2,executeCase:execution,directory:join(output,'run'),trace:trace=>traces.push(trace)});
 if(result.validation?.gatesPassed&&result.state.done)frozen.adopt(result.folder);
 await writeFile(join(output,'result.json'),JSON.stringify({protocol,sourceManifest:result.sourceManifest,state:result.state,disposition:result.disposition,baseline:result.baseline,validation:result.validation,error:'error'in result?result.error:null,ledger:result.ledger,sourceDiff:result.sourceDiff,adopted:frozen.snapshot().digest===result.folder.digest,exchanges,traces}));
 console.log(JSON.stringify({disposition:result.disposition,baseline:result.baseline?.quality,validation:result.validation?.quality,done:result.state.done,changed:result.sourceDiff.length}));
}finally{backend.close();}
