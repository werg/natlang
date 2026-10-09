import {Folder} from '../native/scoped-fs.js';
import {createNatlangRuntime} from '../runtime/node.js';
import {loadVirtualNatlang} from '../runtime/virtual-project.js';
import {AUTHORED_IMPROVER} from './authored-source.js';
import {TOOLS_PROMPT} from '../native/prompt.js';
import {CounterexampleSuite,type IndependentOracle} from './counterexamples.js';
import {SourceEvaluator} from './host.js';
import {UsageGateway} from '../evaluation/usage.js';
import {improveProgram,type ImproveProgramOptions} from './program.js';
import {counterexampleStop,type CounterexampleStopFacts} from './counterexample-stop.js';
export type CounterexampleState={round:number;done:boolean;suite:string;remainingChecks:number;quality:number;reason:string};
/** An authored outer loop, with one shared allocation and an immutable oracle-backed suite per round. */
export async function counterexampleGuidedImprove(options:ImproveProgramOptions & {oracle:IndependentOracle;maxRounds:number;maxChecks:number}){
 if(!Number.isSafeInteger(options.maxRounds)||options.maxRounds<1)throw Error('finite counterexample rounds required');
 let suite=new CounterexampleSuite(options.cases,options.oracle,options.maxChecks);
 const gateway=options.gateway??new UsageGateway(options.budget);
 const evaluator=()=>new SourceEvaluator(options.contract,[...suite.cases],options.executor,gateway,{executorId:options.executorId,signal:options.signal,executeCase:options.executeCase});
 // What the host measured this round: the stop policy reads these, never the model's report of them.
 let round:Omit<CounterexampleStopFacts,'search'|'round'>={admitted:0,remainingChecks:suite.remainingChecks};
 let rounds=0;
 const task=createNatlangRuntime({model:{driver:(request,signal)=>gateway.request(options.improver,request,signal,'reflection')},signal:options.signal,network:false,codeEdits:'deny',seed:{mode:'derived',root:options.seed??0},
  services:{counterexamples:{evidence:async(source:import('../native/scoped-fs.js').FolderSnapshot)=>{const e=evaluator(),report=await e.evaluate(source,{split:'train'});return e.page(report.evidence);},
   admit:async(inputs:unknown[][])=>{const admitted=await suite.admit(inputs);suite=admitted.suite;round={admitted:admitted.admission.accepted.length,remainingChecks:suite.remainingChecks};return {suite:suite.version,admitted:round.admitted,remainingChecks:suite.remainingChecks};},
   repair:async(source:import('../native/scoped-fs.js').FolderSnapshot)=>{const result=await improveProgram({...options,folder:source.branch(),cases:[...suite.cases],gateway,directory:undefined,singleStep:false});const training=result.validation?await evaluator().evaluate(result.folder,{split:'train'}):null;const eligible=!!result.validation?.gatesPassed&&result.state.done;round={...round,repair:{eligible,trainingQuality:training?.quality??0,disposition:result.disposition}};return {folder:result.folder,quality:training?.quality??0,eligible,disposition:result.disposition};},
   stop:async()=>{const facts:CounterexampleStopFacts={search:'counterexample',round:++rounds,...round};round={admitted:0,remainingChecks:round.remainingChecks};return counterexampleStop(options.policy.policies?.shouldStop,facts);}}},
  serviceScopes:{counterexamples:['counterexampleStep.nl']},serviceDeclarations:{counterexamples:'export declare function evidence(source:FolderSnapshot):Promise<unknown[]>; export declare function admit(inputs:unknown[][]):Promise<{suite:string;admitted:number;remainingChecks:number}>; export declare function repair(source:FolderSnapshot):Promise<{folder:FolderSnapshot;quality:number;eligible:boolean;disposition:string}>; export declare function stop():Promise<{stop:boolean;reason:string}>;'}});
 const source={...AUTHORED_IMPROVER};
 const step=loadVirtualNatlang(source,'counterexampleStep.nl');
 const initial:CounterexampleState={round:0,done:options.maxChecks===0,suite:suite.version,remainingChecks:suite.remainingChecks,quality:0,reason:options.maxChecks===0?'independent oracle allowance exhausted':''};
 const result=await task.run(()=>options.folder.iterateOn<CounterexampleState>(step,initial,options.policy.goal).withMeasure(state=>options.maxRounds-state.round).until(state=>state.done||state.round===options.maxRounds));
 return {...result,suite,ledger:gateway.snapshot()};
}
