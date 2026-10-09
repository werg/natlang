/** Explicit application adapter: generic collection has no knowledge of program improvement. */
import {executeProgram,type ProgramRecord,type ExecuteOptions,type ProgramRun} from '../teacher/collector.js';
import {Folder} from '../native/scoped-fs.js';
import {createNatlangRuntime} from '../runtime/node.js';
import {loadVirtualNatlang} from '../runtime/virtual-project.js';
import {SourceEvaluator,sourceFiles} from './host.js';
import {UsageGateway} from '../evaluation/usage.js';
import {isDeepStrictEqual} from 'node:util';
import {callableMeta} from '../runtime/callable.js';
import {EVALUATOR_DECLARATION,PLANS_DECLARATION,planService} from './services.js';
import {improveProgram} from './program.js';
import type {ImproverCase} from './improver-execution.js';
export const identity='natlang.improvement-collection/2';
export async function run(record:ProgramRecord,driver:Parameters<typeof executeProgram>[1],options:ExecuteOptions):Promise<ProgramRun>{
 const fixture=(record.semantics as unknown as {evaluation_fixture?:{kind:string;caseDefinition:ImproverCase;executorId?:string;executorTimeoutMs?:number;scope?:string;populationSources?:Record<string,Record<string,string>>}}).evaluation_fixture;
 if(!fixture)return executeProgram(record,driver,options);
 if(fixture.kind!=='flat-program-evaluator')throw Error('unknown improvement fixture');
 if(fixture.scope==='invocation'){
  const folder=Folder.fromFiles(record.semantics.folder_files!);
  for(const [digest,files] of Object.entries(fixture.populationSources??{})){
   const branch=folder.snapshot().branch();
   for(const path of branch.filePaths())if(!(path in files))branch.remove(path);
   for(const [path,text] of Object.entries(files))branch.writeText(path,text);
   if(branch.snapshot().digest!==digest)throw Error('population fixture source digest mismatch');
  }
  const traces:import('../runtime/runtime.js').InvocationTrace[]=[];
  const gateway=new UsageGateway({maxModelCalls:104,maxRollouts:40,maxProposals:8,maxElapsedMs:600000});
  const target=fixture.caseDefinition;
  const evaluator=new SourceEvaluator(target.contract,target.cases,driver,gateway,{signal:options.signal,timeoutMs:fixture.executorTimeoutMs,executorId:fixture.executorId??'canonical-improvement-fixture',sourcePolicy:{baseline:target.files,mode:target.policy.mode,allowedFiles:target.policy.allowedFiles}});
  const runtime=createNatlangRuntime({model:{driver,contextTokens:options.contextTokens,maxTurns:options.maxTurns??16,maxTokens:24000,turnTokens:2048,maxFailureRepairs:4},signal:options.signal,seed:{mode:'derived',root:options.rootSeed},codeEdits:'deny',network:false,trace:trace=>traces.push(trace),services:{evaluator:{check:evaluator.check.bind(evaluator),evaluate:evaluator.evaluate.bind(evaluator),page:evaluator.page.bind(evaluator)},plans:planService()},serviceDeclarations:{evaluator:EVALUATOR_DECLARATION,plans:PLANS_DECLARATION},serviceScopes:{evaluator:['improveStep.nl'],plans:['improveStep.nl']}});
  const reducer=loadVirtualNatlang(record.semantics.files,record.semantics.root);
  if(record.semantics.root!=='improveStep.nl'){
   const args=callableMeta(reducer)!.definition.params.map(param=>record.semantics.inputs[param.name]);
   const value=await runtime.run(()=>folder.apply(reducer,...args));
   const files=sourceFiles(folder.snapshot());
   const accepted=isDeepStrictEqual(value,record.semantics.expected)&&isDeepStrictEqual(files,record.semantics.expected_files);
   return {outcome:{accepted,kind:accepted?'done':'failed',value,source:files},trace:traces.flatMap(trace=>trace.events) as Record<string,unknown>[]};
  }
  const inputs=record.semantics.inputs as {state:import('./program.js').ImprovementState;policy:import('./program.js').ImprovementPolicy};
  const state=await runtime.run(()=>folder.apply(loadVirtualNatlang(record.semantics.files,record.semantics.root),inputs.state,inputs.policy)) as import('./program.js').ImprovementState;
  const checked=await evaluator.check(folder.snapshot());
  const measured=await evaluator.evaluate(folder.snapshot(),{split:'validation',seed:options.rootSeed});
  const prior=await evaluator.evaluate(folder.at(inputs.state.incumbent),{split:'validation',seed:options.rootSeed});
  let accepted=checked.valid&&state.iteration===inputs.state.iteration+1&&state.incumbent===folder.snapshot().digest&&Math.abs(state.quality-measured.quality)<1e-12&&measured.quality>=prior.quality;
  for(const member of state.population){const actual=await evaluator.evaluate(folder.at(member.source),{split:'validation',seed:options.rootSeed});accepted=accepted&&Math.abs(member.quality-actual.quality)<1e-12;}
  return {outcome:{accepted,kind:accepted?'done':'failed',value:state,source:sourceFiles(folder.snapshot()),quality:measured.quality,baseline:prior.quality,ledger:gateway.snapshot()},trace:traces.flatMap(trace=>trace.events) as Record<string,unknown>[]};
 }
 if(fixture.scope!==undefined)throw Error('unknown improvement collection scope');
 const traces:import('../runtime/runtime.js').InvocationTrace[]=[];
 const result=await improveProgram({folder:Folder.fromFiles(fixture.caseDefinition.files),...fixture.caseDefinition,
  improverSource:Folder.fromFiles(record.semantics.files).snapshot(),improver:driver,executor:driver,executorId:fixture.executorId??'canonical-improvement-fixture',
  signal:options.signal,seed:options.rootSeed,trace:trace=>traces.push(trace),budget:{maxModelCalls:104,maxRollouts:40,maxProposals:2*fixture.caseDefinition.policy.maxExperiments,maxElapsedMs:600000}});
 const accepted=!!result.validation&&!!result.baseline&&result.state.done&&result.validation.gatesPassed&&result.validation.quality>=result.baseline.quality&&result.state.incumbent===result.folder.digest;
 return {outcome:{accepted,kind:accepted?'done':'failed',value:result.state,source:result.sourceManifest,quality:result.validation?.quality??null,baseline:result.baseline?.quality??null,disposition:result.disposition,ledger:result.ledger},trace:traces.flatMap(trace=>trace.events) as Record<string,unknown>[]};
}
export default {identity,run};
