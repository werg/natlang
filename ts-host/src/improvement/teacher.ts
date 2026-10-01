/** Explicit application adapter: generic collection has no knowledge of program improvement. */
import {executeProgram,type ProgramRecord,type ExecuteOptions,type ProgramRun} from '../teacher/collector.js';
import {Folder} from '../native/scoped-fs.js';
import {improveProgram} from './program.js';
import type {ImproverCase} from './improver-execution.js';
export const identity='natlang.improvement-collection/1';
export async function run(record:ProgramRecord,driver:Parameters<typeof executeProgram>[1],options:ExecuteOptions):Promise<ProgramRun>{
 const fixture=(record.semantics as unknown as {evaluation_fixture?:{kind:string;caseDefinition:ImproverCase;executorId?:string}}).evaluation_fixture;
 if(!fixture)return executeProgram(record,driver,options);
 if(fixture.kind!=='flat-program-evaluator')throw Error('unknown improvement fixture');
 const traces:import('../runtime/runtime.js').InvocationTrace[]=[];
 const result=await improveProgram({folder:Folder.fromFiles(fixture.caseDefinition.files),...fixture.caseDefinition,
  improverSource:Folder.fromFiles(record.semantics.files).snapshot(),improver:driver,executor:driver,executorId:fixture.executorId??'canonical-improvement-fixture',
  signal:options.signal,seed:options.rootSeed,trace:trace=>traces.push(trace),budget:{maxModelCalls:104,maxRollouts:40,maxProposals:2*fixture.caseDefinition.policy.maxExperiments,maxElapsedMs:600000}});
 const accepted=!!result.validation&&!!result.baseline&&result.state.done&&result.validation.gatesPassed&&result.validation.quality>=result.baseline.quality&&result.state.incumbent===result.folder.digest;
 return {outcome:{accepted,kind:accepted?'done':'failed',value:result.state,source:result.sourceManifest,quality:result.validation?.quality??null,baseline:result.baseline?.quality??null,disposition:result.disposition,ledger:result.ledger},trace:traces.flatMap(trace=>trace.events) as Record<string,unknown>[]};
}
export default {identity,run};
