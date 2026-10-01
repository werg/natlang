/** A top-level, finite meta-evaluation adapter. It executes a frozen improver with one lower evaluation level. */
import {Folder} from '../native/scoped-fs.js';
import {fingerprint} from '../adaptation/identity.js';
import type {ModelDriver} from '../runtime/runtime.js';
import {improveProgram,type ImprovementPolicy} from './program.js';
import {sourceFiles,type SourceCaseExecution} from './host.js';
import type {ImprovementCase,ProgramContract} from './types.js';
export type ImproverCase = {id:string;files:Record<string,string>;contract:ProgramContract;cases:ImprovementCase[];policy:ImprovementPolicy};
export function improverExecution(cases:readonly ImproverCase[],models:{improver:ModelDriver;executor:ModelDriver;improverId:string;executorId:string},options:{signal?:AbortSignal}={}):SourceCaseExecution {
  const frozen=structuredClone(cases);
  const execute:SourceCaseExecution=Object.assign(async(source:import('../native/scoped-fs.js').FolderSnapshot,row:ImprovementCase,seed:number,gateway:import('../evaluation/usage.js').UsageGateway)=>{
    const target=frozen.find(target=>target.id===row.id);if(!target)throw new Error('foreign meta-evaluation case');
    const result=await improveProgram({folder:Folder.fromFiles(target.files),contract:target.contract,cases:target.cases,policy:target.policy,
      improver:models.improver,executor:models.executor,executorId:models.executorId,improverSource:source,evaluationLevel:1,
      gateway,budget:gateway.limits,signal:options.signal,seed});
    const checked=await result.evaluator.check(result.folder);
    const completed=result.validation!==null && result.state.done;
    // Inner validation chooses a candidate; a separate locked inner suite grades the frozen improver.
    const heldout=completed ? await result.evaluator.confirm(result.folder,{source:result.folder.digest,experiment:fingerprint({improver:source.digest,target:target.id,seed})}) : undefined;
    const quality=completed && checked.valid && heldout?.gatesPassed && heldout.quality===1 ? 1 : 0;
    return {value:{source:result.folder.digest,files:sourceFiles(result.folder),quality,disposition:result.disposition},
      score:{quality,gates:{contract:checked.valid,completed}}};
  },{evaluationLevel:2 as const,identity:fingerprint({cases:frozen,improver:models.improverId,executor:models.executorId,level:2})});
  return execute;
}
