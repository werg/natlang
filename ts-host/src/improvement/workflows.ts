import {improveProgram,type ImproveProgramOptions} from './program.js';
export type BehaviorOptions=Omit<ImproveProgramOptions,'improverSource'>;
/** A specialized application is the ordinary authored improver run with a named transformation (improveStep/transformations.ts). */
export function implementBehavior(options:BehaviorOptions){
 const missing=options.cases.filter(row=>row.expected===undefined).map(row=>row.id);
 if(!options.cases.length||missing.length)return Promise.resolve({disposition:'needs-information' as const,missing:missing.length?missing:['author-supplied independent cases']});
 return improveProgram({...options,policy:{...options.policy,mode:'structural',objective:'quality',transformation:'implementProgram'}});
}
export function repairProgram(options:BehaviorOptions){return improveProgram({...options,policy:{...options.policy,objective:'quality',transformation:'repairProgram'}});}
export function simplifyProgram(options:BehaviorOptions){return improveProgram({...options,policy:{...options.policy,mode:'structural',objective:'source-size',transformation:'simplifyProgram'}});}
