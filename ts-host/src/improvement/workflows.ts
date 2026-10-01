import {Folder} from '../native/scoped-fs.js';
import {AUTHORED_IMPROVER} from './authored-source.js';
import {improveProgram,type ImproveProgramOptions} from './program.js';
export type BehaviorOptions=Omit<ImproveProgramOptions,'improverSource'>;
/** Specialized applications remain normal inspectable authored source, not native schedulers. */
function authoredReducer(name:string){
 const source={...AUTHORED_IMPROVER} as Record<string,string>;
 source['improveStep/rewriteProgram.nl']=source[`reducers/${name}.nl`]!.replace('request: TransformationRequest','request: RewriteRequest').replace('returns: TransformationExplanation','returns: RewriteResult');
 return Folder.fromFiles(source).snapshot();
}
export function implementBehavior(options:BehaviorOptions){
 const missing=options.cases.filter(row=>row.expected===undefined).map(row=>row.id);
 if(!options.cases.length||missing.length)return Promise.resolve({disposition:'needs-information' as const,missing:missing.length?missing:['author-supplied independent cases']});
 return improveProgram({...options,policy:{...options.policy,mode:'structural',objective:'quality'},improverSource:authoredReducer('implementProgram')});
}
export function repairProgram(options:BehaviorOptions){return improveProgram({...options,policy:{...options.policy,objective:'quality'},improverSource:authoredReducer('repairProgram')});}
export function simplifyProgram(options:BehaviorOptions){return improveProgram({...options,policy:{...options.policy,mode:'structural',objective:'source-size'},improverSource:authoredReducer('simplifyProgram')});}
