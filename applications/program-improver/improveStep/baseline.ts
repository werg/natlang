import type {SearchState} from '../types';
type MeasuredSource = {source:string;quality:number;sourceBytes?:number;modelCalls?:number;scores?:{caseId:string;quality:number}[]};
/** Assemble baseline state without asking the model to reproduce its schema. */
export function start(state:SearchState,baseline:MeasuredSource):SearchState {
  return {...state,incumbent:baseline.source,quality:baseline.quality,population:[{source:baseline.source,quality:baseline.quality,parent:'',...(baseline.sourceBytes!==undefined?{cost:baseline.sourceBytes}:{}),...(baseline.modelCalls!==undefined?{modelCalls:baseline.modelCalls}:{}),...(baseline.scores?{scores:baseline.scores}:{})}]};
}
