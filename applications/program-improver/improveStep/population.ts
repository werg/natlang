import type {SearchState,PopulationMember,ExperimentFeedback} from '../types';
import {best,prune} from './selection';
type Experiment = {source:string;parent:string;accepted:boolean;selected:boolean;reason:string};
type MeasuredSource = {source:string;quality:number;sourceBytes?:number;scores?:{caseId:string;quality:number}[]};
/** Exact state assembly. The caller makes every acceptance, selection and stopping decision. */
export function finishStep(state:SearchState,selected:MeasuredSource,population:PopulationMember[],experiment:Experiment,done:boolean,stopReason:string,lastExperiment?:ExperimentFeedback):SearchState {
  if (!population.some(member=>member.source===selected.source && member.quality===selected.quality)) throw Error('selected source and quality must be in the population');
  return {iteration:state.iteration+1,done,incumbent:selected.source,quality:selected.quality,population,
    history:[...state.history,experiment],stopReason,...(lastExperiment?{lastExperiment}:{})};
}

/** Install a measured population member independently of accepting a draft. */
export async function advance(folder:{at(source:string):unknown;select(source:unknown):Promise<void>},evaluator:{evaluate(source:any,request:{split:'validation';seed:number}):Promise<MeasuredSource>},frame:import('../types').SearchFrame,experiment:import('../types').ExperimentResult,policy:import('../types').ImprovementPolicy):Promise<SearchState> {
  const added=experiment.accepted?[...frame.current.population,{source:experiment.source,quality:experiment.quality,parent:experiment.parent,cost:experiment.sourceBytes,...(experiment.modelCalls!==undefined?{modelCalls:experiment.modelCalls}:{}),scores:experiment.scores}]:frame.current.population;
  const selected=best(added,frame.current.incumbent,policy.objective??'quality');
  const population=prune(added,selected,policy.maxPopulation);
  await folder.select(folder.at(selected));
  const measured=await evaluator.evaluate(folder.at(selected),{split:'validation',seed:policy.seed??0});
  const satisfied=measured.quality===1 && (policy.objective??'quality')==='quality';
  const exhaustedHypotheses=experiment.reason==='No supported hypothesis.';
  const fixture=experiment.reason.startsWith('fixture-error:');
  const limit=frame.current.iteration+1>=policy.maxExperiments;
  return finishStep(frame.current,measured,population,{source:experiment.source,parent:experiment.parent,accepted:experiment.accepted,selected:experiment.accepted&&selected===experiment.source,reason:experiment.reason},fixture||satisfied||exhaustedHypotheses||limit,fixture?experiment.reason:satisfied?'Declared objective satisfied.':exhaustedHypotheses?'No further evidenced change.':limit?'Declared experiments completed.':'Continue with a distinct hypothesis.',experiment.feedback);
}
