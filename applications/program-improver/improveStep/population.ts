import {prune} from 'natlang:gepa';
import type {SearchState,PopulationMember,ExperimentFeedback,ExperimentOutcome} from '../types';
import {members} from './crisp';
import type {Decisions} from './capabilities';
type MeasuredSource = {source:string;quality:number;sourceBytes?:number;scores?:{caseId:string;quality:number}[]};
/** Exact state assembly. The caller makes every acceptance, selection and stopping decision. */
export function finishStep(state:SearchState,selected:MeasuredSource,population:PopulationMember[],experiment:ExperimentOutcome,done:boolean,stopReason:string,lastExperiment?:ExperimentFeedback):SearchState {
  if (!population.some(member=>member.source===selected.source && member.quality===selected.quality)) throw Error('selected source and quality must be in the population');
  return {iteration:state.iteration+1,done,incumbent:selected.source,quality:selected.quality,population,
    history:[...state.history,experiment],stopReason,...(lastExperiment?{lastExperiment}:{})};
}

/** The population cut to the limit: the baseline and the selected member stay, then the frontier, then higher quality. */
export function trim(population:PopulationMember[],selected:string,limit:number):PopulationMember[] {
  if(!Number.isSafeInteger(limit)||limit<2)throw Error('population limit must retain baseline and incumbent');
  const baseline=population.filter(member=>member.parent==='').map(member=>member.source);
  const kept=prune(members(population),[selected,...baseline],limit).map(member=>member.id);
  return population.filter((member,index)=>kept.includes(member.source)&&population.findIndex(other=>other.source===member.source)===index);
}

/** Install a measured population member independently of accepting a draft. */
export async function advance(folder:{at(source:string):unknown;select(source:unknown):Promise<void>},evaluator:{evaluate(source:any,request:{split:'validation';seed:number}):Promise<MeasuredSource>},frame:import('../types').SearchFrame,experiment:import('../types').ExperimentResult,policy:import('../types').ImprovementPolicy,decide:Pick<Decisions,'selectIncumbent'|'shouldStop'>):Promise<SearchState> {
  const added=experiment.accepted?[...frame.current.population,{source:experiment.source,quality:experiment.quality,parent:experiment.parent,cost:experiment.sourceBytes,...(experiment.modelCalls!==undefined?{modelCalls:experiment.modelCalls}:{}),scores:experiment.scores}]:frame.current.population;
  const selected=await decide.selectIncumbent(policy,added,frame.current.incumbent);
  const population=trim(added,selected,policy.maxPopulation);
  await folder.select(folder.at(selected));
  const measured=await evaluator.evaluate(folder.at(selected),{split:'validation',seed:policy.seed??0});
  const outcome:ExperimentOutcome={source:experiment.source,parent:experiment.parent,accepted:experiment.accepted,selected:experiment.accepted&&selected===experiment.source,reason:experiment.reason,disposition:experiment.disposition,...(experiment.candidate?{candidate:experiment.candidate}:{}),...(experiment.failing?{failing:experiment.failing}:{})};
  const recent=[...frame.current.history,outcome].slice(-3).map(entry=>({disposition:entry.disposition??(entry.accepted?'accepted':'rejected'),failingCases:entry.failing??[]}));
  const decision=await decide.shouldStop(policy,{disposition:experiment.disposition,reason:experiment.reason,selectedQuality:measured.quality,objective:policy.objective??'quality',iteration:frame.current.iteration,maxExperiments:policy.maxExperiments,recent});
  return finishStep(frame.current,measured,population,outcome,decision.stop,decision.reason,experiment.feedback);
}
