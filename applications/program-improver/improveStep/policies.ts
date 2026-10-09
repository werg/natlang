import {pluggable} from '@natlang/node';
import findOpportunityNl from './findOpportunity.nl';
import chooseParentNl from './chooseParent.nl';
import selectIncumbentNl from './selectIncumbent.nl';
import shouldStopNl from './shouldStop.nl';
import diagnose from './diagnose.nl';
import hypothesize from './hypothesize.nl';
import editSource from './editSource.nl';
import editSourceStructural from './editSourceStructural.nl';
import type {ExperimentOutcome,ImprovementPolicy,Opportunity,OpportunityFacts,PopulationMember,StopDecision,StopFacts} from '../types';
import type {Decisions} from './capabilities';
import {crispIncumbent,crispOpportunity,crispParent,crispStop,incumbentChoices,parentChoices,verifyIncumbent,verifyOpportunity,verifyParent,verifyStop} from './crisp';

/**
 * The decisions of one experiment, each a natural-language function or a pluggable policy. `policy.policies.<name>`
 * selects `crisp` (the default), `nl` or `shadow` (both run, the crisp result is served, and the trace records whether
 * they agree). A natural-language choice is bounded by the crisp verifier beside it (crisp.ts); a choice outside the
 * bound is an error that says what the bound is.
 */
export function findOpportunity(policy:ImprovementPolicy,facts:OpportunityFacts):Promise<Opportunity> {
  return pluggable<[OpportunityFacts],Opportunity>({crisp:async value=>crispOpportunity(value),nl:async value=>verifyOpportunity(await findOpportunityNl(value),value)},
    policy.policies?.findOpportunity,{name:'findOpportunity',default:'crisp',serve:'crisp'})(facts);
}
export function chooseParent(policy:ImprovementPolicy,population:PopulationMember[],history:ExperimentOutcome[],seed:number):Promise<string> {
  return pluggable<[PopulationMember[],ExperimentOutcome[],number],string>({
    crisp:async (all,_history,draw)=>crispParent(all,draw),
    nl:async (all,past)=>verifyParent(await chooseParentNl(parentChoices(all,past)),all)},
    policy.policies?.chooseParent,{name:'chooseParent',default:'crisp',serve:'crisp'})(population,history,seed);
}
export function selectIncumbent(policy:ImprovementPolicy,population:PopulationMember[],incumbent:string):Promise<string> {
  const objective=policy.objective??'quality';
  return pluggable<[PopulationMember[],string],string>({
    crisp:async (all,current)=>crispIncumbent(all,current,objective),
    nl:async (all,current)=>verifyIncumbent(await selectIncumbentNl(incumbentChoices(all,objective),current,objective),all,objective)},
    policy.policies?.selectIncumbent,{name:'selectIncumbent',default:'crisp',serve:'crisp'})(population,incumbent);
}
export function shouldStop(policy:ImprovementPolicy,facts:StopFacts):Promise<StopDecision> {
  return pluggable<[StopFacts],StopDecision>({crisp:async value=>crispStop(value),nl:async value=>verifyStop(await shouldStopNl(value),value)},
    policy.policies?.shouldStop,{name:'shouldStop',default:'crisp',serve:'crisp'})(facts);
}

/** What one experiment decides with: the policies and the semantic stages. */
export const decisions:Decisions={findOpportunity,chooseParent,selectIncumbent,shouldStop,diagnose,hypothesize,
  editor:mode=>mode==='structural'?editSourceStructural:editSource};
