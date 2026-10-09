import type {ImprovementPolicy,SearchState} from '../types';
import {start} from './baseline';
import {advance} from './population';
import type {ExperimentFolder,ExperimentEvaluator,PlanJournal,Snapshot} from './capabilities';
import {test} from './experiment';
import {decisions} from './policies';
type Folder=ExperimentFolder & {snapshot():Snapshot;select(source:unknown):Promise<void>};
/** What is decided before an experiment's edit: recorded in the journal so a resumed run reuses it. */
type Plan={parent:string;chose:{chooseParent:string}};

/**
 * The plan of experiment `iteration`: the journaled one when this experiment was already planned (a resumed run), else
 * the parent the policy chooses now, recorded before the edit begins.
 */
async function plan(plans:PlanJournal,state:SearchState,policy:ImprovementPolicy):Promise<Plan> {
  const recalled=await plans.recall(state.iteration) as Plan|null;
  if(recalled&&state.population.some(member=>member.source===recalled.parent))return recalled;
  const parent=policy.strategy==='gepa'?await decisions.chooseParent(policy,state.population,state.history,(policy.seed??0)+state.iteration+1):state.incumbent;
  const decided:Plan={parent,chose:{chooseParent:policy.strategy==='gepa'?policy.policies?.chooseParent??'crisp':'incumbent'}};
  await plans.record(state.iteration,decided);
  return decided;
}

/** One finite experiment by construction: plan, then one semantic diagnose/hypothesize/edit and test, exact selection. */
export async function step(folder:Folder,evaluator:ExperimentEvaluator,plans:PlanJournal,state:SearchState,policy:ImprovementPolicy):Promise<SearchState> {
  const source=folder.snapshot();
  const validation=await evaluator.evaluate(source,{split:'validation',seed:policy.seed??0});
  const current=state.iteration===0?start(state,validation):state;
  const frame={current,baselineBytes:current.population[0].cost??validation.sourceBytes};
  const planned=await plan(plans,current,policy);
  const experiment=await test(folder,evaluator,planned.parent,policy,current,decisions);
  return await advance(folder,evaluator,frame,experiment,policy,decisions);
}
