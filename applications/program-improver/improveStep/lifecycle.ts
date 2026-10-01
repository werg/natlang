import type {ImprovementPolicy,SearchState,ExperimentContext,ExperimentFeedback} from '../types';
import {start} from './baseline';
import {advance} from './population';
import type {ExperimentFolder,ExperimentEvaluator,Snapshot} from './capabilities';
import {test} from './experiment';
import {inspect} from './planExperiment/feedback';
import {brief} from './planExperiment/bookkeeping';
import {frontierParent} from './planExperiment/selection';
type Folder=ExperimentFolder & {snapshot():Snapshot;select(source:unknown):Promise<void>};
/** One finite experiment by construction: a semantic plan, one edit/test, exact selection. */
export async function step(folder:Folder,evaluator:ExperimentEvaluator,planner:unknown,editor:unknown,state:SearchState,policy:ImprovementPolicy):Promise<SearchState> {
  const source=folder.snapshot();
  const validation=await evaluator.evaluate(source,{split:'validation',seed:policy.seed??0});
  const current=state.iteration===0?start(state,validation):state;
  const frame={current,baselineBytes:current.population[0].cost??validation.sourceBytes};
  const parentId=policy.strategy==='gepa'?frontierParent(current.population,(policy.seed??0)+current.iteration+1):current.incumbent;
  const parent=folder.at(parentId).branch();
  const feedback=await inspect(folder,evaluator,parentId,policy);
  const paths=parent.filePaths();
  const sourceFiles=await Promise.all(paths.map(async path=>({path,text:await parent.readText(path)})));
  const context:ExperimentContext={parent:parentId,sourceFiles,evidence:feedback.evidence,opportunity:feedback.opportunity,history:current.history,...(current.lastExperiment?{lastExperiment:current.lastExperiment as ExperimentFeedback}:{})};
  const hypothesis=await parent.apply(planner,brief(policy,feedback.evidence,sourceFiles),context,policy);
  const experiment=await test(folder,evaluator,parentId,editor,policy,hypothesis);
  return await advance(folder,evaluator,frame,experiment,policy);
}
