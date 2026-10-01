import type {ImprovementPolicy,SearchState} from '../types';
import {start} from './baseline';
import {advance} from './population';
import type {ExperimentFolder,ExperimentEvaluator,Snapshot} from './capabilities';
import {test} from './experiment';
import {frontierParent} from './parents';
type Folder=ExperimentFolder & {snapshot():Snapshot;select(source:unknown):Promise<void>};
/** One finite experiment by construction: one semantic edit/test, exact selection. */
export async function step(folder:Folder,evaluator:ExperimentEvaluator,editor:unknown,state:SearchState,policy:ImprovementPolicy):Promise<SearchState> {
  const source=folder.snapshot();
  const validation=await evaluator.evaluate(source,{split:'validation',seed:policy.seed??0});
  const current=state.iteration===0?start(state,validation):state;
  const frame={current,baselineBytes:current.population[0].cost??validation.sourceBytes};
  const parentId=policy.strategy==='gepa'?frontierParent(current.population,(policy.seed??0)+current.iteration+1):current.incumbent;
  const experiment=await test(folder,evaluator,parentId,editor,policy,'',current);
  return await advance(folder,evaluator,frame,experiment,policy);
}
