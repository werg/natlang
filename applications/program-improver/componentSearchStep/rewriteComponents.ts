import {componentProblem} from 'natlang:gepa';
import editComponents from './editComponents.nl';
import type {Candidate,ComponentRewriteRequest} from '../types';

type Edited={file(path:string):{readText():Promise<string>};apply(reducer:unknown,...args:unknown[]):Promise<unknown>};

/**
 * The component edit: the natural-language edit, then the exact check of its result. The candidate is the components.json
 * the edit leaves in the folder. When it breaks a constraint (only the selected keys differ from the parent; kinds, slot
 * ids and segment counts are unchanged), the edit runs once more with the problem text in `request.problem`. The candidate
 * of the last attempt is returned; the engine's own check decides whether it is valid.
 */
export default async function rewriteComponents(folder:Edited,request:ComponentRewriteRequest):Promise<Candidate> {
  const parent=JSON.parse(await folder.file('components.json').readText()) as Candidate;
  let candidate:Candidate=parent,problem='';
  for(const attempt of [0,1]){
    await folder.apply(editComponents,problem?{...request,problem}:request);
    const text=await folder.file('components.json').readText();
    try{candidate=JSON.parse(text) as Candidate;problem=componentProblem(parent,candidate,request.keys);}
    catch(error){candidate=parent;problem='components.json is not valid JSON ('+(error as Error).message+'); write one JSON object.';}
    if(!problem)return candidate;
  }
  return candidate;
}
