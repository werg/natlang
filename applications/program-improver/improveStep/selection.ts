import type {PopulationMember} from '../types';
/** Exact, baseline-preserving incumbent selection. */
export function best(population: PopulationMember[], incumbent: string, objective: 'quality'|'source-size'|'model-calls'='quality'): string {
  const maximum = Math.max(...population.map(member => member.quality));
  let eligible = population.filter(member => member.quality === maximum);
  if(objective==='source-size'){const minimum=Math.min(...eligible.map(member=>member.cost??Infinity));eligible=eligible.filter(member=>(member.cost??Infinity)===minimum);}
  if(objective==='model-calls'){const minimum=Math.min(...eligible.map(member=>member.modelCalls??Infinity));eligible=eligible.filter(member=>(member.modelCalls??Infinity)===minimum);}
  if(eligible.some(member=>member.source===incumbent))return incumbent;
  return eligible.map(member=>member.source).sort()[0];
}
export function prune(population: PopulationMember[], selected: string, limit: number): PopulationMember[] {
  if(!Number.isSafeInteger(limit)||limit<2)throw Error('population limit must retain baseline and incumbent');
  const unique=population.filter((member,index)=>population.findIndex(other=>other.source===member.source)===index);
  const protectedMembers=unique.filter(member=>member.source===selected||member.parent==='');
  const frontier=new Set<string>();
  for(const caseId of [...new Set(unique.flatMap(member=>(member.scores??[]).map(score=>score.caseId)))]){
    const maximum=Math.max(...unique.map(member=>member.scores?.find(score=>score.caseId===caseId)?.quality??-1));
    for(const member of unique)if(member.scores?.find(score=>score.caseId===caseId)?.quality===maximum)frontier.add(member.source);
  }
  return [...protectedMembers,...unique.filter(member=>!protectedMembers.includes(member)).sort((a,b)=>Number(frontier.has(b.source))-Number(frontier.has(a.source))||b.quality-a.quality||a.source.localeCompare(b.source))].slice(0,limit);
}
