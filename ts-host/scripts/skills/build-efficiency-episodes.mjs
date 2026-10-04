#!/usr/bin/env node
/** Project-owned optimization task corpus; every bound is computed by the host. */
import {mkdir,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {exactObjectiveBounds,scoreSkillObjective} from '../../dist/skills/objective.js';
import {validateEpisode} from '../../dist/skills/episode.js';
const hash=x=>createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const integer=(key,max)=>parseInt(hash(key).slice(0,8),16)%max;
const definitions={
  'weighted-set-cover':['Return JSON {selectedIds:[setId,...]} covering every universe element at minimum total cost.','Use exhaustive subset search for small inputs; otherwise start from cost per newly covered element, then remove redundant sets. Verify complete coverage and sum costs once.'],
  'budgeted-max-coverage':['Return JSON {selectedIds:[itemId,...]} maximizing covered universe weight within budget. Count each covered element once.','Enumerate small subsets or use marginal newly covered weight per cost as a construction heuristic. Recompute unions and costs and improve with swaps; never double-count overlap.'],
  'facility-location':['Return JSON {assignments:{clientId:facilityId,...}} assigning every client. Minimize service cost plus each opened facility cost counted once.','Compare shared facility openings with independently cheapest service choices. For small instances enumerate assignments; charge an opening only once and verify every client.'],
  'bottleneck-assignment':['Return JSON {assignments:{agentId:taskId,...}}, a bijection minimizing the largest assigned cost.','Search feasibility under increasing maximum-cost thresholds, or enumerate small permutations. Do not optimize total cost when the declared objective is the maximum.'],
  'matrix-chain':['Return JSON {tree:...} with leaves {matrix:"A1"},...,{matrix:"An"} in order and internal {left:tree,right:tree}. Minimize scalar multiplication count.','Use interval dynamic programming with recurrence cost(i,j)=min over k of cost(i,k)+cost(k+1,j)+dimensions[i]*dimensions[k+1]*dimensions[j+1]. Reconstruct a full ordered tree.'],
  'boolean-dnf':['Return JSON {terms:[pattern,...]} over 0,1,- with one character per variable. Match the entire truth table exactly, then minimize literalCost per specified literal plus termCost per term. Rows are lexicographic binary order.','Enumerate implicants that cover no false row. Cover all true rows with minimum literal-plus-term cost, remove redundant terms and verify every row including false rows.'],
};
const families=[...Object.keys(definitions).map(kind=>({family:kind,kind,context:''})),
  {family:'test-suite-reduction',kind:'weighted-set-cover',context:'Universe elements are detected mutants; sets are test executions with execution costs. Preserve all required mutant detections.'},
  {family:'sensor-coverage',kind:'budgeted-max-coverage',context:'Select sensors within an installation budget; overlapping locations count once, weighted by their importance.'},
  {family:'cdn-placement',kind:'facility-location',context:'Clients are content regions; facilities are cache sites. Pay for each activated site once and per-region service costs.'},
  {family:'staffing-bottleneck',kind:'bottleneck-assignment',context:'Assign employees to roles to minimize the worst individual effort cost, with one employee per role.'}];
export const EFFICIENCY_FAMILIES=families;
const skill=(name,description,body)=>({'SKILL.md':`---\nname: ${name}\ndescription: ${JSON.stringify(description)}\n---\n${body}\n`});
function library(kind,variant){
 const name='optimization-procedure',body=definitions[kind][1];
 if(variant==='empty')return {kind:'empty',skills:{}};
 if(variant==='metadata')return {kind:'existing',skills:{[name]:skill(name,'Unclassified notes; not useful for typical problems.',body)}};
 return {kind:'existing',skills:{'formatting-notes':skill('formatting-notes','Use for styling a visual document, not numerical optimization.','Use consistent headings, spacing and punctuation.')}};
}
export function makeEfficiencyInstance(kind,key){
 const r=(salt,max)=>integer(key+'/'+salt,max),ids=(prefix,n)=>Array.from({length:n},(_,i)=>prefix+i);
 if(kind==='weighted-set-cover'){
  const universe=ids('u',7);const sets=Array.from({length:8},(_,i)=>({id:'s'+i,cost:2+r('cost'+i,19),elements:universe.filter((_,j)=>r('edge'+i+'/'+j,5)<2)}));
  universe.forEach((id,i)=>sets.push({id:'single'+i,cost:8+r('single'+i,12),elements:[id]}));return {universe,sets};
 }
 if(kind==='budgeted-max-coverage'){
  const universe=ids('u',9).map((id,i)=>({id,weight:1+r('weight'+i,13)}));
  return {budget:15+r('budget',10),universe,items:Array.from({length:10},(_,i)=>({id:'item'+i,cost:1+r('cost'+i,10),elements:universe.filter((_,j)=>r('edge'+i+'/'+j,4)<2).map(x=>x.id)}))};
 }
 if(kind==='facility-location'){
  const clients=ids('client',5);return {clients,facilities:ids('site',4).map((id,i)=>({id,openCost:10+r('open'+i,35),serviceCosts:Object.fromEntries(clients.map((c,j)=>[c,1+r('service'+i+'/'+j,30)]))}))};
 }
 if(kind==='bottleneck-assignment')return {agents:ids('agent',6),tasks:ids('task',6),costs:Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=>1+r('cost'+i+'/'+j,50)))};
 if(kind==='matrix-chain')return {dimensions:Array.from({length:7},(_,i)=>2+r('dimension'+i,39))};
 if(kind==='boolean-dnf'){
  const truthTable=Array.from({length:16},(_,i)=>r('row'+i,3)!==0);truthTable[0]=false;truthTable[15]=true;
  return {variables:ids('v',4),truthTable,literalCost:1+r('literal',3),termCost:1+r('term',4)};
 }
 throw Error('unknown efficiency kind');
}
function construction(kind,x){
 if(kind==='weighted-set-cover')return {selectedIds:x.sets.map(s=>s.id)};
 if(kind==='budgeted-max-coverage'){
  let cost=0;const selectedIds=[];for(const item of x.items)if(cost+item.cost<=x.budget){selectedIds.push(item.id);cost+=item.cost;}return {selectedIds};
 }
 if(kind==='facility-location')return {assignments:Object.fromEntries(x.clients.map(id=>[id,[...x.facilities].sort((a,b)=>a.serviceCosts[id]-b.serviceCosts[id])[0].id]))};
 if(kind==='bottleneck-assignment')return {assignments:Object.fromEntries(x.agents.map((id,i)=>[id,x.tasks[i]]))};
 if(kind==='matrix-chain'){let tree={matrix:'A1'};for(let i=2;i<x.dimensions.length;i++)tree={left:tree,right:{matrix:'A'+i}};return {tree};}
 return {terms:x.truthTable.flatMap((truth,i)=>truth?[i.toString(2).padStart(x.variables.length,'0')]:[])};
}
export function createEfficiencyEpisodes({replicas=3,variants=['empty','metadata','distractor']}={}){
 if(!Number.isSafeInteger(replicas)||replicas<1||replicas>100||!variants.length||new Set(variants).size!==variants.length||variants.some(v=>!['empty','metadata','distractor'].includes(v)))throw Error('invalid corpus allocation');
 const episodes=[],baselines=[];
 for(const item of families)for(let replica=0;replica<replicas;replica++)for(const variant of variants){
  const id=`efficiency-v1-${item.family}-r${replica}-${variant}`;
  const cases=Array.from({length:12},(_,i)=>{
   const group=`efficiency-v1/${item.family}/r${replica}/${variant}/instance-${i}`,x=makeEfficiencyInstance(item.kind,group),expected=exactObjectiveBounds(item.kind,x);
   const baseline=scoreSkillObjective(item.kind,x,construction(item.kind,x),expected);if(!Object.values(baseline.gates).every(Boolean))throw Error('invalid construction baseline');
   baselines.push({episode:id,case:i,kind:item.kind,quality:baseline.quality});
   return {id:'case-'+hash(group).slice(0,20),group,args:[JSON.stringify(x)],expected};
  });
  const episode={version:'natlang.skill-episode/1',id,family:'efficiency-'+item.family,split:'train',license:'project-generated',source_groups:cases.map(row=>row.group),
   target:{kind:'improvement-case',entry:'solve.nl',source:{schema:'natlang.efficiency-target/1',id:item.family},files:{'solve.nl':`---\nargs: { instance: string }\nreturns: string\n---\n${item.context}\nParse instance JSON. ${definitions[item.kind][0]}\nUse helpful bound skills when their descriptions apply. Return only the requested JSON as a string. Host independently checks feasibility and computes the objective.\n`}},
   library:library(item.kind,variant),support:{cases:cases.slice(0,6)},query:{cases:cases.slice(6)},operations:variant==='metadata'?['revise','select','test']:['create','revise','select','test'],limits:{maxSteps:6},
   provenance:{generator:'natlang.efficiency-corpus/1',family:item.family,library_variant:variant,...(variant==='metadata'?{selection_design:'metadata-tuning'}:{}),metric:{schema:'natlang.skill-objective/1',kind:item.kind}}};
  const errors=validateEpisode(episode);if(errors.length)throw Error(JSON.stringify(errors));episodes.push(episode);
 }
 return {episodes,baselines};
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const options={};for(let i=2;i<process.argv.length;i+=2){const key=process.argv[i].replace(/^--/,'');if(!['out','replicas','variants'].includes(key)||process.argv[i+1]===undefined)throw Error('invalid option');options[key]=process.argv[i+1];}
 if(!options.out)throw Error('--out required');const out=resolve(options.out),{episodes,baselines}=createEfficiencyEpisodes({replicas:Number(options.replicas??3),variants:(options.variants??'empty,metadata,distractor').split(',')});
 await mkdir(out,{recursive:true});const body=episodes.map(JSON.stringify).join('\n')+'\n';await writeFile(join(out,'efficiency-episodes.jsonl'),body,{flag:'wx'});
 const manifest={schema:'natlang.efficiency-corpus/1',families:families.map(x=>({family:x.family,objective:x.kind})),episodes:episodes.length,cases:baselines.length,sha256:hash(body),provider_calls:0,publication:'Task candidates only; not generated or admitted training trajectories',construction_baselines:baselines};
 await writeFile(join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({...manifest,construction_baselines:undefined}));
}
