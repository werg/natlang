/** Close a pinned discovery snapshot without confusing exhausted work with admitted coverage. */
import {readFile,writeFile,readdir,mkdir} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
const [root,output=join(root??'.','completion')]=process.argv.slice(2);
if(!root)throw Error('usage: summarize-completion.mjs COLLECTION_ROOT [OUTPUT]');
const read=async p=>JSON.parse(await readFile(p,'utf8'));
const lines=async p=>(await readFile(p,'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
const incidents=await lines(join(root,'index/incidents.jsonl')),inventory=await read(join(root,'index/summary.json'));
const allocation=(await read(join(root,'assignment-allocation/checkpoint.json'))).value;
const linked=new Map(),loops=new Map(),negative=[];
const corpus=await read(resolve('data/teacher/self-improvement/current-manifest.json'));
const admitted=new Set();for(const artifact of corpus.artifacts)for(const row of await lines(resolve(artifact.path)))admitted.add(row.teacher_trajectory_digest);
const historical=[];
for(const entry of await readdir(root,{withFileTypes:true})){
 if(!entry.isDirectory())continue;
 let verification;try{verification=await read(join(root,entry.name,'verification.json'));}catch{continue;}
 for(const id of verification.verified){
  const path=join(root,entry.name,id+'.result.json');let row;try{row=await read(path);}catch{continue;}
  if(!row.accepted)continue;
  const digest=createHash('sha256').update(JSON.stringify(row)).digest('hex');
  if(!admitted.has(digest)){historical.push({id,path,reason:'not represented by the explicitly admitted current corpus'});continue;}
  const existing=loops.get(id);if(!existing||existing.exchanges<row.exchanges.length)loops.set(id,{id,path,exchanges:row.exchanges.length,targetResolved:verification.targetOutcomes?.find(item=>item.id===id)?.resolved??null,changed:row.sourceDiff?.length>0});
  for(const incident of row.incidents??[])linked.set(incident.id,id);
 }
 for(const row of verification.rejected)negative.push({collection:entry.name,...row});
}
const counts={},clusters=new Map();
const exclusions={infrastructure:'infrastructure-excluded','evaluation-only':'heldout-excluded','oracle-review':'independent-oracle-required'};
const terminal=incidents.map(row=>{
 const representative=linked.get(row.id);
 const status=representative?'verified-representative':exclusions[row.route]??'assignment-allocation-exhausted';
 counts[status]=(counts[status]??0)+1;
 const group=clusters.get(row.cluster)??{cluster:row.cluster,incidents:0,statuses:{}};group.incidents++;group.statuses[status]=(group.statuses[status]??0)+1;clusters.set(row.cluster,group);
 return {id:row.id,cluster:row.cluster,route:row.route,source:row.source,status,...(representative?{representative}:{}),reason:representative?'linked independently reproduced source/state trajectory':exclusions[row.route]?'discovery route requires exclusion or independent reconstruction/oracle':'remaining shared provider capacity cannot allocate another default complete-loop collection; no trajectory admitted'};
});
await mkdir(output,{recursive:true});await writeFile(join(output,'incidents.jsonl'),terminal.map(JSON.stringify).join('\n')+'\n');
const summary={schema:'natlang.improvement-completion/1',discovery:inventory,discoverySha256:createHash('sha256').update(await readFile(join(root,'index/incidents.jsonl'))).digest('hex'),incidentStatuses:counts,clusters:[...clusters.values()],verifiedUniqueTrajectories:[...loops.values()],rejected:negative,historicalNotCounted:historical,allocation,remaining:Object.fromEntries(Object.entries(allocation.ceiling).map(([key,value])=>[key,value-(allocation.used[key]??0)])),completeLoopTarget:24,coverageShortfall:Math.max(0,24-loops.size),cashSpend:'not measured for unpriced provider calls; cost ledger zero is not evidence of zero spend',admission:'Only independent verification admits a trajectory. Terminal allocation/exclusion statuses do not establish mastery.'};
await writeFile(join(output,'summary.json'),JSON.stringify(summary,null,2)+'\n');console.log(JSON.stringify({incidents:incidents.length,statuses:counts,verifiedUniqueTrajectories:loops.size,coverageShortfall:summary.coverageShortfall}));
