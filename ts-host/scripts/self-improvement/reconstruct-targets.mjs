/** Reconstruct genuine source and group-disjoint suites; unsupported incidents remain explicit backlog. */
import { readFile,writeFile,mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import YAML from 'yaml';
import { parseType,formatType } from '../../dist/native/types.js';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { createHash } from 'node:crypto';
const [index,output]=process.argv.slice(2);if(!output)throw Error('usage: reconstruct-targets.mjs INDEX_DIR OUTPUT_DIR');
await mkdir(output,{recursive:true});
const incidents=(await readFile(join(index,'incidents.jsonl'),'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
const groups=new Map(),backlog=[];
const hash=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const byTrajectory=new Map();
for(const incident of incidents) {
  if(!['program-improvement','decision-repair','migration'].includes(incident.route)){backlog.push({...incident,disposition:incident.route});continue;}
  const entries=byTrajectory.get(incident.trajectoryId)??[];entries.push(incident);byTrajectory.set(incident.trajectoryId,entries);
}
const inventory=JSON.parse(await readFile(join(index,'inventory.json'),'utf8'));
for(const artifact of inventory) {
  if(artifact.duplicate || artifact.error)continue;
  async function* records() {
    if(artifact.file.endsWith('.jsonl')) {for await(const text of createInterface({input:createReadStream(artifact.file),crlfDelay:Infinity})){if(text.trim()){try{yield JSON.parse(text);}catch{}}}}
    else {try{yield JSON.parse(await readFile(artifact.file,'utf8'));}catch{}}
  }
  for await(const row of records()) {
    const own=byTrajectory.get(row.id)??[],program=row.task?.program_ir??row.program_ir,semantics=program?.semantics,root=semantics?.root?.$lambda;
    if(program?.split==='test' || !semantics)continue;
    if(typeof semantics.root==='string' && semantics.files?.[semantics.root]) {
      try {
        const front=/^---\r?\n([\s\S]*?)\r?\n---/.exec(semantics.files[semantics.root]);
        if(!front || semantics.expected===undefined)throw Error('missing contract or oracle');
        const meta=YAML.parse(front[1]), args=Object.keys(meta.args??{}).map(name=>semantics.inputs?.[name]);
        if(args.some(value=>value===undefined))throw Error('missing concrete arguments');
        const files={...semantics.files};
        const key=hash({family:program.family,files,root:semantics.root});
        const group=groups.get(key)??{id:key,family:program.family,files,entry:semantics.root,cases:[],incidents:[]};
        const sourceGroup=JSON.stringify(program.source_groups??program.source_ids??[program.id]);
        if(group.cases.length<20 && !group.cases.some(c=>c.group===sourceGroup))group.cases.push({id:program.id,group:sourceGroup,args,expected:semantics.expected,services:semantics.services,expectedFiles:semantics.expected_files,...(meta.kind==='directory-reducer'?{folder:semantics.folder_files??semantics.folder??semantics.input_files??{}}:{})});
        for(const incident of own)if(!group.incidents.some(item=>item.id===incident.id))group.incidents.push(incident);
        groups.set(key,group);continue;
      } catch(error){for(const incident of own)backlog.push({...incident,disposition:'reconstruction-required',detail:String(error)});continue;}
    }
    if(!root)continue;
    try {
      if(root.subtype==='directory-reducer' || Object.keys(root.codebase??{}).length || semantics.expected===undefined)throw Error('needs richer source reconstruction');
      const type=parseType(root.type);if(type.kind!=='lambda')throw Error('root is not callable');
      const args=type.params.fields.map(field=>semantics.inputs?.[field.name]);if(args.some(value=>value===undefined))throw Error('missing concrete input');
      const key=hash({family:program.family,instructions:root.instructions,types:root.types??{},type:root.type});
      const group=groups.get(key)??{id:key,family:program.family,root,type,cases:[],incidents:[]};
      const sourceGroup=JSON.stringify(program.source_groups??program.source_ids??[program.id]);
      if(group.cases.length<20 && !group.cases.some(c=>c.group===sourceGroup))group.cases.push({id:program.id,group:sourceGroup,args,expected:semantics.expected});
      for(const incident of own)if(!group.incidents.some(item=>item.id===incident.id))group.incidents.push(incident);
      groups.set(key,group);
    } catch(error) {for(const incident of own)backlog.push({...incident,disposition:'reconstruction-required',detail:String(error)});}
  }
}
// Original source groups that overlap belong to one split unit, even if their arrays differ.
function independentCases(cases) {
 const parent=new Map();const find=x=>{if(!parent.has(x))parent.set(x,x);if(parent.get(x)!==x)parent.set(x,find(parent.get(x)));return parent.get(x);};
 for(const row of cases){const groups=JSON.parse(row.group);for(const group of groups)find(group);for(const group of groups.slice(1))parent.set(find(group),find(groups[0]));}
 const used=new Set();return cases.filter(row=>{const root=find(JSON.parse(row.group)[0]);if(used.has(root))return false;used.add(root);return true;});
}
const targets=[];
for(const group of groups.values()) {
  if(!group.incidents.length)continue;
  group.cases=independentCases(group.cases);
  const sourceGroups=[...new Set(group.cases.flatMap(row=>JSON.parse(row.group)))];
  if(group.cases.length<7){backlog.push({id:group.id,family:group.family,disposition:'insufficient-independent-source-groups',available:group.cases.length,required:7});continue;}
  if(group.files) {
    const cases=group.cases.sort((a,b)=>a.group.localeCompare(b.group)).map((row,index)=>({...row,split:index<3?'train':index<5?'validation':'test'}));
    targets.push({id:group.id,family:group.family,sourceGroups,files:group.files,contract:{entry:group.entry,exportName:'default',programId:'improvement:'+group.id},cases,incidents:group.incidents,goal:'Repair the observed failures while preserving the declared behavior.'});continue;
  }
  const fields=group.type.params.fields;
  const nl='---\n'+YAML.stringify({args:Object.fromEntries(fields.map(field=>[field.name,formatType(field.type)])),returns:formatType(group.type.returns)})+'---\n'+group.root.instructions+'\n';
  const params=fields.map(field=>`${field.name}: ${formatType(field.type)}`).join(', ');
  const files={'solve.nl':nl,'main.ts':`import solveNative from './solve.nl';\nexport async function solve(${params}): Promise<${formatType(group.type.returns)}> { return solveNative(${fields.map(field=>field.name).join(', ')}); }\n`};
  const cases=group.cases.sort((a,b)=>a.group.localeCompare(b.group)).map((row,index)=>({...row,split:index<3?'train':index<5?'validation':'test'}));
  targets.push({id:group.id,family:group.family,sourceGroups,files,contract:{entry:'main.ts',exportName:'solve',programId:'improvement:'+group.id},cases,incidents:group.incidents,goal:'Repair the observed failures while preserving the declared behavior.'});
}
const accounted=new Set([...targets.flatMap(target=>target.incidents.map(incident=>incident.id)),...backlog.map(row=>row.id)]);
for(const incident of incidents)if(!accounted.has(incident.id))backlog.push({...incident,disposition:'needs-source-reconstruction',detail:'No reconstructable source/oracle row was linked in the declared inventory.'});
await writeFile(join(output,'targets.jsonl'),targets.map(JSON.stringify).join('\n')+'\n');
await writeFile(join(output,'backlog.jsonl'),backlog.map(JSON.stringify).join('\n')+'\n');
console.log(JSON.stringify({targets:targets.length,backlog:backlog.length}));
