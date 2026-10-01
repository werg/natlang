/** Index every failed or recovered action, including inside admitted rows. Keep immutable source references. */
import { readdir, readFile, mkdir, writeFile } from 'node:fs/promises';
import { join, resolve, dirname, basename } from 'node:path';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { createHash } from 'node:crypto';
const [output, ...roots] = process.argv.slice(2);
if (!output || !roots.length) throw Error('usage: index-failures.mjs OUTPUT RUN_OR_DATA_ROOT...');
await mkdir(output,{recursive:true});
const digest = value => createHash('sha256').update(value).digest('hex');
async function* files(root) {
  for (const entry of await readdir(root,{withFileTypes:true})) {
    if (entry.isDirectory() && !['node_modules','.git','runtime','dist','self-improvement-data'].includes(entry.name)) yield* files(join(root,entry.name));
    else if (entry.isFile() && /(?:^|\.)(?:result|error)\.json$|(?:results|trajectories|corrections)\.jsonl$|^(?:native|direct|confirm)\.json$/.test(entry.name)) yield join(root,entry.name);
  }
}
const incidents=[], seen=new Set(), inventory=[];
function failures(value,path='',result=[]) {
  if (!value || typeof value!=='object') return result;
  const action = value.surface || value.call_id || value.callId || value.phase==='action' || value.result_text || value.diagnostics;
  if (action && (['error','rejected','failed','quiesced'].includes(value.status) || ['error','rejected','failed','quiesced'].includes(value.outcome))) result.push({path,reason:value.error??value.reason??value.detail??value.result_text??value.status??value.outcome,callId:value.call_id??value.callId});
  for (const [key,child] of Object.entries(value)) if (!['arguments','value','context','messages','expected','inputs'].includes(key) && child && typeof child==='object') failures(child,`${path}/${key}`,result);
  return result;
}
for (const root of roots) for await (const file of files(resolve(root))) {
  const hasher=createHash('sha256');for await(const chunk of createReadStream(file))hasher.update(chunk);const hash=hasher.digest('hex');
  if(seen.has(hash)){inventory.push({file,hash,duplicate:true});continue;}seen.add(hash);
  async function* rows() {
    if(file.endsWith('.jsonl')) {
      let line=0;for await(const text of createInterface({input:createReadStream(file),crlfDelay:Infinity})) {
        if(text.trim()) {try{yield [line,JSON.parse(text)];}catch(error){inventory.push({file,hash,line,error:String(error)});}}line++;
      }
    } else {try{yield [0,JSON.parse(await readFile(file,'utf8'))];}catch(error){inventory.push({file,hash,error:String(error)});}}
  }
  inventory.push({file,hash});
  for await (const [line,row] of rows()) {

    const native=row.sourceManifest?.schema==='natlang.program-source/1';
    const direct=basename(file)==='direct.json'&&typeof row.accepted==='boolean';
    const confirmation=row.freeze?.metric==='paired-binary-success'||basename(file)==='confirm.json'&&!!row.comparisons;
    const program=row.task?.program_ir??row.program_ir??(confirmation?{family:row.family??basename(dirname(file)),split:'test'}:native||direct?{family:row.caseDefinition?.family??row.family??basename(dirname(file)),split:'train'}:undefined);
    const nativeCompleted=native&&row.state?.done===true&&row.validation?.gatesPassed===true&&['improved','transformed','baseline-retained'].includes(row.disposition);
    const directCompleted=direct&&row.accepted&&row.candidate?.gatesPassed===true;
    const admitted=row.outcome?.accepted===true||nativeCompleted||directCompleted, semanticPassed=row.outcome?.oracle?.accepted===true||nativeCompleted||directCompleted;
    const actions=failures({outcome:row.outcome,trajectory:row.trajectory});
    if(native)for(const [index,experiment]of (row.state?.history??[]).entries())if(experiment.accepted===false)actions.push({path:'/state/history/'+index,reason:experiment.reason});
    if (!admitted) actions.unshift({path:'/outcome',reason:row.error??(semanticPassed?'semantic result passed but training admission failed':row.outcome?.detail??'not-admitted')});
    for (const action of actions) {
      const reason=String(action.reason), split=program?.split??'unknown';
      const route=['test','validation','valid','dev'].includes(split)?'evaluation-only':/429|timeout|rate.limit|transport|Cannot find package|cancel/i.test(reason)?'infrastructure':!program?'reconstruct':/oracle|annotation|source.review/i.test(reason)?'oracle-review':/while|recursion|iterateOn|read_code|edit_code|reducer|Folder|obsolete|retired/i.test(reason)?'migration':admitted || semanticPassed?'decision-repair':'program-improvement';
      const family=program?.family??'unknown', cluster=digest(family+'\0'+reason.replace(/\d+/g,'N').slice(0,500)).slice(0,20);
      incidents.push({id:digest(hash+':'+line+':'+action.path),source:{file,hash,line:line+1,path:action.path},trajectoryId:row.id,family,cluster,split,route,recovered:admitted || semanticPassed,admitted,semanticPassed,reason,callId:action.callId??null});
    }
  }
}
await writeFile(join(output,'incidents.jsonl'),incidents.map(JSON.stringify).join('\n')+'\n');
await writeFile(join(output,'inventory.json'),JSON.stringify(inventory,null,2));
const routes={};for(const incident of incidents) routes[incident.route]=(routes[incident.route]??0)+1;
await writeFile(join(output,'summary.json'),JSON.stringify({version:'natlang.failure-index/1',artifacts:inventory.length,incidents:incidents.length,routes},null,2));
console.log(JSON.stringify({artifacts:inventory.length,incidents:incidents.length,routes}));
