#!/usr/bin/env node
/** Provider-free source screening; no episode registration or training admission. */
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {createHash} from 'node:crypto';
import {measureStaticPage,scoreStaticMeasurements} from '../../dist/skills/visual-objective.js';
const options={};
for(let i=2;i<process.argv.length;i+=2){
  const key=process.argv[i]?.replace(/^--/,'');
  if(!['tasks','image','out','limit'].includes(key)||!process.argv[i+1])throw Error('invalid options');
  options[key]=process.argv[i+1];
}
if(!options.tasks||!options.image||!options.out)throw Error('--tasks --image immutable-ID --out fresh-directory required');
const out=resolve(options.out);await mkdir(out,{recursive:false});
const scorerFile=new URL('../../dist/skills/visual-objective.js',import.meta.url);
const scorerHash=createHash('sha256').update(await readFile(scorerFile)).digest('hex');
const body=await readFile(options.tasks,'utf8');
const tasks=body.trim().split('\n').map(JSON.parse).filter(t=>t.blockers.length===1&&t.blockers[0]==='independent_artifact_executor_pending');
const limit=Number(options.limit??tasks.length);
if(!Number.isSafeInteger(limit)||limit<1||limit>tasks.length)throw Error('invalid limit');
const results=[];
for(const task of tasks.slice(0,limit)){
  const html=task.input.files['index.html'];
  let measurementIdentity={};
  try{
    const reference=measureStaticPage(html,options.image);
    const measurementFile=task.id.split('/').at(-1)+'.json',measurementBody=JSON.stringify(reference)+'\n';
    await writeFile(join(out,measurementFile),measurementBody,{flag:'wx'});
    measurementIdentity={measurement_file:measurementFile,measurement_sha256:createHash('sha256').update(measurementBody).digest('hex')};
    const baseline=scoreStaticMeasurements(reference,reference);
    results.push({id:task.id,group:task.group,status:'measured',...measurementIdentity,baseline});
    console.log(JSON.stringify(results.at(-1)));
  }catch(error){
    results.push({id:task.id,group:task.group,status:'unscored',...measurementIdentity,reason:String(error.message)});
    console.log(JSON.stringify(results.at(-1)));
    if(String(error.message).includes('infrastructure failure'))break;
  }
}
const report={schema:'natlang.visual-browser-screen/1',status:'pilot_not_admitted',image:options.image,
  tasks_sha256:createHash('sha256').update(body).digest('hex'),
  scorer_sha256:scorerHash,
  scorer_unchanged:scorerHash===createHash('sha256').update(await readFile(scorerFile)).digest('hex'),
  results,provider_calls:0,remaining:tasks.length-results.length,
  holds:['reward-mutation-audit','content-paint-and-design-preservation-review','role-closed-grouping','native-episode-integration']};
await writeFile(join(out,'report.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
