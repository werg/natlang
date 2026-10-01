/** Finish the original confirmation cohort and replay/export once file development ends. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {spawn} from 'node:child_process';
const root=resolve(process.argv[2]),completion=join(root,'completion-corrected');
const progress=JSON.parse(await readFile(join(completion,'progress.json'),'utf8'));
const design=JSON.parse(await readFile(join(completion,'design.json'),'utf8'));
const deadline=progress.startedAt+design.limits.elapsedMs+60000;
let finished=false;
while(Date.now()<deadline){
 try{await readFile(join(completion,'result.json'));finished=true;break;}catch(error){if(error.code!=='ENOENT')throw error;}
 await new Promise(resolve=>setTimeout(resolve,10000));
}
// Exhaustion still yields a final report and admits only independently verified recordings.
const replay=join(root,'replay-training-v16'),training=join(root,'training-current-v16');
const commands=[
 ['confirm-followup-study.mjs',[root]],
 ['replay-followup-study.mjs',[join(root,'clean-study'),replay]],
 ['export-followup-training.mjs',[replay,training]]
];
const outcomes=[];
for(const [script,args] of commands){
 const status=await new Promise(resolve=>{
  const child=spawn(process.execPath,['scripts/self-improvement/'+script,...args],{cwd:new URL('../../',import.meta.url).pathname,stdio:'inherit'});
  child.once('error',error=>resolve({error:String(error)}));
  child.once('exit',(code,signal)=>resolve({code,signal}));
 });
 outcomes.push({script,...status});
}
await mkdir(join(root,'finalization'),{recursive:true});
await writeFile(join(root,'finalization/result.json'),JSON.stringify({completionFinished:finished,outcomes,replay,training,policy:'One original frozen confirmation cohort; retain closed case results. No new development allocation, confirmation retries, or provider inference during replay/export.'},null,2));
