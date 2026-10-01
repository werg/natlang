/** Bounded correction queue; allocation and per-cluster retry checks live in the shared collector. */
import {readFile,writeFile,readdir,mkdir} from 'node:fs/promises';import {join} from 'node:path';import {spawn} from 'node:child_process';
const [casesPath,previous,output,provider='pi:openai-codex',model='gpt-6-luna']=process.argv.slice(2);
const cases=(await readFile(casesPath,'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse),missing=[];const files=new Set(await readdir(previous));
for(const row of cases){const path=row.id+'.result.json';const result=files.has(path)?JSON.parse(await readFile(join(previous,path),'utf8')):null;if(result?.accepted)continue;
 const next=structuredClone(row);next.policy.outerLesson='Previous execution did not complete: '+String(result?.error??'no admitted trajectory').slice(0,800)+'. Initialize the measured baseline, use only declared argument names, keep helper edits separate from evaluator authority, and return an independently measured selected source.';missing.push(next);
}
await mkdir(output,{recursive:true});const queue=join(output,'queue.jsonl');await writeFile(queue,missing.map(JSON.stringify).join('\n')+'\n');
const child=spawn(process.execPath,[new URL('./collect-improvement.mjs',import.meta.url).pathname,queue,output,provider,model,String(missing.length),'3'],{stdio:'inherit'});await new Promise((resolve,reject)=>{child.on('error',reject);child.on('exit',code=>code===0?resolve():reject(Error('collector exited '+code)));});
