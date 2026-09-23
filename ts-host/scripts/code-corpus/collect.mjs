import { mkdir, readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { SOURCES, acquireGitSource, huggingFaceRequestHeaders } from './sources.mjs';
import { inventory } from './inventory.mjs';
import { importDataset } from './datasets.mjs';
import { digest, readJsonl, writeJsonl } from './common.mjs';
import { writeAssembly } from './assemble.mjs';

async function json(url) {
  const response=await fetch(url,{signal:AbortSignal.timeout(30000),headers:huggingFaceRequestHeaders(url)});
  if(!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
  return response.json();
}
export async function collect(output,{ids,limit=1000}={}) {
  if(!Number.isSafeInteger(limit)||limit<1) throw new Error('positive limit required');
  await mkdir(output,{recursive:false});
  const report=[],all=[];
  for(const id of ids) {
    const source=SOURCES.find(s=>s.id===id);
    if(!source) throw new Error(`unknown source ${id}`);
    console.log(`Collecting ${id}`);
    try {
      const tasksPath=join(output,`${id}.tasks.jsonl`);
      if(source.type==='git') {
        const checkout=join(output,id);
        const manifest=await acquireGitSource(id,checkout);
        const rows=await inventory(checkout,{sourceName:id,revision:manifest.commit,license:source.license},{limit});
        await writeJsonl(tasksPath,rows);
      } else {
        const before=await json(`https://huggingface.co/api/datasets/${source.dataset}`);
        const rows=[];
        for(let offset=0;offset<limit;offset+=100) {
          const url=new URL('https://datasets-server.huggingface.co/rows');
          url.search=new URLSearchParams({dataset:source.dataset,config:source.defaultConfig??'default',split:source.defaultSplit??'train',offset:String(offset),length:String(Math.min(100,limit-offset))});
          const page=await json(url);
          if(!Array.isArray(page.rows)) throw new Error('missing dataset rows');
          rows.push(...page.rows);
          if(page.rows.length<Math.min(100,limit-offset)) break;
        }
        const after=await json(`https://huggingface.co/api/datasets/${source.dataset}`);
        if(before.sha!==after.sha) throw new Error('upstream dataset changed during snapshot');
        const rawPath=join(output,`${id}.raw.jsonl`);
        await writeJsonl(rawPath,rows);
        await writeJsonl(join(output,`${id}.snapshot.jsonl`),[{dataset:source.dataset,revision:before.sha,rows_sha256:digest(rows),
          note:'datasets-server preview snapshot; hash identifies captured data, not guaranteed exact repository revision',offset:0,rows:rows.length}]);
        await importDataset({input:rawPath,output:tasksPath,adapter:id,limit,source:{revision:before.sha,license:source.license,split:source.defaultSplit??'train'}});
      }
      const tasks=await readJsonl(tasksPath);all.push(...tasks);
      report.push({id,status:'collected',tasks:tasks.length,direct:tasks.filter(t=>['javascript','typescript'].includes(t.language)&&t.verification.status!=='rejected').length});
    }catch(error){report.push({id,status:'failed',error:String(error)});console.error(`${id}: ${error}`);}
    await writeJsonl(join(output,'collection.jsonl'),report,{replace:true});
  }
  const assembly=await writeAssembly(join(output,'bundle'),all);
  return {sources:report,assembly};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
  const [output,ids,limit='1000']=process.argv.slice(2);
  if(!output||!ids)throw new Error('Usage: collect.mjs NEW_OUTPUT_DIR SOURCE_IDS_COMMA_SEPARATED [LIMIT_PER_SOURCE]');
  console.log(JSON.stringify(await collect(resolve(output),{ids:ids.split(','),limit:Number(limit)})));
}
