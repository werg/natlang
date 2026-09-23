import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { inventory } from './inventory.mjs';
import { SOURCES } from './sources.mjs';
import { importDataset } from './datasets.mjs';
import { readJsonl, writeJsonl } from './common.mjs';
import { writeAssembly } from './assemble.mjs';

/** Regenerate a bundle from preserved snapshots without acquiring or executing source again. */
export async function rebuild(root, output, limit=1000) {
  const collection=await readJsonl(join(root,'collection.jsonl'));
  const all=[];
  for(const item of collection.filter(i=>i.status==='collected')) {
    const source=SOURCES.find(s=>s.id===item.id);
    let tasks;
    if(source.type==='git') {
      const checkout=join(root,source.id);
      const manifest=JSON.parse(await readFile(join(checkout,'.code-corpus-source.json'),'utf8'));
      tasks=await inventory(checkout,{sourceName:source.id,revision:manifest.commit,license:source.license},{limit});
    } else {
      const [snapshot]=await readJsonl(join(root,`${source.id}.snapshot.jsonl`));
      const temporary=join(root,`${source.id}.rebuild-${Date.now()}.tasks.jsonl`);
      await importDataset({input:join(root,`${source.id}.raw.jsonl`),output:temporary,adapter:source.id,limit,
        source:{revision:snapshot.revision,license:source.license,split:source.defaultSplit??'train'}});
      tasks=await readJsonl(temporary);
    }
    all.push(...tasks);
  }
  return writeAssembly(output,all);
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
 const [root,output,limit='1000']=process.argv.slice(2);
 if(!root||!output)throw new Error('Usage: rebuild.mjs SNAPSHOT_ROOT NEW_BUNDLE_DIR [LIMIT]');
 console.log(JSON.stringify(await rebuild(root,output,Number(limit))));
}
