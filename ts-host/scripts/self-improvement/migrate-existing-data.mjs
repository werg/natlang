/** Migrate canonical IR files and register replacements; historical executions remain replay backlog. */
import { readFile,mkdir,writeFile } from 'node:fs/promises';
import { join,basename,resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
const [output,...inputs]=process.argv.slice(2);if(!inputs.length)throw Error('usage: migrate-existing-data.mjs OUTPUT_DIR INPUT_JSONL...');
await mkdir(output,{recursive:true});const replacements={},manifest=[];
for(const input of inputs) {
 const identity=createHash('sha256').update(resolve(input)).digest('hex').slice(0,12),target=join(output,identity+'-'+basename(input));
 const result=spawnSync(process.execPath,[new URL('./migrate-folder-api.mjs',import.meta.url).pathname,input,target],{stdio:'inherit'});if(result.status!==0)throw Error('migration failed: '+input);
 replacements[input]=target;manifest.push(JSON.parse(await readFile(target+'.manifest.json','utf8')));
}
await writeFile(join(output,'replacements.json'),JSON.stringify(replacements,null,2));
await writeFile(join(output,'manifest.json'),JSON.stringify({version:'natlang.existing-data-folder-migration/1',replacements,files:manifest},null,2));
