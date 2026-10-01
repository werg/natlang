/** Run every arm serially; training headroom is diagnostic, never a gate on implementing/testing the native loop. */
import {spawn} from 'node:child_process';
import {mkdir,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
const [output]=process.argv.slice(2);if(!output)throw Error('usage: run-structural-series.mjs OUTPUT');
await mkdir(output,{recursive:true});const attempts=[];
for(const command of ['probe','native','direct','confirm']){
 const started=new Date().toISOString();console.log(JSON.stringify({command,status:'starting'}));
 const code=await new Promise((resolve,reject)=>{const child=spawn(process.execPath,[new URL('./structural-study.mjs',import.meta.url).pathname,output,command],{stdio:'inherit'});child.on('error',reject);child.on('exit',code=>resolve(code));});
 attempts.push({command,started,finished:new Date().toISOString(),exitCode:code});
 await writeFile(join(output,'series.json'),JSON.stringify({schema:'natlang.structural-series/1',output:resolve(output),attempts,commands:4,retries:0,interpretation:'One development sequence and one final frozen confirmation. All commands run; failed arms remain recorded, never silently replenished.'},null,2)+'\n');
 console.log(JSON.stringify({command,status:'finished',exitCode:code}));
}
