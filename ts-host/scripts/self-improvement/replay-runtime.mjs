/** Pin one current SDK for an offline migration batch, including exporter workers. */
import {cp,mkdir,readFile,writeFile,readdir,symlink} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
export const digest=value=>createHash('sha256').update(value).digest('hex');
export async function pinReplayRuntime(output){
 const directory=resolve(output,'runtime'),manifest=join(directory,'replay-runtime.json');
 try{return JSON.parse(await readFile(manifest,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
 await mkdir(output,{recursive:true});
 await cp(new URL('../../dist/',import.meta.url),directory,{recursive:true});
 await symlink(new URL('../../node_modules/',import.meta.url).pathname,join(directory,'node_modules')).catch(error=>{if(error.code!=='EEXIST')throw error;});
 await cp(new URL('../../prelude.js',import.meta.url),join(resolve(output),'prelude.js'));
 const hashes={};
 async function walk(at){for(const entry of await readdir(at,{withFileTypes:true})){const path=join(at,entry.name);if(entry.isDirectory())await walk(path);else if(entry.name.endsWith('.js'))hashes[path.slice(directory.length+1)]=digest(await readFile(path));}}
 await walk(directory);
 const runtime={directory,hashes,identity:digest(JSON.stringify(hashes))};
 await writeFile(manifest,JSON.stringify(runtime,null,2));return runtime;
}
export const runtimeModule=(runtime,path)=>import(pathToFileURL(join(runtime.directory,path)).href);
/** Migrate context explicitly; provider replies remain original observations. */
export function migrateSystemPrompts(exchanges,previous,current,previousDirectory='',currentDirectory=''){
 let changed=0;
 const records=exchanges.map(exchange=>({...exchange,request:{...exchange.request,messages:exchange.request.messages.map(message=>{
  if(message.role!=='system'||typeof message.content!=='string')return message;
  if(!message.content.startsWith(previous))throw Error('Unrecognized recorded system prompt; recollect this invocation');
  let content=current+message.content.slice(previous.length);
  if(previousDirectory)content=content.replace(previousDirectory,currentDirectory);
  if(content!==message.content)changed++;return {...message,content};
 })}}));
 return {records,migration:{kind:'system-prompt-context-migration',before:digest(previous+previousDirectory),after:digest(current+currentDirectory),requests:changed,providerRepliesRewritten:false}};
}

/** Replace known fixture implementations in opening declarations with their public types. */
export function migrateServiceOpenings(exchanges,replacements){
 let changed=0;
 const records=exchanges.map(exchange=>({...exchange,request:{...exchange.request,messages:exchange.request.messages.map(message=>{
  if(message.role!=='assistant'||!message.tool_calls)return message;
  return {...message,tool_calls:message.tool_calls.map(call=>{
   if(call.id!=='scope_0'||call.function?.name!=='eval')return call;
   const args=JSON.parse(call.function.arguments);if(typeof args.code!=='string')return call;
   let code=args.code;for(const [before,after]of replacements)code=code.replaceAll(before,after);
   if(code===args.code)return call;
   changed++;return {...call,function:{...call.function,arguments:JSON.stringify({...args,code})}};
  })};
 })}}));
 return {records,migration:{kind:'public-service-opening-migration',changedOpenings:changed,providerRepliesRewritten:false}};
}

/** Recover missing case identity from the original, uniquely identifying fixture declaration. */
export function migrateCaseContexts(exchanges,{cases,sources,entry,fingerprint,declarationNamespace,seed=0}){
 const tasks=new Map();
 for(const exchange of exchanges){const task=exchange.request.invocation_id.split('/')[0];if(!tasks.has(task))tasks.set(task,exchange.request);}
 const identities=new Map();
 for(const [task,request]of tasks){
  if(task.startsWith('case:'))continue;
  const opening=request.messages.flatMap(message=>(message.tool_calls??[]).filter(call=>call.id==='scope_0'&&call.function?.name==='eval').map(call=>JSON.parse(call.function.arguments).code)).join('\n');
  const rows=cases.filter(row=>Object.keys(row.services??{}).length&&Object.entries(row.services).every(([name,source])=>opening.includes(declarationNamespace(name,source))));
  const instruction=String(request.messages[1]?.content);
  const programs=[...sources].filter(([,files])=>files[entry]&&instruction.includes('\n\nInstructions:\n'+files[entry].split('---').slice(2).join('---').trim()+'\n\n'));
  if(rows.length===1&&programs.length===1)identities.set(task,'case:'+fingerprint({source:programs[0][0],caseId:rows[0].id,seed}));
 }
 return {records:exchanges.map(exchange=>{
  const task=exchange.request.invocation_id.split('/')[0],context=identities.get(task);
  return context?{...exchange,request:{...exchange.request,invocation_id:context+'/'+exchange.request.invocation_id}}:exchange;
 }),migration:{kind:'case-identity-migration',recoveredTasks:identities.size,basis:'Unique original fixture declaration and exact source instructions; no task inputs or provider replies changed.'}};
}
