/** Syntax-aware reducer call migration. Altered recorded observations must be replayed. */
import ts from 'typescript';
import YAML from 'yaml';
import { createReadStream,createWriteStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { writeFile } from 'node:fs/promises';
import { once } from 'node:events';
import {migrateRevisionCode} from './migrate-revision-code.mjs';
import {migrateIterationCode} from './migrate-iteration-code.mjs';
export function migrateCode(source,reducers) {
  const file=ts.createSourceFile('migration.ts',source,ts.ScriptTarget.ES2022,true), edits=[];
  function visit(node) {
    if(ts.isCallExpression(node) && reducers.has(node.expression.getText(file)) && node.arguments.length) {
      const [folder,...args]=node.arguments;
      const call=`(await ${folder.getText(file)}.propose(${[node.expression.getText(file),...args.map(arg=>arg.getText(file))].join(', ')})).value`;
      if(ts.isAwaitExpression(node.parent))edits.push({start:node.parent.getStart(file),end:node.parent.getEnd(),text:call});
      else edits.push({start:node.getStart(file),end:node.getEnd(),text:`Promise.resolve(${call})`});
      return;
    }
    ts.forEachChild(node,visit);
  }
  visit(file);
  for(const edit of edits.sort((a,b)=>b.start-a.start))source=source.slice(0,edit.start)+edit.text+source.slice(edit.end);
  return {source,changed:edits.length};
}
function reducerNames(value,result=new Set()) {
  if(!value || typeof value!=='object')return result;
  if(value.files) for(const [path,source]of Object.entries(value.files)) {
    if(!path.endsWith('.nl') || typeof source!=='string')continue;
    const front=/^---\r?\n([\s\S]*?)\r?\n---/.exec(source);
    if(front && YAML.parse(front[1])?.kind==='directory-reducer')result.add(path.split('/').at(-1).slice(0,-3));
  }
  if(value.kind==='directory-reducer' || value.subtype==='directory-reducer') {if(value.name)result.add(value.name);if(value.function)result.add(value.function);}
  for(const [name,child]of Object.entries(value)) {
    if(child?.$lambda?.subtype==='directory-reducer')result.add(name);
    if(child && typeof child==='object')reducerNames(child,result);
  }
  return result;
}
export function migrateRow(row) {
  const copy=structuredClone(row), reducers=reducerNames(copy);let changes=0;
  function visit(value) {
    if(!value || typeof value!=='object')return;
    for(const [key,child] of Object.entries(value)) {
      if(['service_scopes','serviceScopes'].includes(key)&&child&&typeof child==='object'&&!Array.isArray(child)){
        for(const [service,paths]of Object.entries(child))if(Array.isArray(paths)){value[key][service]=paths.map(path=>path.endsWith('/**')?path:path+'/**');changes+=paths.filter(path=>!path.endsWith('/**')).length;}
      }else if(typeof child==='string' && (['code','source'].includes(key) || key.endsWith('.ts'))) {
        const result=migrateCode(child,reducers),revisions=migrateRevisionCode(result.source,{populationTypes:JSON.stringify(row).includes('program-improver')}),iteration=migrateIterationCode(revisions.source);
        value[key]=iteration.source;changes+=result.changed+revisions.changes+iteration.changed;
      }else if(key==='arguments'&&typeof child==='string'){
        try{const parsed=JSON.parse(child),prior=changes;visit(parsed);if(changes!==prior)value[key]=JSON.stringify(parsed);}catch{/* Non-JSON content is retained in the migration backlog. */}
      }else if(key==='mode'&&child==='compatibility'){value[key]='derived';if(value.root===undefined)value.root=0;changes++;}
      else if(child && typeof child==='object')visit(child);
    }
  }
  visit(copy);
  if(changes) {
    copy.migration={version:'natlang.folder-api/1',sourceId:row.id,changes,disposition:row.trajectory||row.messages||row.exchanges?'replay-required':'migrated-program'};
    if(copy.outcome)copy.outcome.accepted=false;
    if(copy.training_admission)copy.training_admission.approved=false;
    if(copy.trace_admission)copy.trace_admission.admitted=false;
  }
  return {row:copy,changes};
}
if(process.argv[1]?.endsWith('migrate-folder-api.mjs')) {
  const [input,output]=process.argv.slice(2);if(!output)throw Error('usage: migrate-folder-api.mjs INPUT_JSONL OUTPUT_JSONL');
  const stream=createWriteStream(output);let count=0,changes=0,replay=0;
  for await(const line of createInterface({input:createReadStream(input),crlfDelay:Infinity})) {
    if(!line.trim())continue;
    const result=migrateRow(JSON.parse(line));count++;changes+=result.changes;if(result.row.migration?.disposition==='replay-required')replay++;
    if(!stream.write(JSON.stringify(result.row)+'\n'))await once(stream,'drain');
  }
  stream.end();await once(stream,'finish');
  await writeFile(output+'.manifest.json',JSON.stringify({version:'natlang.folder-api-migration/1',input,output,count,changes,replayRequired:replay},null,2));
  console.log(JSON.stringify({count,changes,replayRequired:replay}));
}
