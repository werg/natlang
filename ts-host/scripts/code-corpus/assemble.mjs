import ts from 'typescript';
import { mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { digest, readJsonl, writeJsonl } from './common.mjs';
import { trainingView } from './views.mjs';

export function assemble(tasks, { maxChars=16000 } = {}) {
  const candidates=[], rejected=[], parent=new Map();
  const find=x=>{if(!parent.has(x)) parent.set(x,x); if(parent.get(x)!==x) parent.set(x,find(parent.get(x))); return parent.get(x);};
  const union=(a,b)=>{a=find(a);b=find(b);if(a!==b) parent.set(a<b?b:a,a<b?a:b);};
  const byCode=new Map();
  for(const task of tasks) {
    if(task.instruction_quality==='name_only') {rejected.push({id:task.id,reason:'missing behavioral description'});continue;}
    const view=trainingView(task);
    if(!view || view.kind!=='code_sft' || !['javascript','typescript'].includes(task.language)) {rejected.push({id:task.id,reason:'no direct code view'});continue;}
    const code=view.completion;
    if(/<\|(?:im_start|im_end|tool_call_start|tool_call_end)\|>/.test(code+view.prompt)) {rejected.push({id:task.id,reason:'reserved template token'});continue;}
    if(code.length>maxChars || view.prompt.length>maxChars) {rejected.push({id:task.id,reason:'length budget'});continue;}
    const file=ts.createSourceFile(task.language==='typescript'?'candidate.ts':'candidate.js',code,ts.ScriptTarget.Latest,true);
    if(file.parseDiagnostics.length) {rejected.push({id:task.id,reason:'syntax',diagnostics:file.parseDiagnostics.map(d=>ts.flattenDiagnosticMessageText(d.messageText,' '))});continue;}
    // Exact AST print equivalence ignores comments/formatting, but not literal values or identifiers.
    const fingerprint=digest(ts.createPrinter({removeComments:true}).printFile(file));
    const group=task.group_id;
    find(group);
    if(byCode.has(fingerprint)) union(group,byCode.get(fingerprint)); else byCode.set(fingerprint,group);
    candidates.push({task,view,fingerprint});
  }
  const heldout=new Map();
  for(const {task} of candidates) {
    const explicit=task.split ?? task.source.split ?? task.source.upstream_split;
    if(['test','validation','valid','dev'].includes(explicit)) heldout.set(find(task.group_id),'test');
  }
  const rows=[], seen=new Set();
  for(const {task,view,fingerprint} of candidates) {
    if(seen.has(fingerprint)) {rejected.push({id:task.id,reason:'duplicate implementation',fingerprint});continue;}
    seen.add(fingerprint);
    const group=find(task.group_id), split=heldout.get(group) ?? (parseInt(digest(group).slice(0,8),16)%100<5?'test':'train');
    rows.push({...view,split,group_id:group,program_id:group,source_groups:[group],implementation_sha256:fingerprint,
      syntax_checked:true,execution_verified:false,messages:[{role:'user',content:view.prompt},{role:'assistant',content:view.completion}]});
  }
  return {rows,rejected,report:{input:tasks.length,kept:rows.length,rejected:rejected.length,train:rows.filter(r=>r.split==='train').length,test:rows.filter(r=>r.split==='test').length,
    sources:Object.fromEntries([...new Set(rows.map(r=>r.source.name))].map(s=>[s,rows.filter(r=>r.source.name===s).length]))}};
}
export async function writeAssembly(output,tasks,options) {
  await mkdir(output); // a new output directory is the transaction boundary
  const result=assemble(tasks,options);
  await writeJsonl(join(output,'train.jsonl'),result.rows.filter(r=>r.split==='train'));
  await writeJsonl(join(output,'test.jsonl'),result.rows.filter(r=>r.split==='test'));
  await writeJsonl(join(output,'rejected.jsonl'),result.rejected);
  await writeJsonl(join(output,'manifest.jsonl'),[{...result.report,version:'natlang.code_corpus_bundle/1',tasks_sha256:digest(tasks)}]);
  return result.report;
}
if(process.argv[1] && import.meta.url===pathToFileURL(resolve(process.argv[1])).href) {
  const [output,...inputs]=process.argv.slice(2);
  if(!output || !inputs.length) throw new Error('Usage: assemble.mjs NEW_OUTPUT_DIR TASKS.jsonl ...');
  const tasks=(await Promise.all(inputs.map(path=>readJsonl(path)))).flat();
  console.log(JSON.stringify(await writeAssembly(output,tasks)));
}
