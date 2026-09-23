import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { digest, readJsonl, writeJsonl } from './common.mjs';

// Separate from native trajectories: these views do NOT claim execution verification.
export function trainingView(task) {
  if (task.verification.status === 'rejected' || !task.instruction) return null;
  const base = {version:'natlang.code_view/1',id:task.id,group_id:task.group_id,source:task.source,
    split:Number.parseInt(digest(task.group_id).slice(0,8),16)%100<5?'test':'train',verification:task.verification};
  if (task.language === 'python') return {...base,kind:'translation_request',prompt:
    `Translate this Python function to a self-contained TypeScript function. Preserve behavior; do not execute code or guess expected outputs. Return source code only.\n\nDescription:\n${task.instruction}\n\nPython:\n${task.function.source}`,
    original_cases:task.raw,completion:null};
  const completion = task.kind === 'tool_calls' ? task.raw?.typescript_call_view : task.function?.source;
  if (!completion) return null;
  return {...base,kind:'code_sft',prompt:task.kind === 'tool_calls'
    ? `${task.instruction}\n\nAvailable tool schemas:\n${JSON.stringify(task.raw.tools)}\n\nPositional signatures:\n${JSON.stringify(task.raw.signatures ?? [])}`
    : `Write ${task.language === 'typescript' ? 'TypeScript' : 'JavaScript'} code.\n\n${task.instruction}`,completion};
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const [input,output,limit='1000'] = process.argv.slice(2);
  if (!input || !output) throw new Error('Usage: views.mjs TASKS OUTPUT [LIMIT]');
  const views=(await readJsonl(input,{limit:Number(limit)})).map(trainingView).filter(Boolean);
  await writeJsonl(output,views); console.log(JSON.stringify({views:views.length}));
}
