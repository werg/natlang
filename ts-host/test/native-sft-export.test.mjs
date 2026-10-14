import assert from 'node:assert/strict';
import { test } from 'node:test';
import { renderSftTurn } from '../scripts/export-native-sft.mjs';

const render = async messages => messages.map(message => {
  const calls = (message.tool_calls ?? []).map(call => `${call.function.name}:${call.function.arguments}`).join('|');
  const reasoning = message.reasoning_content ? `<think>${message.reasoning_content}</think>` : '';
  return `<${message.role}>${reasoning}${message.content ?? ''}${calls}` +
    (message.role === 'assistant' ? '<END>' : '');
}).join('');

test('native SFT export renders approved reasoning and tool choices through the selected template', async () => {
  const row = { id: 'turn-1', program_id: 'p', source_groups: ['p'], family: 'teacher_program', skill: 'eval',
    teacher_trajectory_id: 't', teacher_trajectory_digest: 'd', training_admission: { approved: true },
    messages: [{ role: 'user', content: 'Compute.' }], tools: [], teacher_reasoning: 'Use exact arithmetic.',
    target: { role: 'assistant', content: '', tool_calls: [{ id: 'old', type: 'function',
      function: { name: 'eval', arguments: '{"code":"2+2"}' } }] } };
  const rendered = await renderSftTurn(row, render, '<END>');
  assert.equal(rendered.prompt, '<user>Compute.');
  assert.match(rendered.completion, /<think>Use exact arithmetic.<\/think>eval:/);
  assert.ok(rendered.completion.endsWith('<END>'));
});

test('native SFT export skips decisions denied training admission', async () => {
  assert.equal(await renderSftTurn({ training_admission: { approved: false } }, render, '<END>'), null);
});

test('reasoning in the history is trimmed as the target is, so a later turn extends an earlier one', async () => {
  const base = { program_id: 'p', source_groups: ['p'], family: 'teacher_program', skill: 'eval',
    teacher_trajectory_id: 't', teacher_trajectory_digest: 'd', training_admission: { approved: true }, tools: [] };
  const call = { id: 'c', type: 'function', function: { name: 'eval', arguments: '{"code":"1"}' } };
  const first = await renderSftTurn({ ...base, id: 'turn-1', messages: [{ role: 'user', content: 'Go.' }],
    teacher_reasoning: 'Start.\n', target: { role: 'assistant', content: '', tool_calls: [call] } }, render, '<END>');
  const second = await renderSftTurn({ ...base, id: 'turn-2', messages: [{ role: 'user', content: 'Go.' },
    { role: 'assistant', content: '', reasoning_content: 'Start.\n', tool_calls: [{ ...call, id: 'teacher_0' }] },
    { role: 'tool', tool_call_id: 'teacher_0', content: '1' }], teacher_reasoning: 'Done.',
    target: { role: 'assistant', content: 'ok' } }, render, '<END>');
  assert.ok(second.prompt.startsWith(first.prompt + first.completion));
});

test('reasoning written for a row rather than by a model is masked through the end of reasoning', async () => {
  const row = { id: 'turn-1', program_id: 'p', source_groups: ['p'], family: 'reference_program', skill: 'eval',
    teacher_trajectory_id: 't', teacher_trajectory_digest: 'd', training_admission: { approved: true },
    messages: [{ role: 'user', content: 'Compute.' }], tools: [], teacher_reasoning: 'I run the next step in eval.',
    teacher_reasoning_trained: false,
    target: { role: 'assistant', content: '', tool_calls: [{ id: 'old', type: 'function',
      function: { name: 'eval', arguments: '{"code":"2+2"}' } }] } };
  const rendered = await renderSftTurn(row, render, '<END>', '</think>');
  assert.equal(rendered.completion.slice(rendered.completion_masked), 'eval:{"code":"2+2"}<END>');
  await assert.rejects(renderSftTurn(row, render, '<END>'), /--reasoning-end/);
});

test('native SFT CLI renders Unicode-containing JSONL rows without fragmentation',async()=>{
 const {createServer}=await import('node:http');
 const {mkdtempSync,writeFileSync,readFileSync,rmSync}=await import('node:fs');
 const {tmpdir}=await import('node:os');
 const {join,resolve}=await import('node:path');
 const {spawn}=await import('node:child_process');
 const dir=mkdtempSync(join(tmpdir(),'natlang-sft-unicode-'));
 const server=createServer(async(req,res)=>{
  res.setHeader('content-type','application/json');
  if(req.url==='/props'){res.end(JSON.stringify({chat_template:'fixture'}));return}
  let body='';for await(const chunk of req)body+=chunk;
  res.end(JSON.stringify({prompt:await render(JSON.parse(body).messages)}));
 });
 try{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const row={id:'unicode',messages:[{role:'user',content:'A\u2028B\u2029C'}],tools:[],target:{role:'assistant',content:'Answer\u2028complete\u2029'}};
  const input=join(dir,'input.jsonl'),output=join(dir,'output.jsonl');writeFileSync(input,JSON.stringify(row)+'\n');
  const child=spawn(process.execPath,[resolve(import.meta.dirname,'../scripts/export-native-sft.mjs'),input,output,'--server',`http://127.0.0.1:${server.address().port}`,'--end-token','<END>']);
  let stderr='';child.stderr.on('data',chunk=>stderr+=chunk);child.stdout.resume();
  const code=await new Promise((resolve,reject)=>{child.on('error',reject);child.on('close',resolve)});
  assert.equal(code,0,stderr);const result=JSON.parse(readFileSync(output,'utf8'));
  assert.equal(result.prompt,'<user>'+row.messages[0].content);assert.equal(result.completion,'<assistant>'+row.target.content+'<END>');
 }finally{await new Promise(resolve=>server.close(resolve));rmSync(dir,{recursive:true,force:true})}
});
