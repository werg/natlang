import test from 'node:test';
import assert from 'node:assert/strict';
import {renderCodeRow} from '../scripts/code-corpus/render.mjs';
test('code rows use actual renderer interface and preserve provenance',async()=>{
 const row={id:'a',kind:'code_sft',syntax_checked:true,prompt:'Double x',completion:'return x*2;',program_id:'g',source_groups:['g'],source:{name:'fixture'},split:'test'};
 const render=async messages=>messages.map(m=>`<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join('')+'<|im_start|>assistant\n';
 const pair=await renderCodeRow(row,render);assert.equal(pair.completion,'return x*2;<|im_end|>');assert.equal(pair.execution_verified,false);assert.equal(pair.split,'test');
});
