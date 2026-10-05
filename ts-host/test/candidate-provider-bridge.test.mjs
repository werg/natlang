import test from 'node:test';import assert from 'node:assert/strict';import {reasoningAliases} from '../scripts/candidate-provider-bridge.mjs';
test('reasoning aliases preserve exact history and do not mutate source',()=>{
 const body={messages:[{role:'assistant',content:null,reasoning_content:' Think\n',tool_calls:[{id:'x'}]},{role:'user',content:'input'}]};
 const result=reasoningAliases(body);assert.equal(result.messages[0].reasoning,' Think\n');assert.equal(result.messages[0].reasoning_content,' Think\n');assert.equal(body.messages[0].reasoning,undefined);assert.deepEqual(result.messages[1],body.messages[1]);
 assert.throws(()=>reasoningAliases({messages:[{role:'assistant',reasoning:'a',reasoning_content:'b'}]}),/conflicting/);
});
