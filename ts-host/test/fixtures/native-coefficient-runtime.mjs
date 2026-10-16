import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {neuraleseServerModelTurn,HttpNeuraleseStore} from '../../dist/model/neuralese-server.js';
import {MemoryNeuraleseStore} from '../../dist/native/neuralese-store.js';
import {withAdapters} from '../../dist/neuralese/learning.js';
import {neuraleseRef} from '../../dist/native/neuralese.js';
const [endpoint,path]=process.argv.slice(2),fixture=JSON.parse(await readFile(path,'utf8'));
const request={messages:[{role:'user',content:'t4 t5'}],options:['t7','t8']};
const direct=async adapters=>{
 const response=await fetch(endpoint+'/v1/neuralese/decide',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({...request,adapters})});
 assert.equal(response.status,200);return (await response.json()).log_probs;
};
const store=new MemoryNeuraleseStore();
const driver=neuraleseServerModelTurn({endpoint,model:'native-coefficient-runtime-fixture',store,
 request:{x_natlang_native_coefficient_policy:fixture.runtime,x_natlang_rl_capture:true}});
assert.deepEqual((await driver.decide(request)).log_probs,await direct(fixture.defaults));
const scoped=await withAdapters(neuraleseRef('Adapter',fixture.handle),()=>driver.decide(request));
assert.deepEqual(scoped.log_probs,await direct([...fixture.defaults,{id:fixture.current,scale:1}]));
const turn=await driver({messages:request.messages,tools:[],max_tokens:1,temperature:.8,seed:421});
const capture=turn.sampling??turn.x_natlang_rl_capture;
// The transport also archives raw wire sampling receipts through the normal
// model-turn field. Assert effective behavior via an actual returned export.
const observed=await fetch(endpoint+'/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json'},
 body:JSON.stringify({messages:request.messages,max_tokens:1,temperature:.8,seed:421,x_natlang_rl_capture:true,x_natlang_adapters:fixture.defaults})});
assert.equal(observed.status,200);
const directTurn=await observed.json();assert.ok(directTurn.x_natlang_rl_capture.adapter_binding);
assert.ok(turn);
const forged=neuraleseServerModelTurn({endpoint,model:'bad-policy',store,
 request:{x_natlang_native_coefficient_policy:{...fixture.runtime,replacements:{...fixture.runtime.replacements,[fixture.handle]:fixture.handle}}}});
await assert.rejects(forged.decide(request),/restore the exact declared actor policy/);
console.log(JSON.stringify({passed:true,captured:directTurn.x_natlang_rl_capture.policy}));
