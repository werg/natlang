import assert from 'node:assert/strict';
import { test } from 'node:test';
import { subtreeEffects } from '../dist/calls/replay.js';
import { graphCustodyDigest } from '../dist/calls/graph-custody.js';
import { hexDigest } from '../dist/native/hash.js';
const ref=value=>({complete:true,value,hash:hexDigest(JSON.stringify(value)),bytes:new TextEncoder().encode(JSON.stringify(value)).byteLength});

function graph(depth=24) {
  const rows=new Map();
  for(let i=0;i<depth;i++) {
    const call_id=`t/${i}`, child_call_ids=i+1<depth?[`t/${i+1}`]:[];
    rows.set(call_id,{call_id,task_id:'t',parent_call_id:i?`t/${i-1}`:null,
      graph_custody:{schema:'natlang.call-graph-chronology/1',task_id:'t',child_call_ids},
      effects:[{order:0,graph_order:depth-i-1,service:'s',method:'x',
        args:ref([i]),result:ref(i)}]});
  }
  for(const row of rows.values()) row.graph_custody.sha256=graphCustodyDigest(row.task_id,row.call_id,row.parent_call_id,row.graph_custody.child_call_ids,row.effects);
  const store={call:id=>rows.get(id),children:id=>(rows.get(id)?.graph_custody.child_call_ids??[]).map(call_id=>({call_id})),
    value:ref=>ref?.value};
  return {rows,store,root:rows.get('t/0')};
}

test('authenticates every child and sorts actual task-wide requests beyond depth sixteen',()=>{
  const {store,root}=graph();
  const effects=subtreeEffects(store,root);
  assert.equal(effects.length,24);
  assert.deepEqual(effects.map(effect=>effect.result),Array.from({length:24},(_,i)=>23-i));
});

test('missing children, duplicate chronology, and unknown historical cross-call order fail closed',()=>{
  let g=graph(2);g.rows.delete('t/1');
  assert.throws(()=>subtreeEffects(g.store,g.root),/missing|inventory/);
  g=graph(2);const changed=g.rows.get('t/1');changed.effects[0].graph_order=g.root.effects[0].graph_order;
  changed.graph_custody.sha256=graphCustodyDigest(changed.task_id,changed.call_id,changed.parent_call_id,changed.graph_custody.child_call_ids,changed.effects);
  assert.throws(()=>subtreeEffects(g.store,g.root),/missing or repeated/);
  g=graph(2);const target=g.rows.get('t/1').effects[0].result;
  g.store.value=ref=>ref===target?'forged':ref?.value;
  assert.throws(()=>subtreeEffects(g.store,g.root),/content binding/);
  g=graph(2);for(const row of g.rows.values())delete row.graph_custody;
  g.store.children=id=>id==='t/0'?[{call_id:'t/1'}]:[];
  assert.throws(()=>subtreeEffects(g.store,g.root),/historical cross-call effect chronology is unknown/);
});

test('real nested calls record task-wide service request chronology and child custody',async()=>{
  const { mkdtempSync,rmSync }=await import('node:fs');
  const { tmpdir }=await import('node:os');
  const { join }=await import('node:path');
  const { createNatlangRuntime,loadVirtualNatlang,CallStore }=await import('../dist/index.js');
  const dir=mkdtempSync(join(tmpdir(),'natlang-graph-order-'));const store=CallStore.open(dir);
  try {
    const files={'root.nl':'---\nargs: { id: string }\nreturns: number\n---\nAsk the child.\n',
      'root/price.nl':'---\nargs: { id: string }\nreturns: number\n---\nLook up order.\n'};
    let parentTurn=0,childTurn=0;
    const model=async request=>{
      const opening=String(request.messages.find(message=>message.role==='user')?.content??'');
      if(opening.includes('Look up order')) return {calls:[childTurn++===0?
        ['eval',{code:"return orders.lookup('child').total"}]:['return_result',{status:'success',value:5}]]};
      return {calls:[parentTurn++===0?['eval',{code:"orders.lookup('before'); const value = await price(id); orders.lookup('after'); return value"}]:
        ['return_result',{status:'success',value:5}]]};
    };
    const service={lookup:id=>({id,total:5})};const root=loadVirtualNatlang(files,'root.nl');
    assert.equal(await createNatlangRuntime({model,calls:store,services:{orders:service}}).run(()=>root('q')),5);
    const calls=store.calls(),parent=calls.find(call=>call.definition_name==='root');
    const record=store.call(parent.call_id);assert.equal(record.graph_custody.child_call_ids.length,1);
    const effects=subtreeEffects(store,record);
    assert.deepEqual(effects.map(effect=>effect.args[0]),['before','child','after']);
  } finally {store.close();rmSync(dir,{recursive:true,force:true});}
});
