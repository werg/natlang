import assert from 'node:assert/strict';
import {test} from 'node:test';
import {BrowserCallStore} from '../dist/browser/call-store.js';

test('browser checkpoints cross the real client protocol and wait for the worker commit acknowledgement',async()=>{
  const original=globalThis.Worker;let worker;
  class Worker {
    constructor(){worker=this;}
    postMessage(value){
      const message=structuredClone(value);
      if(message.op==='checkpoint'){this.checkpoint=message;return;}
      if('id' in message)queueMicrotask(()=>this.onmessage({data:{type:'reply',id:message.id,ok:true}}));
    }
    terminate(){}
  }
  globalThis.Worker=Worker;
  const store=await BrowserCallStore.open();
  try {
    let settled=false;
    const pending=store.checkpoint('call',{call_id:'call'},new Map([['hash','exact bytes']])).then(()=>{settled=true;});
    await Promise.resolve();assert.equal(settled,false);
    assert.equal(worker.checkpoint.op,'checkpoint');
    assert.deepEqual([...worker.checkpoint.args[2]],[['hash','exact bytes']]);
    worker.onmessage({data:{type:'reply',id:worker.checkpoint.id,ok:true}});
    await pending;assert.equal(settled,true);
    const failed=store.checkpoint('call',{call_id:'call'},new Map());
    worker.onmessage({data:{type:'reply',id:worker.checkpoint.id,ok:false,error:'OPFS commit failed'}});
    await assert.rejects(failed,/OPFS commit failed/);
  } finally {await store.close();globalThis.Worker=original;}
});
