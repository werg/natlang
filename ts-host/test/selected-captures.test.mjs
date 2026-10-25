import assert from 'node:assert/strict';
import test from 'node:test';
import {withSelectedCaptureAccessors} from '../dist/runtime/selected-captures.js';
import {makeCallable} from '../dist/runtime/callable.js';
import {createNatlangRuntime} from '../dist/runtime/node.js';
import {declareSelectedSurface} from '../dist/native/selected-surface.js';

const plan={captures:[{name:'count',mutable:true,source:'local',type:{text:'number',natlang:'number'}}]};
const nativeFactory=(action)=>accessors=>makeCallable({kind:'inline',definition:{name:'capture-step'},
  invoke:async()=>action(accessors.count)});

test('selected live capture calls serialize refresh and flush without losing sibling updates',async()=>{
  let value=1;const writes=[],runtime=createNatlangRuntime({calls:false});
  try{
    await runtime.run(async()=>{
      const step=await withSelectedCaptureAccessors(plan,{count:[async()=>value,async next=>{writes.push(next);value=next;}]},
        nativeFactory(async cell=>{const before=cell[0]();await Promise.resolve();cell[1](before+1);return before+1;}));
      assert.deepEqual(await Promise.all([step(),step()]),[2,3]);
    });
    assert.equal(value,3);assert.deepEqual(writes,[2,3]);
  }finally{runtime.close();}
});

test('capture conflict rejects before write and a later invocation can recover from current state',async()=>{
  let value=1,conflict=true;const writes=[],runtime=createNatlangRuntime({calls:false});
  try{
    await runtime.run(async()=>{
      const step=await withSelectedCaptureAccessors(plan,{count:[async()=>value,async next=>{writes.push(next);value=next;}]},
        nativeFactory(async cell=>{const before=cell[0]();if(conflict)value=9;cell[1](before+1);return before+1;}));
      await assert.rejects(step(),/capture-conflict/);assert.deepEqual(writes,[]);assert.equal(value,9);
      conflict=false;assert.equal(await step(),10);assert.deepEqual(writes,[10]);
    });
  }finally{runtime.close();}
});

test('native failure commits no capture writes and mutable object snapshots detect in-place updates',async()=>{
  let value={items:[1]},fail=true;const writes=[],runtime=createNatlangRuntime({calls:false});
  const objectPlan={captures:[{name:'count',mutable:true,source:'local',type:{text:'{items:number[]}'}}]};
  try{
    await runtime.run(async()=>{
      const step=await withSelectedCaptureAccessors(objectPlan,{count:[async()=>structuredClone(value),async next=>{writes.push(next);value=next;}]},
        nativeFactory(async cell=>{cell[0]().items.push(2);if(fail)throw new Error('native child failed');return 'ok';}));
      await assert.rejects(step(),/native child failed/);assert.deepEqual(value,{items:[1]});assert.deepEqual(writes,[]);
      fail=false;assert.equal(await step(),'ok');assert.deepEqual(value,{items:[1,2]});
    });
  }finally{runtime.close();}
});

test('typed capture resources retain identity and only replacing their handle writes the binding',async()=>{
  let reads=0;
  class Box {
    #value=1;
    constructor(){declareSelectedSurface(this,'Box',{increment:'synchronous'},['value']);}
    get value(){reads++;return this.#value;}
    increment(){this.#value++;}
  }
  const first=new Box(),second=new Box(),writes=[],runtime=createNatlangRuntime({calls:false});
  let value={box:first,alias:first},replace=false;
  try{
    await runtime.run(async()=>{
      const step=await withSelectedCaptureAccessors(plan,{count:[async()=>value,async next=>{writes.push(next);value=next;}]},
        nativeFactory(async cell=>{
          assert.equal(cell[0]().box,first);assert.equal(cell[0]().alias,first);
          if(replace)cell[1]({box:second,alias:second});else cell[0]().box.increment();
          return 'ok';
        }));
      assert.equal(await step(),'ok');assert.deepEqual(writes,[]);
      assert.equal(reads,0,'snapshot and conflict checks never invoke native resource getters');
      replace=true;assert.equal(await step(),'ok');assert.equal(writes.length,1);
      assert.equal(value.box,second);assert.equal(value.alias,second);
    });
  }finally{runtime.close();}
});
