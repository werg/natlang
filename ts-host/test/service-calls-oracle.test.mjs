import assert from 'node:assert/strict';
import test from 'node:test';
import {observeRequiredServiceCalls} from '../dist/teacher/service-calls.js';
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);

test('records actual successful service calls without counting failures or invented arguments', async () => {
  let calls=0;
  const services={board:{commit_move(input){calls++; if(input.card==='bad')throw Error('bad'); return {revision: calls}}}};
  const spec=[{service:'board',method:'commit_move',args:[{card:'a',to:'done'}],min_successes:1,max_successes:1}];
  const check=observeRequiredServiceCalls(spec,services);
  assert.equal(check(same),false);
  assert.throws(()=>services.board.commit_move({card:'bad',to:'done'}));
  assert.equal(check(same),false);
  assert.deepEqual(services.board.commit_move({card:'a',to:'done'}),{revision:2});
  assert.equal(check(same),true);
  services.board.commit_move({card:'other',to:'done'});
  assert.equal(check(same),false);
});

test('a declared blocked operation requires a real failed attempt and no successful commit', () => {
  const services={board:{commit_move(){throw Error('locked')}}};
  const spec=[{service:'board',method:'commit_move',args:[{card:'a',to:'done'}],
    min_attempts:1,min_successes:0,max_successes:0}];
  const check=observeRequiredServiceCalls(spec,services);
  assert.equal(check(same),false);
  assert.throws(()=>services.board.commit_move({card:'a',to:'done'}));
  assert.equal(check(same),true);
});
