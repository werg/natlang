import test from 'node:test';
import assert from 'node:assert/strict';
import {assemble} from '../scripts/code-corpus/assemble.mjs';
const task=(id,code,split)=>({id,group_id:id,kind:'instruction',language:'typescript',instruction:'Double x',source:{name:'fixture',split},function:{source:code},verification:{status:'unverified'}});
test('dedup links groups and preserves upstream holdout across duplicate implementations',()=>{
 const a=task('a','function f(x:number) { return x*2; }','train');
 const b=task('b','/* doc */ function f(x: number){return x * 2;}','test');
 const c=task('b','function g(x:number) { return x*3; }','train');c.id='c';
 const r=assemble([a,b,c]);assert.equal(r.rows.length,2);assert.ok(r.rows.every(v=>v.split==='test'));assert.equal(r.rows[0].group_id,r.rows[1].group_id);
});
test('syntax errors and non-code translation requests cannot enter direct SFT',()=>{
 const r=assemble([task('a','function {'),{...task('b','def f(): pass'),language:'python'}]);assert.equal(r.rows.length,0);
});
