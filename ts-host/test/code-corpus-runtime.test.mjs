import test from 'node:test';
import assert from 'node:assert/strict';
import {compileScopeSnippet} from '../dist/scope-compiler.js';
import {replayIsolated} from '../scripts/code-corpus/replay.mjs';
test('library throw guards compile but catching runtime failures remains forbidden',()=>{
 assert.equal(compileScopeSnippet('if (x < 0) throw new RangeError("negative"); return x;', {inputBindings:['x']}).ok,true);
 assert.equal(compileScopeSnippet('try { throw new Error("x"); } catch {}').ok,false);
});
test('guarded library bodies replay on valid input and fail closed on invalid input',async()=>{
 const record={id:'guard',group_id:'guard',kind:'function',instruction:'Return nonnegative x.',source:{name:'fixture'},function:{parameters:[{name:'x'}],body:'{ if (x < 0) throw new RangeError("negative"); return x; }'},cases:[{args:[3],expected:3,outcome:'return'}]};
 assert.equal((await replayIsolated(record,0)).outcome.accepted,true);
 record.cases=[{args:[-1],expected:-1,outcome:'return'}];
 const failed=await replayIsolated(record,0);assert.equal(failed.outcome.accepted,false);assert.match(failed.outcome.detail,/negative/);
});
test('directly returned empty array locals use the declared result type',async()=>{
 const record={id:'empty',group_id:'empty',kind:'function',instruction:'Copy the input array.',source:{name:'fixture'},function:{parameters:[{name:'x'}],body:'{ const result = x.slice(); return result; }'},cases:[{args:[[]],expected:[],outcome:'return'},{args:[[1]],expected:[1],outcome:'return'}]};
 assert.equal((await replayIsolated(record,0)).outcome.accepted,true);
});
