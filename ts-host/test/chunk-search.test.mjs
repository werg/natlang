import test from 'node:test';import assert from 'node:assert/strict';
import {replaceChunk,trajectoryNll,firstDifficultChunk,candidateDecision,correctivePrefix} from '../scripts/chunk-search.mjs';
test('one code line is replaced without mutating suffix or original evidence',()=>{
 const original={calls:[['eval',{code:'let a=1;\nreturn a;'}]],raw_response:{old:true}};
 const out=replaceChunk(original,{kind:'code_line',call_index:0,argument_name:'code',value_start:0,value_end:9},'let a=2;\n');
 assert.equal(out.calls[0][1].code,'let a=2;\nreturn a;');assert.equal(original.calls[0][1].code,'let a=1;\nreturn a;');assert.equal(out.raw_response,undefined);
});
test('whole mean cannot hide first difficult token and invalid proposals never win',()=>{
 const current={turns:[{score:{token_count:10,sum_logprob:-10,chunks:[{mean_nll:1,max_nll:9}]}}]};
 const controls={chunk_nll:2,token_nll:8,regression_tolerance:0};assert.equal(trajectoryNll(current.turns),1);assert.equal(firstDifficultChunk(current.turns,controls).turnIndex,0);
 assert.equal(candidateDecision({...current,admitted:false,ranking_nll:.1,chunk_nll:.1,chunk_max_nll:.1},current,controls).accepted,false);
 assert.equal(candidateDecision({turns:[{score:{token_count:10,sum_logprob:-20}}],admitted:true,ranking_nll:.1,chunk_nll:.1,chunk_max_nll:.1},current,controls).reason,'trajectory_likelihood_regressed');
 assert.equal(correctivePrefix([{decision_index:0,training_admission:{approved:true}},{decision_index:1,training_admission:{approved:true}}],0).length,1);
});
test('argument types and full-action boundary are explicit',()=>{
 assert.throws(()=>replaceChunk({calls:[['eval',{finish:true}]]},{kind:'argument',call_index:0,argument_name:'finish'},'true'),/changed type/);
 assert.deepEqual(replaceChunk({}, {kind:'action'}, {role:'assistant',content:null,tool_calls:[{function:{name:'eval',arguments:'{"code":"return 1"}'}}]}),{text:'',calls:[['eval',{code:'return 1'}]]});
});
test('failed/context decisions do not drive rewriting or training-target perplexity',()=>{
 const turns=[{training_target:false,score:{token_count:100,sum_logprob:-1000,chunks:[{mean_nll:10,max_nll:10}]}},{training_target:true,score:{token_count:2,sum_logprob:-1,chunks:[{mean_nll:.5,max_nll:.5}]}}];
 assert.equal(trajectoryNll(turns),.5);
 assert.equal(firstDifficultChunk(turns,{chunk_nll:1,token_nll:4}),null);
});
