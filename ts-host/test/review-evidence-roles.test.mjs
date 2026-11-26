import test from 'node:test';import assert from 'node:assert/strict';
import {semanticReviewRolesBound} from '../scripts/observed-goal-completion.mjs';
function fixture(){const input={actual:{goal:'Explain the unchanged state',artifacts:{},requestedReturn:{reason:'Window withdrawn'},attemptedArguments:[{path:'/effects/0/args',method:'apply',arguments:['old',{notice:'shift-board'}]}]},expected:{goal:'Explain the unchanged state',requestedReturnRequired:true,evidenceRoleLayout:'native-reference/1',observations:{current:{notice:'paper'},effects:[{method:'apply',result:{status:'changed'},attemptedArgumentsPath:'/effects/0/args'}]}},rubric:'Judge requested meaning against native facts'};const grade={accepted:true,verdict:'TEST-only local reviewer fixture'};return {input,review:{schema:'natlang.live-workflow-review/1',model:'TEST-only:reviewer',teacher:'TEST-only:teacher',input:structuredClone(input),verdict:structuredClone(grade),exchange:{request:{messages:[{role:'user',content:JSON.stringify({rubric:input.rubric,reference:input.expected,candidate:input.actual})}]},reply:{calls:[['grade',structuredClone(grade)]],raw_calls:[{function:{name:'grade',arguments:JSON.stringify(grade)}}],raw_response:{provider:'TEST-only',model:'reviewer'}}}}};}
test('exact new native-reference request and raw grade bind without model inference',()=>{const{input,review}=fixture();assert.equal(semanticReviewRolesBound(review,input),true);});
for(const[label,alter]of[
 ['changed native reference',(r)=>{r.input.expected.observations.current.notice='shift-board';}],
 ['changed requested answer',(r)=>{r.input.actual.requestedReturn={reason:'Proceed'};}],
 ['changed attempted input',(r)=>{r.input.actual.attemptedArguments[0].arguments=[];}],
 ['substituted request',(r)=>{r.exchange.request.messages[0].content='{}';}],
 ['unknown role schema',(r)=>{r.input.expected.evidenceRoleLayout='unknown';}],
 ['forged verdict',(r)=>{r.verdict.verdict='Unissued verdict';}],
 ['forged raw grade',(r)=>{r.exchange.reply.raw_calls[0].function.arguments=JSON.stringify({accepted:false,verdict:'Rejected'});}],
 ['wrong provider',(r)=>{r.exchange.reply.raw_response.provider='foreign';}],
 ['teacher self review',(r)=>{r.teacher=r.model;}],
 ['missing actual reply',(r)=>{delete r.exchange.reply;}],
])test(`new review role guard refuses ${label}`,()=>{const{input,review}=fixture();alter(review);assert.equal(semanticReviewRolesBound(review,input),false);});
test('new format requires declared source input; legacy declarations remain exact',()=>{const{input,review}=fixture();assert.equal(semanticReviewRolesBound(review),false);assert.equal(semanticReviewRolesBound(review,{}),false);const legacy={input:{actual:{old:'value'},expected:{goal:'old'}}};assert.equal(semanticReviewRolesBound(legacy),true);assert.equal(semanticReviewRolesBound(legacy,legacy.input),true);assert.equal(semanticReviewRolesBound(legacy,{actual:{old:'other'}}),false);});
