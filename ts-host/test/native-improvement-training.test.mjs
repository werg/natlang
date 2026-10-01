import test from 'node:test';
import assert from 'node:assert/strict';
import {Folder} from '../dist/index.js';
import {AUTHORED_IMPROVER} from '../dist/improvement/authored-source.js';
import {run} from '../dist/improvement/teacher.js';
import {observedActions,successfulTurn} from '../scripts/self-improvement/export-followup-training.mjs';
import {recordedDriver} from '../scripts/self-improvement/replay-followup-study.mjs';

test('terminal and intermediate successful actions are admitted from actual trace boundaries',()=>{
 const exchange={turn:{calls:[['eval',{code:'return 7;',finish:true}]]}};
 const action={kind:'action',name:'eval',arguments:exchange.turn.calls[0][1],outcome:'completed'};
 const trace={outcome:'done',events:[{kind:'model_request',phase:'end',turn:1},action,{kind:'state',phase:'final'}]};
 assert.deepEqual(observedActions(trace,1),[action]);
 assert.equal(successfulTurn(trace,exchange,1),true);
 action.outcome='ok';assert.equal(successfulTurn(trace,exchange,1),true);
 action.outcome='rejected';assert.equal(successfulTurn(trace,exchange,1),false);
 action.outcome='completed';action.arguments={code:'return 8;',finish:true};assert.equal(successfulTurn(trace,exchange,1),false);
});

test('recorded execution refuses changed input and preserves observed parallel completion order',async()=>{
 const request=value=>({invocation_id:'old/'+value,messages:[{role:'user',content:value}],tools:[],seed:1,max_tokens:10});
 const driver=recordedDriver([{request:request('first'),turn:{text:'1'}},{request:request('second'),turn:{text:'2'}}]);
 const completed=[];
 const second=driver({...request('second'),invocation_id:'fresh/2'}).then(()=>completed.push('second'));
 const first=driver({...request('first'),invocation_id:'fresh/1'}).then(()=>completed.push('first'));
 await Promise.all([second,first]);assert.deepEqual(completed,['first','second']);
 assert.equal(driver.audit().unconsumedRequests,0);
 await assert.rejects(driver(request('changed')),/No recorded invocation matches/);
});

test('app collection adapter executes the supplied reducer state without restarting its improvement loop',async()=>{
 const files={'solve.nl':'---\nargs: {}\nreturns: number\n---\nReturn seven.'};
 const source=Folder.fromFiles(files).snapshot().digest;
 const policy={maxExperiments:3,maxPopulation:2,strategy:'adaptive',mode:'instruction',goal:'Return seven.',allowedFiles:['solve.nl']};
 const state={iteration:1,done:false,incumbent:source,quality:1,population:[],history:[],stopReason:''};
 const record={version:'natlang.program/2',id:'continuation',kind:'lambda_source',semantics:{root:'improveStep.nl',files:AUTHORED_IMPROVER,inputs:{state,policy},folder_files:files,expected:null,evaluation_fixture:{kind:'flat-program-evaluator',scope:'invocation',caseDefinition:{id:'seven',files,policy,contract:{programId:'seven',entry:'solve.nl',exportName:'solve'},cases:[{id:'train',group:'seven-train',split:'train',args:[],expected:7},{id:'validation',group:'seven-validation',split:'validation',args:[],expected:7}]}}}};
 const result=await run(record,async request=>String(request.messages[1].content).split('\n')[0].includes('improveStep(')?{calls:[['eval',{code:"return {...state,iteration:state.iteration+1,done:true,quality:1,stopReason:'No supported hypothesis.'};",finish:true}]]}:{text:'7'}, {rootSeed:0,runId:'adapter-test',systemPrompt:'',contextTokens:16384,signal:AbortSignal.timeout(10000)});
 assert.equal(result.outcome.accepted,true);
 assert.equal(result.outcome.value.iteration,2);
 assert.equal(result.outcome.source['solve.nl'],files['solve.nl']);
});
