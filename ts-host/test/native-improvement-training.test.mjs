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

test('evaluation recordings distinguish identical visible calls with different host fixtures',async()=>{
 const {SourceEvaluator}=await import('../dist/index.js');
 const {UsageGateway}=await import('../dist/evaluation/usage.js');
 const {canonicalRequest}=await import('../scripts/self-improvement/replay-followup-study.mjs');
 const source=Folder.fromFiles({'solve.nl':'---\nkind: directory-reducer\nargs: {}\nreturns: number\n---\nReturn the number in value.txt.'}).snapshot();
 const cases=[{id:'alpha',group:'alpha',split:'train',args:[],folder:{'value.txt':'1'},expected:1},{id:'beta',group:'beta',split:'validation',args:[],folder:{'value.txt':'2'},expected:2}];
 const exchanges=[];
 const driver=async request=>{
  const turn={calls:[['eval',{code:"return Number(await folder.file('value.txt').readText());",finish:true}]]};
  exchanges.push({request,turn});return turn;
 };
 const contract={programId:'fixture-identities',entry:'solve.nl',exportName:'default'};
 const evaluator=new SourceEvaluator(contract,cases,driver,new UsageGateway({maxModelCalls:20,maxRollouts:4,maxProposals:0}),{executorId:'scripted',timeoutMs:10000});
 assert.equal((await evaluator.evaluate(source,{split:'train'})).quality,1);
 assert.equal((await evaluator.evaluate(source,{split:'validation'})).quality,1);
 assert.deepEqual(exchanges[0].request.messages,exchanges[1].request.messages);
 assert.notEqual(canonicalRequest(exchanges[0].request),canonicalRequest(exchanges[1].request));
 const replay=recordedDriver(exchanges);
 const repeated=new SourceEvaluator(contract,cases,replay,new UsageGateway({maxModelCalls:20,maxRollouts:4,maxProposals:0}),{executorId:'scripted',timeoutMs:10000});
 assert.equal((await repeated.evaluate(source,{split:'validation'})).quality,1);
 assert.equal((await repeated.evaluate(source,{split:'train'})).quality,1);
 assert.equal(replay.audit().unconsumedRequests,0);
});

test('file reducer diagnostics expose actual training effects without opening validation outputs',async()=>{
 const {SourceEvaluator}=await import('../dist/index.js');
 const {UsageGateway}=await import('../dist/evaluation/usage.js');
 const source=Folder.fromFiles({'solve.nl':'---\nkind: directory-reducer\nargs: {}\nreturns: string\n---\nReturn done.'}).snapshot();
 const files={'notes/a.txt':'No rush.'},expected={...files,'report.json':'{"urgent":[]}'};
 const cases=['train','validation'].map(split=>({id:split,group:split,split,args:[],folder:files,expected:'done',expectedFiles:expected}));
 const evaluator=new SourceEvaluator({programId:'file-effects',entry:'solve.nl',exportName:'default'},cases,async()=>({calls:[['return_result',{status:'success',value:'done'}]]}),new UsageGateway({maxModelCalls:20,maxRollouts:4,maxProposals:0}),{executorId:'scripted',timeoutMs:10000});
 const train=await evaluator.evaluate(source,{split:'train'});
 const evidence=evaluator.page(train.evidence)[0];
 assert.equal(evidence.passed,false);assert.deepEqual(evidence.files,files);
 assert.deepEqual(evidence.expectedFiles,expected);
 const runtime=await import('../dist/runtime/node.js');
 const {compileVirtualProject}=await import('../dist/runtime/virtual-project.js');
 const compiled=compileVirtualProject({files:{'main.ts':AUTHORED_IMPROVER['improveStep/context.ts'].replace("'../types'","'./types'"),'types.ts':AUTHORED_IMPROVER['types.ts']}},runtime,{constrained:true,target:'node'});
 assert.equal(compiled.ok,true);
 const brief=compiled.require('main.ts').brief({goal:'Write report.json.',mode:'structural',objective:'quality',allowedFiles:['solve.nl']},[evidence],[]);
 assert.match(brief,/File effects:.*report.json.*<missing>/);
 const validation=await evaluator.evaluate(source,{split:'validation'});
 assert.equal(validation.outcomes,undefined);assert.throws(()=>evaluator.page(validation.evidence),/Training evidence reference unavailable/);
});


test('student service openings contain public types without fixture implementation',async()=>{
 const {SourceEvaluator}=await import('../dist/index.js');
 const {UsageGateway}=await import('../dist/evaluation/usage.js');
 const source=Folder.fromFiles({'solve.nl':'---\nargs: {}\nreturns: number\n---\nReturn store.read().'}).snapshot();
 const requests=[];
 const driver=async request=>{requests.push(request);return {calls:[['eval',{code:'return await store.read();',finish:true}]]};};
 const cases=[{id:'train',group:'train',split:'train',args:[],expected:7,services:{store:'export function read():number{return 7;}'}}];
 const evaluator=new SourceEvaluator({programId:'public-fixture',entry:'solve.nl',exportName:'default'},cases,driver,new UsageGateway({maxModelCalls:4,maxRollouts:1,maxProposals:0}),{executorId:'scripted',timeoutMs:10000});
 assert.equal((await evaluator.evaluate(source,{split:'train'})).quality,1);
 const opening=JSON.stringify(requests[0].messages);
 assert.match(opening,/read\(\): number/);assert.doesNotMatch(opening,/return 7/);
});
