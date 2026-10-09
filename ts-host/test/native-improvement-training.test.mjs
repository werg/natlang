import test from 'node:test';
import assert from 'node:assert/strict';
import {Folder} from '../dist/index.js';
import {AUTHORED_IMPROVER} from '../dist/improvement/authored-source.js';
import {run} from '../dist/improvement/teacher.js';
import {observedActions,successfulTurn,sourceEditAdmission} from '../scripts/self-improvement/export-followup-training.mjs';
import {recordedDriver} from '../scripts/self-improvement/replay-followup-study.mjs';
import {migrateSystemPrompts,migrateServiceOpenings,migrateCaseContexts} from '../scripts/self-improvement/replay-runtime.mjs';

test('rejected source edits remain context rather than positive file-edit targets',()=>{
 const before={'solve.nl':'original'},after={'solve.nl':'candidate'};
 const parent=accepted=>({events:[{kind:'state',phase:'final',value:{$lambda:{return:{history:[{accepted,reason:'Measured against independent cases.'}]}}}}]});
 assert.equal(sourceEditAdmission(parent(false),before,after).approved,false);
 assert.equal(sourceEditAdmission(parent(true),before,after).approved,true);
 assert.equal(sourceEditAdmission(parent(false),before,before).approved,true);
 assert.throws(()=>sourceEditAdmission(undefined,before,after),/independently measured parent/);
});

test('fixture declaration migration retains case identity when public openings become identical',async()=>{
 const source={'solve.nl':'---\nargs: {}\nreturns: number\n---\nReturn store.read().'};
 const cases=[{id:'train',services:{store:'export function read():number{return 1;}'}},{id:'validation',services:{store:'export function read():number{return 2;}'}}];
 const exchanges=cases.map((row,i)=>({request:{invocation_id:'old-'+i+'/1',messages:[{role:'system',content:'guidance'},{role:'user',content:'solve()\n\nInstructions:\nReturn store.read().\n\nIn eval use store.'},{role:'assistant',tool_calls:[{id:'scope_0',function:{name:'eval',arguments:JSON.stringify({code:row.services.store})}}]}],tools:[],seed:1,max_tokens:10},turn:{text:String(i+1)}}));
 const before=structuredClone(exchanges);
 const identified=migrateCaseContexts(exchanges,{cases,sources:new Map([['source-id',source]]),entry:'solve.nl',fingerprint:row=>row.caseId,declarationNamespace:(_name,value)=>value});
 assert.equal(identified.migration.recoveredTasks,2);
 const migrated=migrateServiceOpenings(identified.records,new Map(cases.map(row=>[row.services.store,'export function read():number;'])));
 assert.equal(migrated.migration.changedOpenings,2);assert.deepEqual(exchanges,before);
 assert.deepEqual(migrated.records[0].request.messages,migrated.records[1].request.messages);
 const driver=recordedDriver(migrated.records);
 assert.deepEqual(await driver({...migrated.records[1].request,invocation_id:'case:validation/fresh/1'}),{text:'2'});
 assert.deepEqual(await driver({...migrated.records[0].request,invocation_id:'case:train/fresh/1'}),{text:'1'});
 assert.equal(driver.audit().unconsumedRequests,0);
});

test('offline prompt migration preserves provider replies and refuses changed task inputs',async()=>{
 const original=[{request:{invocation_id:'old/1',messages:[{role:'system',content:'old guidance\nold folder help'},{role:'user',content:'Return seven.'}],tools:[],seed:1,max_tokens:10},turn:{calls:[['eval',{code:'return 7;',finish:true}]],reasoning:'Recorded reasoning.'}}];
 const before=structuredClone(original);
 const {records,migration}=migrateSystemPrompts(original,'old guidance','new guidance','old folder help','new folder help');
 assert.deepEqual(original,before);assert.deepEqual(records[0].turn,before[0].turn);
 assert.equal(records[0].request.messages[0].content,'new guidance\nnew folder help');
 assert.equal(migration.requests,1);assert.equal(migration.providerRepliesRewritten,false);
 const driver=recordedDriver(records);
 await assert.rejects(driver({...records[0].request,messages:[records[0].request.messages[0],{role:'user',content:'Return eight.'}]}),/No recorded invocation matches/);
 assert.deepEqual(await driver(records[0].request),original[0].turn);
 assert.throws(()=>migrateSystemPrompts([{...original[0],request:{...original[0].request,messages:[{role:'system',content:'unknown guidance'}]}}],'old guidance','new guidance'),/Unrecognized recorded system prompt/);
});

test('directory guidance names real scoped file handles and computed writes finish with exact effects',async()=>{
 const {createNatlangRuntime}=await import('../dist/index.js');
 const {loadVirtualNatlang}=await import('../dist/runtime/virtual-project.js');
 const folder=Folder.fromFiles({'keep.txt':'Preserve this.'});
 const target=loadVirtualNatlang({'save.nl':'---\nkind: directory-reducer\nargs: {}\nreturns: string\n---\nWrite report.json with count two and return done.'},'save.nl');
 const runtime=createNatlangRuntime({model:{driver:async request=>{
  const prompt=String(request.messages[0].content);
  assert.doesNotMatch(prompt,/The fs helper provides/);
  assert.match(prompt,/folder\.file\("report\.json"\)\.writeText/);
  return {calls:[['eval',{code:'const report={count:2}; await folder.file("report.json").writeText(JSON.stringify(report)); return "done";',finish:true}]]};
 }}});
 assert.equal(await runtime.run(()=>folder.apply(target)),'done');
 assert.equal(await folder.file('report.json').readText(),'{"count":2}');
 assert.equal(await folder.file('keep.txt').readText(),'Preserve this.');
});

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
 const compiled=compileVirtualProject({files:{'main.ts':AUTHORED_IMPROVER['improveStep/context.ts'].replace("'../types'","'./types'"),'types.ts':AUTHORED_IMPROVER['types.ts'],'transformations.ts':AUTHORED_IMPROVER['improveStep/transformations.ts']}},runtime,{constrained:true,target:'node'});
 assert.equal(compiled.ok,true,JSON.stringify(compiled.diagnostics));
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


test('isolated editor supervision requires the exact measured candidate effects', async()=>{
 const {verifyMeasuredEditorEffects}=await import('../scripts/self-improvement/export-followup-training.mjs');
 const before={'solve.nl':'old'},after={'solve.nl':'new'};
 const parent={events:[{kind:'state',phase:'final',value:{$lambda:{return:{lastExperiment:{sourceFiles:[{path:'solve.nl',text:'new'}]}}}}}]};
 assert.doesNotThrow(()=>verifyMeasuredEditorEffects(parent,before,after));
 assert.throws(()=>verifyMeasuredEditorEffects(parent,before,{'solve.nl':'invented'}),/measured candidate/);
 assert.throws(()=>verifyMeasuredEditorEffects(undefined,before,after),/measured candidate/);
 assert.doesNotThrow(()=>verifyMeasuredEditorEffects(undefined,before,before));
});
