import test from 'node:test';
import assert from 'node:assert/strict';
import { Folder, SourceEvaluator, improveProgram } from '../dist/index.js';
import { UsageGateway } from '../dist/evaluation/index.js';
import { scriptedModel } from './support/natlang.mjs';
const contract = { entry: 'main.ts', exportName: 'solve', programId: 'repair-fixture', signature: 'solve(value: number): number' };
const cases = ['train', 'validation', 'test'].flatMap(split => [1, 2].map(value => ({id: `${split}-${value}`, group: `${split}-${value}`, split, args: [value], expected: value + 1})));
const baseline = () => Folder.fromFiles({ 'main.ts': 'export function solve(value: number): number { return value; }' });
const budget = { maxRollouts: 30, maxModelCalls: 20, maxProposals: 5 };

test('the native optimizer cannot resample by changing the pinned evaluation seed',async()=>{
 const seeds=[];
 const executeCase=Object.assign(async(_folder,row,seed)=>{seeds.push(seed);return {value:row.expected,modelCalls:0};},{identity:'pinned-seed',evaluationLevel:1});
 const model=scriptedModel(()=> 'let reason="";try{await evaluator.evaluate(folder.snapshot(),{split:"train",seed:10});}catch(error){reason=String(error);}const source=folder.snapshot();const report=await evaluator.evaluate(source,{split:"validation",seed:9});return {...state,iteration:1,done:true,quality:report.quality,population:[{source:source.digest,quality:report.quality,parent:""}],stopReason:reason};');
 const result=await improveProgram({folder:baseline(),contract,cases,seed:9,executeCase,policy:{maxExperiments:1,maxPopulation:2,strategy:'adaptive',mode:'structural',goal:'increment',allowedFiles:['main.ts']},improver:model.driver,executor:()=>{throw Error('unexpected inference');},executorId:'pinned',budget});
 assert.match(result.state.stopReason,/seed is pinned.*9/);assert.equal(result.validation.quality,1);assert.deepEqual(seeds,[9,9]);
});

test('invalid joint checkpoints roll back before durable publication and on resume',async()=>{
  const {mkdtempSync}=await import('node:fs');const {tmpdir}=await import('node:os');const {join}=await import('node:path');
  const {OperationJournal}=await import('../dist/index.js');
  for(const failure of ['identity','quality','iteration','compile']){
    const directory=mkdtempSync(join(tmpdir(),'natlang-invalid-checkpoint-'));
    const code='await folder.file("main.ts").writeText("export function solve(value: number): number {'+(failure==='compile'?'while(true){}':'return value+1;')+'}"); const current=folder.snapshot(); return {iteration:'+(failure==='iteration'?2:1)+',done:true,incumbent:'+(failure==='identity'?'state.incumbent':'current.digest')+',quality:'+(failure==='quality'||failure==='compile'?0:1)+',population:[],history:[],stopReason:"claimed success"};';
    const model=scriptedModel(()=>code);
    const options={folder:baseline(),contract,cases,policy:{maxExperiments:3,maxPopulation:2,strategy:'adaptive',mode:'structural',goal:'increment',allowedFiles:['main.ts']},improver:model.driver,executor:()=>{throw Error('no inference');},executorId:'exact',budget,directory};
    const result=await improveProgram(options);
    assert.equal(result.disposition,'incomplete-search');assert.equal(result.folder.digest,options.folder.snapshot().digest);assert.equal(result.state.iteration,0);
    assert.equal(new OperationJournal(directory).checkpoint(),undefined);
    const resumed=await improveProgram(options);assert.equal(resumed.state.iteration,0);assert.equal(resumed.folder.digest,result.folder.digest);
  }
});

test('evaluation executes source bytes in fresh workers and seals validation and confirmation evidence', async () => {
  const gateway = new UsageGateway(budget);
  const evaluator = new SourceEvaluator(contract, cases, () => {throw Error('no inference needed');}, gateway, {executorId: 'exact'});
  const source = baseline();
  assert.equal((await evaluator.evaluate(source.snapshot(), {split:'train'})).quality, 0);
  source.writeText('main.ts', 'export function solve(value: number): number { return value + 1; }');
  const snapshot = source.snapshot();
  assert.equal((await evaluator.evaluate(snapshot, {split:'train'})).quality, 1);
  const validation = await evaluator.evaluate(snapshot, {split:'validation'});
  assert.equal(validation.quality, 1); assert.equal(validation.outcomes, undefined);
  assert.throws(() => evaluator.page(validation.evidence), /unavailable/);
  assert.throws(() => evaluator.evaluate(snapshot, {split:'test'}), /locked/);
  assert.equal((await evaluator.confirm(snapshot, {source:snapshot.digest, experiment:'frozen-final'})).quality, 1);
  await assert.rejects(() => evaluator.confirm(snapshot, {source:snapshot.digest, experiment:'retry'}), /only once/);
  assert.equal(gateway.ledger.rollouts, 8);
  source.writeText('main.ts', 'export function solve(value: number): number { while(true) {} }');
  assert.equal((await evaluator.check(source.snapshot())).valid, false);
});

test('installed authored reducer performs an actual propose-check-evaluate-accept-select loop', async () => {
  const source = baseline(), traces = [];
  const model = scriptedModel(opening => opening.includes('test request.hypothesis') ?
    'await folder.file("main.ts").writeText("export function solve(value: number): number { return value + 1; }"); return await bookkeeping.finish(folder,"fix increment",["number contract"]);' :
    'const base = folder.snapshot(); const before = await evaluator.evaluate(base,{split:"train"}); const parent = base.branch(); const candidate = await parent.propose(rewriteProgram,{goal:policy.goal,mode:policy.mode,hypothesis:"increment",brief:"exact increment repair",sourceFiles:[],evidence:before.outcomes,allowedFiles:policy.allowedFiles}); const checked = await evaluator.check(candidate.folder); if (!checked.valid) throw Error(checked.diagnostics.join("\\n")); const measured = await evaluator.evaluate(candidate.folder,{split:"validation"}); await parent.accept(candidate); const selected = (parent.snapshot()).digest; await folder.select(folder.at(selected)); return {iteration:state.iteration+1,done:true,incumbent:selected,quality:measured.quality,population:[{source:selected,quality:measured.quality,parent:base.digest}],history:[{source:selected,parent:base.digest,accepted:true,selected:true,reason:"improved"}],stopReason:"completed"}');
  const result = await improveProgram({folder:source,contract,cases,policy:{maxExperiments:2,maxPopulation:3,strategy:'adaptive',mode:'structural',goal:'increment',allowedFiles:['main.ts']}, improver:async(request,signal)=>{const turn=await model.driver(request,signal);for(const [name,args]of turn.calls??[])if(name==='eval')args.finish=true;return turn;}, executor:()=>{throw Error('no model');},executorId:'exact',budget,trace:trace=>traces.push(trace)});
  assert.equal(result.state.quality, 1,result.error); assert.equal(result.state.incumbent, result.folder.digest);
  assert.match(await result.folder.readText('main.ts'), /value \+ 1/);
  assert.equal(await source.readText('main.ts'), 'export function solve(value: number): number { return value; }');
  assert.ok(traces.some(trace=>trace.name==='rewriteProgram'));
});

test('confirmation survives evaluator restart and cannot change the frozen experiment',async()=>{
  const {mkdtempSync}=await import('node:fs');const {tmpdir}=await import('node:os');const {join}=await import('node:path');
  const {OperationJournal}=await import('../dist/index.js');
  const journal=new OperationJournal(mkdtempSync(join(tmpdir(),'natlang-confirmation-')));
  const source=baseline().snapshot(),gateway=new UsageGateway(budget);
  const make=()=>new SourceEvaluator(contract,cases,()=>{throw Error('no inference');},gateway,{executorId:'exact',journal});
  const freeze={source:source.digest,experiment:'one-final-cohort'};
  const first=await make().confirm(source,freeze);assert.equal(gateway.ledger.rollouts,2);
  assert.deepEqual(await make().confirm(source,freeze),first);assert.equal(gateway.ledger.rollouts,2);
  await assert.rejects(()=>make().confirm(source,{...freeze,experiment:'significance-retry'}),/different candidate or experiment/);
  const changed=source.branch();changed.writeText('main.ts','export function solve(value: number): number { return value+1; }');
  await assert.rejects(()=>make().confirm(changed.snapshot(),{source:changed.snapshot().digest,experiment:freeze.experiment}),/different candidate or experiment/);
});

test('paired confirmation reports the predeclared effect without retrying significance',async()=>{
 const source=baseline().snapshot(),selected=source.branch();selected.writeText('main.ts','export function solve(value: number): number { return value+1; }');
 const evaluator=new SourceEvaluator(contract,cases,()=>{throw Error('no inference');},new UsageGateway(budget),{executorId:'exact'});
 const report=await evaluator.confirmPair(source,selected.snapshot(),'frozen-paired');
 assert.equal(report.effect,1);assert.equal(report.wins,2);assert.equal(report.losses,0);
 assert.equal(report.pValue,0.25);assert.equal(report.supported,false);
 await assert.rejects(()=>evaluator.confirmPair(source,selected.snapshot(),'retry'),/repeated/);
});

test('a signature in a comment cannot conceal an edited external function contract',async()=>{
 const source=Folder.fromFiles({'main.ts':'// solve(value: number): number\nexport function solve(value: string): number {return 0;}'});
 const evaluator=new SourceEvaluator(contract,cases,()=>{throw Error('no inference');},new UsageGateway(budget),{executorId:'exact'});
 const checked=await evaluator.check(source.snapshot());assert.equal(checked.valid,false);assert.ok(checked.diagnostics.includes('pinned external signature changed'));
});

test('selected frozen improver source is executed and pinned in resume identity',async()=>{
 const {AUTHORED_IMPROVER}=await import('../dist/improvement/authored-source.js');
 const {mkdtempSync}=await import('node:fs');const {tmpdir}=await import('node:os');const {join}=await import('node:path');
 const authored=Folder.fromFiles(AUTHORED_IMPROVER);authored.writeText('improveStep.nl',(await authored.readText('improveStep.nl'))+'\nSelected experimental improver revision.\n');
 const improverSource=authored.snapshot();
 const model=scriptedModel(opening=>{assert.ok(opening.includes('Selected experimental improver revision'));return 'const snapshot=folder.snapshot(); const quality=await evaluator.evaluate(snapshot,{split:"validation"}); const identity=(snapshot).digest; return {iteration:1,done:true,incumbent:identity,quality:quality.quality,population:[],history:[],stopReason:"no justified change"};';});
 const options={folder:baseline(),contract,cases,policy:{maxExperiments:1,maxPopulation:2,strategy:'adaptive',mode:'structural',goal:'increment',allowedFiles:['main.ts']},improver:model.driver,executor:()=>{throw Error('no inference');},executorId:'exact',budget,directory:mkdtempSync(join(tmpdir(),'natlang-meta-source-')),improverSource};
 const result=await improveProgram(options);assert.match(result.authored.files['improveStep.nl'],/Selected experimental/);
 const next=improverSource.branch();next.writeText('improveStep.nl',(await next.readText('improveStep.nl'))+'Different revision.');
 await assert.rejects(()=>improveProgram({...options,improverSource:next.snapshot()}),/same frozen source/);
});

test('portfolio pins complete member sources and preserves its declared result contract',async()=>{
 const {composePortfolio}=await import('../dist/index.js');
 const source=Folder.fromFiles({'main.ts':'export function solve(value: number): number {return value+1;}'}).snapshot();
 const portfolio=composePortfolio([{name:'increment',source,contract}],{parameters:'value: number',arguments:['value'],returns:'number'});
 const {members}=JSON.parse(await portfolio.readText('portfolio.json'));assert.equal(members[0].source,source.digest);
 const model=scriptedModel(()=> 'return choices[0]');
 const evaluator=new SourceEvaluator({...contract,signature:'solve(value: number): Promise<number>'},cases,model.driver,new UsageGateway({...budget,maxModelCalls:30}),{executorId:'router-exact'});
 assert.equal((await evaluator.check(portfolio)).valid,true);
 assert.equal((await evaluator.evaluate(portfolio,{split:'train'})).quality,1);
 assert.ok(model.openings.every(opening=>!opening.includes('train-') && !opening.includes('validation-')));
});

test('targets cannot recover evaluation authority through imports',async()=>{
 const evaluator=new SourceEvaluator(contract,cases,()=>{throw Error('no inference');},new UsageGateway(budget),{executorId:'exact'});
 const source=Folder.fromFiles({'main.ts':'import {SourceEvaluator} from "@natlang/node"; export function solve(value: number): number {return value;}'});
 assert.ok((await evaluator.check(source.snapshot())).diagnostics.some(message=>message.includes('evaluation authority')));
 source.writeText('main.ts','import vm from "node:vm"; export function solve(value: number): number {return value;}');
 assert.ok((await evaluator.check(source.snapshot())).diagnostics.some(message=>message.includes('outside the declared checked core')));
});

test('numeric loop rounding cannot create an unbounded checked-core computation',async()=>{
 const source=Folder.fromFiles({'main.ts':'export function solve(value: number): number {let result=0; for(let n=9007199254740992;n<9007199254740994;n++){result++;} return result;}'});
 const evaluator=new SourceEvaluator(contract,cases,()=>{throw Error('no inference');},new UsageGateway(budget),{executorId:'exact'});
 assert.equal((await evaluator.check(source.snapshot())).valid,true);
 const report=await evaluator.evaluate(source.snapshot(),{split:'train'});
 assert.equal(report.quality,0);assert.ok(report.outcomes.every(row=>row.error.includes('numeric loop counter did not advance')));
});

test('meta-evaluation executes the edited frozen improver and shares its parent allocation',async()=>{
 const {AUTHORED_IMPROVER}=await import('../dist/improvement/authored-source.js');
 const {improverExecution}=await import('../dist/index.js');
 const original=Folder.fromFiles(AUTHORED_IMPROVER);
 const bad=original.snapshot().branch();bad.writeText('improveStep.nl',(await bad.readText('improveStep.nl')).replace(/Improve this source[\s\S]*/,'Frozen baseline-only improver: retain the input source and report its validation.'));
 assert.match(await bad.readText('improveStep.nl'),/Frozen baseline-only/);
 const driver=scriptedModel(opening=>opening.includes('Frozen baseline-only')?
  'const snapshot=folder.snapshot(); const measured=await evaluator.evaluate(snapshot,{split:"validation"}); const source=(snapshot).digest; return {iteration:1,done:true,incumbent:source,quality:measured.quality,population:[{source,quality:measured.quality,parent:""}],history:[],stopReason:"baseline-only"};':
  opening.includes('test request.hypothesis')?
  'await folder.file("main.ts").writeText("export function solve(value: number): number {return value+1;}"); return {summary:"increment",changed:["main.ts"],preserves:["numeric contract"]};':
  'const base=folder.snapshot(); const parent=base.branch(); const proposal=await parent.propose(rewriteProgram,{goal:policy.goal,mode:policy.mode,hypothesis:"increment",brief:"exact increment repair",sourceFiles:[],evidence:[],allowedFiles:policy.allowedFiles}); const measured=await evaluator.evaluate(proposal.folder,{split:"validation"}); await parent.accept(proposal); const source=(parent.snapshot()).digest; await folder.select(folder.at(source)); return {iteration:1,done:true,incumbent:source,quality:measured.quality,population:[{source,quality:measured.quality,parent:base.digest}],history:[{source,parent:base.digest,accepted:true,selected:true,reason:"measured improvement"}],stopReason:"completed"};');
 const metaCases=[{id:'meta-train',group:'meta-train',split:'train',args:[],expected:null}];
 const executeCase=improverExecution([{id:'meta-train',files:{'main.ts':'export function solve(value: number): number {return value;}'},contract,cases,
  policy:{maxExperiments:1,maxPopulation:2,strategy:'adaptive',mode:'structural',goal:'increment',allowedFiles:['main.ts']}}],{improver:driver.driver,executor:()=>{throw Error('no inference');},improverId:'scripted-editor',executorId:'exact'});
 const gateway=new UsageGateway({maxRollouts:100,maxModelCalls:200,maxProposals:12});
 assert.throws(()=>new SourceEvaluator({entry:'improveStep.nl',exportName:'default',programId:'meta-improver'},metaCases,driver.driver,gateway,{executorId:'meta',executeCase}),/evaluation level/);
 const evaluator=new SourceEvaluator({entry:'improveStep.nl',exportName:'default',programId:'meta-improver'},metaCases,driver.driver,gateway,{executorId:'meta',executeCase,evaluationLevel:2});
 assert.equal((await evaluator.evaluate(bad.snapshot(),{split:'train'})).quality,0);
 assert.equal((await evaluator.evaluate(original.snapshot(),{split:'train'})).quality,1);
 assert.ok(gateway.ledger.rollouts>2);assert.ok(gateway.ledger.usage.modelCalls>0);assert.equal(gateway.ledger.proposals,1);
});


test('checked loops keep a finite domain when bounds or collections change through aliases', async () => {
 for (const body of [
   'const limits={n:2}; const alias=limits; for(let i=0;i<limits.n;i++){alias.n++;}',
   'const keys=new Set([0]); const alias=keys; for(const key of keys){alias.delete(key); alias.add(key+1);}',
   'const keys=new Map([[0,0]]); const alias=keys; for(const [key] of keys){alias.delete(key); alias.set(key+1,0);}'
 ]) {
   const source=Folder.fromFiles({'main.ts':`export function solve(value: number): number {${body} return value+1;}`});
   const evaluator=new SourceEvaluator(contract,cases,()=>{throw Error('no inference');},new UsageGateway(budget),{executorId:'exact'});
   assert.equal((await evaluator.check(source.snapshot())).valid,true);
   const report=await evaluator.evaluate(source.snapshot(),{split:'train'});
   assert.equal(report.quality,1,JSON.stringify(report.outcomes));
 }
});

test('measured target request counts survive repeated immutable evaluations',async()=>{
 const folder=Folder.fromFiles({'solve.nl':'---\nargs:\n  value: number\nreturns: number\n---\nReturn value incremented by one.\n'}).snapshot();
 const model=scriptedModel(()=> 'return value+1;');
 const gateway=new UsageGateway({...budget,maxRollouts:10});
 const evaluator=new SourceEvaluator({entry:'solve.nl',exportName:'default',programId:'measured-target'},cases,model.driver,gateway,{executorId:'scripted'});
 const first=await evaluator.evaluate(folder,{split:'validation'});assert.equal(first.modelCalls,4);assert.equal(first.quality,1);
 const second=await evaluator.evaluate(folder,{split:'validation'});assert.deepEqual(second,first);assert.equal(gateway.ledger.usage.modelCalls,4);
 const independent=await evaluator.evaluate(folder,{split:'validation',seed:1});assert.equal(independent.modelCalls,4);assert.equal(gateway.ledger.usage.modelCalls,8);
});

test('independent fixture errors expose diagnostics without claiming target model failure',async()=>{
 const folder=Folder.fromFiles({'solve.nl':'---\nargs: {}\nreturns: number\n---\nReturn 1.\n'}).snapshot();
 const row={id:'bad-fixture',group:'fixture',split:'train',args:[],expected:1,services:{data:'const ROWS=[{status:"late"}]; export function page():{status:"late"}[]{return ROWS;}'}};
 const evaluator=new SourceEvaluator({entry:'solve.nl',exportName:'default',programId:'fixture-error'},[row],()=>{throw Error('target must not run');},new UsageGateway(budget),{executorId:'none'});
 const report=await evaluator.evaluate(folder,{split:'train'});assert.equal(report.modelCalls,0);assert.equal(report.outcomes[0].failureKind,'fixture');assert.match(report.outcomes[0].error,/not assignable/);
});

test('bounded target timeouts are diagnosable cached outcomes, not lost evidence',async()=>{
 const folder=Folder.fromFiles({'solve.nl':'---\nargs: {}\nreturns: number\n---\nReturn 1.\n'}).snapshot();
 const row={id:'timeout',group:'timeout',split:'train',args:[],expected:1};
 const gateway=new UsageGateway(budget);
 const evaluator=new SourceEvaluator({entry:'solve.nl',exportName:'default',programId:'timeout-evidence'},[row],()=>new Promise(()=>{}),gateway,{executorId:'never-finishes',timeoutMs:50});
 const first=await evaluator.evaluate(folder,{split:'train'});assert.equal(first.gatesPassed,false);assert.equal(first.outcomes[0].failureKind,'timeout');assert.match(first.outcomes[0].error,/timed out/);assert.deepEqual(evaluator.page(first.evidence),first.outcomes);
 const rollouts=gateway.ledger.rollouts;assert.deepEqual(await evaluator.evaluate(folder,{split:'train'}),first);assert.equal(gateway.ledger.rollouts,rollouts);
});

test('one response can compute exactly and finish; action traces stay training-only',async()=>{
 const folder=Folder.fromFiles({'solve.nl':'---\nargs:\n  value: number\nreturns: number\n---\nReturn value incremented by one.\n'}).snapshot();
 const driver=async()=>({calls:[['eval',{code:'return value+1;'}],['return_result',{status:'success'}]]});
 const evaluator=new SourceEvaluator({entry:'solve.nl',exportName:'default',programId:'one-response'},cases,driver,new UsageGateway(budget),{executorId:'one-response'});
 const training=await evaluator.evaluate(folder,{split:'train'});assert.equal(training.quality,1);assert.equal(training.modelCalls,2);
 assert.equal(training.outcomes[0].modelTrace[0].calls[0].name,'eval');assert.equal(training.outcomes[0].modelTrace[0].calls[1].name,'return_result');
 const validation=await evaluator.evaluate(folder,{split:'validation'});assert.equal(validation.quality,1);assert.equal(validation.modelCalls,2);assert.equal(validation.outcomes,undefined);assert.throws(()=>evaluator.page(validation.evidence),/unavailable/);
});

test('training feedback contains actual public service types without fixture implementation',async()=>{
 const folder=Folder.fromFiles({'solve.nl':'---\nargs: {}\nreturns: number\n---\nCount late shipments in shipments.page(1).\n'}).snapshot();
 const services={shipments:'type Shipment={status:"late"|"on_time"|"cancelled"}; const PRIVATE_FIXTURE_MARKER=42; export function page(n:number):Shipment[]{return [{status:"late"}];}'};
 const rows=['train','validation'].map(split=>({id:split,group:split,split,args:[],expected:1,services}));
 const driver=async()=>({calls:[['eval',{code:'return shipments.page(1).filter(row=>row.status==="late").length;'}],['return_result',{status:'success'}]]});
 const evaluator=new SourceEvaluator({entry:'solve.nl',exportName:'default',programId:'service-feedback'},rows,driver,new UsageGateway(budget),{executorId:'service-feedback'});
 const training=await evaluator.evaluate(folder,{split:'train'});assert.equal(training.quality,1);
 const declaration=evaluator.page(training.evidence)[0].serviceDeclarations.shipments;
 assert.match(declaration,/status.*"late"/s);assert.match(declaration,/page\(n: number\)/);assert.doesNotMatch(declaration,/PRIVATE_FIXTURE_MARKER|return \[/);
 const validation=await evaluator.evaluate(folder,{split:'validation'});assert.equal(validation.quality,1);assert.equal(validation.outcomes,undefined);assert.throws(()=>evaluator.page(validation.evidence),/unavailable/);
 const {EVALUATOR_DECLARATION}=await import('../dist/improvement/services.js');assert.match(EVALUATOR_DECLARATION,/modelCalls\?:number/);assert.match(EVALUATOR_DECLARATION,/serviceDeclarations\?/);
});

test('a failed computation prevents finishing an older staged result in the same response',async()=>{
 const folder=Folder.fromFiles({'solve.nl':'---\nargs:\n  value: number\nreturns: number\n---\nReturn value incremented by one.\n'}).snapshot();
 const turns=new Map();
 const driver=async request=>{
  const id=request.invocation_id,turn=(turns.get(id)??0)+1;turns.set(id,turn);
  if(turn===1)return {calls:[['eval',{code:'return value;'}]]};
  if(turn===2)return {calls:[['eval',{code:'throw new Error("computation failed");'}],['return_result',{status:'success'}]]};
  return {calls:[['eval',{code:'return value+1;'}],['return_result',{status:'success'}]]};
 };
 const evaluator=new SourceEvaluator({entry:'solve.nl',exportName:'default',programId:'batch-repair'},cases,driver,new UsageGateway(budget),{executorId:'batch-repair'});
 const training=await evaluator.evaluate(folder,{split:'train'});
 assert.equal(training.quality,1);assert.equal(training.modelCalls,6);
});

 test('simplified native lifecycle invokes diagnosis, editing and selection with real capabilities',async()=>{
 const model=scriptedModel(opening=>{
  if(opening.includes('test request.hypothesis'))return 'await folder.file("main.ts").writeText("export function solve(value: number): number {return value+1;}");return await bookkeeping.finish(folder,"increment",["number contract"]);';
  if(opening.includes('Plan one evidenced source change'))return 'if(context.evidence.length===0 || context.sourceFiles[0].path!=="main.ts")throw Error("missing prepared context");return "Return the increment rather than the input.";';
  return 'return await lifecycle.step(folder,evaluator,planExperiment,rewriteProgram,state,policy);';
 });
 const result=await improveProgram({folder:baseline(),contract,cases,policy:{maxExperiments:3,maxPopulation:3,strategy:'adaptive',mode:'structural',goal:'increment',allowedFiles:['main.ts']},improver:async(request,signal)=>{if(String(request.messages[1].content).includes("Plan one evidenced source change")){const opening=request.messages.map(message=>String(message.content)).join("\n");assert.match(opening,/brief: string/);assert.match(opening,/main\.ts/);return {text:"Return the increment rather than the input."};}const turn=await model.driver(request,signal);for(const [name,args]of turn.calls??[])if(name==='eval')args.finish=true;return turn;},executor:()=>{throw Error('no inference');},executorId:'exact',budget:{...budget,maxModelCalls:60}});
 assert.equal(result.disposition,'improved',result.error);assert.equal(result.state.iteration,1);assert.equal(result.state.history[0].accepted,true);assert.equal(result.state.history[0].selected,true);assert.equal(result.validation.quality,1);assert.equal(result.ledger.roles.reflection.modelCalls,3);
 });

test('declared allocation permits more than 24 requests across finite native experiments',async()=>{
 const {modelTurnsSoFar}=await import('../dist/native/agent.js');
 const driver=async({messages})=>{
  const turn=modelTurnsSoFar(messages);
  return {calls:[['eval',{code:turn<9?'state.iteration':'const report=await evaluator.evaluate(folder.snapshot(),{split:"validation"});return {...state,iteration:state.iteration+1,done:state.iteration===2,quality:report.quality,population:[],stopReason:"finite experiment allowance completed"};',finish:turn>=9}]],completion_tokens:1,prompt_tokens:1};
 };
 const result=await improveProgram({folder:baseline(),contract,cases,policy:{maxExperiments:3,maxPopulation:3,strategy:'adaptive',mode:'structural',goal:'increment',allowedFiles:['main.ts']},improver:driver,executor:()=>{throw Error('no inference');},executorId:'exact',budget:{...budget,maxModelCalls:40}});
 assert.equal(result.state.iteration,3,result.error);assert.equal(result.ledger.roles.reflection.modelCalls,30);assert.equal(result.state.done,true);
});

test('the next semantic plan sees the actual rejected source and train trace without validation answers',async()=>{
 let plans=0;
 const model=scriptedModel(opening=>{
  if(opening.includes('Plan one evidenced source change')){plans++;return plans===1?'return "Try adding two.";':'if(!context.lastExperiment.sourceFiles[0].text.includes("value+2")||context.lastExperiment.training[0].passed!==false)throw Error("rejected evidence lost");if("expected" in context.lastExperiment.validation)throw Error("validation answer leaked");return "The trial overshot; add one instead.";';}
  if(opening.includes('test request.hypothesis'))return 'const delta=request.hypothesis.includes("two")?2:1;await folder.file("main.ts").writeText("export function solve(value:number):number{return value+"+delta+";}");return await bookkeeping.finish(folder,"test delta",["number contract"]);';
  return 'return await lifecycle.step(folder,evaluator,planExperiment,rewriteProgram,state,policy);';
 });
 const result=await improveProgram({folder:baseline(),contract,cases,policy:{maxExperiments:3,maxPopulation:3,strategy:'adaptive',mode:'structural',goal:'increment',allowedFiles:['main.ts']},improver:async(request,signal)=>{const turn=await model.driver(request,signal);for(const [name,args]of turn.calls??[])if(name==='eval')args.finish=true;return turn;},executor:()=>{throw Error('no inference');},executorId:'exact',budget:{...budget,maxModelCalls:60}});
 assert.equal(result.disposition,'improved',result.error);assert.equal(plans,2);assert.deepEqual(result.state.history.map(row=>row.accepted),[false,true]);assert.equal(result.ledger.roles.reflection.modelCalls,6);assert.equal(result.state.lastExperiment.validation.quality,1);
});

test('cost search continues after its first gain and lets the semantic planner stop when no opportunity remains',async()=>{
 let plans=0;
 const model=scriptedModel(opening=>{
  if(opening.includes('Plan one evidenced source change')){plans++;return 'return context.opportunity.kind==="none"?"":context.evidence[0].modelCalls===5?"cost3":"cost1";';}
  if(opening.includes('test request.hypothesis'))return 'await folder.file("main.ts").writeText("export function solve(value:number):number{return value+1;} //"+request.hypothesis);return await bookkeeping.finish(folder,"lower measured cost",["number contract"]);';
  return 'return await lifecycle.step(folder,evaluator,planExperiment,rewriteProgram,state,policy);';
 });
 const executeCase=Object.assign(async(folder,row)=>{const text=await folder.readText('main.ts');return {value:row.expected,modelCalls:text.includes('cost1')?1:text.includes('cost3')?3:5};},{identity:'deterministic-cost-fixture',evaluationLevel:1});
 const result=await improveProgram({folder:Folder.fromFiles({'main.ts':'export function solve(value:number):number{return value+1;}'}),contract,cases:cases.filter(row=>row.id.endsWith('1')),policy:{maxExperiments:4,maxPopulation:4,strategy:'adaptive',mode:'structural',objective:'model-calls',goal:'reduce requests',allowedFiles:['main.ts']},executeCase,improver:async(request,signal)=>{const turn=await model.driver(request,signal);for(const [name,args]of turn.calls??[])if(name==='eval')args.finish=true;return turn;},executor:()=>{throw Error('no inference');},executorId:'exact-cost',budget:{...budget,maxModelCalls:60}});
 assert.equal(result.disposition,'transformed',result.error);assert.equal(plans,3);assert.deepEqual(result.state.history.map(row=>row.accepted),[true,true,false]);assert.equal(result.validation.modelCalls,1);assert.equal(result.state.stopReason,'No further evidenced change.');
});

test('training traces pair each action with its own outcome, including atomic terminal actions',async()=>{
 const folder=Folder.fromFiles({'solve.nl':'---\nargs: {value: number}\nreturns: number\n---\nReturn value incremented by one.'}).snapshot(),turns=new Map();
 const driver=async request=>{const turn=(turns.get(request.invocation_id)??0)+1;turns.set(request.invocation_id,turn);return {calls:[['eval',{code:turn===1?'throw Error("OBSERVED_ACTION_FAILURE");':'return value+1;',finish:turn!==1}]]};};
 const evaluator=new SourceEvaluator({entry:'solve.nl',exportName:'default',programId:'causal-trace'},[cases[0]],driver,new UsageGateway(budget),{executorId:'causal-trace'});
 const report=await evaluator.evaluate(folder,{split:'train'});assert.equal(report.quality,1);
 const trace=report.outcomes[0].modelTrace;assert.equal(trace.length,2);assert.match(trace[0].observation,/OBSERVED_ACTION_FAILURE/);assert.doesNotMatch(trace[1].observation,/OBSERVED_ACTION_FAILURE/);assert.match(trace[1].observation,/eval.*completed.*2/s);assert.equal(trace[1].calls[0].arguments.finish,true);
});
