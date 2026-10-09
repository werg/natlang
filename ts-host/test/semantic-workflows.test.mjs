import test from 'node:test';
import assert from 'node:assert/strict';
import {Folder,SourceEvaluator,checkTransformation,admitCounterexamples,CounterexampleSuite,counterexampleGuidedImprove,implementBehavior,composePortfolio} from '../dist/index.js';
import {scriptedModel} from './support/natlang.mjs';
import {stagedImprover} from './support/improver.mjs';
import {UsageGateway} from '../dist/evaluation/usage.js';
const contract={entry:'main.ts',exportName:'solve',programId:'semantic-workflows'};
const cases=[{id:'train',group:'train',split:'train',args:[1],expected:2},{id:'validation',group:'validation',split:'validation',args:[2],expected:3},{id:'test',group:'test',split:'test',args:[3],expected:4}];
const source=()=>Folder.fromFiles({'main.ts':'export function solve(value:number):number{return value+1;}'}).snapshot();
test('an unchanged source cannot support a requested transformation',async()=>{
 const before=source(),evaluator=new SourceEvaluator(contract,cases,()=>{throw Error('no inference');},new UsageGateway({maxModelCalls:0,maxProposals:0,maxRollouts:10}),{executorId:'exact'});
 const report=await checkTransformation(evaluator,before,before,{id:'no-change',intent:'extract helper',preserves:['behavior'],changes:['extract helper'],checks:['compile']});
 assert.equal(report.eligible,false);assert.equal(report.obligations.find(row=>row.obligation==='extract helper').status,'violated');
});
test('transformation obligations preserve scope and do not certify unknown behavioral checks',async()=>{
 const before=source(),branch=before.branch();branch.writeText('main.ts','export function solve(value:number):number{const result=value+1;return result;}');
 const evaluator=new SourceEvaluator(contract,cases,()=>{throw Error('unexpected inference');},new UsageGateway({maxModelCalls:0,maxProposals:0,maxRollouts:10}),{executorId:'exact'});
 const spec={id:'extract',intent:'extract local',preserves:['increment behavior'],changes:[],checks:['compile','validation-regressions','universal equivalence'],allowedFiles:['main.ts']};
 const report=await checkTransformation(evaluator,before,branch.snapshot(),spec);
 assert.equal(report.status,'unverified');assert.equal(report.eligible,false);
 assert.equal(report.obligations.find(row=>row.obligation==='increment behavior').status,'empirically-supported');
 const outside=before.branch();outside.writeText('extra.ts','export const bad=1;');
 const rejected=await checkTransformation(evaluator,before,outside.snapshot(),{...spec,checks:['compile']});assert.equal(rejected.status,'violated');
});
test('counterexample admission versions only train data and rejects contradictory or protected gold',async()=>{
 let calls=0;const oracle={identity:'independent-increment-v1',expected:async args=>{calls++;return args[0]+1;}};
 const admitted=await admitCounterexamples(cases,[[3],[7],[7],[1]],oracle,4);
 assert.equal(admitted.accepted.length,1);assert.equal(calls,3);assert.equal(admitted.accepted[0].expected,8);
 assert.deepEqual(admitted.cases.filter(row=>row.split!=='train'),cases.filter(row=>row.split!=='train'));
 assert.ok(admitted.rejected.some(row=>row.reason==='protected input overlap'));
 const contradicted=await admitCounterexamples(cases,[[1]],{identity:'bad-oracle',expected:async()=>9},1);
 assert.match(contradicted.rejected[0].reason,/contradicts/);
});
test('implementation requires independently supplied expected behavior',async()=>{
 const result=await implementBehavior({cases:[]});assert.equal(result.disposition,'needs-information');
});
test('counterexample suite versions consume a shared finite oracle allowance without changing protected cases',async()=>{
 const initial=new CounterexampleSuite(cases,{identity:'increment',expected:async args=>args[0]+1},1);
 const first=await initial.admit([[8],[9]]);
 assert.equal(first.suite.remainingChecks,0);assert.equal(first.admission.accepted.length,1);
 assert.notEqual(first.suite.version,initial.version);assert.equal(initial.cases.length,3);
 const second=await first.suite.admit([[10]]);assert.equal(second.admission.accepted.length,0);
 assert.deepEqual(second.suite.cases.filter(row=>row.split!=='train'),cases.filter(row=>row.split!=='train'));
});
test('authored counterexample loop admits independent gold and executes the normal authored repair',async()=>{
 const model=scriptedModel(stagedImprover({other:opening=>opening.includes('Perform one counterexample-guided')?
  'const observed=await counterexamples.evidence(folder.snapshot());const suggestion=await suggestCounterexamples(goal,observed);const admitted=await counterexamples.admit(suggestion.inputs);const repaired=await counterexamples.repair(folder.snapshot());await folder.select(repaired.folder);return {round:state.round+1,done:repaired.eligible&&repaired.quality===1,suite:admitted.suite,remainingChecks:admitted.remainingChecks,quality:repaired.quality,reason:repaired.disposition};':
  opening.includes('Suggest concrete deployment inputs')?'return {inputs:[[7]],reason:"probe a larger value"}':null,
  edit:'await folder.file("main.ts").writeText("export function solve(value:number):number{return value+1;}");return {summary:"repair",preserves:["number signature"]};'}));
 const folder=Folder.fromFiles({'main.ts':'export function solve(value:number):number{return value;}'});
 const result=await counterexampleGuidedImprove({folder,contract,cases,policy:{maxExperiments:1,maxPopulation:3,mode:'structural',strategy:'adaptive',goal:'increment',allowedFiles:['main.ts']},improver:model.driver,executor:()=>{throw Error('no inference');},executorId:'exact',budget:{maxModelCalls:20,maxRollouts:24,maxProposals:3},oracle:{identity:'independent-increment',expected:async args=>args[0]+1},maxRounds:2,maxChecks:2});
 assert.equal(result.state.quality,1);assert.equal(result.state.round,1);assert.equal(result.suite.cases.length,4);
 assert.equal(result.suite.remainingChecks,1);assert.match(await result.folder.readText('main.ts'),/value\+1/);
 assert.match(await folder.readText('main.ts'),/return value;/);
});
test('natlang external contracts resolve aliases and reject structural edits to their types',async()=>{
 const files={'main.nl':'---\nargs: {value: Input}\nreturns: number\n---\nReturn the count.','types.ts':'export type Input={count:number};'};
 const evaluator=new SourceEvaluator({...contract,entry:'main.nl',exportName:'default'},cases,()=>{throw Error('no inference');},new UsageGateway({maxModelCalls:0,maxRollouts:0,maxProposals:0}),{executorId:'exact',sourcePolicy:{baseline:files,mode:'structural',allowedFiles:['main.nl','types.ts']}});
 assert.equal((await evaluator.check(Folder.fromFiles({...files,'main.nl':files['main.nl'].replace('Return the count.','Return value.count.')}).snapshot())).valid,true);
 const changed=await evaluator.check(Folder.fromFiles({...files,'types.ts':'export type Input={count:string};'}).snapshot());
 assert.equal(changed.valid,false);assert.match(changed.diagnostics.join('\n'),/natlang function contract changed/);
});
test('portfolio includes a checked named fallback and deployment-only router inputs',async()=>{
 const portfolio=composePortfolio([{name:'incumbent',source:source(),contract}],{parameters:'value:number',arguments:['value'],returns:'number'},'incumbent');
 assert.match(await portfolio.readText('main.ts'),/return incumbent\(value\);/);
 assert.doesNotMatch(await portfolio.readText('main.ts'),/throw new Error/);
 assert.equal(JSON.parse(await portfolio.readText('portfolio.json')).fallback,'incumbent');
});

test('partial quality remains measurable while independently required regressions are gates',async()=>{
 const partial=Folder.fromFiles({'main.ts':'export function solve(value:number):number{return value===1?2:0;}'}).snapshot();
 const evaluator=new SourceEvaluator(contract,[...cases,{id:'required',group:'required',split:'validation',args:[1],expected:2,required:true}],()=>{throw Error('no inference');},new UsageGateway({maxModelCalls:0,maxProposals:0,maxRollouts:8}),{executorId:'exact'});
 const report=await evaluator.evaluate(partial,{split:'validation'});assert.equal(report.quality,0.5);assert.equal(report.gatesPassed,true);
 const broken=Folder.fromFiles({'main.ts':'export function solve(value:number):number{return 0;}'}).snapshot();
 assert.equal((await evaluator.evaluate(broken,{split:'validation'})).gatesPassed,false);
});
test('directory source grading checks edited files and supports finite exact TypeScript reducers',async()=>{
 const folderCases=[{id:'files',group:'files',split:'train',args:[],folder:{'input.txt':'hello'},expected:'done',expectedFiles:{'input.txt':'hello','output.txt':'HELLO'}}];
 const code='import type {Folder} from "@natlang/node"; export async function solve(folder:Folder):Promise<string>{const text=await folder.file("input.txt").readText();await folder.file("output.txt").writeText(text.toUpperCase());return "done";}';
 const evaluator=new SourceEvaluator(contract,folderCases,()=>{throw Error('no inference');},new UsageGateway({maxModelCalls:0,maxProposals:0,maxRollouts:4}),{executorId:'exact'});
 assert.equal((await evaluator.evaluate(Folder.fromFiles({'main.ts':code}).snapshot(),{split:'train'})).quality,1);
 assert.equal((await evaluator.evaluate(Folder.fromFiles({'main.ts':code.replace('text.toUpperCase()','text')}).snapshot(),{split:'train'})).quality,0);
});

test('implementation workflow executes its specialized editor with prepared source and evidence',async()=>{
 let specialized=false;
 const model=scriptedModel(stagedImprover({edit:()=>{specialized=true;return 'if(!request.transformation.includes("Implement the requested application in native natlang"))throw Error("missing transformation");if(request.sourceFiles[0].path!=="main.ts")throw Error("missing source");await folder.file("main.ts").writeText("export function solve(value:number):number{return value+1;}");return {summary:"implement increment",preserves:["number contract"]};';}}));
 const result=await implementBehavior({folder:Folder.fromFiles({'main.ts':'export function solve(value:number):number{return value;}'}),contract,cases,policy:{maxExperiments:2,maxPopulation:3,mode:'structural',strategy:'adaptive',goal:'increment',allowedFiles:['main.ts']},improver:async(request,signal)=>{const turn=await model.driver(request,signal);for(const [name,args]of turn.calls??[])if(name==='eval')args.finish=true;return turn;},executor:()=>{throw Error('no inference');},executorId:'exact',budget:{maxModelCalls:20,maxRollouts:20,maxProposals:3}});
 assert.equal(result.disposition,'improved',result.error);assert.equal(specialized,true);assert.equal(result.validation.quality,1);assert.equal(result.ledger.roles.reflection.modelCalls,4);
});
