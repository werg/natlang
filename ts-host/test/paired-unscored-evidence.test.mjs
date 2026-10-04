import assert from 'node:assert/strict';
import test from 'node:test';
import {Folder} from '../dist/native/scoped-fs.js';
import {SourceEvaluator} from '../dist/improvement/host.js';
import {UsageGateway} from '../dist/evaluation/usage.js';

const before=Folder.fromFiles({'solve.ts':'export function solve():number{return 0;}'}).snapshot();
const after=Folder.fromFiles({'solve.ts':'export function solve():number{return 1;}'}).snapshot();
function evaluator(failureKind){
 const executeCase=Object.assign(async folder => folder.digest===before.digest
   ? {error:'diagnostic from executor',failureKind} : {value:1},{identity:'test-unscored-executor'});
 return new SourceEvaluator({entry:'solve.ts',exportName:'solve',programId:'paired-failure'},
  [{id:'sealed',group:'sealed',split:'test',args:[],expected:1}],()=>{throw Error('no inference expected');},
  new UsageGateway({maxModelCalls:1,maxRollouts:2,maxProposals:0}),{executorId:'fixture',executeCase});
}
test('fixture and timeout records cannot create paired quality gains or sign-test wins',async()=>{
 for(const failureKind of ['fixture','timeout']){
  await assert.rejects(evaluator(failureKind).confirmQuality(before,after,'quality'),/unscored paired baseline/);
  await assert.rejects(evaluator(failureKind).confirmPair(before,after,'sign'),/unscored paired confirmation/);
 }
});
test('independently observed target errors remain repairable measured failures',async()=>{
 const report=await evaluator('target').confirmQuality(before,after,'quality');
 assert.equal(report.effect,1);assert.equal(report.selected.gatesPassed,true);
});
