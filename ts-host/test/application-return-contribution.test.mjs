import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {declaredReturnSelectors,validateReturnContribution} from '../scripts/observed-workspace-evidence.mjs';
const hash=text=>createHash('sha256').update(text).digest('hex');
const output=value=>{const text=JSON.stringify(value);return {complete:true,bytes:Buffer.byteLength(text),hash:hash(text)};};
function fixture(kind='projection'){
 const value={answer:'wrong',diagnostics:'actual producer metadata'},actual=kind==='forwarded'?value:{answer:value.answer};
 const records=new Map([
  ['root',{call_id:'root',parent_call_id:null,executor:{kind:'crisp'},outcome:'done',definition:{name:'compose',source:'compose.ts'},output:output(actual)}],
  ['producer',{call_id:'producer',parent_call_id:'root',executor:{kind:'agent'},outcome:'done',definition:{name:'judge',source:'judge.nl'},output:output(value),model_turns:[{kind:'generation',reply:{}}]}],
 ]);
 const valueJson=JSON.stringify(value),rootJson=JSON.stringify(actual);
 const selector={definition:structuredClone(records.get('producer').definition),valueJson,sha256:hash(valueJson),required:false,
 contribution:{schema:'natlang.application-return-contribution/1',kind,root_call_id:'root',producer_call_id:'producer',root_value_json:rootJson,root_sha256:hash(rootJson),root_path:kind==='forwarded'?[]:['answer'],producer_path:kind==='forwarded'?[]:['answer'],checks:['answerCorrect'],lineage:['root','producer']}};
 const proof={actualResult:actual,outcomeDimensions:{returnedValue:{checks:{answerCorrect:false}},incidentalValues:{passed:false,checks:{},incorrectReturns:[selector]}}};
 return {selector,proof,records};
}
const check=({selector,proof,records})=>{assert.deepEqual(declaredReturnSelectors(proof),[selector]);validateReturnContribution(selector,records,['root']);};
test('projected sampled return binds actual producer whole output and explicit crisp ancestry',()=>{
 check(fixture());
 for(const corrupt of [
  f=>{f.selector.contribution.root_path=['absent'];},
  f=>{f.selector.contribution.producer_path=['diagnostics'];},
  f=>{f.selector.contribution.producer_call_id='foreign';},
  f=>{f.selector.contribution.root_call_id='foreign';},
  f=>{f.selector.contribution.checks=['nativeWriteCorrect'];},
  f=>{f.selector.contribution.root_sha256='0'.repeat(64);},
  f=>{f.selector.contribution.lineage=['root','foreign','producer'];},
  f=>{f.records.get('root').executor.kind='agent';},
  f=>{f.records.get('producer').parent_call_id='foreign';},
  f=>{f.records.get('producer').executor.kind='crisp';},
  f=>{f.records.get('producer').model_turns=[];},
  f=>{f.records.get('producer').definition.source='wrong.nl';},
  f=>{f.records.get('producer').output=output('wrong');},
 ]){const f=fixture();corrupt(f);assert.throws(()=>check(f));}
});
test('whole-return forwarding admits only a sole child and exact whole-output identity',()=>{
 check(fixture('forwarded'));
 const siblings=fixture('forwarded');siblings.records.set('sibling',{call_id:'sibling',parent_call_id:'root'});assert.throws(()=>check(siblings),/sole child/);
 const projected=fixture();projected.selector.contribution.kind='forwarded';assert.throws(()=>check(projected));
});
test('root contribution preserves existing actual-root semantics without guessing equal branches',()=>{
 const f=fixture();const c=f.selector.contribution;c.kind='root';c.producer_call_id='root';c.lineage=['root'];c.root_path=[];c.producer_path=[];
 f.selector.definition=f.records.get('root').definition;f.selector.valueJson=c.root_value_json;f.selector.sha256=c.root_sha256;check(f);
 c.producer_path=['answer'];assert.throws(()=>check(f));
});
test('legacy return selectors retain their original containment validation',()=>{
 const f=fixture('forwarded');delete f.selector.contribution;check(f);
 const wrong=fixture();delete wrong.selector.contribution;assert.throws(()=>check(wrong),/outside the actual application result/);
});
