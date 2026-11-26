import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {declaredReturnSelectors,validateReturnContribution,validateReturnContributions,validateUnreviewedReturnCoverage} from '../scripts/observed-workspace-evidence.mjs';
const hash=text=>createHash('sha256').update(text).digest('hex');
const output=value=>{const text=JSON.stringify(value);return {complete:true,bytes:Buffer.byteLength(text),hash:hash(text)};};
function fixture(kind='projection'){
 const value={answer:'wrong',diagnostics:'actual producer metadata'},actual=kind==='forwarded'?value:{answer:value.answer};
 const records=new Map([
  ['root',{call_id:'root',parent_call_id:null,executor:{kind:'crisp'},outcome:'done',definition:{name:'compose',source:'compose.ts'},output:output(actual)}],
  ['producer',{call_id:'producer',parent_call_id:'root',executor:{kind:'agent',model_id:'offline-reference'},cost:{model_requests:1},outcome:'done',definition:{name:'judge',source:'judge.nl'},output:output(value),model_turns:[{kind:'generation',reply:{}}]}],
 ]);
 const valueJson=JSON.stringify(value),rootJson=JSON.stringify(actual);
 const selector={definition:structuredClone(records.get('producer').definition),valueJson,sha256:hash(valueJson),required:false,
 contribution:{schema:'natlang.application-return-contribution/1',kind,root_call_id:'root',producer_call_id:'producer',root_value_json:rootJson,root_sha256:hash(rootJson),root_path:kind==='forwarded'?[]:['answer'],producer_path:kind==='forwarded'?[]:['answer'],checks:['answerCorrect'],lineage:['root','producer']}};
 if(kind==='projection')Object.assign(selector.contribution,{producer_source_suffix:'judge.nl',candidates:[{call_id:'producer',value_json:valueJson,sha256:hash(valueJson)}]});
 const proof={actualResult:actual,outcomeDimensions:{returnedValue:{passed:false,checks:{answerCorrect:false}},incidentalValues:{passed:false,checks:{},incorrectReturns:[selector]}}};
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
test('projection witnesses use JSON keys and indices, never JS coercions or array properties',()=>{
 for(const [value,path,actual] of [[{answer:['wrong']},['answer','length'],{answer:1}],[{answer:['wrong']},['answer','0'],{answer:'wrong'}],[{answer:{'0':'wrong'}},['answer',0],{answer:'wrong'}]]){
  const f=fixture(),text=JSON.stringify(value),rootText=JSON.stringify(actual);
  f.proof.actualResult=actual;f.records.get('root').output=output(actual);f.records.get('producer').output=output(value);
  f.selector.valueJson=text;f.selector.sha256=hash(text);f.selector.contribution.candidates=[{call_id:'producer',value_json:text,sha256:hash(text)}];Object.assign(f.selector.contribution,{producer_path:path,root_value_json:rootText,root_sha256:hash(rootText)});
  assert.throws(()=>check(f),/projection/);
 }
 const valid=fixture(),value={answer:['wrong']},text=JSON.stringify(value);valid.selector.valueJson=text;valid.selector.sha256=hash(text);valid.selector.contribution.producer_path=['answer',0];valid.selector.contribution.candidates=[{call_id:'producer',value_json:text,sha256:hash(text)}];valid.records.get('producer').output=output(value);check(valid);
});

test('projection uniqueness rechecks all actual candidate outputs and conservatively holds ambiguous returns',()=>{
 const f=fixture(),siblingValue={answer:'wrong',diagnostics:'second actual result'},text=JSON.stringify(siblingValue);
 f.records.set('sibling',{...structuredClone(f.records.get('producer')),call_id:'sibling',output:output(siblingValue)});
 f.selector.contribution.candidates.push({call_id:'sibling',value_json:text,sha256:hash(text)});
 assert.throws(()=>check(f),/ambiguous/);
 const second=structuredClone(f.selector);second.valueJson=text;second.sha256=hash(text);second.contribution.producer_call_id='sibling';second.contribution.lineage=['root','sibling'];
 for(const item of [f.selector,second])Object.assign(item.contribution,{kind:'uncertain-projection',uncertainty:'ambiguous-actual-producer'});
 f.proof.outcomeDimensions.incidentalValues.incorrectReturns=[f.selector,second];
 const selected=declaredReturnSelectors(f.proof);validateReturnContributions(selected,f.records,['root']);
 assert.throws(()=>validateReturnContributions([f.selector],f.records,['root']),/omitted/);
 for(const item of selected)item.contribution.candidates.pop();
 assert.throws(()=>validateReturnContributions(selected,f.records,['root']),/inventory/);
});
test('unique projection retains a sibling with a different projected result; unbound returns hold all possible producers',()=>{
 const f=fixture(),good={answer:'good'},text=JSON.stringify(good);
 f.records.set('sibling',{...structuredClone(f.records.get('producer')),call_id:'sibling',output:output(good)});
 f.selector.contribution.candidates.push({call_id:'sibling',value_json:text,sha256:hash(text)});check(f);
 const actual={answer:'unbound'},rootText=JSON.stringify(actual);f.proof.actualResult=actual;f.records.get('root').output=output(actual);
 Object.assign(f.selector.contribution,{kind:'uncertain-projection',uncertainty:'no-verified-actual-producer',root_value_json:rootText,root_sha256:hash(rootText)});
 const second=structuredClone(f.selector);second.valueJson=text;second.sha256=hash(text);second.contribution.producer_call_id='sibling';second.contribution.lineage=['root','sibling'];
 f.proof.outcomeDimensions.incidentalValues.incorrectReturns=[f.selector,second];validateReturnContributions(declaredReturnSelectors(f.proof),f.records,['root']);
 assert.throws(()=>validateReturnContributions([f.selector],f.records,['root']),/omitted/);
});

test('optional returned-value masks work without inventing an incidental outcome role',()=>{
 const f=fixture();delete f.proof.outcomeDimensions.incidentalValues;
 f.proof.outcomeDimensions.returnedValue.incorrectReturns=[f.selector];check(f);
});

test('independently checked wrong incidental child binds actual bytes without a false root projection',()=>{
 const f=fixture();f.selector.contribution.kind='incidental-child';f.selector.contribution.root_path=[];f.selector.contribution.producer_path=[];
 f.proof.outcomeDimensions.incidentalValues.checks={answerCorrect:false};f.proof.outcomeDimensions.returnedValue={passed:true,checks:{}};
 f.records.get('root').executor={kind:'agent',model_id:'offline-reference'};check(f);
 f.proof.outcomeDimensions.incidentalValues.checks={};f.proof.outcomeDimensions.returnedValue={passed:false,checks:{answerCorrect:false}};
 assert.throws(()=>check(f),/incidental checks/);
});
function uncheckedFixture(){
 const f=fixture(),c=f.selector.contribution;
 f.records.get('root').executor={kind:'agent',model_id:'offline-reference'};
 Object.assign(f.selector,{trainingEvidence:'format-only'});
 f.proof={actualResult:f.proof.actualResult,outcomeContract:{returnedValue:{required:false,trainingEvidence:'format-only'}},outcomeDimensions:{returnedValue:{required:false,passed:true,checks:{answerCorrect:true},trainingEvidence:'format-only',unreviewedReturns:[f.selector]}}};
 const rootSelector={...structuredClone(f.selector),definition:f.records.get('root').definition,valueJson:c.root_value_json,sha256:c.root_sha256,contribution:{...structuredClone(c),kind:'root',producer_call_id:'root',root_path:[],producer_path:[],lineage:['root']}};
 f.proof.outcomeDimensions.returnedValue.unreviewedReturns.unshift(rootSelector);
 return f;
}
test('unreviewed format-only holds bind optional original outputs without claiming they are wrong',()=>{
 const f=uncheckedFixture(),selectors=declaredReturnSelectors(f.proof);
 assert.equal(f.proof.outcomeDimensions.returnedValue.passed,true);assert.equal(selectors.length,2);
 validateReturnContributions(selectors,f.records,['root']);
 for(const corrupt of [f=>{f.proof.outcomeDimensions.returnedValue.required=true;},f=>{delete f.proof.outcomeContract.returnedValue.trainingEvidence;},f=>{f.selector.contribution.checks=['nativeWrite'];},f=>{f.selector.contribution.root_sha256='0'.repeat(64);},f=>{f.records.get('producer').output=output('other');},f=>{f.proof.outcomeDimensions.returnedValue.unreviewedReturns.shift();}]){
  const bad=uncheckedFixture();corrupt(bad);assert.throws(()=>validateReturnContributions(declaredReturnSelectors(bad.proof),bad.records,['root']));
 }
});
test('nonnull format-only returns require complete holds while null and native receipts do not',()=>{
 assert.throws(()=>declaredReturnSelectors({actualResult:'unsupported',outcomeDimensions:{returnedValue:{required:false,passed:true,trainingEvidence:'format-only',checks:{answer:true}}}}),/root training hold/);
 assert.deepEqual(declaredReturnSelectors({actualResult:null,outcomeDimensions:{returnedValue:{required:false,passed:true,trainingEvidence:'format-only',checks:{answer:true}}}}),[]);
 assert.deepEqual(declaredReturnSelectors({actualResult:'independently bound receipt',outcomeDimensions:{returnedValue:{required:false,passed:true,checks:{receipt:true}}}}),[]);
});

test('a missing or claimed null optional readout cannot hide the actual nonnull root',()=>{
 const f=uncheckedFixture(),proof={actualResult:null,outcomeDimensions:{returnedValue:{required:false,trainingEvidence:'format-only'}}};
 assert.throws(()=>validateUnreviewedReturnCoverage([proof],f.records),/actual captured root/);
 f.records.get('root').output=output(null);assert.doesNotThrow(()=>validateUnreviewedReturnCoverage([proof],f.records));
});
