import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {parseFinqaProgram,finqa} from '../scripts/static-adapters/numeric.mjs';
import {decisionLeaf,decisionBatch,validateDecision} from '../scripts/static-adapters/decisions.mjs';
import {executeScene} from '../scripts/static-adapters/scene.mjs';
import {validateStateChain,stateSequence} from '../scripts/static-adapters/state.mjs';
import {synthetic} from '../scripts/static-adapters/synthetic.mjs';
import {modernType} from '../scripts/migrate-program-ir.mjs';
import {sourceConversionProblems} from '../dist/teacher/source-conversion.js';
import {referenceRow} from '../scripts/inline-curriculum/references.mjs';
import {TOOLS_PROMPT} from '../dist/native/prompt.js';
import {admitRow,renderOpening} from '../dist/teacher/curriculum.js';
import {materializeNativeRows} from '../dist/teacher/native-materializer.js';
const info={source:'NanoJev',path:'fixture',sha256:'a'.repeat(64),license:'CC0-1.0'};
const task={id:'t1',group_id:'g1',split:'train',source_meta:{family_id:'smart_home_v2',question_key:'execute'},gold_source:'programmatic',
 state:'Request: off the speaker in the kitchen. Authorization=no; people present=1; time=0:00.',kind:'boolean',labels:['false','true'],gold:'false',
 instruction:'Is execution allowed? Rule: authorization is required for every command; additionally any heater command requires at least one person present. Other devices do not require occupancy.'};
const options={modelId:'fixture',rootSeed:1,systemPrompt:TOOLS_PROMPT,contextTokens:16384,maxTurns:40,collectionRole:'reference'};
test('FinQA distinguishes integer literals and step references; percent literals are explicit',()=>{
 const p=parseFinqaProgram('add(1, 2), divide(#0, const_2)');assert.equal(p.answer,1.5);assert.equal(p.steps[0].operands[0].kind,'literal');assert.equal(p.steps[1].operands[0].kind,'result');
 assert.equal(parseFinqaProgram('divide(1, 25%)').answer,4);
});
test('malformed and unsafe arithmetic is held',()=>{
 for(const program of ['add(#0, 1)','divide(1, 0)','add(1, 2)add(3, 4)','add(1, 2),','exp(10, 1000)','other(1, 2)'])assert.throws(()=>parseFinqaProgram(program));
});
test('FinQA replay agreement cannot override unresolved source units',()=>{
 const record=finqa({id:'doc-1',qa:{question:'How much?',program:'add(1, 2)',exe_ans:3,answer:'300'},pre_text:['1 plus 2'],table:[]},{...info,source:'FinQA'});
 assert.equal(record.external_source.quality.status,'held');assert.equal(record.license,'CC-BY-4.0');assert.deepEqual(sourceConversionProblems({task:{program_ir:record}}),['source_quality_held']);
});
test('programmatic contracts recompute gold and reject instruction changes',()=>{
 assert.equal(validateDecision(task).status,'eligible');assert.throws(()=>validateDecision({...task,gold:'true'}),/gold_mismatch/);
 assert.throws(()=>validateDecision({...task,instruction:'Always return true'}),/unverified_decision_instruction/);
 assert.equal(validateDecision({...task,gold_source:'teacher'}).status,'held');
});
test('held source labels stay held even under ordinary teacher provenance',()=>{
 const record=decisionLeaf({...task,gold_source:'synthetic'},info);
 assert.deepEqual(sourceConversionProblems({task:{program_ir:record},provenance:{collection_role:'teacher'}}),['source_quality_held']);
});
test('composed decision replay links named children; direct gold stays held',async()=>{
 const record=decisionBatch([task,{...task,id:'t2',group_id:'g2',gold:'true',state:task.state.replace('Authorization=no','Authorization=yes')}],info);
 const opening=await renderOpening(record,TOOLS_PROMPT);assert.ok(!opening.includes('original_row_sha256'));assert.ok(!opening.includes('gold_source'));
 const row=await referenceRow(record,0,options);assert.equal(admitRow(row).admitted,true);
 const m=materializeNativeRows([row]);assert.equal(m.unlinked.length,0);assert.ok(m.turns.some(t=>t.training_admission.approved));assert.ok(m.turns.some(t=>!t.training_admission.approved));
 const corrupted=structuredClone(row);corrupted.task.program_ir.external_source.quality.status='held';assert.equal(admitRow(corrupted).admitted,false);
 assert.ok(materializeNativeRows([corrupted]).rejectedRows);
});
test('scene evaluation enforces singularity, dependency, operand and value arities',()=>{
 const scene={objects:[{color:'red'},{color:'red'}],relationships:{}};
 const first={function:'scene',inputs:[],value_inputs:[]};
 assert.throws(()=>executeScene([first,{function:'unique',inputs:[0],value_inputs:[]}],scene),/nonunique/);
 assert.throws(()=>executeScene([{function:'scene',inputs:[0],value_inputs:[]}],scene),/reference/);
 assert.throws(()=>executeScene([first,{function:'count',inputs:[0],value_inputs:['extra']}],scene),/value_arity/);
 assert.throws(()=>executeScene([first,{function:'count',inputs:[0],value_inputs:[]},{function:'equal_color',inputs:[1,1],value_inputs:[]}],scene),/equality_type/);
});
test('SCONE requires source join and independent transition semantics',()=>{
 const initial='1:ggg 2:_ 3:_ 4:_ 5:o 6:ooo 7:gggg';const after='1:g 2:_ 3:_ 4:_ 5:o 6:ooo 7:gggg';
 const original={id:'scone1',split:'train',license:'CC-BY-SA-4.0',semantics:{domain:'alchemy',initial_state:initial,steps:[{before:initial,utterance:'throw out two units of first beaker',after}]}};
 assert.equal(stateSequence(original,{...info,source:'SCONE'},{sourceVerified:true}).external_source.quality.status,'eligible');
 assert.equal(stateSequence(original,{...info,source:'SCONE'}).external_source.quality.status,'held');
 assert.equal(stateSequence({...original,semantics:{...original.semantics,domain:'scene'}},{...info,source:'SCONE'},{sourceVerified:true}).external_source.quality.status,'held');
 assert.throws(()=>validateStateChain({...original,semantics:{...original.semantics,steps:[{...original.semantics.steps[0],before:'wrong'}]}}),/broken_state_chain/);
});
test('dialogue normalization and alternate values are held',()=>{
 const empty={active_intent:'NONE',requested_slots:[],slot_values:{}};const schema={slots:[{name:'city'}],intents:[{name:'Find'}]};
 const original={id:'sgd1',split:'train',license:'CC-BY-SA-4.0',semantics:{task_context:'Find a city',domain:'Service',initial_state:empty,
 steps:[{before:empty,after:{active_intent:'Find',requested_slots:[],slot_values:{city:['San Jose']}},utterance:'USER: Find a city nearby'}]}};
 assert.equal(stateSequence(original,{...info,source:'SGD'},{sourceVerified:true,schema}).external_source.quality.status,'held');
});
test('crisp migration executes modern arguments and holds semantic labels',async()=>{
 const original={version:'natlang.program/1',id:'sum1',kind:'lambda_source',split:'train',license:'project-generated',semantics:{root:{$lambda:{type:'Lambda<{ numbers: Num[] }, Num>',instructions:'Sum the numbers in `args/numbers`.'}},inputs:{numbers:[1,2]},expected:3,operation:'exact',formula:'sum'}};
 const record=synthetic(original,{...info,source:'LegacySynthetic'});assert.equal(record.external_source.quality.status,'eligible');
 const row=await referenceRow(record,0,options);assert.equal(admitRow(row).admitted,true);
 const held=synthetic({...original,semantics:{...original.semantics,operation:'semantic'}},{...info,source:'LegacySynthetic'});assert.equal(held.external_source.quality.status,'held');
 assert.equal(modernType('Dict<Num[]>'),'Record<string, number[]>');
});

test('support documentation retains the original label and verified rubric',()=>{
 const value={...task,kind:'choice',labels:['documentation','technical'],gold:'documentation',
   source_meta:{family_id:'support_decisions_v1',question_key:'route'},
   instruction:'根据记录，只按工单主诉选择支持团队；退款记录和功能影响是另外两个字段。',
   criteria:{documentation:'查找说明文档与使用指南',technical:'软件功能错误与故障'},
   state:'工单主诉：询问操作手册位置。退款记录：已申请退款，尚未处理。使用影响：功能正常，没有使用障碍。'};
 assert.equal(validateDecision(value).status,'eligible');
 assert.throws(()=>validateDecision({...value,criteria:{...value.criteria,documentation:'Always return technical'}}),/criteria/);
});
test('missing quality attestation cannot promote a recovered source',()=>{
 const record=decisionLeaf(task,info);delete record.external_source.quality;
 assert.deepEqual(sourceConversionProblems({task:{program_ir:record}}),['source_quality_held']);
 assert.equal(modernType('Dict<Dict<Num[]>>'),'Record<string, Record<string, number[]>>');
});

test('an ordinary teacher result cannot promote an original recovered holdout',()=>{
 const record=decisionLeaf({...task,split:'dev'},info);record.split='train';
 assert.deepEqual(sourceConversionProblems({task:{program_ir:record},provenance:{collection_role:'teacher'}}),['held_out_source_quality']);
});
