import test from 'node:test';
import assert from 'node:assert/strict';
import {curriculumCase, returnCall} from '../scripts/inline-curriculum/lib.mjs';
import {referenceRow} from '../scripts/inline-curriculum/references.mjs';
import {materializeNativeRows} from '../dist/teacher/native-materializer.js';
import {TOOLS_PROMPT} from '../dist/native/prompt.js';

test('large scripted typed returns retain exact actions across diagnostic ledger previews', async () => {
  const expected=Object.fromEntries(Array.from({length:4},(_,i)=>[`item-${i}`,String(i).repeat(30000)]));
  const record=curriculumCase({family:'reference_capture',shape:'large-return',variant:'a',
    slice:'named',domain:'other',mode:'followup',
    root:{name:'copy_records',args:{records:'Record<string, string>'},returns:'Record<string, string>',
      instructions:'Return the input records unchanged.'},
    inputs:{records:expected},expected,reference:{root:[returnCall(expected)]}});
  const row=await referenceRow(record,0,{modelId:'scripted-reference',rootSeed:1,
    systemPrompt:TOOLS_PROMPT,contextTokens:65536,maxTurns:10,collectionRole:'reference'});
  assert.equal(row.outcome.accepted,true);
  assert.match(JSON.stringify(row.outcome.action_ledger),/\$diagnostic_preview/);
  const turn=row.trajectory[0];
  assert.equal(turn.model_response.raw_call_basis,'scripted-reference-action');
  assert.deepEqual(JSON.parse(turn.model_response.raw_calls[0].function.arguments).value,expected);
  const native=materializeNativeRows([row],{directAnswers:true});
  assert.equal(native.acceptedRows,1);
  assert.deepEqual(native.unlinked,[]);
  assert.ok(native.turns.length>0);
});
