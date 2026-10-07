import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ChildResultIndexBuilder, childCallIds, childFunctionNames, childReturn, convertTrajectory, instructionsDigest, invocationOf, printedResults } from '../dist/compiler/neuralese-conversion.js';
import { COMPACTION_NOTICE, GENERATION_GUIDANCE, HANDOVER_NOTE_CLOSE, HANDOVER_NOTE_OPEN, TOOLS_PROMPT } from '../dist/native/prompt.js';
import { programGuidance } from '../dist/adaptation/prompts.js';

const note = 'Checked lines 1-3; line 2 is a fee with no PO line. Left: prior invoices.';
const compact = (id, text) => ({ role: 'assistant', content: '', tool_calls: [{ id, type: 'function',
  function: { name: 'compact_history', arguments: JSON.stringify({ note: text }) } }] });
const record = () => ({ id: 'r1', messages: [
  { role: 'system', content: TOOLS_PROMPT + GENERATION_GUIDANCE + programGuidance('Prefer the contract over emails.') },
  { role: 'user', content: 'You are inside this call: judge(state: unknown): boolean\n\nInstructions:\nIs line 2 a fee?\n\nIn eval you can use state.' },
  { role: 'user', content: HANDOVER_NOTE_OPEN + note + HANDOVER_NOTE_CLOSE },
  compact('c1', note),
  { role: 'tool', tool_call_id: 'c1', content: 'Compacted.' },
  { role: 'assistant', content: '', tool_calls: [{ id: 'e1', type: 'function', function: { name: 'eval',
    arguments: JSON.stringify({ code: 'const v = await nl<boolean>`Is it a fee?`(state);' }) } }] },
  { role: 'tool', tool_call_id: 'e1', content: 'console:\ntrue' + COMPACTION_NOTICE },
], target: compact('c2', 'Line 2 is a fee. Return true.') });

test('prompts, guidance and handover notes become Neuralese; the rest is counted', () => {
  const { record: out, pieces } = convertTrajectory(record());
  const system = out.messages[0].content;
  assert.deepEqual(system.filter(p => p.type === 'soft').map(p => p.name).slice(0, 2), ['prompt:interpreter', 'prompt:generation-guidance']);
  assert.ok(system.some(p => p.type === 'soft' && p.name.startsWith('guidance@')), 'program guidance is its own soft parameter');
  assert.ok(!JSON.stringify(system).includes('You are running one call'));

  const pinned = out.messages[2].content;
  assert.deepEqual(pinned.map(p => p.type), ['soft', 'read', 'soft']);
  const write = JSON.parse(out.messages[3].tool_calls[0].function.arguments).note.$write;
  assert.equal(write.name, pinned[1].name, 'the pinned note reads the block the compaction call writes');
  assert.equal(write.source, note);
  assert.equal(write.type, 'Neuralese<HandoverNote>');
  assert.ok(JSON.parse(out.target.tool_calls[0].function.arguments).note.$write, 'a compaction target is a write');

  assert.ok(out.messages[6].content.some(p => p.type === 'soft' && p.name === 'prompt:compaction-notice'));
  assert.equal(typeof out.messages[1].content, 'string', 'single-use instructions stay text');
  const { sites } = out.neuralese_conversion;
  assert.equal(sites['handover-write'].converted, 2);
  assert.equal(sites['handover-read'].converted, 1);
  assert.equal(sites['nl-literal'].exact['later-curriculum-step'], 1);
  assert.equal(sites['tool-output'].exact['single-use'], 1);
  assert.equal(sites['child-result'].exact['producer-missing'], 1, 'without the run\'s child returns a printed result stays exact');
  assert.equal(sites.instructions.exact['single-use'], 1);
  assert.ok(pieces.some(p => p.name === 'prompt:interpreter' && p.text === TOOLS_PROMPT), 'pieces carry their initial text');
});

test('instructions used by several calls become one shared soft parameter; unregistered system text is versioned', () => {
  const input = record();
  input.messages[0].content = 'An older runtime prompt.';
  const digest = instructionsDigest('Is line 2 a fee?');
  const { record: out, pieces } = convertTrajectory(input, { instructionCalls: new Map([[digest, 3]]), instructionsShare: 0 });
  assert.match(out.messages[0].content[0].name, /^prompt:system@[0-9a-f]{12}$/);
  const body = out.messages[1].content.find(p => p.type === 'soft');
  assert.equal(body.name, `instructions@${digest}`);
  assert.equal(pieces.find(p => p.name === body.name).text, 'Is line 2 a fee?');
  assert.equal(out.neuralese_conversion.sites['instructions-reused'].converted, 1);
  const single = convertTrajectory(record(), { instructionCalls: new Map([[digest, 1]]), instructionsShare: 1 }).record;
  assert.equal(single.neuralese_conversion.sites['instructions-coverage'].converted, 1, 'a share of single-use instructions converts for coverage');
});

test('a cut-off value in the root call\'s opening listing becomes a digest site of the full value', () => {
  const full = { task: 'Review the packet.', lines: Array.from({ length: 50 }, (_, i) => ({ id: `L${i}`, amount: i * 10 })) };
  const input = { id: 'r3', task: { program_ir: { semantics: { root: 'judge.nl', inputs: { state: full } } } }, messages: [
    { role: 'system', content: TOOLS_PROMPT },
    { role: 'user', content: 'You are inside this call: judge(state: unknown): boolean\n\nInstructions:\nDecide.\n\nIn eval you can use state.' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'scope_0', type: 'function', function: { name: 'eval', arguments: '{"code":"const state = read_inputs().state;"}' } }] },
    { role: 'tool', tool_call_id: 'scope_0', content: 'state: unknown = { task: "Review the packet.", <<cut off: 1 of 2 fields not shown; state holds all of it>> }\nDeclared state for the rest of this call.' },
  ], target: { role: 'assistant', content: 'true' } };
  const { record: out } = convertTrajectory(input);
  const parts = out.messages[3].content;
  const digest = parts.find(p => p.type === 'digest');
  assert.deepEqual(JSON.parse(digest.source), full);
  assert.match(digest.preview, /cut off/);
  assert.equal(parts[0].text, 'state: unknown = ');
  assert.equal(out.neuralese_conversion.sites.digest.converted, 1);
  const nested = convertTrajectory({ ...input, task: undefined }).record;
  assert.equal(nested.neuralese_conversion.sites.digest.exact['full-value-unavailable'], 1);
});

test('a child call\'s returned value that its caller prints becomes a write in the child and a read in the caller', () => {
  const summary = 'Line 2 bills 7.5 hours of on-site work at the contract rate.';
  const run = { source_ref: { trajectory_id: 'run-1' }, task: { program_ir: { semantics: { root: 'judge.nl' } } } };
  const child = { ...run, id: 'child', messages: [
    { role: 'system', content: TOOLS_PROMPT },
    { role: 'user', content: 'You are inside this call: summarise(line: string): string\n\nInstructions:\nSummarise the line.\n\nIn eval you can use line.' },
  ], target: { role: 'assistant', content: '', tool_calls: [{ id: 'r1', type: 'function', function: { name: 'return_result',
    arguments: JSON.stringify({ status: 'success', value: summary }) } }] } };
  const caller = { ...run, id: 'caller', messages: [
    { role: 'system', content: TOOLS_PROMPT },
    { role: 'user', content: 'You are inside this call: judge(state: unknown): boolean\n\nInstructions:\nIs line 2 a fee?\n\nIn eval you can use state.' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'e1', type: 'function', function: { name: 'eval',
      arguments: JSON.stringify({ code: 'const s = await nl`Summarise the line.`(state.lines[1]); console.log(s)' }) } }] },
    { role: 'tool', tool_call_id: 'e1', content: `console:\n${summary}\nStored local s.` },
  ], target: { role: 'assistant', content: 'true' } };
  assert.equal(childReturn(child), summary);
  assert.equal(childReturn({ ...child, messages: [child.messages[0], caller.messages[1]] }), undefined, 'the root call returns no child result');
  const multipartRoot = { ...caller, target: { role: 'assistant', tool_calls: [{ id: 'root-return', function: {
    name: 'return_result', arguments: JSON.stringify({ status: 'success', value: summary }),
  } }] }, messages: [
    { role: 'user', content: [
      { type: 'text', text: 'You are inside this call: judge(state: unknown): string\n\nInstructions:\n' },
      { type: 'neuralese', id: 'root-body' },
      { type: 'text', text: '\n\nIn eval you can use state.' },
    ] },
  ] };
  assert.equal(childReturn(multipartRoot), undefined, 'multipart root openings still cannot become child producers');
  assert.deepEqual([...childCallIds(caller.messages)], ['e1']);
  const read = new Set(printedResults(caller.messages[3].content, [summary, 'true']));
  const childResults = new Map([['run-1', { returned: [summary, 'true'], read }]]);

  const written = convertTrajectory(child, { childResults }).record;
  const write = JSON.parse(written.target.tool_calls[0].function.arguments);
  assert.equal(write.status, 'success');
  assert.equal(write.value.$write.source, summary);
  assert.equal(write.value.$write.type, 'Neuralese<string>');

  const reading = convertTrajectory(caller, { childResults }).record;
  const parts = reading.messages[3].content;
  assert.deepEqual(parts.map(p => p.type), ['text', 'read', 'text']);
  assert.equal(parts[1].name, write.value.$write.name, 'the caller reads the block the child writes');
  assert.equal(parts.map(p => p.type === 'read' ? p.source : p.text).join(''), caller.messages[3].content);
  assert.equal(reading.neuralese_conversion.sites['child-result'].converted, 1);

  const short = convertTrajectory(caller, { childResults: new Map([['run-1', { returned: ['true'], read: new Set() }]]) }).record;
  assert.equal(short.neuralese_conversion.sites['child-result'].exact['crisp-value'], 1, 'a short value is its exact form');
});

test('a parent eval console log links an exact returned child array before the parent aggregates it', () => {
  const value = '["ARC-201","ARC-215"]';
  const run = 'folder-reducer-run';
  const producer = { id: 'teacher:decision:0004', decision: { index: 4 },
    source_ref: { trajectory_id: run, invocation_id: 'folder-reducer', parent_invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl' } } },
    messages: [{ role: 'user', content: [
      { type: 'text', text: 'You are inside this call: soft@eval:1(folder: Folder): string[]\n\nInstructions:\n' },
      { type: 'neuralese', id: 'writer-body' },
      { type: 'text', text: '\n\nIn eval you can use folder.' },
    ] }],
    target: { role: 'assistant', tool_calls: [{ id: 'result', function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value: JSON.parse(value) }) } }] } };
  const caller = { id: 'teacher:decision:0007', decision: { index: 7 },
    source_ref: { trajectory_id: run, invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl' } } }, messages: [
      { role: 'user', content: 'You are inside this call: root(): string[]' },
      { role: 'assistant', tool_calls: [{ id: 'aggregate', function: { name: 'eval',
        arguments: JSON.stringify({ code: "const perFolder = await reduceFolders(); console.log('Per-folder reducer results', perFolder); return perFolder.flat();" }) } }] },
      { role: 'tool', tool_call_id: 'aggregate', content: `console:\nPer-folder reducer results ${JSON.stringify([JSON.parse(value)])}\nStaged ["ARC-201", "ARC-215"] as the result.` },
    ], target: { role: 'assistant', content: '["ARC-201","ARC-215"]' } };
  const index = new ChildResultIndexBuilder();
  index.add(caller);
  index.add(producer);
  const childResults = index.finish().get(run);
  assert.deepEqual(childResults.readers.map(({ tool_call_id, value: read }) => [tool_call_id, read]), [['aggregate', value]],
    'the eval console is a real parent observation and authoritative decision indexes establish producer order');
  const written = convertTrajectory(producer, { childResults: new Map([[run, childResults]]) }).record;
  const write = JSON.parse(written.target.tool_calls[0].function.arguments).value.$write;
  assert.equal(write.source, value);
  const reading = convertTrajectory(caller, { childResults: new Map([[run, childResults]]) }).record;
  assert.ok(reading.messages[2].content.some(part => part.type === 'read' && part.source === value));
});

test('eval output text alone does not link future, unrelated-parent, or ambiguous child returns', () => {
  const value = 'the exact returned value is visible here';
  const makeProducer = (id, index, invocation, parent = 'root') => ({ id, decision: { index },
    source_ref: { trajectory_id: 'same-run', invocation_id: invocation, parent_invocation_id: parent },
    task: { program_ir: { semantics: { root: 'root.nl' } } },
    messages: [{ role: 'user', content: `You are inside this call: child${index}(): string\n\nInstructions:\nReturn the value.\n\nIn eval you can use it.` }],
    target: { role: 'assistant', tool_calls: [{ id: `return-${id}`, function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value }) } }] } });
  const makeCaller = index => ({ id: `caller-${index}`, decision: { index },
    source_ref: { trajectory_id: 'same-run', invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl' } } }, messages: [
      { role: 'user', content: 'You are inside this call: root(): string' },
      { role: 'assistant', tool_calls: [{ id: 'root-eval', function: { name: 'eval',
        arguments: JSON.stringify({ code: 'console.log("unrelated computation");' }) } }] },
      { role: 'tool', tool_call_id: 'root-eval', content: `console:\n${value}` },
    ], target: { role: 'assistant', content: value } });
  const cases = [
    { name: 'future producer', callerIndex: 4, producers: [makeProducer('future', 7, 'child-future')] },
    { name: 'different parent', callerIndex: 7, producers: [makeProducer('elsewhere', 4, 'child-other', 'other-root')] },
    { name: 'ambiguous identical producers', callerIndex: 7, producers: [
      makeProducer('same-a', 4, 'child-a'), makeProducer('same-b', 5, 'child-b'),
    ] },
  ];
  for (const scenario of cases) {
    const index = new ChildResultIndexBuilder();
    const caller = makeCaller(scenario.callerIndex);
    index.add(caller);
    for (const producer of scenario.producers) index.add(producer);
    const childResults = index.finish().get('same-run');
    assert.deepEqual(childResults.readers, [], `${scenario.name} must stay unlinked`);
    const result = convertTrajectory(caller, { childResults: new Map([['same-run', childResults]]) }).record;
    assert.equal(result.messages[2].content, `console:\n${value}`, `${scenario.name} remains crisp text`);
    if (scenario.name === 'ambiguous identical producers')
      assert.equal(result.neuralese_conversion.sites['child-result'].exact['ambiguous-producer'], 1);
  }
});


test('a structured result\'s text field passed into another call is written field by field and read from its argument listing', () => {
  const facts = 'It was co-founded in 1973 by former astronaut Edgar Mitchell.';
  const run = { source_ref: { trajectory_id: 'run-2' }, task: { program_ir: { semantics: { root: 'research.nl' } } } };
  const opening = (signature, scope) => [{ role: 'system', content: TOOLS_PROMPT },
    { role: 'user', content: `You are inside this call: ${signature}\n\nInstructions:\nRead it.\n\nIn eval you can use x.` },
    { role: 'assistant', content: '', tool_calls: [{ id: 'scope_0', type: 'function', function: { name: 'eval', arguments: '{"code":"x"}' } }] },
    { role: 'tool', tool_call_id: 'scope_0', content: scope }];
  const judge = { ...run, id: 'judge', messages: opening('nl@eval:2(file: unknown): { relevant: boolean, facts: string }',
    `file: unknown = folder.file("a.md")\n  // File contents:\n"# IONS\\n\\nThe institute. ${facts} More."`),
    target: { role: 'assistant', content: '', tool_calls: [{ id: 'r', type: 'function', function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value: { relevant: true, facts } }) } }] } };
  const answer = { ...run, id: 'answer', messages: opening('nl@eval:8(notes: string[]): string', `notes: string[] = ["${facts}"]`),
    target: { role: 'assistant', content: '1973' } };
  const producer = { id: 'judge#facts', invocation: invocationOf(judge), field: 'facts', value: facts, renderings: [facts] };
  const childResults = new Map([['run-2', { returned: [JSON.stringify({ relevant: true, facts }), facts], read: new Set([facts]),
    producers: [producer], readers: [{ invocation: invocationOf(answer), value: facts, producer_id: producer.id }] }]]);
  assert.equal(invocationOf(judge), invocationOf({ ...judge, id: 'judge-turn-2' }), 'every turn of one call shares its invocation');
  assert.notEqual(invocationOf(judge), invocationOf(answer));

  const written = JSON.parse(convertTrajectory(judge, { childResults }).record.target.tool_calls[0].function.arguments);
  assert.equal(written.value.relevant, true, 'the branch field stays exact');
  assert.equal(written.value.facts.$write.source, facts);
  assert.equal(written.value.facts.$write.type, 'Neuralese<string>');

  const read = convertTrajectory(answer, { childResults }).record;
  const parts = read.messages[3].content;
  assert.deepEqual(parts.map(p => p.type), ['text', 'read', 'text']);
  assert.equal(parts[1].name, written.value.facts.$write.name, 'the consumer reads the block the producer writes');
  assert.equal(read.neuralese_conversion.sites['argument-read'].converted, 1);

  // The producer's own listing shows the quoted text inside its file, not as a passed value: no read.
  const own = convertTrajectory(judge, { childResults: new Map([['run-2', { ...childResults.get('run-2'),
    readers: [{ invocation: invocationOf(judge), value: facts, producer_id: producer.id }] }]]) }).record;
  assert.equal(typeof own.messages[3].content, 'string');
});

test('named file calls are recognized and equal returns in independent runs stay separate', () => {
  const value = 'A private observation supports the contractual claim.';
  const make = run => ({ source_ref: { trajectory_id: run }, task: { program_ir: { semantics: {
    root: 'folder/review.nl', files: { 'folder/review.nl': '', 'folder/review/assess.nl': '' } } } },
    messages: [{ role: 'user', content: 'You are inside this call: review(): string' },
      { role: 'assistant', tool_calls: [{ id: 'e1', function: { name: 'eval', arguments: JSON.stringify({ code: 'await assess()' }) } }] },
      { role: 'tool', tool_call_id: 'e1', content: value }] });
  const a = make('run-a'), b = make('run-b');
  assert.deepEqual([...childCallIds(a.messages, childFunctionNames(a))], ['e1']);
  const childResults = new Map(['run-a', 'run-b'].map(id => [id, { returned: [value], read: new Set([value]) }]));
  const read = row => convertTrajectory(row, { childResults }).record.messages.at(-1).content.find(p => p.type === 'read');
  assert.equal(read(a).source, value);
  assert.notEqual(read(a).name, read(b).name, 'equal text in different executions is not one producer');
  assert.equal(childReturn({ ...a, target: { role: 'assistant', tool_calls: [{ function: {
    name: 'return_result', arguments: JSON.stringify({ status: 'success', value }) } }] } }), undefined,
    'a root inside a folder is still a root');
});

test('invocation-indexed results preserve ambiguous equal returns as text and never turn a root into a writer', () => {
  const value = 'This sufficiently detailed conclusion is shared by two independent invocations.';
  const root = {id:'reader', source_ref:{trajectory_id:'run',invocation_id:'root'},
    messages:[{role:'user',content:'You are inside this call: root(): string'},
      {role:'assistant',tool_calls:[{id:'e',function:{name:'eval',arguments:JSON.stringify({code:'return await nl`Assess item.`();'})}}]},
      {role:'tool',tool_call_id:'e',content:value}],
    target:{role:'assistant',tool_calls:[{id:'r',function:{name:'return_result',arguments:JSON.stringify({status:'success',value})}}]}};
  const producers=[{id:'child-a',invocation:'a',value},{id:'child-b',invocation:'b',value}];
  const ambiguous=new Map([['run',{returned:[value],read:new Set(),producers}]]);
  const held=convertTrajectory(root,{childResults:ambiguous}).record;
  assert.equal(held.messages[2].content,value);
  assert.equal(held.neuralese_conversion.sites['child-result'].exact['ambiguous-producer'],1);
  const unique=new Map([['run',{returned:[value],read:new Set([value]),producers:producers.slice(0,1)}]]);
  const reader=convertTrajectory(root,{childResults:unique}).record;
  assert.equal(reader.messages[2].content[0].type,'read');
  assert.equal(JSON.parse(reader.target.tool_calls[0].function.arguments).value,value,'root cannot write a child result');
  const child=convertTrajectory({...root,id:'child-a',source_ref:{trajectory_id:'run',invocation_id:'a'}},{childResults:unique}).record;
  assert.equal(JSON.parse(child.target.tool_calls[0].function.arguments).value.$write.name,reader.messages[2].content[0].name);
});

test('observed parentage distinguishes equal returns along a nested chain', () => {
 const value='A long exact evidence record retained through a nested call chain.';
 const producers=[{id:'leaf',invocation:'leaf',parent:'middle',value},{id:'middle',invocation:'middle',parent:'root',value}];
 const readers=[{invocation:'middle',value,producer_id:'leaf'},{invocation:'root',value,producer_id:'middle'}];
 const childResults=new Map([['run',{returned:[value],read:new Set([value]),producers,readers}]]);
 const make=invocation=>({id:invocation,source_ref:{trajectory_id:'run',invocation_id:invocation},messages:[
   {role:'assistant',tool_calls:[{id:'e',function:{name:'eval',arguments:JSON.stringify({code:'return await nl`Read evidence.`();'})}}]},
   {role:'tool',tool_call_id:'e',content:value}],target:{role:'assistant',tool_calls:[{id:'r',function:{name:'return_result',arguments:JSON.stringify({status:'success',value})}}]}});
 const middle=convertTrajectory(make('middle'),{childResults}).record,root=convertTrajectory(make('root'),{childResults}).record;
 const written=JSON.parse(middle.target.tool_calls[0].function.arguments).value.$write;
 assert.equal(root.messages[1].content[0].name,written.name);
 assert.notEqual(middle.messages[1].content[0].name,written.name,'middle reads its child and writes its own distinct return');
 assert.equal(JSON.parse(root.target.tool_calls[0].function.arguments).value,value);
});

test('all exact printer renderings are replaced so a structured result cannot leak beside its block',()=>{
 const value='{"source":"packet_0","quote":"A current report establishes completion."}';
 const printed='{ source: "packet_0", quote: "A current report establishes completion." }';
 const producers=[{id:'child',invocation:'child',parent:'root',value,renderings:[printed]}];
 const childResults=new Map([['run',{returned:[value],read:new Set([value]),producers,readers:[{invocation:'root',value,producer_id:'child'}]}]]);
 const record={id:'root',source_ref:{trajectory_id:'run',invocation_id:'root'},messages:[
  {role:'assistant',tool_calls:[{id:'e',function:{name:'eval',arguments:JSON.stringify({code:'return await nl`Extract.`();'})}}]},
  {role:'tool',tool_call_id:'e',content:`console:\n${value}\nStaged ${printed} as the result.`}]};
 const result=convertTrajectory(record,{childResults}).record.messages[1].content;
 assert.equal(result.filter(p=>p.type==='read').length,2);
 assert.ok(!result.filter(p=>p.type==='text').map(p=>p.text).join('').includes('packet_0'));
});

test('child result indexing never links a future producer to an earlier scope listing', () => {
  const value = 'Which test summary proves that at least 85% of the 20 devices passed?';
  const run = 'causal-run';
  const scope = `objective: string = "${value}"`;
  const root = { id: 'root-at-5', decision: { index: 5 }, source_ref: { trajectory_id: run, invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl', files: { 'root.nl': '', 'child.nl': '' } } } }, messages: [
      { role: 'user', content: 'You are inside this call: root(): string' },
      { role: 'assistant', tool_calls: [{ id: 'scope_0', function: { name: 'eval', arguments: '{"code":"x"}' } }] },
      { role: 'tool', tool_call_id: 'scope_0', content: scope },
    ], target: { role: 'assistant', content: value } };
  const producer = { id: 'future-child', decision: { index: 11 },
    source_ref: { trajectory_id: run, invocation_id: 'child', parent_invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl', files: { 'root.nl': '', 'child.nl': '' } } } }, messages: [
      { role: 'user', content: 'You are inside this call: find(): string' },
    ], target: { role: 'assistant', tool_calls: [{ id: 'ret', function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value }) } }] } };
  const index = new ChildResultIndexBuilder();
  index.add(producer);
  index.add(root);
  const childResults = index.finish();
  assert.equal(childResults.get(run).read.size, 0);
  assert.equal(childResults.get(run).readers.length, 0);
  const convertedRoot = convertTrajectory(root, { childResults }).record;
  assert.equal(convertedRoot.messages[2].content, scope, 'the original scope input stays exact');
  const convertedProducer = convertTrajectory(producer, { childResults }).record;
  assert.equal(JSON.parse(convertedProducer.target.tool_calls[0].function.arguments).value, value,
    'an unread future result does not become a write');
});

test('child result indexing requires authoritative order when decision indexes are absent', () => {
  const value = 'A later child conclusion with enough detail to become a Neuralese value.';
  const run = 'unknown-order-run';
  const makeRoot = id => ({ id, source_ref: { trajectory_id: run, invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl' } } }, messages: [
      { role: 'user', content: 'You are inside this call: root(): string' },
      { role: 'assistant', tool_calls: [{ id: 'scope_0', function: { name: 'eval', arguments: '{"code":"x"}' } }] },
      { role: 'tool', tool_call_id: 'scope_0', content: `objective: string = "${value}"` },
    ], target: { role: 'assistant', content: value } });
  const makeChild = id => ({ id, source_ref: { trajectory_id: run, invocation_id: 'child', parent_invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl' } } }, messages: [
      { role: 'user', content: 'You are inside this call: find(): string' },
    ], target: { role: 'assistant', tool_calls: [{ id: 'ret', function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value }) } }] } });
  const unknown = new ChildResultIndexBuilder();
  unknown.add(makeChild('imported-child'));
  unknown.add(makeRoot('imported-root'));
  const noOrder = unknown.finish().get(run);
  assert.equal(noOrder.readers.length, 0, 'file/ingestion order cannot prove a recurrence edge');
  assert.equal(noOrder.read.size, 0);

  const indexed = new ChildResultIndexBuilder();
  indexed.add(makeChild('teacher:decision:0011'));
  indexed.add(makeRoot('teacher:decision:0005'));
  assert.equal(indexed.finish().get(run).readers.length, 0,
    'an unambiguous decision suffix is authoritative even when rows are shuffled');
  const causal = new ChildResultIndexBuilder();
  causal.add(makeRoot('teacher:decision:0005'));
  causal.add(makeChild('teacher:decision:0004'));
  assert.equal(causal.finish().get(run).readers.length, 1,
    'the suffix recovers a genuinely earlier producer despite reverse ingestion');
});

test('reader links use tool message IDs and choose whole-object blocks over overlapping field blocks', () => {
  const facts = 'Eighteen of twenty devices passed the receiving test at the required threshold.';
  const structured = { facts, status: 'complete' };
  const value = JSON.stringify(structured);
  const run = 'field-whole-run';
  const root = { id: 'root', decision: { index: 3 }, source_ref: { trajectory_id: run, invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl', files: { 'root.nl': '', 'child.nl': '' } } } }, messages: [
      { role: 'user', content: 'You are inside this call: root(): string' },
      { role: 'assistant', tool_calls: [
        { id: 'scope_0', function: { name: 'eval', arguments: '{"code":"x"}' } },
        { id: 'e_facts', function: { name: 'eval', arguments: '{"code":"await child()"}' } },
      ] },
      { role: 'tool', tool_call_id: 'scope_0', content: `evidence: unknown = "${value}"` },
      { role: 'tool', tool_call_id: 'e_facts', content: `console:\n${facts}` },
    ], target: { role: 'assistant', content: 'done' } };
  const child = { id: 'child', decision: { index: 1 },
    source_ref: { trajectory_id: run, invocation_id: 'child', parent_invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl', files: { 'root.nl': '', 'child.nl': '' } } } }, messages: [
      { role: 'user', content: 'You are inside this call: child(): Evidence' },
    ], target: { role: 'assistant', tool_calls: [{ id: 'ret', function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value: structured }) } }] } };
  const index = new ChildResultIndexBuilder();
  index.add(root);
  index.add(child);
  const childResults = index.finish().get(run);
  assert.deepEqual(childResults.readers.map(({ tool_call_id, value: read }) => [tool_call_id, read]), [['scope_0', value]],
    'the specific whole-object observation wins; the overlapping facts field is not an orphan read');
  const convertedChild = convertTrajectory(child, { childResults: new Map([[run, childResults]]) }).record;
  const result = JSON.parse(convertedChild.target.tool_calls[0].function.arguments).value;
  assert.ok(result.$write, 'the whole-object reader has a matching whole-object writer');
  assert.equal(result.$write.source, value, 'the whole-object writer owns the complete structured result');
  const convertedRoot = convertTrajectory(root, { childResults: new Map([[run, childResults]]) }).record;
  assert.ok(convertedRoot.messages[2].content.some(part => part.type === 'read'));
  assert.equal(convertedRoot.messages[3].content, `console:\n${facts}`, 'a separate tool message has no unrelated field link');
});
